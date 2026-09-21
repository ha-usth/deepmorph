"""
bench_efficiency.py - Model size, FLOPs, memory and inference speed.

Produces the numbers for the computational-efficiency subsection: parameter
counts, multiply-accumulate cost, peak inference memory, and per-image latency
on both GPU and CPU, for the heatmap framework and the regression ablation.

Timing methodology: CUDA is asynchronous, so every measured region is bracketed
by torch.cuda.synchronize(); warmup iterations are discarded so cuDNN autotuning
and lazy CUDA context creation do not land inside the measurement; and the
median over repeats is reported rather than the mean, which a single scheduling
hiccup would distort.

Usage:
    python bench_efficiency.py
    python bench_efficiency.py --image-size 512 --num-landmarks 15
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
import json
import argparse
import statistics
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.mae import mae_vit_base
from models.landmark_head import WingLandmarkModel
from models.regression_head import RegressionLandmarkModel


def build(head, image_size, num_landmarks, heatmap_size):
    mae = mae_vit_base(img_size=image_size)
    if head == 'heatmap':
        return WingLandmarkModel(
            mae_encoder=mae, num_landmarks=num_landmarks, embed_dim=768,
            patch_size=16, img_size=image_size, heatmap_size=heatmap_size,
            freeze_encoder=False).eval()
    return RegressionLandmarkModel(
        mae_encoder=mae, num_landmarks=num_landmarks, embed_dim=768,
        freeze_encoder=False).eval()


def count_params(model):
    """
    Split the encoder into the part inference actually uses and the dead weight.

    WingLandmarkModel wraps a whole MaskedAutoencoderViT, but only calls
    forward_features(), so the MAE decoder is carried in the checkpoint and
    never executed. Reporting the two separately keeps 'model size' honest:
    the released file is the larger number, the deployed model is the smaller.
    """
    used, dead = 0, 0
    for name, p in model.encoder.named_parameters():
        if name.startswith('decoder') or name == 'mask_token':
            dead += p.numel()
        else:
            used += p.numel()
    head = sum(p.numel() for p in model.head.parameters())
    return used, dead, head


@torch.no_grad()
def count_flops(model, image_size, device):
    """MACs for one forward pass. FlopCounterMode reports MACs*2 for matmuls."""
    x = torch.randn(1, 3, image_size, image_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    return counter.get_total_flops()


@torch.no_grad()
def latency(model, image_size, device, batch=1, warmup=10, repeats=50):
    """Median per-image latency in ms, plus throughput in images/s."""
    x = torch.randn(batch, 3, image_size, image_size, device=device)
    cuda = device.startswith('cuda')

    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    per_batch = statistics.median(samples)
    return per_batch / batch, batch / (per_batch / 1000.0)


@torch.no_grad()
def peak_memory(model, image_size, device, batch=1):
    if not device.startswith('cuda'):
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model(torch.randn(batch, 3, image_size, image_size, device=device))
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--heatmap-size', type=int, default=128)
    p.add_argument('--num-landmarks', type=int, default=15)
    p.add_argument('--batch', type=int, default=8,
                   help="batch used for the throughput row")
    p.add_argument('--repeats', type=int, default=50)
    p.add_argument('--out', default='./results/efficiency.json')
    args = p.parse_args()

    devices = ['cuda', 'cpu'] if torch.cuda.is_available() else ['cpu']
    print("=== Hardware ===")
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")
    print(f"  torch {torch.__version__}, threads={torch.get_num_threads()}")
    print(f"  input {args.image_size}x{args.image_size}, "
          f"K={args.num_landmarks}\n")

    report = {'image_size': args.image_size,
              'num_landmarks': args.num_landmarks,
              'gpu': (torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None),
              'torch': torch.__version__, 'heads': {}}

    for head in ('heatmap', 'regression'):
        model = build(head, args.image_size, args.num_landmarks,
                      args.heatmap_size)
        used, dead, hd = count_params(model)
        entry = {'params_encoder_used_M': used / 1e6,
                 'params_decoder_unused_M': dead / 1e6,
                 'params_head_M': hd / 1e6,
                 'params_deployed_M': (used + hd) / 1e6,
                 'params_checkpoint_M': (used + dead + hd) / 1e6,
                 'devices': {}}

        print(f"--- {head} head ---")
        print(f"  deployed   : encoder {used/1e6:.1f}M + head {hd/1e6:.2f}M "
              f"= {(used+hd)/1e6:.1f}M")
        print(f"  checkpoint : + {dead/1e6:.1f}M unused MAE decoder "
              f"= {(used+dead+hd)/1e6:.1f}M")

        for dev in devices:
            model = model.to(dev)
            flops = count_flops(model, args.image_size, dev)
            ms1, ips1 = latency(model, args.image_size, dev, 1,
                                repeats=args.repeats if dev == 'cuda' else 10)
            mem = peak_memory(model, args.image_size, dev, 1)
            row = {'gflops': flops / 1e9, 'ms_per_image_batch1': ms1,
                   'img_per_s_batch1': ips1, 'peak_mem_gb_batch1': mem}

            if dev == 'cuda':
                msb, ipsb = latency(model, args.image_size, dev, args.batch,
                                    repeats=args.repeats)
                row.update({'batch': args.batch, 'ms_per_image_batched': msb,
                            'img_per_s_batched': ipsb,
                            'peak_mem_gb_batched': peak_memory(
                                model, args.image_size, dev, args.batch)})

            entry['devices'][dev] = row
            print(f"  {dev:<5}: {flops/1e9:7.1f} GFLOPs   "
                  f"{ms1:7.2f} ms/img   {ips1:6.1f} img/s"
                  + (f"   peak {mem:.2f} GB" if mem else ""))
            if dev == 'cuda':
                print(f"         batched x{args.batch}: {msb:.2f} ms/img, "
                      f"{ipsb:.1f} img/s, peak {row['peak_mem_gb_batched']:.2f} GB")

        report['heads'][head] = entry
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Written to {args.out}")


if __name__ == '__main__':
    main()
