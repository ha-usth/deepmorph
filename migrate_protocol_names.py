"""
migrate_protocol_names.py - one-off rename of the Stage 1 protocol labels.

    P1 -> PA   (full biological pool)
    P2 -> PB   (target-excluded)
    P0 -> PC   (ImageNet only)

The label is not only a name: it is embedded in Stage 1 checkpoint filenames,
in the 'protocol' field of every cached run, and in the per-run prediction and
log directory names. Renaming the code alone would orphan all of that, and the
runners would silently recompute every cached run from scratch. This script
migrates the three together.

It is idempotent and defaults to a dry run.

Usage:
    python migrate_protocol_names.py              # show what would change
    python migrate_protocol_names.py --apply
"""
import os
import re
import json
import glob
import shutil
import argparse
from pathlib import Path

RENAME = {'P1': 'PA', 'P2': 'PB', 'P0': 'PC'}


def new_label(old):
    return RENAME.get(old, old)


def migrate_checkpoints(apply_changes):
    """checkpoints/mae_P1_s0_best.pth -> checkpoints/mae_PA_s0_best.pth"""
    moves = []
    for p in sorted(Path('checkpoints').glob('mae_P*')):
        m = re.match(r'^mae_(P[012])(_.*)$', p.name)
        if not m:
            continue
        target = p.with_name(f"mae_{RENAME[m.group(1)]}{m.group(2)}")
        if target.exists():
            print(f"  [skip] {target.name} already exists")
            continue
        moves.append((p, target))

    print(f"\n=== Stage 1 checkpoints: {len(moves)} file(s) ===")
    for src, dst in moves:
        print(f"  {src.name}  ->  {dst.name}")
        if apply_changes:
            shutil.move(str(src), str(dst))
    return len(moves)


def migrate_run_dirs(apply_changes):
    """results/*/pred/P1_n15_s0 -> .../PA_n15_s0, and the matching logs."""
    moves = []
    for sub in ('pred', 'logs'):
        for d in sorted(glob.glob(f'results/*/{sub}/*')):
            p = Path(d)
            m = re.match(r'^(stage[23]_)?(P[012])(_.*)$', p.name)
            if not m:
                continue
            prefix = m.group(1) or ''
            target = p.with_name(f"{prefix}{RENAME[m.group(2)]}{m.group(3)}")
            if target.exists():
                continue
            moves.append((p, target))

    print(f"\n=== prediction / log directories: {len(moves)} entr(y/ies) ===")
    for src, dst in moves[:6]:
        print(f"  {src.parent.parent.name}/{src.parent.name}/{src.name}"
              f"  ->  {dst.name}")
    if len(moves) > 6:
        print(f"  ... and {len(moves) - 6} more")
    if apply_changes:
        for src, dst in moves:
            shutil.move(str(src), str(dst))
    return len(moves)


def migrate_results(apply_changes):
    """
    Rewrite the 'protocol' field and the checkpoint paths inside results.json.

    A backup is written next to each file the first time it is touched, since
    these records hold hours of GPU time that cannot be recovered from the
    checkpoints alone.
    """
    total_rows, files = 0, 0
    print("\n=== results.json ===")
    for f in sorted(glob.glob('results/*/results.json')):
        payload = json.load(open(f))
        changed = 0
        for r in payload.get('runs', []):
            if r.get('protocol') in RENAME:
                r['protocol'] = new_label(r['protocol'])
                changed += 1
            for key in ('mae_checkpoint', 'finetune_checkpoint'):
                if isinstance(r.get(key), str):
                    r[key] = re.sub(r'(mae_|ft_[^/\\]*?_)(P[012])(_)',
                                    lambda m: m.group(1) + RENAME[m.group(2)]
                                    + m.group(3), r[key])
        for s in payload.get('summary', []):
            if s.get('protocol') in RENAME:
                s['protocol'] = new_label(s['protocol'])
        if not changed:
            continue
        files += 1
        total_rows += changed
        print(f"  {Path(f).parent.name:<24} {changed} run(s)")
        if apply_changes:
            backup = f + '.pre_rename_backup'
            if not os.path.exists(backup):
                shutil.copy2(f, backup)
            with open(f, 'w') as fh:
                json.dump(payload, fh, indent=2, default=str)
    print(f"  total: {total_rows} run(s) in {files} file(s)")
    return total_rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help="perform the rename (default: dry run)")
    args = ap.parse_args()

    mode = "APPLYING" if args.apply else "DRY RUN (nothing is changed)"
    print(f"=== Protocol rename: " +
          ", ".join(f"{k} -> {v}" for k, v in RENAME.items()) + f" [{mode}]")

    n_ckpt = migrate_checkpoints(args.apply)
    n_dirs = migrate_run_dirs(args.apply)
    n_rows = migrate_results(args.apply)

    print(f"\nSummary: {n_ckpt} checkpoint(s), {n_dirs} directory(ies), "
          f"{n_rows} cached run(s)")
    if not args.apply:
        print("Re-run with --apply to perform the rename.")
    else:
        print("Done. results.json backups are kept as *.pre_rename_backup.")
        print("runs.csv and summary.csv refresh on the next run of e2/e3.")


if __name__ == '__main__':
    main()
