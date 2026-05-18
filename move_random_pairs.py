import argparse
import math
import random
import shutil
from pathlib import Path


def find_pairs(source_dir: Path, image_exts=None):
    if image_exts is None:
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    files = list(source_dir.iterdir())
    base_map = {}

    for p in files:
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        name = p.stem
        base_map.setdefault(name, set()).add(ext)

    matched = []
    for base, exts in base_map.items():
        if ".txt" in exts and exts.intersection(image_exts):
            matched.append(base)

    return matched


def move_pairs(base_names, source_dir: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for base in base_names:
        image_src = None
        text_src = None

        for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            candidate = source_dir / f"{base}{ext}"
            if candidate.exists():
                image_src = candidate
                break

        text_candidate = source_dir / f"{base}.txt"
        if text_candidate.exists():
            text_src = text_candidate

        if not image_src or not text_src:
            continue

        shutil.move(str(image_src), str(dest_dir / image_src.name))
        shutil.move(str(text_src), str(dest_dir / text_src.name))
        moved.append((image_src.name, text_src.name))

    return moved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly move a percentage of paired .jpg and .txt files from one folder to another."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source folder containing paired .jpg and .txt files.",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination folder where sampled pairs will be moved.",
    )
    parser.add_argument(
        "--percent",
        type=float,
        default=20.0,
        help="Percentage of matched pairs to move (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which pairs would be moved without actually moving files.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum number of pairs to move if percent results in less than 1 (default: 1).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source_dir = Path(args.source).expanduser().resolve()
    dest_dir = Path(args.dest).expanduser().resolve()

    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    if args.seed is not None:
        random.seed(args.seed)

    pairs = find_pairs(source_dir)
    total_pairs = len(pairs)

    if total_pairs == 0:
        print("No matched .jpg and .txt file pairs found in source folder.")
        return

    num_to_copy = math.floor(total_pairs * args.percent / 100.0)
    if num_to_copy < args.min_count:
        num_to_copy = min(args.min_count, total_pairs)

    selected = random.sample(pairs, num_to_copy)

    print(f"Found {total_pairs} matched pairs.")
    print(f"Selecting {num_to_copy} pairs ({args.percent}% of total).\n")

    for base in selected:
        print(f"  {base}.jpg + {base}.txt")

    if args.dry_run:
        print("\nDry run enabled. No files were moved.")
        return

    moved = move_pairs(selected, source_dir, dest_dir)
    print(f"\nMoved {len(moved)} pairs to {dest_dir}")


if __name__ == "__main__":
    main()
