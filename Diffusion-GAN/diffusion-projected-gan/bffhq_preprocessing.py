#!/usr/bin/env python3
"""
bffhq_preprocessing.py

Pipeline:
  1. Create balanced BFFHQ subsets for multiple rho values (LargeBFFHQ + create_balanced_dataset2)
  2. Copy selected images to per-rho directories with collision-safe filenames + export metadata CSV
  3. Generate dataset.json
  4. Run dataset_tool.py (optional GAN step)

Safety: run with --dry_run first to verify subset sizes before committing disk writes.

NOTE: images are *copied* (not symlinked) so each rho directory is self-contained for
dataset_tool.py. Disk cost: ~size_of_bffhq × len(RHO_VALUES). Replace shutil.copy2
with os.symlink(src_path, dst_path) to avoid this if your dataset_tool.py handles symlinks.

NOTE: create_balanced_dataset2 is imported from dataset_utils2, which is what the existing
data-prep scripts use. get_balanced_bffhq (exposed in bffhq.py) calls create_balanced_dataset
instead — verify these are equivalent before switching.
"""

import os
import json
import shutil
import argparse
import subprocess
import sys
import torch
import pandas as pd

from bffhq.bffhq import LargeBFFHQ
from bffhq.dataset_utils2 import create_balanced_dataset2

RHO_VALUES = [0.95, 0.995]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "bffhq/data/bffhq")

OUT_IMAGES = os.path.join(SCRIPT_DIR, "bffhq/data/bffhq_rho_images")
OUT_META   = os.path.join(SCRIPT_DIR, "bffhq/data/bffhq_rho_csv")
OUT_GAN    = os.path.join(SCRIPT_DIR, "bffhq/data/bffhq_rho_gan")

RESOLUTION = "256x256"  


def create_rho_csv(
    dataset: LargeBFFHQ,
    rho: float,
    dry_run: bool = True,
    max_total_per_class: int = None,
):
    print(f"\n=== Processing rho={rho:.2f} ===")

    # max_total_per_class is passed through only if create_balanced_dataset2 supports it.
    # If not, remove the kwarg and handle normalization separately.
    kwargs = {}
    if max_total_per_class is not None:
        kwargs["max_total_per_class"] = max_total_per_class

    subset = create_balanced_dataset2(dataset, rho=rho, **kwargs)
    indices = subset.indices
    print(f"  Subset size: {len(indices)}")

    img_dir = os.path.join(OUT_IMAGES, f"rho_{rho:.2f}")
    if not dry_run:
        os.makedirs(img_dir, exist_ok=True)

    rows = []
    for new_idx, orig_idx in enumerate(indices):
        sample = dataset[int(orig_idx)]

        src_path    = sample["name"]
        class_label = int(sample["class_label"])
        bias_label  = int(sample["bias_label"])

        # Rename to avoid collisions across train/val/test splits.
        # Pattern mirrors BiasedMNIST: {sequential_idx}_y{class}_b{bias}.ext
        ext   = os.path.splitext(src_path)[1]   # .jpg for BFFHQ
        fname = f"{new_idx}_y{class_label}_b{bias_label}{ext}"

        if not dry_run:
            dst_path = os.path.join(img_dir, fname)
            shutil.copy2(src_path, dst_path)

        rows.append({
            "img_filename": fname,
            "src_path":     src_path,
            "class_label":  class_label,
            "bias_label":   bias_label,
            "rho":          rho,
        })

    df = pd.DataFrame(rows)
    print(df.groupby(["class_label", "bias_label"]).size().to_string())

    meta_path = os.path.join(OUT_META, f"bffhq_rho_{rho:.2f}.csv")
    if not dry_run:
        os.makedirs(OUT_META, exist_ok=True)
        df.to_csv(meta_path, index=False)
        print(f"  Metadata → {meta_path}")

    return meta_path, img_dir


def write_json(img_dir: str, meta_path: str):
    df = pd.read_csv(meta_path)
    df = df.sort_values(by="img_filename")

    labels = [
        [row["img_filename"], int(row["class_label"])]
        for _, row in df.iterrows()
    ]

    json_path = os.path.join(img_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump({"labels": labels}, f, indent=2)

    print(f"  dataset.json → {json_path}")


def run_gan(dataset_tool_path: str, img_dir: str, out_dir: str, resolution: str):
    cmd = [
        sys.executable,
        dataset_tool_path,
        f"--source={img_dir}",
        f"--dest={out_dir}",
        f"--resolution={resolution}",
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="BFFHQ preprocessing pipeline")
    parser.add_argument("--dataset_tool_path", default="dataset_tool.py")
    parser.add_argument("--skip_gan",        action="store_true",
                        help="Skip dataset_tool.py step")
    parser.add_argument("--dry_run",         action="store_true",
                        help="Print subset statistics only; do not write any files")
    parser.add_argument("--normalize_sizes", action="store_true",
                        help="Truncate all rho subsets to the smallest per-class count "
                             "(requires create_balanced_dataset2 to accept max_total_per_class)")
    parser.add_argument("--resolution",      default=RESOLUTION)
    args = parser.parse_args()

    if args.dry_run:
        print("[DRY RUN] No files will be written.\n")

    print("Loading LargeBFFHQ...")
    dataset = LargeBFFHQ(root=DATA_ROOT, transform=None)
    print(f"Total dataset size: {len(dataset)}")

    # Pass 1: natural sizes to find the binding constraint
    print("\nComputing natural subset sizes...")
    natural = {}
    for rho in RHO_VALUES:
        bd = create_balanced_dataset2(dataset, rho=rho)
        natural[rho] = bd
        print(f"  rho={rho}: natural size={len(bd)}")

    # The smaller dataset (rho=0.95) sets the per-class cap for all rhos
    min_size_per_class = min(len(bd) for bd in natural.values()) // 2
    print(f"  min_size_per_class={min_size_per_class} → total per dataset={min_size_per_class * 2}")

    # Pass 2: re-create subsets with the cap applied, preserving each rho's own conflict ratio
    for rho in RHO_VALUES:
        meta_path, img_dir = create_rho_csv(
            dataset,
            rho,
            dry_run=args.dry_run,
            max_total_per_class=min_size_per_class,
        )

        if not args.dry_run:
            write_json(img_dir, meta_path)

            if not args.skip_gan:
                out_dir = os.path.join(OUT_GAN, f"bffhq_{int(rho * 100)}")
                run_gan(args.dataset_tool_path, img_dir, out_dir, args.resolution)
    
    print("\nDone.")


if __name__ == "__main__":
    main()