#!/usr/bin/env python3
"""
waterbirds_preprocessing.py

Orchestrates the full Waterbirds preprocessing pipeline:
  1. Generate per-rho metadata CSVs  (waterbirds_dynamic_rho.py)
  2. Copy images into per-rho folders (waterbirds_rho_train.py)
  3. Generate dataset.json per folder (csv_json_waterb.py)
  4. Run dataset_tool.py for each rho

Usage:
    python waterbirds_preprocessing.py 
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import pandas as pd

# ── Single source of truth for rho values ────────────────────────────────────
RHO_VALUES = [0.70, 0.80, 0.95]

# ── Paths (all relative to this script's directory) ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

WB_DIR = os.path.join(SCRIPT_DIR, "waterbirds")
DATA_ROOT = os.path.join(WB_DIR, "data")

CSV_DIR = os.path.join(DATA_ROOT, "waterbirds_rho_csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "waterbirds_rho_images")
GAN_DIR = os.path.join(DATA_ROOT, "waterbirds_rho_gan")
IMAGE_ROOT = os.path.join(
    DATA_ROOT,
    "waterbirds",
    "waterbird_complete95_forest2water2"
)

def load_module(name, rel_path):
    """Load a sibling script as a module without installing it."""
    path = os.path.join(SCRIPT_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def step0_download():
    print("\n=== STEP 0: Downloading Waterbirds dataset ===")
    if os.path.exists(os.path.join(IMAGE_ROOT, "metadata.csv")):
        print("  [SKIP] Dataset already downloaded.")
        return
    dyn = load_module("waterbirds_dynamic_rho", "waterbirds_dynamic_rho.py")
    dyn.WaterbirdsDynamicRho(env="test", root=DATA_ROOT)
    print("  Download complete.")


def step1_generate_csvs():
    print("\n=== STEP 1: Generating per-rho metadata CSVs ===")
    os.makedirs(CSV_DIR, exist_ok=True)
    dyn = load_module("waterbirds_dynamic_rho", "waterbirds_dynamic_rho.py")

    original_df = pd.read_csv(os.path.join(IMAGE_ROOT, "metadata.csv"))

    held_out_path = os.path.join(DATA_ROOT, "waterbirds_rho", "metadata_with_held_out.csv")
    os.makedirs(os.path.dirname(held_out_path), exist_ok=True)
    _, remainder_df = dyn.WaterbirdsDynamicRho.create_held_out_test_split(
        root=DATA_ROOT, n_per_group=100, random_seed=42, output_path=held_out_path
    )

    pool_df = original_df.query("split == 0 or split == 1").copy()
    pool_df = pd.concat([pool_df, remainder_df]).reset_index(drop=True)

    # --- compute raw counts for all rhos ---
    all_counts = {}
    for rho in RHO_VALUES:
        counts = dyn.WaterbirdsDynamicRho.compute_group_counts_for_rho(rho, pool_df)
        all_counts[rho] = counts
        print(f"rho={rho:.2f}: counts={counts}, total={sum(counts.values())}")

    min_per_class_per_class = {
        0: min(c[(0,0)] + c[(0,1)] for c in all_counts.values()),
        1: min(c[(1,0)] + c[(1,1)] for c in all_counts.values()),
    }
    shared_min = min(min_per_class_per_class.values())
    min_per_class = {0: shared_min, 1: shared_min}
    print(f"shared_min per class = {shared_min}, total dataset size = {shared_min * 2}")

    # --- rescale per-rho counts to shared_min, then generate ---
    for rho in RHO_VALUES:

        counts  = all_counts[rho]
        n_class = {
            0: counts[(0,0)] + counts[(0,1)],
            1: counts[(1,0)] + counts[(1,1)],
        }
        final_counts = {}
        for y in [0, 1]:
            aligned_key  = (y, y)
            conflict_key = (y, 1 - y)
            current_total_y = n_class[y]

            if current_total_y > min_per_class[y]:   # <-- per-class min
                scale   = min_per_class[y] / current_total_y
                n_align = int(counts[aligned_key] * scale)
                n_conf  = min_per_class[y] - n_align
            else:
                n_align = counts[aligned_key]
                n_conf  = counts[conflict_key]

            effective_rho = n_align / (n_align + n_conf)
            final_counts[aligned_key]  = n_align
            final_counts[conflict_key] = n_conf
            print(f"rho={rho:.2f}, class={y}: align={n_align}, conflict={n_conf}, "
                  f"effective_rho={effective_rho:.3f}")

        out_path = os.path.join(CSV_DIR, f"waterbirds_{rho:.2f}_fixed.csv")
        if os.path.exists(out_path):
            print(f"  [SKIP] {out_path} already exists.")
            continue
        dyn.WaterbirdsDynamicRho.create_and_save_metadata(
            output_path=out_path,
            strategy="borrow_from_val_fixed",
            bias_amount=rho,
            n_per_class=min_per_class,
            root=DATA_ROOT,
            extra_pool_df=remainder_df,
        )
        train = dyn.WaterbirdsDynamicRho(env="train", metadata_path=out_path)
        print(f"total dataset: {len(train)} samples, groups={train.get_group_counts().tolist()}\n")


def step2_copy_images():
    print("\n=== STEP 2: Copying images per rho ===")
    train_mod = load_module("waterbirds_rho_train", "waterbirds_rho_train.py")

    for rho in RHO_VALUES:
        rho_tag  = f"{int(rho * 100)}"
        csv_path = os.path.join(CSV_DIR, f"waterbirds_{rho:.2f}_fixed.csv")
        out_dir  = os.path.join(IMAGES_DIR, f"waterbirds_{rho_tag}")
        print(f"  rho={rho:.2f} -> {out_dir}")
        train_mod.images_from_csv(csv_path, IMAGE_ROOT, out_dir, dry_run=False)


def step3_generate_jsons():
    """
    Corrected version of csv_json_waterb.py logic, using Waterbirds columns.
    Writes dataset.json directly into each image folder.
    """
    print("\n=== STEP 3: Writing dataset.json files ===")

    for rho in RHO_VALUES:
        rho_tag  = f"{int(rho * 100)}"
        csv_path = os.path.join(CSV_DIR, f"waterbirds_{rho:.2f}_fixed.csv")
        img_dir  = os.path.join(IMAGES_DIR, f"waterbirds_{rho_tag}")
        json_out = os.path.join(img_dir, "dataset.json")

        if os.path.exists(json_out):
            print(f"  [SKIP] {json_out} already exists.")
            continue

        df = pd.read_csv(csv_path).sort_values(by="img_id")  # Waterbirds column
        labels_list = [
            [row["img_filename"], int(row["y"])]              # Waterbirds columns
            for _, row in df.iterrows()
        ]
        with open(json_out, "w") as f:
            json.dump({"labels": labels_list}, f, indent=2)
        print(f"  Written: {json_out}  ({len(labels_list)} entries)")


def step4_run_dataset_tool(dataset_tool_path):
    print("\n=== STEP 4: Running dataset_tool.py ===")
    for rho in RHO_VALUES:
        rho_tag = f"{int(rho * 100)}"
        src     = os.path.join(IMAGES_DIR, f"waterbirds_{rho_tag}")
        dest    = os.path.join(GAN_DIR,    f"waterbirds_64_{rho_tag}")
        cmd = [
            sys.executable, dataset_tool_path,
            f"--source={src}",
            f"--dest={dest}",
            "--width=64",
            "--height=64"
        ]
        print(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Waterbirds preprocessing pipeline")
    parser.add_argument(
        "--dataset_tool_path", required=False, default="dataset_tool.py",
        help="Path to dataset_tool.py from your GAN repo"
    )
    parser.add_argument(
        "--skip_step", type=int, nargs="*", default=[],
        help="Steps to skip, e.g. --skip_step 1 2"
    )
    args = parser.parse_args()

    if 0 not in args.skip_step: step0_download()
    if 1 not in args.skip_step: step1_generate_csvs()
    if 2 not in args.skip_step: step2_copy_images()
    if 3 not in args.skip_step: step3_generate_jsons()
    if 4 not in args.skip_step: step4_run_dataset_tool(args.dataset_tool_path)

    print("\nDone.")


if __name__ == "__main__":
    main()