#!/usr/bin/env python3
"""
dogsandcats_preprocessing.py

Orchestrates the full Dogs and Cats preprocessing pipeline:
  1. Generate per-rho metadata CSVs  (dogsandcats_dynamic_rho.py)
  2. Copy images into per-rho folders (dogsandcats_rho_train.py)
  3. Generate dataset.json per folder (csv_json_dogsandcats.py)
  4. Run dataset_tool.py for each rho

Usage:
    python dogsandcats_preprocessing.py 
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
RHO_VALUES = [95, 99]

# ── Paths (all relative to this script's directory) ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DAC_DIR = os.path.join(SCRIPT_DIR, "dogsandcats")
DATA_ROOT = os.path.join(DAC_DIR, "data")

CSV_DIR = os.path.join(DATA_ROOT, "dogscats_rho_csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "dogscats_rho_images")
GAN_DIR = os.path.join(DATA_ROOT, "dogscats_rho_gan")
IMAGE_ROOT = os.path.join(
    DATA_ROOT,
    "dogs_and_cats"
)

def load_module(name, rel_path):
    """Load a sibling script as a module without installing it."""
    path = os.path.join(DAC_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def step0_download():
    print("\n=== STEP 0: Downloading Dogs and Cats dataset ===")
    dac = load_module("dogsandcats", "dogsandcats.py")
    for rho in RHO_VALUES:
        dac.DogsAndCats(root=DATA_ROOT, bias_amount=rho)
    print("  Download complete.")
    
def step1_generate_csvs():
    print("\n=== STEP 1: Generating per-rho metadata CSVs ===")
    mod = load_module("dogsandcats_metadata_csv", "dogsandcats_metadata_csv.py")
    os.makedirs(CSV_DIR, exist_ok=True)
    old_argv = sys.argv
    sys.argv = [
        "dogsandcats_metadata_csv.py",
        "--root", IMAGE_ROOT,
        "--out_dir", CSV_DIR,
    ]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    print("  CSV generation complete.")
    
def step2_copy_images():
    print("\n=== STEP 2: Copying images into per-rho folders ===")
    for rho in RHO_VALUES:
        csv_path = os.path.join(CSV_DIR, f"dogscats_rho{rho/100:.2f}.csv")
        out_dir  = os.path.join(IMAGES_DIR, f"dogsandcats_{rho}")

        if not os.path.exists(csv_path):
            sys.exit(f"[FATAL] Missing CSV for rho={rho}: {csv_path} "
                      f"(did step 1 run first?)")

        os.makedirs(out_dir, exist_ok=True)
        df = pd.read_csv(csv_path)

        n_copied, n_missing = 0, 0
        for _, row in df.iterrows():
            src = row["img_path"]          # already absolute, unlike Waterbirds
            dst = os.path.join(out_dir, row["img_filename"])
            if not os.path.exists(src):
                print(f"  [WARN] source missing, skipping: {src}")
                n_missing += 1
                continue
            shutil.copy2(src, dst)
            n_copied += 1

        print(f"  rho={rho} -> {out_dir}  ({n_copied} copied, {n_missing} missing)")
    print("  Image copying complete.")


def step3_generate_jsons():
    print("\n=== STEP 3: Writing dataset.json files ===")
    for rho in RHO_VALUES:
        csv_path = os.path.join(CSV_DIR, f"dogscats_rho{rho/100:.2f}.csv")
        img_dir  = os.path.join(IMAGES_DIR, f"dogsandcats_{rho}")
        json_out = os.path.join(img_dir, "dataset.json")

        if os.path.exists(json_out):
            print(f"  [SKIP] {json_out} already exists.")
            continue

        df = pd.read_csv(csv_path).sort_values(by="img_filename")  # no img_id here
        labels_list = [
            [row["img_filename"], int(row["class_label"])]         # not "y"
            for _, row in df.iterrows()
        ]
        with open(json_out, "w") as f:
            json.dump({"labels": labels_list}, f, indent=2)
        print(f"  Written: {json_out}  ({len(labels_list)} entries)")
    print("  JSON generation complete.")
    
def step4_run_dataset_tool(dataset_tool_path):
    print("\n=== STEP 4: Running dataset_tool.py ===")
    for rho in RHO_VALUES:
        src     = os.path.join(IMAGES_DIR, f"dogsandcats_{rho}")
        dest    = os.path.join(GAN_DIR,    f"dogsandcats_{rho}")
        cmd = [
            sys.executable,
            dataset_tool_path,
            f"--source={src}",
            f"--dest={dest}",
            "--resolution=256x256",
        ]
        print(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
def main():
    parser = argparse.ArgumentParser(description="Dogs and Cats preprocessing pipeline")
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