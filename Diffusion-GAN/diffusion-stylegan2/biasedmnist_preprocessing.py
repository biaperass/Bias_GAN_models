#!/usr/bin/env python3
"""
biasedmnist_preprocessing.py

Pipeline:
  1. Generate BiasedMNIST datasets for multiple rho values
  2. Export images + metadata directly
  3. Generate dataset.json
  4. Run dataset_tool.py (optional GAN step)
"""

import os
import json
import argparse
import subprocess
import sys
import torch
import pandas as pd

from biasedmnist.biased_mnist import get_dataloader  # your script

RHO_VALUES = [0.70, 0.80, 0.95]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT   = os.path.join(SCRIPT_DIR, "biasedmnist/data")

OUT_IMAGES  = os.path.join(DATA_ROOT, "biasedmnist_images")
OUT_META    = os.path.join(DATA_ROOT, "biasedmnist_metadata")
OUT_GAN     = os.path.join(DATA_ROOT, "biasedmnist_gan")


def create_rho_csv(rho):
    print(f"\n=== Processing rho={rho} ===")

    loader = get_dataloader(
        root=os.path.join(DATA_ROOT, "mnist"),
        train=True,
        batch_size=1,
        shuffle=False,
        rho=rho,
        num_workers=0
    )

    dataset = loader.dataset

    img_dir = os.path.join(OUT_IMAGES, f"rho_{rho:.2f}")
    os.makedirs(img_dir, exist_ok=True)

    rows = []

    for i, (img, target, bias) in enumerate(dataset):
        img = (img * 0.5 + 0.5) # renormalization
        img = (img * 255).byte().permute(1, 2, 0).numpy() # scale to [0, 255] and convert to HWC format

        fname = f"{i}_y{target}_b{bias}.png"
        path = os.path.join(img_dir, fname)

        # save
        from PIL import Image
        Image.fromarray(img).save(path)

        rows.append({
            "img_id": i,
            "img_filename": fname,
            "y": int(target),
            "bias": int(bias),
            "rho": rho
        })

    df = pd.DataFrame(rows)

    meta_path = os.path.join(OUT_META, f"biasedmnist_{rho:.2f}.csv")
    os.makedirs(OUT_META, exist_ok=True)
    df.to_csv(meta_path, index=False)

    return meta_path, img_dir


def write_json(img_dir, meta_path):
    df = pd.read_csv(meta_path)

    labels = [
        [row["img_filename"], int(row["y"])]
        for _, row in df.iterrows()
    ]

    with open(os.path.join(img_dir, "dataset.json"), "w") as f:
        json.dump({"labels": labels}, f, indent=2)


def run_gan(dataset_tool_path, img_dir, out_dir):
    cmd = [
        sys.executable,
        dataset_tool_path,
        f"--source={img_dir}",
        f"--dest={out_dir}",
        "--height=32",
        "--width=32"
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_tool_path", default="dataset_tool.py")
    parser.add_argument("--skip_gan", action="store_true")
    args = parser.parse_args()

    for rho in RHO_VALUES:
        meta_path, img_dir = create_rho_csv(rho)
        write_json(img_dir, meta_path)

        if not args.skip_gan:
            out_dir = os.path.join(OUT_GAN, f"biasedmnist_{int(rho*100)}")
            run_gan(args.dataset_tool_path, img_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()