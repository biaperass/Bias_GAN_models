#!/usr/bin/env python3
"""
dogsandcats_metadata_csv.py

Recovers class_label and bias_label directly from the pre-built DogsAndCats
folder structure (align/conflict/test), using the exact same filename-
parsing logic as DogsAndCats.assign_class_label / assign_bias_label, and
writes one CSV per rho (bias_amount) plus a shared test CSV.

The two per-rho train sets are downsampled to the same total size, per
class, without changing the align:conflict ratio (i.e. without moving rho):
the smaller rho's per-class total becomes the cap, and the larger rho's
align/conflict counts are scaled down proportionally to hit that cap.

WHAT THIS DOES NOT SOLVE:
  - The bias label recovered here is just the `z` token baked into every
    filename -- the same value the dataset class already parses at load
    time. It does not tell you HOW `z` was assigned (manual judgment,
    brightness heuristic, VLM output, ...). Provenance of `z` itself is
    still unknown.
  - This does not reproduce `dogs_and_cats_metadata_aug.csv` / the `ddb`
    column. If that was meant to be a different, refined label, it's still
    missing.
  - Filename parsing mirrors the original code's `filename.split(".")[0]`,
    which only takes everything before the FIRST dot. Check n_failed counts
    below before trusting the output.

Usage:
    python build_dogscats_metadata_csv.py --root /path/to/dogs_and_cats
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BIAS_FOLDERS = {"1pct": 99, "5pct": 95}  # matches DogsAndCats.bias_folder_dict
SEED = 42


def parse_label(filename):
    """Same logic as DogsAndCats.assign_class_label / assign_bias_label."""
    stem = filename.split(".")[0]
    parts = stem.split("_")
    if len(parts) != 3:
        return None, None
    _, y, z = parts
    try:
        return int(y), int(z)
    except ValueError:
        return None, None


def walk_split(root, bias_folder, split_name, subdir, has_class_folders,
                bias_amount, align_conflict=None):
    rows = []
    n_failed = 0
    base = os.path.join(root, bias_folder, subdir)
    if not os.path.isdir(base):
        print(f"  [WARN] Missing directory, skipping: {base}")
        return rows

    if has_class_folders:
        file_iter = [
            (cf, fname)
            for cf in sorted(os.listdir(base))
            for fname in sorted(os.listdir(os.path.join(base, cf)))
        ]
    else:
        file_iter = [(None, fname) for fname in sorted(os.listdir(base))]

    for cf, fname in file_iter:
        path = os.path.join(base, cf, fname) if cf else os.path.join(base, fname)
        y, z = parse_label(fname)
        if y is None:
            print(f"  [WARN] Could not parse filename, skipping: {path}")
            n_failed += 1
            continue
        rows.append({
            "img_path": path,
            "img_filename": fname,
            "class_label": y,
            "bias_label": z,
            "align_conflict": align_conflict,
            "bias_amount": bias_amount,
            "split": split_name,
        })

    if n_failed:
        print(f"  [WARN] {n_failed} unparseable filenames in {base}")
    return rows


def group_by_class(rows):
    by_class = {}
    for r in rows:
        by_class.setdefault(r["class_label"], []).append(r)
    return by_class


def compute_balanced_targets(counts_by_bias_amount):
    """
    counts_by_bias_amount: {bias_amount: {y: {'align': n, 'conflict': n}}}
    Returns (targets, shared_min_per_class) where targets has the same shape,
    with align/conflict counts capped at the smaller bias_amount's per-class
    total and rescaled proportionally to preserve each bias_amount's own
    align:conflict ratio as closely as integer counts allow.
    """
    bias_amounts = list(counts_by_bias_amount.keys())
    classes = sorted({y for b in bias_amounts for y in counts_by_bias_amount[b]})
    
    targets = {}
    totals = {}

    # ----------------------------------------------------------
    # Largest feasible total per class across all rho datasets
    # ----------------------------------------------------------

    shared_total = float("inf")

    for b in bias_amounts:
        rho = b / 100.0

        for y in classes:
            avail_conf = counts_by_bias_amount[b][y]["conflict"]

            max_total = int(avail_conf / (1.0 - rho))

            shared_total = min(shared_total, max_total)

    print(f"\nShared total per class = {shared_total}")

    # ----------------------------------------------------------
    # Build targets
    # ----------------------------------------------------------

    for b in bias_amounts:

        rho = b / 100.0

        conflict_target = int(round(shared_total * (1.0 - rho)))
        align_target = shared_total - conflict_target

        targets[b] = {
            0: {
                "align": align_target,
                "conflict": conflict_target,
            },
            1: {
                "align": align_target,
                "conflict": conflict_target,
            },
        }

        totals[b] = shared_total

    return targets, totals
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/dogs_and_cats",
                        help="Path to the extracted dogs_and_cats/ folder")
    parser.add_argument("--out_dir", default=".",
                        help="Directory to write the per-rho CSVs into")
    parser.add_argument("--skip_test", action="store_true",
                        help="Don't write the shared test-split CSV")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- pass 1: walk both bias_folders, keep align/conflict pools separate ---
    pools = {}       # bias_amount -> {'align': {y: [rows]}, 'conflict': {y: [rows]}}
    counts = {}       # bias_amount -> {y: {'align': n, 'conflict': n}}

    for bias_folder, bias_amount in BIAS_FOLDERS.items():
        print(f"\n=== bias_folder={bias_folder} (bias_amount={bias_amount}) ===")
        align_rows = walk_split(args.root, bias_folder, "train", "align",
                                has_class_folders=True, bias_amount=bias_amount,
                                align_conflict="align")
        conflict_rows = walk_split(args.root, bias_folder, "train", "conflict",
                                   has_class_folders=True, bias_amount=bias_amount,
                                   align_conflict="conflict")

        align_by_class = group_by_class(align_rows)
        conflict_by_class = group_by_class(conflict_rows)
        pools[bias_amount] = {"align": align_by_class, "conflict": conflict_by_class}

        classes = sorted(set(align_by_class) | set(conflict_by_class))
        counts[bias_amount] = {
            y: {"align": len(align_by_class.get(y, [])), "conflict": len(conflict_by_class.get(y, []))}
            for y in classes
        }
        for y in classes:
            print(f"  available class={y}: align={counts[bias_amount][y]['align']}, "
                  f"conflict={counts[bias_amount][y]['conflict']}")

    # --- pass 2: compute shared per-class cap and rescaled targets ---
    
    """ 
    # global minimum version: one shared minimum across all classes, but conflicting imbalance
    targets, shared_min_per_class = compute_balanced_targets(counts)
    print(f"\n=== shared_min_per_class = {shared_min_per_class} ===")
    """
    
    # Same number of aligned/conflicting examples for both classes
    targets, totals = compute_balanced_targets(counts)
    
    for b in sorted(totals):
        print(f"rho={b/100:.2f}: {totals[b]} images per class")
    
    shared_total = next(iter(totals.values()))
    total_dataset_size = 2 * shared_total

    print(
        f"each rho's train set will total {total_dataset_size} images "
        f"({shared_total} per class)"
    )

    # --- pass 3: subsample to targets and write CSVs ---
    for bias_amount, bias_folder in {v: k for k, v in BIAS_FOLDERS.items()}.items():
        sampled_rows = []
        for y, tgt in targets[bias_amount].items():
            align_pool = pools[bias_amount]["align"].get(y, [])
            conflict_pool = pools[bias_amount]["conflict"].get(y, [])
            n_align, n_conf = tgt["align"], tgt["conflict"]

            if len(align_pool) < n_align or len(conflict_pool) < n_conf:
                sys.exit(f"[FATAL] bias_amount={bias_amount}, class={y}: "
                         f"need align={n_align}/conflict={n_conf}, "
                         f"have align={len(align_pool)}/conflict={len(conflict_pool)}")

            align_idx = rng.choice(len(align_pool), size=n_align, replace=False) if n_align else []
            conflict_idx = rng.choice(len(conflict_pool), size=n_conf, replace=False) if n_conf else []
            sampled_rows += [align_pool[i] for i in align_idx]
            sampled_rows += [conflict_pool[i] for i in conflict_idx]

        df = pd.DataFrame(sampled_rows)
        out_path = os.path.join(args.out_dir, f"dogscats_rho{bias_amount/100:.2f}.csv")
        df.to_csv(out_path, index=False)
        print(f"\n  Wrote {len(df)} rows -> {out_path}")

        claimed_rho = bias_amount / 100
        for y in sorted(df.class_label.unique()):
            y_sub = df[df.class_label == y]
            n_align = int((y_sub.align_conflict == "align").sum())
            n_conf = int((y_sub.align_conflict == "conflict").sum())
            total = n_align + n_conf
            actual_rho = n_align / total if total else float("nan")
            flag = "  <-- MISMATCH" if abs(actual_rho - claimed_rho) > 0.01 else ""
            print(f"    class={y}: align={n_align}, conflict={n_conf}, "
                  f"actual_rho={actual_rho:.4f}, claimed_rho={claimed_rho:.4f}{flag}")

    # --- shared test split, not specific to either rho, not resized ---
    if not args.skip_test:
        print("\n=== test split (shared across bias amounts) ===")
        test_rows = walk_split(args.root, "", "test", "test",
                               has_class_folders=True, bias_amount=None,
                               align_conflict=None)
        test_df = pd.DataFrame(test_rows)
        test_path = os.path.join(args.out_dir, "dogscats_test.csv")
        test_df.to_csv(test_path, index=False)
        print(f"  Wrote {len(test_df)} rows -> {test_path}")


if __name__ == "__main__":
    main()