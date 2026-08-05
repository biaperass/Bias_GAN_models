import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate FID CSVs across seeds (mean/std per rho,trunc,condition)."
    )
    ap.add_argument(
        "--input_glob",
        required=True,
        help="Glob for input CSVs, e.g. 'fid_diff-gan_unique/*.csv'",
    )
    ap.add_argument(
        "--output_csv",
        required=True,
        help="Output aggregated CSV",
    )
    args = ap.parse_args()

    files = sorted(Path().glob(args.input_glob))

    if not files:
        raise FileNotFoundError(f"No files matched: {args.input_glob}")

    # Load and concatenate all seeds
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Columns for which we want mean/std
    value_cols = [
        "fid_class0",
        "fid_class1",
        "fid_mean",
        "n_real_class0",
        "n_real_class1",
        "n_gen_class0",
        "n_gen_class1",
    ]

    # Keep only columns that actually exist
    value_cols = [c for c in value_cols if c in df.columns]

    # Group by experimental setting
    grouped = (
        df.groupby(["rho", "trunc", "condition"], dropna=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten MultiIndex columns
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    # Add number of contributing seeds
    n_runs = (
        df.groupby(["rho", "trunc", "condition"], dropna=False)
        .size()
        .reset_index(name="n_runs")
    )

    result = grouped.merge(
        n_runs,
        on=["rho", "trunc", "condition"],
        how="left",
    )

    # Nice ordering
    result = result.sort_values(["rho", "trunc", "condition"])

    result.to_csv(args.output_csv, index=False)

    print(f"Wrote {args.output_csv}")
    print(result.head())


if __name__ == "__main__":
    main()