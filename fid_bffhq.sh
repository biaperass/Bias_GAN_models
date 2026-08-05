#!/bin/bash

set -e

for seed in 0 1 2; do
    python fid_bffhq.py \
        --gen_root="Diffusion-GAN/diffusion-stylegan2/generated_10k_bffhq_seed${seed}" \
        --detections_dir="results_siglip_diff-stylegan/result_diff-stylegan_bffhq_siglip2_seed${seed}" \
        --output_csv="fid_diff-gan/fid_diff-gan_bffhq_aligned_vs_conflicting_seed_${seed}.csv"
done

python fid_mean_std.py \
    --input_glob="fid_diff-gan/*.csv" \
    --output_csv="fid_diff-gan/fid_diff-gan_bffhq_aligned_vs_conflicting_aggregated.csv"