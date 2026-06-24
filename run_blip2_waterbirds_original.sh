#!/bin/bash

for rho in 70 80 95; do
    python blip2_waterbirds_original.py \
        --csv_path Bias_GAN_models/projected-gan/waterbirds/data/waterbirds_rho_csv/waterbirds_0.${rho}_fixed.csv \
        --image_root Bias_GAN_models/projected-gan/waterbirds/data/waterbirds_rho_images/waterbirds_${rho} \
        --output_csv results_waterbirds_original_${rho}.csv
done