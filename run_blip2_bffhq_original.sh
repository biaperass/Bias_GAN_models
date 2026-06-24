#!/bin/bash

for rho in 95 99; do
    python blip2_bffhq_original.py \
        --csv_path Bias_GAN_models/projected-gan/bffhq/data/bffhq_rho_csv/bffhq_rho_0.${rho}.csv \
        --output_csv results_bffhq_original_${rho}.csv
done