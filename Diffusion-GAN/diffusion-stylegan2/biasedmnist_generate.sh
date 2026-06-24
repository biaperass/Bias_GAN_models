#!/bin/bash

set -e

NETWORK_PATH="biasedmnist/output_biasedmnist_90/00001-biasedmnist_90-cond-stylegan2-gamma8-batch64-target0.6-ada_kimg100-ts_dist-priority-image_augno-noise_sd0.05/network-snapshot-001400.pkl"


for rho in 90; do #70 80 95   

    for y in 0 1 2 3 4 5 6 7 8 9; do
        for phi in 1.0 0.75 0.50; do

            python generate.py \
                --network="$NETWORK_PATH" \
                --seeds=0-10000 \
                --class="$y" \
                --trunc="$phi" \
                --outdir="generated_10k_rho${rho}/biasedmnist_trunc${phi}_class${y}"                
        done
    done
done