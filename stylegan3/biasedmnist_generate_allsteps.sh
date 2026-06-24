#!/bin/bash

set -e

rho=90

for steps in "000403" "000604" "000806" "001008" 001209" 001400"; do
    
    NETWORK_PATH="biasedmnist/output_biasedmnist_90/00000-stylegan2-biasedmnist_90-gpus1-batch64-gamma8/network-snapshot-${steps}.pkl"    

    for y in 0 1 2 3 4 5 6 7 8 9; do
        for phi in 1.0 0.75 0.50; do

            python gen_images.py \
                --network="$NETWORK_PATH" \
                --seeds=0-10000 \
                --class="$y" \
                --trunc="$phi" \
                --outdir="generated_10k_rho${rho}_steps${steps}/biasedmnist_trunc${phi}_class${y}"
                
        done
    done
done