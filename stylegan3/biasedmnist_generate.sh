#!/bin/bash

set -e

for rho in 90; do #70 80 95
    
    # Determine the network path based on the value of rho
    if [ "$rho" -eq 90 ]; then 
        NETWORK_PATH="biasedmnist/output_biasedmnist_90/00000-stylegan2-biasedmnist_90-gpus1-batch64-gamma8/network-snapshot-001400.pkl"    
    elif [ "$rho" -eq 95 ]; then
        NETWORK_PATH="output_biasedmnist_95/00003-stylegan2-biasedmnist_95-gpus1-batch64-gamma8/network-snapshot-003500.pkl"
    else
        NETWORK_PATH="output_biasedmnist_${rho}/00000-stylegan2-biasedmnist_${rho}-gpus1-batch64-gamma8/network-snapshot-002000.pkl"
    fi

    for y in 0 1 2 3 4 5 6 7 8 9; do
        for phi in 1.0 0.75 0.50; do

            python gen_images.py \
                --network="$NETWORK_PATH" \
                --seeds=0-10000 \
                --class="$y" \
                --trunc="$phi" \
                --outdir="generated_10k_rho${rho}/biasedmnist_trunc${phi}_class${y}"
                
        done
    done
done