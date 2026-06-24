#!/bin/bash

for rho in 95 99; do
    for trunc in 1 0.75 0.50; do
        for cls in 0 1; do
            echo "Running rho=$rho truncation=$trunc class=$cls"

            python blip2_bffhq_cuda.py \
                --rho "$rho" \
                --truncation "$trunc" \
                --class "$cls"
        done
    done
done
