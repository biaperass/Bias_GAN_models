#!/bin/bash

for rho in 70 80 95; do
    for trunc in 0.5 0.75 1; do
        for cls in 0 1; do
            echo "Running rho=$rho truncation=$trunc class=$cls"

            python blip2_waterbirds.py \
                --rho "$rho" \
                --truncation "$trunc" \
                --class "$cls"
        done
    done
done