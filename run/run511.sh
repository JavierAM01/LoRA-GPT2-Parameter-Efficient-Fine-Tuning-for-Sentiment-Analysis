#!/bin/bash

for p in 0 1 2; do
    python train.py --init_from="gpt2-medium" \
        --out_dir="gpt-lora-prompt_$p" \
        --lora_rank=128 \
        --lora_alpha=512 \
        --prompt=$p
    
    python generate.py --init_from="resume" \
        --out_dir="gpt-lora-prompt_$p" \
        --prompt=$p
done
