#!/bin/bash

for p in -1 0 1 2 3 4 5 6 7 8 9; do
    python train.py --init_from="gpt2-medium" \
        --out_dir="gpt-lora-prompt_$p" \
        --lora_rank=128 \
        --lora_alpha=512 \
        --prompt=$p
    python generate.py --init-from="resume" \
        --out_dir="gpt-lora-prompt_$p"
done