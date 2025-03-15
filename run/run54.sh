#!/bin/bash

r=(16 128 196)
a=(64 512 784)

for i in 0 1 2; do
    python train.py --init_from="gpt2-medium" \
        --out_dir="gpt-lora-r${r[$i]}_a${a[$i]}" \
        --lora_rank=${r[$i]} \
        --lora_alpha=${a[$i]}
done


python train.py --init_from="gpt2-medium" \
        --out_dir="gpt-lora-default" \
        --lora_rank=0 \
        --lora_dropout=0.05