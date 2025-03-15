#!/bin/bash

r=(16 128 196)
a=(64 512 784)

for i in 0 1 2; do
    python train.py --init_from="gpt2-medium" \
        --out_dir="gpt-lora-r${r[$i]}_a${a[$i]}" \
        --lora_rank=${r[$i]} \
        --lora_alpha=${a[$i]}
    python generate.py --init-from="resume" \
        --out_dir="generate-gpt-lora-r${r[$i]}_a${a[$i]}"
done


python train.py --init_from="gpt2-medium" \
        --out_dir="gpt-lora-full-finetune" \
        --lora_rank=0 \
        --lora_dropout=0.05
python generate.py --init_from="resume" \
        --out_dir="generate-gpt-lora-full-finetune"