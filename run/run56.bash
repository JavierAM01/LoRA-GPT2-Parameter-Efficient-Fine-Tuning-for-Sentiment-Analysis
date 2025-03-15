#!/bin/bash

#SBATCH --job-name=JavierAbollado_PINNs_HeatEqn_1d
#SBATCH -A cis240109p  # Specify allocation
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-16:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/GenAI_HW3_run56_%j.out
#SBATCH --error=logs/GenAI_HW3_run56_%j.err
#SBATCH --mail-type=ALL  # Send email on job BEGIN, END, and FAIL
#SBATCH --mail-user=jabollad@andrew.cmu.edu

# Variables
FOLDER=/ocean/projects/cis240109p/abollado/GenAI/HW3/LoRA-GPT2-Parameter-Efficient-Fine-Tuning-for-Sentiment-Analysis/

# Change to the target working directory
cd $FOLDER

# Create logs directory if it doesn't exist
mkdir -p logs

# activate environment
module load anaconda3/2024
source /ocean/projects/cis240109p/abollado/GenAI/HW2/Small_Difussion_Model/venv/bin/activate

# Run the Python script
r=(16 16)
a=(64 256)

for i in 0 1; do
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