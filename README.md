# LoRA-GPT2: Parameter-Efficient Fine-Tuning for Sentiment Analysis  

## Introduction  
Fine-tuning large pre-trained models can be computationally expensive, requiring significant memory and training time. This repository implements **Low-Rank Adaptation (LoRA)**, a **Parameter-Efficient Fine-Tuning (PEFT)** method, to efficiently fine-tune a pre-trained **GPT-2** model. Instead of updating all model parameters, LoRA introduces trainable low-rank matrices, drastically reducing computational overhead while maintaining high performance.  

## Dataset  
The model is trained on the **Rotten Tomatoes Dataset**, sourced from Hugging Face. This dataset consists of balanced movie reviews labeled as **positive** or **negative**, making it ideal for sentiment analysis. The dataset will be automatically downloaded when running `train.py`.  

## Features  
✅ Implement LoRA from scratch for efficient fine-tuning  
✅ Apply LoRA to a pre-trained GPT-2 model  
✅ Train on Rotten Tomatoes sentiment analysis dataset  
✅ Reduce memory and compute requirements compared to full fine-tuning  

Stay tuned for setup instructions, training details, and evaluation metrics! 🚀
