#!/bin/bash

# Quick start training script for VQ-VAE
# Adjust parameters as needed for your setup

echo "Starting VQ-VAE Training..."
echo "=========================="
echo ""

# Training with smaller batch size for Mac/limited GPU memory
python train_vqvae.py \
    --video-dir 20bn-something-something-v2/filtered_videos \
    --event-dir events_output \
    --output-dir vqvae_output \
    --epochs 30 \
    --batch-size 8 \
    --lr 1e-4 \
    --clip-length 8 \
    --stride 32 \
    --num-embeddings 512 \
    --latent-dim 64 \
    --num-workers 4 \
    --save-every 5 \
    --vis-every 5

echo ""
echo "Training complete! Check vqvae_output/ for results."
