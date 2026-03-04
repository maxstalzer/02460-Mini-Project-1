#!/bin/sh
#BSUB -J 02460_mini_project_1
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=16G]"
#BSUB -W 08:00
#BSUB -o output_%J.out
#BSUB -e error_%J.err

echo "=========================================================="
echo "Starting Mini-Project 1 Pipeline"
echo "=========================================================="

# 1. U-Net DDPM
echo "\n--- 1. U-Net DDPM ---"
echo "Training U-Net DDPM..."
uv run ddpm.py train --network unet --model checkpoints/unet_ddpm.pt --device cuda --epochs 50 --batch-size 128

echo "Sampling U-Net DDPM (Computes Wall-clock time & FID)..."
uv run ddpm.py sample --network unet --model checkpoints/unet_ddpm.pt --samples samples/unet_samples.png --device cuda


# 2. Latent DDPM (Beta = 0.5)
echo "\n--- 2. Latent DDPM (Beta = 0.5) ---"
echo "Training Beta-VAE (Beta=0.5)..."
uv run latent_ddpm.py --mode train_vae --data mnist --vae-model checkpoints/beta_vae_05.pt --device cuda --epochs 50 --batch-size 128 --latent-dim 32 --beta 0.5

echo "Training Latent DDPM (Beta=0.5)..."
uv run latent_ddpm.py --mode train_ddpm --data mnist --vae-model checkpoints/beta_vae_05.pt --model checkpoints/latent_ddpm_05.pt --device cuda --epochs 50 --batch-size 128 --latent-dim 32

echo "Sampling Latent DDPM (Beta=0.5) (Computes Wall-clock time & FID)..."
uv run latent_ddpm.py --mode sample --data mnist --vae-model checkpoints/beta_vae_05.pt --model checkpoints/latent_ddpm_05.pt --samples samples/latent_samples_05.png --device cuda --latent-dim 32


# 3. Latent DDPM (Beta = 10^-6)
echo "\n--- 3. Latent DDPM (Beta = 1e-6) ---"
echo "Training Beta-VAE (Beta=1e-6)..."
uv run latent_ddpm.py --mode train_vae --data mnist --vae-model checkpoints/beta_vae_1e6.pt --device cuda --epochs 50 --batch-size 128 --latent-dim 32 --beta 1e-6

echo "Training Latent DDPM (Beta=1e-6)..."
uv run latent_ddpm.py --mode train_ddpm --data mnist --vae-model checkpoints/beta_vae_1e6.pt --model checkpoints/latent_ddpm_1e6.pt --device cuda --epochs 50 --batch-size 128 --latent-dim 32

echo "Sampling Latent DDPM (Beta=1e-6) (Computes Wall-clock time & FID)..."
uv run latent_ddpm.py --mode sample --data mnist --vae-model checkpoints/beta_vae_1e6.pt --model checkpoints/latent_ddpm_1e6.pt --samples samples/latent_samples_1e6.png --device cuda --latent-dim 32


# 4. Bernoulli VAE with Flow Prior
echo "\n--- 4. Bernoulli VAE with Flow Prior ---"
echo "Training Bernoulli Flow VAE..."
uv run vae_bernoulli.py --mode train --model checkpoints/vae_flow_model.pt --device cuda --epochs 50 --batch-size 64 --latent-dim 32

echo "Evaluating Bernoulli Flow VAE (Computes ELBO, Wall-clock time, FID, and PCA)..."
uv run vae_bernoulli.py --mode evaluate --model checkpoints/vae_flow_model.pt --device cuda --batch-size 64 --latent-dim 32

echo "\n=========================================================="
echo "Pipeline Complete! Check output files for FID scores and times."
echo "=========================================================="