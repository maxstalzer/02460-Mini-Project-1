import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# --- NEW IMPORTS ---
from fid import compute_fid
from data_utils import get_mnist_dataloaders

# 1. BETA-VAE MODULES
class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, log_std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        # log_std to std
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(log_std)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        # We assume a fixed variance of 1.0 for the likelihood p(x|z). 
        # Maximizing this log-prob is equivalent to minimizing MSE.
        mean = self.decoder_net(z)
        return td.Independent(td.Normal(loc=mean, scale=torch.ones_like(mean)), 1)

class BetaVAE(nn.Module):
    def __init__(self, encoder, decoder, beta=1.0):
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.latent_dim = None # Will be inferred

    def elbo(self, x):
        # 1. Encode
        q = self.encoder(x)
        z = q.rsample()

        # 2. Standard Normal Prior p(z) ~ N(0, I)
        if self.latent_dim is None:
            self.latent_dim = z.shape[-1]
        prior = td.Independent(td.Normal(
            loc=torch.zeros_like(z), 
            scale=torch.ones_like(z)
        ), 1)

        # 3. KL Divergence: D_KL(q(z|x) || p(z))
        kl = td.kl_divergence(q, prior)
        
        # 4. Reconstruction Log-Likelihood: log p(x|z)
        recon_loss = self.decoder(z).log_prob(x)
        
        # Beta-ELBO
        return torch.mean(recon_loss - self.beta * kl, dim=0)

    def forward(self, x):
        return -self.elbo(x)


# 2. DDPM MODULES (Operating on Latent Z)
class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden=256):
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + 1, num_hidden), nn.ReLU(), 
            nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
            nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
            nn.Linear(num_hidden, input_dim)
        )

    def forward(self, z, t):
        z_t_cat = torch.cat([z, t], dim=1)
        return self.network(z_t_cat)

class LatentDDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=1000):
        super(LatentDDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, z):
        batch_size = z.shape[0]
        t = torch.randint(0, self.T, (batch_size,), device=z.device)
        epsilon = torch.randn_like(z)
        alpha_bar_t = self.alpha_cumprod[t].unsqueeze(1)

        z_t = torch.sqrt(alpha_bar_t) * z + torch.sqrt(1 - alpha_bar_t) * epsilon
        t_input = (t.float() / self.T).unsqueeze(1)
        
        epsilon_theta = self.network(z_t, t_input)
        return torch.sum((epsilon - epsilon_theta) ** 2, dim=1)

    def loss(self, z):
        return self.negative_elbo(z).mean()

    def sample(self, shape):
        z_t = torch.randn(shape).to(self.alpha.device)
        for t in range(self.T-1, -1, -1):
            t_tensor = torch.full((shape[0], 1), t, dtype=torch.float32, device=z_t.device)
            t_input = t_tensor / self.T
            epsilon_theta = self.network(z_t, t_input)
            
            z = torch.randn_like(z_t) if t > 0 else torch.zeros_like(z_t)
            z_t = (1/torch.sqrt(self.alpha[t])) * (z_t - (1 - self.alpha[t])/torch.sqrt(1 - self.alpha_cumprod[t])*epsilon_theta) + torch.sqrt(self.beta[t])*z
        return z_t

# 3. MAIN RUN
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_ddpm', choices=['train_vae', 'train_ddpm', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='mnist', choices=['mnist'], help='dataset to use (default: %(default)s)')
    
    # Model/sample paths
    parser.add_argument('--vae-model', type=str, default='checkpoints/beta_vae.pt', help='file to save/load VAE model (default: %(default)s)')
    parser.add_argument('--model', type=str, default='checkpoints/latent_ddpm.pt', help='file to save/load DDPM model (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples/latent_samples.png', help='file to save samples in (default: %(default)s)')
    
    # Standard hyperparameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')
    
    # Latent-specific arguments
    parser.add_argument('--latent-dim', type=int, default=32, help='dimension of the VAE latent space (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=0.5, help='beta value for the VAE (default: %(default)s)')

    args = parser.parse_args()
    
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # Automatically create directories based on the provided file paths
    for path in [args.vae_model, args.model, args.samples]:
        dir_name = os.path.dirname(path)
        if dir_name:  # Only create if there's an actual directory in the path
            os.makedirs(dir_name, exist_ok=True)

    # 1. GENERATE THE DATA
    if args.data == 'mnist':
        print("Loading MNIST (Continuous, Flatten=True)...")
        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=args.batch_size, 
            binarize=False, 
            flatten=True
        )
        D = 784             # 28x28 flattened

    # 2. DEFINE THE NETWORK AND MODEL
    M = args.latent_dim
    encoder_net = nn.Sequential(
        nn.Linear(D, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M * 2)
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, D)
    )
    vae = BetaVAE(GaussianEncoder(encoder_net), GaussianDecoder(decoder_net), beta=args.beta).to(args.device)
    
    network = FcNetwork(input_dim=M)
    T = 1000
    ddpm = LatentDDPM(network, T=T).to(args.device)

    # 3. CHOOSE MODE TO RUN
    if args.mode == 'train_vae':
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr)
        vae.train()
        print(f"\nTraining Beta-VAE (saving to {args.vae_model})...")
        for epoch in range(args.epochs):
            pbar = tqdm(train_loader)
            for x, _ in pbar:
                x = x.to(args.device)
                optimizer.zero_grad()
                loss = vae(x)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{args.epochs}")
        torch.save(vae.state_dict(), args.vae_model)
        print(f"Saved {args.vae_model}")

    elif args.mode == 'train_ddpm':
        vae.load_state_dict(torch.load(args.vae_model, map_location=args.device))
        vae.eval() 
        
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
        ddpm.train()
        print(f"\nTraining Latent DDPM (saving to {args.model})...")
        for epoch in range(args.epochs):
            pbar = tqdm(train_loader)
            for x, _ in pbar:
                x = x.to(args.device)
                
                # Encode into latent space
                with torch.no_grad():
                    z = vae.encoder(x).sample() 

                optimizer.zero_grad()
                loss = ddpm.loss(z)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{args.epochs}")
        torch.save(ddpm.state_dict(), args.model)
        print(f"Saved {args.model}")

    elif args.mode == 'sample':
        vae.load_state_dict(torch.load(args.vae_model, map_location=args.device))
        ddpm.load_state_dict(torch.load(args.model, map_location=args.device))
        vae.eval()
        ddpm.eval()

        print(f"\nSampling (saving to {args.samples})...")
        with torch.no_grad():
            # Start timer
            start_time = time.time()
            
            # 1. Generate new latents via DDPM
            sampled_z = ddpm.sample((64, M))
            
            # 2. Decode latents into images
            sampled_images = vae.decoder(sampled_z).mean 
            
            # End timer
            sampling_time = time.time() - start_time
            print(f"Sampling 64 images took: {sampling_time:.4f} seconds ({64/sampling_time:.2f} samples/sec)")
            
            # 3. Reshape generated images to (64, 1, 28, 28)
            # Note: We keep them in the [-1, 1] range for the FID calculation
            sampled_images_fid = sampled_images.view(64, 1, 28, 28)
            
            # 4. Get a batch of real images from the test loader for comparison
            x_real, _ = next(iter(test_loader))
            x_real = x_real[:64].view(64, 1, 28, 28).to(args.device)
            
            print("\nComputing Frechet Inception Distance on Test Set...")
            # 5. Compute FID using [-1, 1] images
            fid = compute_fid(
                x_real=x_real, 
                x_gen=sampled_images_fid, 
                device=args.device, 
                classifier_ckpt="checkpoints/mnist_classifier.pth"
            )
            print(f"FID = {np.real(fid):.4f}")

            # 6. Un-normalize [-1, 1] -> [0, 1] purely for saving the visualization
            sampled_images_save = sampled_images_fid / 2.0 + 0.5
            save_image(sampled_images_save, args.samples, nrow=8)
            print(f"Saved {args.samples}")