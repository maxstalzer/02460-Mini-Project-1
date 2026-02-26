# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import numpy as np
import time
from torchvision.utils import save_image

# --- NEW IMPORTS ---
from fid import compute_fid
from data_utils import get_mnist_dataloaders
from unet import Unet

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.
        """
        batch_size = x.shape[0]

        # Sample a random timestep t uniformly for each sample in the batch
        t = torch.randint(0, self.T, (batch_size,), device=x.device)

        # Sample noise epsilon ~ N(0, I)
        epsilon = torch.randn_like(x)

        # Retrieve the cumulative product of alphas (alpha_bar) for the sampled timesteps
        alpha_bar_t = self.alpha_cumprod[t].unsqueeze(1)

        # Forward Process: Compute x_t (the noisy data)
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon

        # Format t for the network: shape (batch_size, 1), scaled by T
        t_input = (t.float() / self.T).unsqueeze(1)
        
        # Predict the noise using the neural network
        epsilon_theta = self.network(x_t, t_input)

        # Calculate the squared error ||epsilon - epsilon_theta||^2
        # Since x is always flattened to 2D [batch, 784], we just sum over dim=1
        neg_elbo = torch.sum((epsilon - epsilon_theta) ** 2, dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            # Create timestep tensor and scale it exactly as done in training
            t_tensor = torch.full((shape[0], 1), t, dtype=torch.float32, device=x_t.device)
            t_input = t_tensor / self.T
            
            epsilon_theta = self.network(x_t, t_input)
            
            if t > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)

            alpha_t = self.alpha[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            beta_t = self.beta[t]

            x_t = (1/torch.sqrt(alpha_t)) * (x_t - (1 - alpha_t)/torch.sqrt(1 - alpha_cumprod_t)*epsilon_theta) + torch.sqrt(beta_t)*z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f" {loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


class FcNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden):
        """
        Initialize a fully connected network for the DDPM.
        """
        super(FcNetwork, self).__init__()
        self.network = nn.Sequential(nn.Linear(input_dim+1, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, num_hidden), nn.ReLU(), 
                                     nn.Linear(num_hidden, input_dim))

    def forward(self, x, t):
        """"
        Forward function for the network.
        """
        x_t_cat = torch.cat([x, t], dim=1)
        return self.network(x_t_cat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--network', type=str, default='fc', choices=['fc', 'unet'], help='network architecture to use (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # 1. DEFINE THE NETWORK AND SAMPLE SHAPE
    if args.network == 'unet':
        print("Using U-Net Architecture...")
        network = Unet()
        flatten = True
        sample_shape = (64, 784)
    else:
        print("Using Fully Connected Architecture...")
        D = 784 
        num_hidden = 512 
        network = FcNetwork(D, num_hidden)
        flatten = True
        sample_shape = (64, D)

    # 2. GENERATE THE DATA
    # Binarize is False because we are training on standard MNIST
    print(f"Loading MNIST (Flatten={flatten})...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size, 
        binarize=False, 
        flatten=flatten
    )
        
    T = 1000
    model = DDPM(network, T=T).to(args.device)

    # 3. CHOOSE MODE TO RUN
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, args.device)
        torch.save(model.state_dict(), args.model)
        print(f"Model saved to {args.model}")

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        print(f"\nSampling (saving to {args.samples})...")
        with torch.no_grad():
            # Start timer for wall-clock sampling time 
            start_time = time.time()
            
            # Generate latents/images via DDPM
            samples = model.sample(sample_shape)
            
            # End timer
            sampling_time = time.time() - start_time
            print(f"Sampling 64 images took: {sampling_time:.4f} seconds ({64/sampling_time:.2f} samples/sec)")
            
            # Keep samples in [-1, 1] range for FID calculation
            samples_fid = samples.view(64, 1, 28, 28).to(args.device)
            
            # Un-normalize [-1, 1] -> [0, 1] purely for saving the visualization
            samples_save = samples_fid / 2.0 + 0.5
            save_image(samples_save, args.samples, nrow=8)
            print(f"Saved MNIST samples grid to {args.samples}")

            # --- FID CALCULATION ON TEST SET ---
            print("\nComputing Frechet Inception Distance on Test Set...")
            # Get a batch of real images from the test loader
            x_real, _ = next(iter(test_loader))
            
            # Reshape real images to ensure they match (64, 1, 28, 28) regardless of FC or U-Net
            x_real_fid = x_real[:64].view(64, 1, 28, 28).to(args.device)
            
            # Compute FID using [-1, 1] images 
            fid = compute_fid(
                x_real=x_real_fid, 
                x_gen=samples_fid, 
                device=args.device, 
                classifier_ckpt="checkpoints/mnist_classifier.pth"
            )
            print(f"FID = {np.real(fid):.4f}")