# Code for DTU course 02460 (Advanced Machine Learning Spring)
# Exercise 2.6: VAE with Flow Prior

import torch
import torch.nn as nn
import torch.distributions as td
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import time
import os

from fid import compute_fid
from data_utils import get_mnist_dataloaders

# ==========================================
# 1. FLOW MODULES (From Ex 2.4/2.5)
# ==========================================

class GaussianBase(nn.Module):
    def __init__(self, D):
        super(GaussianBase, self).__init__()
        self.D = D
        self.mean = nn.Parameter(torch.zeros(self.D), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.D), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class MaskedCouplingLayer(nn.Module):
    def __init__(self, scale_net, translation_net, mask):
        super(MaskedCouplingLayer, self).__init__()
        self.scale_net = scale_net
        self.translation_net = translation_net
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, z):
        z_frozen = z * self.mask
        z_active = z * (1 - self.mask)
        s = self.scale_net(z_frozen)
        t = self.translation_net(z_frozen)
        x = z_frozen + (1 - self.mask) * (z_active * torch.exp(s) + t)
        log_det_J = torch.sum((1 - self.mask) * s, dim=1)
        return x, log_det_J
    
    def inverse(self, x):
        x_frozen = x * self.mask
        x_active = x * (1 - self.mask)
        s = self.scale_net(x_frozen)
        t = self.translation_net(x_frozen)
        z = x_frozen + (1 - self.mask) * (x_active - t) * torch.exp(-s)
        log_det_J = -torch.sum((1 - self.mask) * s, dim=1)
        return z, log_det_J

class Flow(nn.Module):
    def __init__(self, base, transformations):
        super(Flow, self).__init__()
        self.base = base
        self.transformations = nn.ModuleList(transformations)

    def forward(self, z):
        sum_log_det_J = 0
        for T in self.transformations:
            x, log_det_J = T(z)
            sum_log_det_J += log_det_J
            z = x
        return x, sum_log_det_J
    
    def inverse(self, x):
        sum_log_det_J = 0
        for T in reversed(self.transformations):
            z, log_det_J = T.inverse(x)
            sum_log_det_J += log_det_J
            x = z
        return z, sum_log_det_J
    
    def log_prob(self, x):
        z, log_det_J = self.inverse(x)
        return self.base().log_prob(z) + log_det_J
    
    def sample(self, sample_shape=(1,)):
        z = self.base().sample(sample_shape)
        return self.forward(z)[0]

# ==========================================
# 2. VAE MODULES
# ==========================================

class FlowPrior(nn.Module):
    def __init__(self, M, num_transformations=5, num_hidden=64):
        """
        Define a Flow-based prior over the latent space.
        """
        super(FlowPrior, self).__init__()
        self.M = M
        
        # Base distribution: Standard Gaussian
        base = GaussianBase(M)
        
        # Transformations
        transformations = []
        # Simple alternating mask
        mask = torch.zeros(M)
        mask[::2] = 1 # [1, 0, 1, 0, ...]
        
        for i in range(num_transformations):
            mask = 1 - mask # Flip mask
            
            scale_net = nn.Sequential(
                nn.Linear(M, num_hidden), nn.Tanh(),
                nn.Linear(num_hidden, num_hidden), nn.Tanh(),
                nn.Linear(num_hidden, M), nn.Tanh() # Tanh for stability
            )
            translation_net = nn.Sequential(
                nn.Linear(M, num_hidden), nn.Tanh(),
                nn.Linear(num_hidden, num_hidden), nn.Tanh(),
                nn.Linear(num_hidden, M)
            )
            transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask.clone()))
            
        self.flow = Flow(base, transformations)

    def forward(self):
        """
        Return the flow object itself (which implements log_prob and sample).
        """
        return self.flow

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

class VAE(nn.Module):
    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        q = self.encoder(x)
        z = q.rsample()

        # Get the prior (Flow)
        p = self.prior()

        # KL Divergence: log q(z|x) - log p(z)
        # p(z) is computed by the Flow's log_prob method
        kl = q.log_prob(z) - p.log_prob(z)
        
        # Reconstruction term
        recon_loss = self.decoder(z).log_prob(x)
        
        return torch.mean(recon_loss - kl, dim=0)

    def sample(self, n_samples=1):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        return -self.elbo(x)

# ==========================================
# 3. TRAINING & EVALUATION
# ==========================================

def train(model, optimizer, data_loader, epochs, device):
    model.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate_elbo(model, data_loader, device):
    model.eval()
    total_elbo = 0
    total_samples = 0
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            total_elbo += model.elbo(x).item() * x.size(0)
            total_samples += x.size(0)
    return total_elbo / total_samples

def plot_posterior_vs_prior(model, data_loader, device, save_path='posterior_prior.png'):
    """
    Plots samples from the approximate posterior q(z|x) vs samples from the learned prior p(z).
    """
    model.eval()
    posterior_z = []
    
    print("Collecting posterior samples...")
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            posterior_z.append(z.cpu().numpy())
    
    posterior_z = np.concatenate(posterior_z, axis=0)
    
    print("Collecting prior samples...")
    with torch.no_grad():
        # Sample same amount from prior as we have from posterior
        prior_z = model.prior().sample(torch.Size([posterior_z.shape[0]])).cpu().numpy()

    # PCA Projection
    print(f"Projecting to 2D (Latent Dim: {posterior_z.shape[1]})...")
    pca = PCA(n_components=2)
    # Fit PCA on posterior, transform both
    posterior_2d = pca.fit_transform(posterior_z)
    prior_2d = pca.transform(prior_z)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(posterior_2d[:, 0], posterior_2d[:, 1], alpha=0.1, label='Posterior q(z|x)', s=5, c='blue')
    plt.scatter(prior_2d[:, 0], prior_2d[:, 1], alpha=0.1, label='Prior p(z) (Flow)', s=5, c='red')
    plt.legend()
    plt.title("Posterior vs Learned Flow Prior (PCA)")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    import argparse
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode')
    parser.add_argument('--model', type=str, default='vae_flow_model.pt', help='Model file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--latent-dim', type=int, default=32)
    args = parser.parse_args()

    # Automatically create directory for the model if it doesn't exist
    dir_name = os.path.dirname(args.model)
    if dir_name:  
        os.makedirs(dir_name, exist_ok=True)

    # [cite_start]Load Data via data_utils (Binarized for Bernoulli VAE) [cite: 27]
    print("Loading Binarized MNIST (Flatten=True)...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size, 
        binarize=True,   # Must be True for Bernoulli VAE
        flatten=True     # Must be True for FC networks
    )

    # Instantiate Components
    M = args.latent_dim
    
    # 1. Flow Prior
    prior = FlowPrior(M, num_transformations=5, num_hidden=64)
    
    # 2. Encoder
    encoder_net = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, M*2),
    )
    encoder = GaussianEncoder(encoder_net)
    
    # 3. Bernoulli Decoder
    decoder_net = nn.Sequential(
        nn.Linear(M, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 784)
    )
    decoder = BernoulliDecoder(decoder_net)

    # 4. VAE
    model = VAE(prior, decoder, encoder).to(args.device)

    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(model, optimizer, train_loader, args.epochs, args.device)
        torch.save(model.state_dict(), args.model)
        print(f"Model saved to {args.model}")
        
    elif args.mode == 'evaluate':
        model.load_state_dict(torch.load(args.model, map_location=args.device))
        model.eval() # Make sure to set eval mode!
        
        # 1. Evaluate ELBO
        elbo = evaluate_elbo(model, test_loader, args.device)
        print(f"Test Set ELBO: {elbo:.4f}")
        
        # 2. Plot Posterior vs Prior
        plot_posterior_vs_prior(model, test_loader, args.device)
        
        # 3. Generate Samples and Measure Time
        print("\nSampling and computing FID...")
        with torch.no_grad():
            # Start timer
            start_time = time.time()
            
            # Sample from the VAE (outputs are in {0, 1})
            samples = model.sample(64)
            
            # End timer
            sampling_time = time.time() - start_time
            print(f"Sampling 64 images took: {sampling_time:.4f} seconds ({64/sampling_time:.2f} samples/sec)")

            # Reshape generated samples
            samples = samples.view(64, 1, 28, 28).to(args.device)
            
            # Save the {0, 1} samples as an image
            save_image(samples, 'vae_flow_samples.png')
            print("Samples saved to vae_flow_samples.png")

            # --- FID CALCULATION ---
            
            # Map {0, 1} generated samples to {-1, 1} for FID
            samples_fid = (samples * 2.0) - 1.0
            
            # Get a batch of real images from the test loader
            # Note: Your train_loader for this VAE outputs {0, 1}
            x_real, _ = next(iter(test_loader))
            x_real = x_real[:64].view(64, 1, 28, 28).to(args.device)
            
            # Map {0, 1} real samples to {-1, 1} for FID
            x_real_fid = (x_real * 2.0) - 1.0
            
            print("Computing Frechet Inception Distance...")
            # Ensure the classifier_ckpt path points to your actual checkpoint file!
            fid = compute_fid(
                x_real=x_real_fid, 
                x_gen=samples_fid, 
                device=args.device, 
                classifier_ckpt="checkpoints/mnist_classifier.pth" 
            )
            print(f"FID = {np.real(fid):.4f}")