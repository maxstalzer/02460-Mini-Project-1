# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm


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

            x_t = (1/torch.sqrt(self.alpha[t])) * (x_t - (1 - self.alpha[t])/torch.sqrt(1 - self.alpha_cumprod[t])*epsilon_theta) + torch.sqrt(self.beta[t])*z

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
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    import ToyData
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Import your new Unet class
    from unet import Unet

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--data', type=str, default='tg', choices=['tg', 'cb', 'mnist'], help='dataset to use (default: %(default)s)')
    parser.add_argument('--network', type=str, default='fc', choices=['fc', 'unet'], help='network architecture to use (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=10000, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='V', help='learning rate for training (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    # 1. GENERATE THE DATA
    if args.data == 'mnist':
        if args.batch_size == 10000:
            print("\nWARNING: Batch size 10000 is too large for MNIST. Defaulting to 128 to prevent OOM errors.\n")
            args.batch_size = 128

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255),
            transforms.Lambda(lambda x: (x - 0.5) * 2.0),
            transforms.Lambda(lambda x: x.flatten())
        ])
        
        train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        
        D = 784 
        num_hidden = 512 
    else:
        n_data = 10000000
        toy = {'tg': ToyData.TwoGaussians, 'cb': ToyData.Chequerboard}[args.data]()
        transform = lambda x: (x-0.5)*2.0
        train_loader = torch.utils.data.DataLoader(transform(toy().sample((n_data,))), batch_size=args.batch_size, shuffle=True)
        
        D = next(iter(train_loader)).shape[1]
        num_hidden = 64

    # 2. DEFINE THE NETWORK AND MODEL
    if args.network == 'unet':
        if args.data != 'mnist':
            raise ValueError("The provided Unet is hardcoded for 28x28 MNIST images. Please use --data mnist.")
        print("Using U-Net Architecture...")
        network = Unet()
    else:
        print("Using Fully Connected Architecture...")
        network = FcNetwork(D, num_hidden)
        
    T = 1000
    model = DDPM(network, T=T).to(args.device)

    # 3. CHOOSE MODE TO RUN
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, args.device)
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()

        if args.data == 'mnist':
            with torch.no_grad():
                samples = model.sample((64, D)).cpu()
            
            samples = samples / 2 + 0.5
            samples = samples.view(-1, 1, 28, 28)
            save_image(samples, args.samples, nrow=8)
            print(f"Saved MNIST samples grid to {args.samples}")

        else:
            with torch.no_grad():
                samples = (model.sample((10000, D))).cpu() 

            samples = samples / 2 + 0.5
            coordinates = [[[x,y] for x in np.linspace(*toy.xlim, 1000)] for y in np.linspace(*toy.ylim, 1000)]
            prob = torch.exp(toy().log_prob(torch.tensor(coordinates)))

            fig, ax = plt.subplots(1, 1, figsize=(7, 5))
            im = ax.imshow(prob, extent=[toy.xlim[0], toy.xlim[1], toy.ylim[0], toy.ylim[1]], origin='lower', cmap='YlOrRd')
            ax.scatter(samples[:, 0], samples[:, 1], s=1, c='black', alpha=0.5)
            ax.set_xlim(toy.xlim)
            ax.set_ylim(toy.ylim)
            ax.set_aspect('equal')
            fig.colorbar(im)
            plt.savefig(args.samples)
            plt.close()