import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Fix 404 Error using Resource Override ---
datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bb3b8fb87f'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]

def get_mnist_dataloaders(batch_size=128, binarize=False, flatten=True, data_dir='data/'):
    """
    Loads the MNIST dataset and returns training and testing dataloaders.
    
    Args:
        batch_size (int): The number of samples per batch.
        binarize (bool): If True, binarizes the images to {0, 1} (for Bernoulli VAE).
                         If False, dequantizes and scales to [-1, 1] (for DDPM).
        flatten (bool): If True, flattens the images to 1D vectors of size (784,).
                        If False, keeps them as 2D tensors of size (1, 28, 28).
        data_dir (str): Directory to store the downloaded dataset.
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform_list = [transforms.ToTensor()]
    
    if binarize:
        # Binarize: Map [0, 1] -> {0, 1}
        transform_list.append(transforms.Lambda(lambda x: (x > 0.5).float()))
    else:
        # Continuous: Dequantize and map to [-1, 1]
        transform_list.append(transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255.0))
        transform_list.append(transforms.Lambda(lambda x: (x - 0.5) * 2.0))
        
    if flatten:
        # Flatten (1, 28, 28) -> (784,)
        transform_list.append(transforms.Lambda(lambda x: x.flatten()))
        
    # Compose all selected transformations
    transform = transforms.Compose(transform_list)
    
    # Load datasets
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader