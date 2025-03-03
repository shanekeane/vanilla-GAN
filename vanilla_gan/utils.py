import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as cm

def plot_samples(model, no_samples, noise_dim):
    """
    Plots horizontal array of samples
    
    Input:  model      - generator model
            no_samples - how many samples to print (int)
            noise_dim  - dimension of generator input noise (int)
   
    """   

    model.eval()
    gen_noise = torch.randn(no_samples, noise_dim)
    gen_images = model(gen_noise)
    gen_images = gen_images.reshape(gen_images.shape[0], 28, 28)
    fig, ax = plt.subplots(1,no_samples)
    for i in range(no_samples):
        ax[i].imshow((-gen_images[i]).squeeze().detach().numpy(), cmap=plt.cm.binary)
        ax[i].axis('off')
    plt.show()

def load_data(data="mnist", batch_size=128):
    """
    Returns data loader for MNIST or Fashion MNIST
    
    Input:  data       - "mnist" or "fmnist"
            batch_size - batch size for data loader

    Output: data loader
   
    """   

    #Load training and testing data
    if data=="mnist":
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    elif data=="fmnist":
        train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
        test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())    
        
    all_data = torch.utils.data.ConcatDataset([train_data, test_data])
    dataloader = DataLoader(all_data, batch_size=batch_size, shuffle=True)
    return dataloader
