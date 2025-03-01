from .models import Generator, Discriminator
import torch
import torch.nn as nn
import numpy as np
from .utils import plot_samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_epoch(gen, disc, gen_opt, disc_opt, loss_fn, 
                data_ldr, batch_sz, noise_dim):
    gen.train()
    disc.train()
    disc_epoch_loss = list()
    gen_epoch_loss = list()
    for batch_num, input_data in enumerate(data_ldr):
        #discriminator step
        disc_opt.zero_grad()
        for param in gen.parameters():
            param.requires_grad = False
        for param in disc.parameters():
            param.requires_grad = True
        X, y = input_data
        X = torch.flatten(X, 1).float()
        noise = torch.randn(batch_sz, noise_dim)
        noise_data = gen(noise)
        real_pred = disc(X.float()).squeeze()
        fake_pred = disc(noise_data).squeeze()
        labels1 = torch.ones_like(y).float()
        labels2 = torch.zeros(batch_sz).float()
        disc_loss1 = loss_fn(real_pred, labels1)
        disc_loss2 = loss_fn(fake_pred, labels2)
        disc_loss = (disc_loss1+disc_loss2)/2.
        disc_epoch_loss.append(disc_loss.detach().numpy())
        disc_loss.backward()
        disc_opt.step()

        #generator step
        gen_opt.zero_grad()
        for param in gen.parameters():
            param.requires_grad = True
        for param in disc.parameters():
            param.requires_grad = False
        noise = torch.randn(batch_sz, noise_dim)
        output = disc(gen(noise)).squeeze()
        labels = torch.ones(batch_sz).float()
        gen_loss = loss_fn(output, labels)
        gen_epoch_loss.append(gen_loss.detach().numpy())
        gen_loss.backward()
        gen_opt.step()

        ge_loss = np.asarray(gen_epoch_loss)
        de_loss = np.asarray(disc_epoch_loss)

    return ge_loss, de_loss

def train_vanilla_gan(dataloader, epochs=100, samples_to_plot=6, batch_sz=128, noise_dim=100):
    #Setup
    gen = Generator(noise_dim).to(device)
    disc = Discriminator().to(device)

    #Optimizers
    gen_opt = torch.optim.Adam(gen.parameters())
    disc_opt = torch.optim.Adam(disc.parameters())

    #Set loss
    loss_fn = nn.BCELoss()

    #Losses
    gen_loss=[]
    disc_loss = []

    for epoch in range(epochs):

        #train epoch        
        gen_epoch_loss, disc_epoch_loss = train_epoch(gen, disc, gen_opt, disc_opt, loss_fn, 
                                                      dataloader, batch_sz, noise_dim)

        plot_samples(gen, samples_to_plot, noise_dim)

        gen_loss.append(gen_epoch_loss)
        disc_loss.append(disc_epoch_loss)

    return gen, np.asarray(gen_loss), np.asarray(disc_loss)
