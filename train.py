import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import ImageDataset
from model import MappingNetwork, Generator, Discriminator

from tqdm import tqdm
from PIL import Image
import numpy as np

initial_channels = 512
channels_growning =   [ 256, 128, 64, 32, 16, 8]
batch_size_growning = [  16,  16,  8,  4,  2, 1] 
style_dim = 512 
num_epoch_per_resolution = 1
num_workers=15

pathes = [
    "/mnt/d/local-develop/lineart2image_data_generator/colorized/"
    #"/mnt/d/local-develop/AnimeIconGenerator128x_v3/small_dataset128x/"
]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def main():
    if os.path.exists("./mapping.pt"):
        m = torch.load("./mapping.pt")
        print("Loaded mapping network.")
    else:
        m = MappingNetwork(style_dim, num_layers=8, activation=nn.LeakyReLU)
        print("Initialized mapping network.")

    if os.path.exists("./generator.pt"):
        g = torch.load("./generator.pt")
        print("Loaded generator network.")
    else:
        g = Generator(style_dim=style_dim, initial_channels=initial_channels)
        print("Initialized generator network.")

    if os.path.exists("./discriminator.pt"):
        d = torch.load("./discriminator.pt")
        print("Loaded discriminator network.")
    else:
        d = Discriminator(initial_channels=initial_channels)
        print("Initialized discriminator network.")

    dataset = ImageDataset(pathes, size=4)

    for res_id in range(len(g.layers)-1, len(channels_growning)-1):
        g.add_layer(channels_growning[res_id])
        d.add_layer(channels_growning[res_id])
        resolution = 4 * (2 ** len(g.layers))
        dataset.set_size(resolution)
        batch_size = batch_size_growning[res_id]
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        optimizer_m = optim.Adam(m.parameters(), lr=1e-4, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(g.parameters(), lr=1e-4, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(d.parameters(), lr=1e-4, betas=(0.5, 0.999))
        criterion = nn.BCELoss()
        
        d = d.to(device)
        g = g.to(device)
        m = m.to(device)
        
        print("Started train resolution: {}x{}".format(resolution, resolution))
        bar = tqdm(total=len(dataset) * num_epoch_per_resolution)
        len_ds = len(dataset)
        for epoch in range(num_epoch_per_resolution):
            for i, (image) in enumerate(dataloader):
                image = image.to(device)
                g.zero_grad()
                m.zero_grad()
                z = torch.randn(batch_size, style_dim)
                z = z.to(device)
                style = m(z)
                
                # train Generator
                gout = g(style)
                dout = d(gout)
                g_loss = criterion(dout, torch.ones(batch_size, 1).to(device))
                g_loss.backward()
                optimizer_g.step()
                optimizer_m.step()
                
                # train Discriminator
                d.zero_grad()
                fake = gout.detach()
                real = image
                d_loss = criterion(d(real), torch.ones(batch_size, 1).to(device)) + criterion(d(fake), torch.zeros(batch_size, 1).to(device))
                d_loss.backward()
                optimizer_d.step()
                
                bar.set_description(f"Epoch {epoch} Batch {i}, d_loss:{round(d_loss.item(), 5)} g_loss:{round(g_loss.item(), 5)} alpha: {round(g.alpha, 5)}")
                bar.update(batch_size)
                if i % 100 == 0:
                    save_model(d, g, m)
                    # output fake image to ./results
                    img = gout.detach().cpu().numpy()[0]
                    img = np.transpose(img, (1, 2, 0))
                    img = img * 255
                    img = img.astype(np.uint8)
                    Image.fromarray(img).save(f"./results/{i}.png")
                g.alpha=((epoch - num_epoch_per_resolution) / num_epoch_per_resolution) + 1 + (1/num_epoch_per_resolution/len_ds*batch_size) * i
        bar.close()
        print("Finished train resolution: {}x{}".format(resolution, resolution))
        save_model(d, g, m)
    
def save_model(d, g, m):
    torch.save(m, "mapping.pt")
    torch.save(g, "generator.pt")
    torch.save(d, "discriminator.pt")
        
if __name__ == "__main__":
    main()
    