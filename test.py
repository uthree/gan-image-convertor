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

num_generate_images = 10

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    m = torch.load("./mapping.pt").to(device)
    print("Loaded mapping network.")
    g = torch.load("./generator.pt").to(device)
    print("Loaded generator network.")
    
    with torch.no_grad():
        for i in tqdm(range(num_generate_images)):
            z = torch.randn(1, 512).to(device)
            style = m(z)
            image = g(style)
            
            # save image
            image = image.cpu().numpy()[0]
            image = np.transpose(image, (1, 2, 0))
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(f"./tests/{i}.png")

if __name__ == "__main__":
    main()
        
        