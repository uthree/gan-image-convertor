import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import numpy as np
import os 

from tqdm import tqdm

class ImageDataset(torch.utils.data.Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, dir_pathes, size=4):
        super(ImageDataset, self).__init__()
        self.dir_pathes = dir_pathes
        self.pathes = []
        self.size = size
        for dir_path in tqdm(dir_pathes):
            tqdm.write(f"loading {dir_path}...")
            p = os.listdir(dir_path)
            p = [os.path.join(dir_path, i) for i in p]
            self.pathes += p
            tqdm.write(f"loaded {len(p)} images.")
        self.len = len(self.pathes)
    
    def set_size(self, size):
        self.size = size
        
    def __getitem__(self, index):
        path = self.pathes[index]
        img = Image.open(path)
        img = img.resize((self.size, self.size))
        img = img.convert('RGB')
        img = img.resize([self.size, self.size])
        img = np.transpose(np.array(img), (2, 0, 1)) / 255.0
        img = torch.from_numpy(np.array(img)).float()
        return img

    def __len__(self):
        return self.len