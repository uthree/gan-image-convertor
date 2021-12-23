import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import PIL
import glob
import numpy as np

class ImageDataset(torch.utils.data.Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, dir_pathes, size=4):
        super(ImageDataset, self).__init__()
        self.dir_pathes = dir_pathes
        self.pathes = []
        self.size = size
        for dir_path in dir_pathes:
            self.pathes += glob.glob(dir_path)
        self.len = len(self.pathes)
        
    def __getitem__(self, index):
        path = self.pathes[index]
        img = PIL.Image.open(path)
        img = img.resize((self.size, self.size))
        img = img.convert('RGB')
        img = torch.from_numpy(np.array(img)).float()
        return img

    def __len__(self):
        return self.len