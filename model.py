import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MappingNetwork(nn.Module):
    """Some Information about MappingNetwork"""
    def __init__(self, dim, num_layers=8, activation=nn.LeakyReLU):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential([
            nn.Sequential(
                nn.Linear(dim, dim),
                activation()
            ) for _ in range(num_layers) ]
        )
    def forward(self, x):
        return self.seq(x)

class AdaptiveInstanceNormalization(nn.Module):
    """Some Information about AdaptiveInstanceNormalization"""
    def __init__(self, channels, style_dim):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.affine = nn.Linear(style_dim, channels * 2)
        self.norm = nn.InstanceNorm2d(channels)
        
    def forward(self, x, style):
        scale, bias = self.affine(style).chunk(2, dim=1)
        scale = scale.unsqueeze(2).unsqueeze(3)
        bias = bias.unsqueeze(2).unsqueeze(3)
        x = self.norm(x)
        x = x * scale + bias
        return x
    
class NoiseLayer(nn.Module):
    """Some Information about AddNoiseLayer"""
    def __init__(self, channel, gain = 1.0):
        super(NoiseLayer, self).__init__()
        self.shape = (1, channel, 1, 1)
        
    def forward(self, x):
        return x + self.weight * torch.randn(self.shape) * self.gain

# generate and upscale
class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, in_channel, out_channel, style_dim, noise_gain=1.0):
        super(GeneratorBlock, self).__init__()
        self.noise = NoiseLayer(in_channel, gain=noise_gain)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU()
        self.adain = AdaptiveInstanceNormalization(in_channel, style_dim)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU()
        
    def forward(self, x, style):
        x = self.noise(x)
        x = self.upsample(x)
        x = self.lrelu1(self.conv1(x)) + x
        x = self.adain(x, style)
        x = self.lrelu2(self.conv2(x)) + x
        x = self.upsample(x)
        return x
    
