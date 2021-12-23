import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MappingNetwork(nn.Module):
    """Some Information about MappingNetwork"""
    def __init__(self, dim, num_layers=8, activation=nn.LeakyReLU):
        super(MappingNetwork, self).__init__()
        self.seq = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    activation()
                ) for _ in range(num_layers)
            ]
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
    def __init__(self, channels, gain = 1.0):
        super(NoiseLayer, self).__init__()
        self.shape = (1, channels, 1, 1)
        self.gain = gain
        
    def forward(self, x):
        noise = torch.randn(self.shape)
        device = x.device
        noise = noise.to(device)
        return x + noise * self.gain

class ToRGB(nn.Module):
    """Some Information about ToRGB"""
    def __init__(self, channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv(x))


# generate and upscale
class GeneratorBlock(nn.Module):
    """Some Information about GeneratorBlock"""
    def __init__(self, in_channels, out_channels, style_dim, noise_gain=1.0):
        super(GeneratorBlock, self).__init__()
        self.noise = NoiseLayer(in_channels, gain=noise_gain)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU()
        self.adain = AdaptiveInstanceNormalization(in_channels, style_dim)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.lrelu2 = nn.LeakyReLU()
        self.torgb = ToRGB(out_channels)
        
    def forward(self, x, style):
        x = self.noise(x)
        x = self.upsample(x)
        x = self.lrelu1(self.conv1(x)) + x
        x = self.adain(x, style)
        x = self.lrelu2(self.conv2(x)) + x
        x = self.channel_conv(x)
        return x

    
class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, style_dim, initial_channels=512):
        super(Generator, self).__init__()
        self.style_dim = style_dim
        self.const = nn.Parameter(torch.ones(1, 1, 4, 4))
        self.last_channels = initial_channels
        self.layers = nn.ModuleList()
        self.alpha = 0
        
    def forward(self, style):
        if type(style) != list:
            style = [style for _ in range(len(self.layers))]
        num_layers = len(self.layers)
        outs = []
        x = self.const
        for i in range(len(self.layers)):
            x = self.layers[i](x, style[i])
            outs.append(x)
        if len(outs) == 1:
            return self.layers[-1].torgb(outs[-1])
        else:
            a = self.layers[-1].torgb(outs[-1]) * self.alpha 
            b = self.layers[-2].torgb(outs[-2]) * (1 - self.alpha)
            b = F.upsample(b, scale_factor=2, mode='bilinear', align_corners=True)
            return a + b
            
    def add_layer(self, channels):
        self.layers.append(GeneratorBlock(self.last_channels, channels, self.style_dim))
        self.last_channels = channels

class FromRGB(nn.Module):
    """Some Information about FromRGB"""
    def __init__(self, channels):
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, channels, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        return self.conv(x)
        
class DiscriminatorBlock(nn.Module):
    """Some Information about DiscriminatorBlock"""
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.in_channels = in_channels
        self.fromRGB = FromRGB(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.lrelu2 = nn.LeakyReLU()
        self.channel_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.down_sample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        
        x = self.lrelu1(self.conv1(x)) + x
        x = self.norm(x)
        x = self.lrelu2(self.conv2(x)) + x
        x = self.channel_conv(x)
        x = self.down_sample(x)
        return x
    
class Discriminator(nn.Module):
    """Some Information about Discriminator"""
    def __init__(self, initial_channels=512):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList([])
        self.last_channels = initial_channels
        self.affine = nn.Linear(initial_channels * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.layers[0].fromRGB(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.view(x.size(0), -1)
        x = self.affine(x)
        x = self.sigmoid(x)
        return x
    
    def add_layer(self, channels):
        self.layers.insert(0, DiscriminatorBlock(channels, self.last_channels))
        self.last_channels  = channels
