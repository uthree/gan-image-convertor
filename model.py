import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import revtorch as rv

class ReversibleConvSequence(nn.Module):
    """Some Information about ReversibleConvSequence"""
    def __init__(self, channels, groups=1, num_layers=1):
        super(ReversibleConvSequence, self).__init__()
        blocks = nn.ModuleList()
        for i in range(num_layers):
            blocks.append(
                rv.ReversibleBlock(
                    nn.Sequential(
                        nn.Conv2d(channels, channels*2, 3, padding=0, groups=groups),
                        nn.ConvTranspose2d(channels*2, channels, kernel_size=3, padding=0, groups=groups),
                        nn.LeakyReLU()
                    ),
                    nn.Sequential(
                        nn.Conv2d(channels, channels*2, 3, padding=0, groups=groups),
                        nn.ConvTranspose2d(channels*2, channels, kernel_size=3, padding=0, groups=groups),
                        nn.LeakyReLU()
                    ),
                    split_along_dim=1
                )
            )
        self.seq = rv.ReversibleSequence(blocks)

    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x 

class MappingNetwork(nn.Module):
    """Some Information about MappingNetwork"""
    def __init__(self, dim, num_layers=8, activation=nn.LeakyReLU):
        super(MappingNetwork, self).__init__()
        blocks = nn.ModuleList()
        for i in range(num_layers):
            blocks.append(
                rv.ReversibleBlock(
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim),
                        activation(),
                    ),
                    nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, dim),
                        activation(),
                    ),
                    split_along_dim=1
                )
            )
        self.seq = rv.ReversibleSequence(blocks)

    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=1)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x = (x1 + x2) / 2
        return x

class AddChannelWiseBias(nn.Module):
    """Some Information about AddChannelWiseBias"""
    def __init__(self, channels):
        super(AddChannelWiseBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        oC,*_ = self.bias.shape
        shape = (1,oC) if x.ndim==2 else (1,oC,1,1)
        y = x + self.bias.view(*shape)
        return y

class AdaptiveInstanceNormalization(nn.Module):
    """Some Information about AdaptiveInstanceNormalization"""
    def __init__(self, feature_channels, map_channels):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.add_bias = AddChannelWiseBias(feature_channels*2)
        self.fc = nn.Linear(map_channels, feature_channels*2)
        
    def forward(self, x, style):
        #N,D = w.shape
        N,C,H,W = x.shape
        _vec = self.add_bias( self.fc(style) ).view(N,2*C,1,1) # (N,2C,1,1)
        scale, shift = _vec[:,:C,:,:], _vec[:,C:,:,:] # (N,C,1,1), (N,C,1,1)
        return (scale+1) * F.instance_norm(x, eps=1e-8) + shift

class GeneratorLayer(nn.Module):
    """Some Information about GeneratorLayer"""
    def __init__(self, in_channels, out_channels, map_channels, upscale_factor=2, num_layers=1):
        super(GeneratorLayer, self).__init__()
        self.upscale = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
        self.conv = ReversibleConvSequence(in_channels, num_layers=num_layers)
        self.adain = AdaptiveInstanceNormalization(in_channels, map_channels) 
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.torgb = ToRGB(out_channels)
    def forward(self, x, style):
        x = self.upscale(x)
        x = self.conv(x)
        x = self.adain(x, style)
        x = self.conv_out(x)
        return x

class DiscriminatorLayer(nn.Module):
    """Some Information about DiscriminatorLayer"""
    def __init__(self, input_channels, output_channels, num_layers):
        super(DiscriminatorLayer, self).__init__()
        self.conv = ReversibleConvSequence(input_channels, num_layers=num_layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(input_channels, output_channels, kernel_size=1, padding=0)
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.conv_out(x)
        return x

class ToRGB(nn.Module):
    """Some Information about ToRGB"""
    def __init__(self, input_channels):
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(input_channels, 3, kernel_size=1, padding=0)
    def forward(self, x):
        return self.conv(x)

class FromRGB(nn.Module):
    """Some Information about FromRGB"""
    def __init__(self, output_channels):
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, output_channels, kernel_size=1, padding=0)
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    """Some Information about Generator"""
    def __init__(self, num_layers_per_resolution=2, channels=512, map_channels=512):
        super(Generator, self, ).__init__()
        self.layers = nn.ModuleList()
        self.num_layers_per_resolution = num_layers_per_resolution
        self.const_4x4 = nn.Parameter(torch.ones(1,channels,4,4))
        self.const_4x4_channels = channels
        self.map_channels = map_channels
        
    def forward(self, styles, noise_seed=0, noise_gains=None):
        torch.manual_seed(noise_seed)
        if noise_gains == None:
            noise_gains = [0.1]*len(styles)
        x = self.const_4x4
        for i, layer in enumerate(self.layers):
            x = x + torch.randn(x.shape) * noise_gains[i]
            x = x + layer(x, styles[i])
        return x
    
    def add_layer(self, channels = 256):
        self.last_layer_channels = channels
        if len(self.layers) == 0:
            self.layers.append(
                GeneratorLayer(
                    self.const_4x4_channels, channels, self.map_channels, upscale_factor=1, num_layers=self.num_layers_per_resolution
                )
            )
        else:
            self.layers.append(
                GeneratorLayer(self.last_layer_channels, out_channels=channels, map_channels=self.map_channels, upscale_factor=2, num_layers=self.num_layers_per_resolution)
            )


g = Generator()
print(g)