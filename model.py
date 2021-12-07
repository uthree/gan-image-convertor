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
                        nn.GroupNorm(channels, num_groups=groups),
                        nn.Conv2d(channels, channels, 3, padding=1, groups=groups, padding_mode="replicate"),
                        nn.GELU(),
                    ),
                    nn.Sequential(
                        nn.GroupNorm(channels, num_groups=groups),
                        nn.Conv2d(channels, channels, 3, padding=1, groups=groups, padding_mode="replicate"),
                        nn.GELU(),
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
    def __init__(self, dim, num_layers=8, activation=nn.GELU):
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

