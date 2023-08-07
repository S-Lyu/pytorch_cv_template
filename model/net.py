import torch.nn as nn
import torch


class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x