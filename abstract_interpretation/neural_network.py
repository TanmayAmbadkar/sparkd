import torch
import torch.nn as nn
from abstract_interpretation.domains import Zonotope, Box

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        if isinstance(x, Zonotope) or isinstance(x, Box):
            W = self.linear.weight.detach().numpy()
            b = self.linear.bias.detach().numpy()
            return x.affine_transform(W, b)
        else:
            return self.linear(x)

class ReLULayer(nn.Module):
    def forward(self, x):
        if isinstance(x, Zonotope) or isinstance(x, Box):
            return x.relu()
        else:
            return torch.relu(x)

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x