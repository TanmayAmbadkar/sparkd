import torch
import torch.nn as nn
from abstract_interpretation.domains import Zonotope, Box, DeepPoly

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        if isinstance(x, Zonotope) or isinstance(x, Box) or isinstance(x, DeepPoly):
            W = self.linear.weight.detach().numpy()
            b = self.linear.bias.detach().numpy()
            return x.affine_transform(W, b)
        else:
            return self.linear(x)
    
    def return_w_b(self):
        return self.linear.weight.detach().numpy(), self.linear.bias.detach().numpy()
        
    def __str__(self):
        return f"Linear Layer, Weights Shape: {self.linear.weight.detach().numpy().shape}, Bias Shape: {self.linear.bias.detach().numpy().shape}"

class ReLULayer(nn.Module):
    def forward(self, x):
        if isinstance(x, Zonotope) or isinstance(x, Box) or isinstance(x, DeepPoly):
            return x.relu()
        else:
            return torch.relu(x)
        
    def __str__(self):
        return "ReLU Layer"
        
class TanhLayer(nn.Module):
    def forward(self, x):
        if isinstance(x, Zonotope) or isinstance(x, Box)or isinstance(x, DeepPoly):
            return x.tanh()
        else:
            return torch.tanh(x)
    
    def __str__(self):
        return "Tanh Layer"
        

class SigmoidLayer(nn.Module):
    def forward(self, x):
        if isinstance(x, Zonotope) or isinstance(x, Box) or isinstance(x, DeepPoly):
            return x.sigmoid()
        else:
            return torch.sigmoid(x)
    
    
    def __str__(self):
        return "Sigmoid Layer"

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x