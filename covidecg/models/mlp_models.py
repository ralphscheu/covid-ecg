import numpy as np
import torch.functional as F
import torch.nn as nn


'''
    Multilayer Perceptron
'''
class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size), 
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)