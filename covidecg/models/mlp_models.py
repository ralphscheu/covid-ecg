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
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(), 
            nn.Linear(hidden_size // 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


'''
    CNN 2D ( leads x time )
'''
class CNN2D(nn.Module):

    def __init__(self, dense_hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 2)),
            nn.Dropout(0.1),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 2)),
            nn.Dropout(0.1),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 2)),
            nn.Dropout(0.1),
            
            nn.Flatten(),
            nn.LazyLinear(dense_hidden_size),  # automatically infers input shape
            nn.Linear(dense_hidden_size, dense_hidden_size),
            nn.Linear(dense_hidden_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        '''Forward pass'''
        # print("CNN input:", x.shape)
        return self.layers(x[:, np.newaxis, :, :])


'''
    CNN 1D ( leads as channels )
'''
class CNN1D(nn.Module):

    def __init__(self, dense_hidden_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=24, kernel_size=10, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            
            nn.Conv1d(in_channels=24, out_channels=48, kernel_size=10, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            
            nn.Conv1d(in_channels=48, out_channels=96, kernel_size=10, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.Dropout(0.5),
            
            nn.Flatten(),
            nn.LazyLinear(dense_hidden_size),  # automatically infers input shape
            nn.LazyLinear(dense_hidden_size // 2),
            nn.LazyLinear(2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        '''Forward pass'''
        x = x.reshape(x.shape[0], 12, -1)  # undo channel flattening
        return self.layers(x)
