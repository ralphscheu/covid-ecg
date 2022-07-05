import numpy as np
import torch.functional as F
import torch.nn as nn
import torchvision.models
import torch


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
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 10), stride=(1, 2), padding=(1,4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 3)),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 10), stride=(1, 2), padding=(0, 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 3)),
            
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 10), stride=(1, 2), padding=(1, 4)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 3)),
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
            nn.Dropout(0.1),
            
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


'''
    ECG Image classification model using VGG16 for feature extraction and FC layers for classification
'''
class VGG16(nn.Module):
    
    def __init__(self):
        super().__init__()
        vgg16_pretrained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        
        self.vgg_feature_extractor = nn.Sequential(
            # torchvision.transforms.Resize((224, 224)),
            vgg16_pretrained.features#,
            # vgg16_pretrained.avgpool
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),            
            # Binary Classifier
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        '''Forward pass'''

        # print("========================")
        # print("x:", x.shape)  # batch_size, *
        x = x.reshape(x.shape[0], 3, 224, 224)  # undo channel flattening
        # print("x:", x.shape)  # batch_size, channels, image_height, image_width

        x = self.vgg_feature_extractor(x)
        # print("x after feature extraction:", x.shape)
        x = nn.Flatten()(x)
        # print("x before classifier:", x.shape)
        x = self.classifier(x)
        # print("========================")
        
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 12, -1)  # undo channel/lead flattening
        # print("RNN input shape:", x.shape)  # batch_size, leads, timesteps
        x = x.swapaxes(1, 2)
        # print("RNN input shape:", x.shape)  # batch_size, timesteps, leads
        
        # The RNN also returns its hidden state but we don't use it.
        # While the RNN can also take a hidden state as input, the RNN
        # gets passed a hidden state initialized with zeros by default.
        x, _ = self.lstm(x)
        print("LSTM output:", x.shape)
        x = self.linear(x)
        x = x[:, -1, :]  # only return last state
        print("model output:", x.shape)
        return x
