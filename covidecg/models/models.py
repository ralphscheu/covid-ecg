import os
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
    CNN 2D ( 1(channels) x leads(height) x time(width) )
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
    CNN2DImage ( leads(channels) x height x width )
'''
class CNN2DImage(nn.Module):

    def __init__(self, dense_hidden_size, img_size):
        super().__init__()
        self.img_size = img_size

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            
            nn.Flatten(),
            nn.LazyLinear(dense_hidden_size),  # automatically infers input shape
            nn.Linear(dense_hidden_size, dense_hidden_size),
            nn.Linear(dense_hidden_size, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        '''Forward pass'''
        x = x.reshape(x.shape[0], 3, self.img_size, self.img_size)  # undo channel flattening
        return self.layers(x.float())


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
class PretrainedVGG16Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        # specify download location for pretrained weights
        os.environ['TORCH_HOME'] = './.torch'
        vgg16_pretrained = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES)
        
        self.feature_extractor = nn.Sequential(
            vgg16_pretrained.features,
            # vgg16_pretrained.avgpool
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.LazyLinear(out_features=4096),
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
        x = x.reshape(x.shape[0], 3, 224, 224)  # undo flattening
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


'''
    ECG Image classification model using VGG16 for feature extraction and FC layers for classification
'''
class PretrainedResNet18Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18_pretrained = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.feature_extractor = nn.Sequential(
            resnet18_pretrained.conv1,
            resnet18_pretrained.bn1,
            resnet18_pretrained.relu,
            resnet18_pretrained.maxpool,
            resnet18_pretrained.layer1,
            resnet18_pretrained.layer2,
            resnet18_pretrained.layer3,
            resnet18_pretrained.layer4,
            nn.Flatten()   
        )
        self.classifier = nn.Sequential(
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
        x = x.reshape(x.shape[0], 3, 224, 224)  # undo channel flattening
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x
