import os
import numpy as np
import torch.functional as F
import torch.nn as nn
import torchvision.models
import torch
import logging


def cnn3dseq_conv_layer(dropout, **kwargs):
    return nn.Sequential(
        nn.Conv3d(**kwargs),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        nn.Dropout(dropout)
    )

class MeanStdPool(nn.Module):
    def forward(self, x):
        std, mean = torch.std_mean(x, dim=1)
        x = torch.concat([mean, std], dim=1)  # concat mean and std vectors for each sample
        return x

class CNN3DSeqPool(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.conv1 = cnn3dseq_conv_layer(dropout=dropout, in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pooling = MeanStdPool()
        self.classifier = nn.Sequential(
            nn.LazyLinear(2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """Forward step

        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)

        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"Model input shape (batch_size, timesteps, d1, d2, d3): {x.shape}")
        batch_size, timesteps, d1, d2, d3 = x.size()
        x = x.view(batch_size * timesteps, d1, d2, d3)
        x = x[:, None, :, :, :]  # insert channels dimension for Conv3D
        logging.debug(f"CNN input: {x.shape}")
        x = self.conv1(x)
        logging.debug(f"conv1 output: {x.shape}")
        x = self.conv2(x)
        logging.debug(f"conv2 output: {x.shape}")
        x = self.conv3(x)
        logging.debug(f"CNN output: {x.shape}")
        x = x.view(batch_size, timesteps, -1)  # restore timesteps and flatten CNN output
        logging.debug(f"CNN output reshaped: {x.shape}")
        x = self.pooling(x)
        logging.debug(f"Pooling output: {x.shape}")
        x = self.classifier(x)
        logging.debug(f"Model output shape: {x.shape}")
        return x


class CNN3DSeqLSTM(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.conv1 = cnn3dseq_conv_layer(dropout=dropout, in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.rnn = nn.LSTM(input_size=10368, hidden_size=200, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LazyLinear(2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """Forward step

        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)

        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"Model input shape: {x.shape} ({x.type()})")
        batch_size, timesteps, d1, d2, d3 = x.shape
        x = x.view(batch_size * timesteps, d1, d2, d3)
        x = x[:, None, :, :, :]  # insert channels dimension for Conv3D
        logging.debug(f"CNN input: {x.shape}")
        x = self.conv1(x)
        logging.debug(f"conv1 output: {x.shape}")
        x = self.conv2(x)
        logging.debug(f"conv2 output: {x.shape}")
        x = self.conv3(x)
        logging.debug(f"CNN output: {x.shape}")
        x = x.view(batch_size, timesteps, -1)  # restore timesteps and flatten CNN output
        logging.debug(f"CNN output reshaped: {x.shape}")
        
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        logging.debug(f"RNN output: {x.shape}")
        
        x = self.classifier(x)
        
        logging.debug(f"Model output shape: {x.shape}")
        return x
