import os
import numpy as np
import torch.functional as F
import torch.nn as nn
import torchvision.models
import torch
import logging


class TimeDistributed(nn.Module):
    """ Equivalent to Keras TimeDistributed
    taken from https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    """
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        logging.info(f"x_reshape: {x_reshape.shape}")

        y = self.module(x_reshape)
        logging.info(f"y: {y.shape}")

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
            
        logging.info(f"y: {y.shape}")

        return y


class CNNSeqPool(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            # Conv Layer 1
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Conv Layer 2
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Conv Layer 3
            nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(num_features),
            nn.ReLU()
        )
        self.timedistributed = TimeDistributed(self.cnn, batch_first=True)

    def forward(self, x):
        """Forward step

        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)

        Returns:
            np.ndarray: Softmax output
        """
        logging.info(f"Model input shape: {x.shape}")
        x = self.timedistributed(x)
        logging.info(f"TD output: {x.shape}")
        
        return self.cnn(x)
