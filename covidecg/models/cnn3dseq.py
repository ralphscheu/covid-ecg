import os
import numpy as np
import torch.functional as F
import torch.nn as nn
import torchvision.models
import torch
import logging


###########
# MODULES #
###########

class MeanStdPool(nn.Module):
    """ Mean+Std Pooling layer """
    def forward(self, x):
        std, mean = torch.std_mean(x, dim=1)
        x = torch.concat([mean, std], dim=1)  # concat mean and std vectors for each sample
        return x

class MeanPool(nn.Module):
    """ Mean Pooling layer """
    def forward(self, x):
        return torch.mean(x, dim=1)

def cnn3dseq_conv_layer(dropout, **kwargs):
    return nn.Sequential(
        nn.Conv3d(**kwargs),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        nn.Dropout(dropout)
    )


class CNN3DSeq(nn.Module):
    def __init__(self, dropout, reduction_size, conv_kernel_size=(3, 3, 3)):
        super().__init__()
        self.conv1 = cnn3dseq_conv_layer(dropout=dropout, in_channels=1, out_channels=8, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.conv2 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.conv3 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.reduction = nn.LazyLinear(reduction_size)
        self.classifier = nn.Sequential(nn.LazyLinear(2), nn.Softmax(dim=-1))
    
    def forward_cnn3d(self, x):
        batch_size, timesteps, d1, d2, d3 = x.size()
        x = x.view(batch_size * timesteps, d1, d2, d3)  # merge timesteps and batch dimension
        x = x[:, None, :, :, :]  # insert channels dimension for Conv3D
        logging.debug(f"conv1 input: {x.shape}")
        x = self.conv1(x)
        logging.debug(f"conv1 output: {x.shape}")
        x = self.conv2(x)
        logging.debug(f"conv2 output: {x.shape}")
        x = self.conv3(x)
        logging.debug(f"conv3 output: {x.shape}")
        x = x.view(batch_size, timesteps, -1)  # restore timesteps and flatten CNN output
        logging.debug(f"conv3 output reshaped: {x.shape}")
        return x



###########
#  MODELS #
###########

class CNN3DSeqMeanStdPool(CNN3DSeq):
    """ CNN3DSeq variant applying Mean+Std Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=1024, conv_kernel_size=(3, 3, 3)):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.pooling = MeanStdPool()

    def forward(self, x):
        """Forward step
        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)
        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"model input: {x.shape} (batch_size, timesteps, d1, d2, d3)")
        x = self.forward_cnn3d(x)
        x = self.reduction(x)
        logging.debug(f"reduction layer output: {x.shape}")
        x = self.pooling(x)
        logging.debug(f"pooling output: {x.shape}")
        x = self.classifier(x)
        logging.debug(f"classifier output: {x.shape}")
        return x


class CNN3DSeqLSTM(CNN3DSeq):
    """ CNN3DSeq variant applying one unidirectional LSTM layer across timesteps """
    def __init__(self, dropout=0.1, reduction_size=1024, conv_kernel_size=(3, 3, 3), lstm_hidden_size=200):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.rnn = nn.LSTM(input_size=1024, hidden_size=lstm_hidden_size, batch_first=True)

    def forward(self, x):
        """Forward step
        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)
        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"model input: {x.shape} (batch_size, timesteps, d1, d2, d3)")
        x = self.forward_cnn3d(x)
        x = self.reduction(x)
        logging.debug(f"reduction layer output: {x.shape}")
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # use last LSTM output
        logging.debug(f"rnn output: {x.shape}")        
        x = self.classifier(x)       
        logging.debug(f"classifier output: {x.shape}")
        return x


# TODO: add CNN3DSeqAttnPool model


# TODO: add CNN3DSeqAttnLSTM model
