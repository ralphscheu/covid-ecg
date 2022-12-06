import os
import numpy as np
import torch.functional as F
from torch import nn
import torch
import logging
from covidecg.models.cnn3dseq import MeanStdPool, MeanPool, SelfAttentionPooling

REDUCE_SIZE = 1024


###########
# MODULES #
###########

def cnn2dseq_conv_layer(dropout, **kwargs):
    return nn.Sequential(
        nn.Conv2d(**kwargs),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Dropout(dropout)
    )



###########
#  MODELS #
###########


class CNN2DSeq(nn.Module):
    def __init__(self, dropout, reduction_size, conv_kernel_size):
        super().__init__()
        self.conv1 = cnn2dseq_conv_layer(dropout=dropout, in_channels=12, out_channels=36, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.conv2 = cnn2dseq_conv_layer(dropout=dropout, in_channels=36, out_channels=36, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.conv3 = cnn2dseq_conv_layer(dropout=dropout, in_channels=36, out_channels=36, kernel_size=conv_kernel_size, stride=1, padding='same')
        
        self.reduction_size = reduction_size
        if self.reduction_size > 0:
            self.reduction = nn.LazyLinear(reduction_size)
        
        self.classifier = nn.Sequential(nn.LazyLinear(2), nn.Softmax(dim=-1))
        
    
    def forward_cnn2d(self, x):
        batch_size, timesteps, dleads, dy, dx = x.size()
        x = x.view(batch_size * timesteps, dleads, dy, dx)  # merge timesteps and batch dimension
        # x = x[:, None, :, :, :]  # insert channels dimension for Conv3D
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

class CNN2DSeqMeanStdPool(CNN2DSeq):
    """ CNN2DSeq variant applying Mean+Std Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3)):
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
        x = self.forward_cnn2d(x)
        if self.reduction_size > 0:
            x = self.reduction(x)
            logging.debug(f"reduction layer output: {x.shape}")
        x = self.pooling(x)
        logging.debug(f"pooling output: {x.shape}")
        x = self.classifier(x)
        logging.debug(f"classifier output: {x.shape}")
        return x


class CNN2DSeqReducedMeanStdPool(CNN2DSeqMeanStdPool):
    def __init__(self, dropout=0.1, conv_kernel_size=(3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)



class CNN2DSeqMeanPool(CNN2DSeqMeanStdPool):
    """ CNN2DSeq variant applying Mean Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3)):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.pooling = MeanPool()

class CNN2DSeqReducedMeanPool(CNN2DSeqMeanPool):
    """ CNN2DSeq variant applying Mean Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)



class CNN2DSeqAttnPool(CNN2DSeq):
    """ CNN2DSeq variant applying Self-Attention Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3)):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.pooling = SelfAttentionPooling()  # input_dim=self.reduction_size if self.reduction_size > 0 else 3456

    def forward(self, x):
        """Forward step
        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)
        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"model input: {x.shape} (batch_size, timesteps, d1, d2, d3)")
        x = self.forward_cnn2d(x)
        
        if self.reduction_size > 0:
            x = self.reduction(x)
            logging.debug(f"reduction layer output: {x.shape}")
        x = self.pooling(x)
        logging.debug(f"pooling output: {x.shape}")
        x = self.classifier(x)
        logging.debug(f"classifier output: {x.shape}")
        return x


class CNN2DSeqReducedAttnPool(CNN2DSeqAttnPool):
    def __init__(self, dropout=0.1, conv_kernel_size=(3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)


class CNN2DSeqLSTM(CNN2DSeq):
    """ CNN2DSeq variant applying one unidirectional LSTM layer across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3), lstm_hidden_size=200):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.rnn = nn.LSTM(input_size=self.reduction_size if self.reduction_size > 0 else 1296, 
                           hidden_size=lstm_hidden_size, batch_first=True)

    def forward(self, x):
        """Forward step
        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)
        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"model input: {x.shape} (batch_size, timesteps, d1, d2, d3)")
        x = self.forward_cnn2d(x)
        
        if self.reduction_size > 0:
            x = self.reduction(x)
            logging.debug(f"reduction layer output: {x.shape}")
        
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # use last LSTM output
        logging.debug(f"rnn output: {x.shape}")        
        x = self.classifier(x)       
        logging.debug(f"classifier output: {x.shape}")
        return x


class CNN2DSeqReducedLSTM(CNN2DSeqLSTM):
    def __init__(self, dropout=0.1, conv_kernel_size=(3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)
