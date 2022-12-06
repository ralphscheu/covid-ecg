import os
import numpy as np
import torch.functional as F
from torch import nn
import torch
import logging
import mlflow

REDUCE_SIZE = 1024

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


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self):
        super().__init__()
        self.W = nn.LazyLinear(1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)  # compute weighted sum using Self Attention weights for each timestep

        return utter_rep, att_w


class CNN3DSeq(nn.Module):
    def __init__(self, dropout, reduction_size, conv_kernel_size):
        super().__init__()
        self.conv1 = cnn3dseq_conv_layer(dropout=dropout, in_channels=1, out_channels=8, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.conv2 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=conv_kernel_size, stride=1, padding='same')
        self.conv3 = cnn3dseq_conv_layer(dropout=dropout, in_channels=8, out_channels=8, kernel_size=conv_kernel_size, stride=1, padding='same')
        
        self.reduction_size = reduction_size
        if self.reduction_size > 0:
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
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3, 3)):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.pooling = MeanStdPool()

    def forward(self, x):
        """Forward step
        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)
        Returns:
            np.ndarray: Softmax output
        """
                
        # print(f"model input: {x.shape} (batch_size, timesteps, d1, d2, d3)  ||  {torch.sum(torch.isnan(x))}")
        x = self.forward_cnn3d(x)
        # print(f"conv layers output: {x.shape}  ||  {torch.sum(torch.isnan(x))}")
        if self.reduction_size > 0:
            x = self.reduction(x)
            # print(f"reduction layer output: {x.shape}  ||  {torch.sum(torch.isnan(x))}")
        x = self.pooling(x)
        # print(f"pooling output: {x.shape}  ||  {torch.sum(torch.isnan(x))}")
        x = self.classifier(x)
        # print(f"classifier output: {x.shape}  ||  {torch.sum(torch.isnan(x))}")
        
        return x


class CNN3DSeqReducedMeanStdPool(CNN3DSeqMeanStdPool):
    def __init__(self, dropout=0.1, conv_kernel_size=(3, 3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)



class CNN3DSeqMeanPool(CNN3DSeqMeanStdPool):
    """ CNN3DSeq variant applying Mean Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3, 3)):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.pooling = MeanPool()


class CNN3DSeqReducedMeanPool(CNN3DSeqMeanPool):
    """ CNN3DSeq variant applying Mean Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)



class CNN3DSeqAttnPool(CNN3DSeq):
    """ CNN3DSeq variant applying Self-Attention Pooling across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3, 3)):
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

        
        # Export x and attention weights if model is being evaluated on test set
        if self.training:
            np.savez_compressed('/tmp/covidecg_x.npz', x=x.detach().cpu().numpy())
            mlflow.log_artifact('/tmp/covidecg_x.npz')
        
        x = self.forward_cnn3d(x)
        
        if self.reduction_size > 0:
            x = self.reduction(x)
            logging.debug(f"reduction layer output: {x.shape}")
        
        x, att_w = self.pooling(x)
        
        # Export x and attention weights if model is being evaluated on test set
        if self.training:
            np.savez_compressed('/tmp/covidecg_att_w.npz', att_w=att_w.detach().cpu().numpy())
            mlflow.log_artifact('/tmp/covidecg_att_w.npz')
        
        logging.debug(f"pooling output: {x.shape}")
        x = self.classifier(x)
        logging.debug(f"classifier output: {x.shape}")
        return x


class CNN3DSeqReducedAttnPool(CNN3DSeqAttnPool):
    def __init__(self, dropout=0.1, conv_kernel_size=(3, 3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)


class CNN3DSeqLSTM(CNN3DSeq):
    """ CNN3DSeq variant applying one unidirectional LSTM layer across timesteps """
    def __init__(self, dropout=0.1, reduction_size=-1, conv_kernel_size=(3, 3, 3), lstm_hidden_size=200):
        super().__init__(dropout=dropout, reduction_size=reduction_size, conv_kernel_size=conv_kernel_size)
        self.rnn = nn.LSTM(input_size=self.reduction_size if self.reduction_size > 0 else 3456, 
                           hidden_size=lstm_hidden_size, batch_first=True)

    def forward(self, x):
        """Forward step
        Args:
            x (np.ndarray): Input array of shape (batch, timesteps, leads, height, width)
        Returns:
            np.ndarray: Softmax output
        """
        logging.debug(f"model input: {x.shape} (batch_size, timesteps, d1, d2, d3)")
        x = self.forward_cnn3d(x)
        
        if self.reduction_size > 0:
            x = self.reduction(x)
            logging.debug(f"reduction layer output: {x.shape}")
        
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # use last LSTM output
        logging.debug(f"rnn output: {x.shape}")        
        x = self.classifier(x)       
        logging.debug(f"classifier output: {x.shape}")
        return x


class CNN3DSeqReducedLSTM(CNN3DSeqLSTM):
    def __init__(self, dropout=0.1, conv_kernel_size=(3, 3, 3)):
        super().__init__(reduction_size=REDUCE_SIZE)
