import os
import neurokit2 as nk
import pandas as pd
import numpy as np
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import spafe.features.lfcc
import biosppy.signals.tools
import biosppy.signals.ecg as ecg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


##########################################################
#               PIPELINE BUILDING BLOCKS                 #
##########################################################

class EcgLfccFeatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        lfcc_feats = []
        for lead_i in range(x.shape[1]):
            lead_signals = x[:, lead_i]
            lead_lfcc_feats = [spafe.features.lfcc.lfcc(signal, fs=self.sampling_rate) for signal in lead_signals]
            lead_lfcc_feats = np.stack(lead_lfcc_feats)
            lfcc_feats.append(lead_lfcc_feats)
        lfcc_feats = np.stack(lfcc_feats, axis=1)
        return lfcc_feats


class EcgSignalToImageConverter(BaseEstimator, TransformerMixin):
    def __init__(self, height, width, dpi=96):
        self.height = height
        self.width = width
        self.dpi = dpi

    def fit(self, x:np.ndarray, y=None):
        return self
    
    def get_lead_signal_img(self, lead_signal:np.ndarray, crop_horizontal_padding:int=0):
        
        # Make a line plot
        lead_img_height = self.height // 12
        fig = plt.figure(figsize=(self.width / self.dpi, lead_img_height / self.dpi), dpi=self.dpi)
        fig.gca().plot(lead_signal, linewidth=.6, c='black')
        fig.tight_layout(pad=0)
        plt.axis('off')
        fig.canvas.draw()
        plt.savefig('./data/processed/lead_signal.png')

        # Save it to a numpy array.
        lead_signal_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) 
        plt.close()

        lead_signal_image = lead_signal_image.reshape(fig.canvas.get_width_height()[::-1] + (3,)) 
        assert len(np.unique(lead_signal_image[:, :, 0])) > 1  # make sure not all pixels have the same value (otherwise something went wrong)

        # Crop the image
        lead_signal_image = ~lead_signal_image
        nonzero_coords = cv2.findNonZero(lead_signal_image[:, :, 0]) # Find all non-zero points
        x, y, w, h = cv2.boundingRect(nonzero_coords) # Find minimum spanning bounding box
        lead_signal_image = lead_signal_image[:, x - crop_horizontal_padding:x + w + crop_horizontal_padding, :]
        lead_signal_image = ~lead_signal_image

        lead_signal_image = cv2.resize(lead_signal_image, (self.width, lead_img_height))
        return lead_signal_image


    def transform(self, x:np.ndarray) -> np.ndarray:
        """Convert signal values in x to image representation of ECG signal

        Args:
            x (np.ndarray): Input array of shape (n_samples, leads, timesteps)
        
        Returns:
            out (np.ndarray): Output array of shape (n_samples, leads, y_resolution, timesteps)
        """
        out = []
        
        for recording in tqdm(x, "Convert recordings to signal images"):
            # print("recording:", recording.shape)
            
            recording_image = np.concatenate([self.get_lead_signal_img(recording[lead_i, :]) for lead_i in range(recording.shape[0])], axis=0)
            # print("recording_image.shape", recording_image.shape)
            out.append(recording_image)
        
        out = np.stack(out)    
        # print("out:", out.shape)
        
        plt.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)
        plt.imshow(out[0], cmap='binary')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.gcf().canvas.draw()
        plt.savefig('./data/processed/example_signal_to_image.png')
        
        out = np.moveaxis(out, 3, 1)  # convert to channel-first representation for PyTorch processing
        
        print("EcgSignalToImageConverter__out.shape:", out.shape)
        
        return out    
