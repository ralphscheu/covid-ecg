import os
import neurokit2 as nk
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import covidecg.data.utils as data_utils
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import cv2
from PIL import Image
import pathlib

PAT_GROUP_TO_NUMERIC_TARGET = {'postcovid': 1, 'ctrl': 0, 'covid': 1, 'normal': 0}


class ScaleGrayscale(object):
    def __call__(self, im):
        im = im / 255.0
        return im

class InvertGrayscale(object):
    def __call__(self, im):
        im = 1.0 - im
        return im
    
    

class SliceEcgGrid(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        sample = sample.squeeze()
        # print(f"SliceEcgGrid: {sample.shape}")
        # slice ecggrid image into ECG leads
        ecggrid_row_height = sample.shape[0] // 3
        ecggrid_col_width = sample.shape[1] // 4
        leads = [sample[y:y + ecggrid_row_height, x:x + ecggrid_col_width] for x in range(0, sample.shape[1], ecggrid_col_width) for y in range(0, sample.shape[0], ecggrid_row_height)]
        leads = np.stack(leads, axis=0)
        return torch.Tensor(leads)


class SliceTimesteps(object):
    
    def slice_image(self, signal, window_size_ms=300, step_size=100, sampling_rate=500):
        window_size_px = int( window_size_ms // (1000.0 / sampling_rate) ) // 2  # convert ms to pixels in image
        step_size = int( step_size // (1000.0 / sampling_rate) )  # convert ms to number of samples in signal
        signal_len = signal.shape[2]

        # right-pad signal to multiple of step_size for equally sized windows
        right_pad_len = window_size_px - (signal_len % window_size_px) if signal_len % window_size_px > 0 else 0
        signal = np.pad(signal, ((0,0), (0,0), (0, right_pad_len)), mode='constant', constant_values=1.0)

        for start in range(0, signal_len - window_size_px + 1, step_size):
            next_slice = signal[:, :, start:start + window_size_px]
            # print(start, start + window_size_px, "slice:", next_slice.shape)
            yield next_slice
    
    def __call__(self, sample):
        # print(f"SliceTimesteps: {sample.shape}")
        img_slices = list(self.slice_image(sample))
        img_slices = np.stack(img_slices, axis=0)
        img_slices = torch.Tensor(img_slices.astype(np.float32))
        return img_slices