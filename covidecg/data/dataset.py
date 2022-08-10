import os
import neurokit2 as nk
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import covidecg.data.utils as data_utils
from torch.utils.data import Dataset
import cv2

PAT_GROUP_TO_NUMERIC_TARGET = {'postcovid': 1, 'ctrl': 0}


class EcgDataset(Dataset):
    """ PyTorch Dataset for loading ECG signals as arrays of voltage values """
    
    def __init__(self, recordings_file, recordings_dir, min_length=5000, max_length=5000, transform=None):
        self.recordings = pd.read_csv(recordings_file, sep=';')
        self.recordings = self.recordings.loc[self.recordings.ecg_length >= min_length]
        self.min_length = min_length
        self.max_length = max_length
        self.recordings_dir = recordings_dir
        self.transform = transform

    def __len__(self):
        return len(self.recordings.index)

    def __getitem__(self, idx):
        """ Fetch a single recording referenced by index """
        signal_path = os.path.join(self.recordings_dir, self.recordings.iloc[idx]['recording'] + '.csv')
        signal = data_utils.load_signal(signal_path)
        signal = data_utils.clean_signal(signal)
        if self.transform:
            signal = self.transform(signal)
        pat_group = self.recordings.iloc[idx]['pat_group']
        target = PAT_GROUP_TO_NUMERIC_TARGET[pat_group]
        return signal, target


class EcgImageDataset(EcgDataset):
    """ PyTorch Dataset for loading ECG signal images of full recordings """

    def __init__(self, recordings_file, ecg_img_data_file, min_length=100):
        self.recordings = pd.read_csv(recordings_file, sep=';')
        self.recordings = self.recordings.loc[self.recordings.ecg_length >= min_length]
        self.ecg_img_data = np.load(ecg_img_data_file)
        # print(list(self.ecg_img_data.keys()))

    def __getitem__(self, idx):
        img = self.ecg_img_data[self.recordings.iloc[idx].recording]
        img = img / 255.0  # normalize values between 0 (black) and 1 (white)
        img = np.moveaxis(img, 2, 0)
        target = self.recordings.iloc[idx].pat_group
        target = PAT_GROUP_TO_NUMERIC_TARGET[target]
        return img, target


class EcgImageSequenceDataset(EcgImageDataset):
    """ PyTorch Dataset for loading ECG signal images sliced into fixed-length timesteps """

    def __init__(self, recordings_file, ecg_img_data_file, min_length=100):
        super().__init__(recordings_file, ecg_img_data_file, min_length)

    def slice_image(self, signal, window_size=30, step_size=10, sampling_rate=500):
        window_size = int( window_size // (1000.0 / sampling_rate) )  # convert ms to number of samples in signal
        step_size = int( step_size // (1000.0 / sampling_rate) )  # convert ms to number of samples in signal
        signal_len = signal.shape[2]

        # right-pad signal to multiple of step_size for equally sized windows
        right_pad_len = window_size - (signal_len % window_size) if signal_len % window_size > 0 else 0
        signal = np.pad(signal, ((0,0), (0,0), (0, right_pad_len)), mode='constant', constant_values=1.0)

        for start in range(0, signal_len - window_size + 1, step_size):
            next_slice = signal[:, :, start:start + window_size]
            print(start, start + window_size, "slice:", next_slice.shape)
            yield next_slice

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_slices = list(self.slice_image(img))
        img_slices = np.stack(img_slices, axis=0)
        return img_slices, target
