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

PAT_GROUP_TO_NUMERIC_TARGET = {'postcovid': 1, 'ctrl': 0}


class EcgDataset(Dataset):
    """ PyTorch Dataset for loading ECG signals as arrays of voltage values """
    
    def __init__(self, recordings_file, recordings_dir, min_length=5000, max_length=5000, transform=None):
        self.recordings = pd.read_csv(recordings_file, sep=';')
        self.recordings = self.recordings.loc[self.recordings.ecg_length >= min_length]
        if max_length:
            self.recordings = self.recordings.loc[self.recordings.ecg_length <= max_length]
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

    def __init__(self, recordings_file, ecg_img_data_file, min_length=5000, max_length=5000):
        self.recordings = pd.read_csv(recordings_file, sep=';')
        self.recordings = self.recordings.loc[self.recordings.ecg_length >= min_length]
        if max_length:
            self.recordings = self.recordings.loc[self.recordings.ecg_length <= max_length]
        self.min_length = min_length
        self.max_length = max_length
        self.ecg_img_data = np.load(ecg_img_data_file)
    
    def get_targets(self):
        targets = [PAT_GROUP_TO_NUMERIC_TARGET[pat_group] for pat_group in self.recordings.pat_group]
        targets = np.array(targets)
        return targets

    def __getitem__(self, idx):
        img = self.ecg_img_data[self.recordings.iloc[idx].recording]
        img = img / 255.0  # normalize values between 0 (black) and 1 (white)
        img = np.moveaxis(img, 2, 0)
        target = self.recordings.iloc[idx].pat_group
        target = PAT_GROUP_TO_NUMERIC_TARGET[target]
        return img, target


class EcgImageSequenceDataset(EcgImageDataset):
    """ PyTorch Dataset for loading ECG signal images sliced into fixed-length timesteps """

    def __init__(self, recordings_file, ecg_img_data_file, min_length=100, max_length=None):
        super().__init__(recordings_file, ecg_img_data_file, min_length, max_length)

    def slice_image(self, signal, window_size_ms=30, step_size=10, sampling_rate=500):
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

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        img_slices = list(self.slice_image(img))
        img_slices = np.stack(img_slices, axis=0)
        return img_slices, target


class ConcatEcgDataset(ConcatDataset):
    def get_targets(self):
        return np.concatenate([d.get_targets() for d in self.datasets])
