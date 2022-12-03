import os
import neurokit2 as nk
import pandas as pd
import numpy as np
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


##########################################################
#                 DATA LOADING HELPERS                   #
##########################################################


def clean_signal(signal, sampling_rate):
    cleaned_signals = [nk.ecg_clean(lead, method='biosppy', sampling_rate=sampling_rate) for lead in signal]
    cleaned_signals = np.stack(cleaned_signals)
    return cleaned_signals

def load_signal(filepath, return_cleaned_signal=False):
    raw_signal = pd.read_csv(filepath, index_col=0).to_numpy().T
    if return_cleaned_signal:
        return clean_signal(raw_signal, int(os.getenv('SAMPLING_RATE', 500)))
    else:
        return raw_signal

def generate_ecg_leads_grid(imgdata):
    imgdata = np.reshape(imgdata, (4, 3, imgdata.shape[1], imgdata.shape[2]))
    imgdata = np.swapaxes(imgdata, 0, 1)
    col0 = np.concatenate(list(imgdata[:, 0]), axis=0)
    col1 = np.concatenate(list(imgdata[:, 1]), axis=0)
    col2 = np.concatenate(list(imgdata[:, 2]), axis=0)
    col3 = np.concatenate(list(imgdata[:, 3]), axis=0)
    del imgdata
    # print(f"cols: {col0.shape}, {col1.shape}, {col2.shape}, {col3.shape}")
    ecg_leads_grid = np.concatenate([col0, col1, col2, col3], axis=1)
    return ecg_leads_grid
