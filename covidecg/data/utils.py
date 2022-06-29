import os
import neurokit2 as nk
import pandas as pd
import numpy as np
import sklearn.preprocessing
import sklearn.pipeline
from sklearn.base import BaseEstimator, TransformerMixin


##########################################################
#                 DATA LOADING HELPERS                   #
##########################################################

def load_signal(filepath):
    return pd.read_csv(filepath, index_col=0).to_numpy().T


def load_runs(runs_list, root_dir, min_length=5000, max_length=5000, return_pat_ids=True):
    """ Load raw ECG Signals from runs dir """
    
    runs_list = runs_list.loc[runs_list.ecg_length >= min_length]
    runs_list = runs_list.loc[runs_list.ecg_length <= max_length]
    
    signals, targets, pat_ids = [], [], []
    for i in range(len(runs_list.index)):        
        signal_path = os.path.join(root_dir, runs_list.iloc[i]['recording'] + '.csv')
        signal = load_signal(signal_path)
        signals.append(signal)
        targets.append(runs_list.iloc[i]['pat_group'])
        pat_ids.append(runs_list.iloc[i]['pat_id'])
    
    if return_pat_ids:
        return np.stack(signals).astype(np.float32), np.array(targets), np.array(pat_ids)
    else:
        return np.stack(signals).astype(np.float32), np.array(targets)

def load_all_runs(runs_csv, root_dir):
    runs_list = pd.read_csv(runs_csv, sep=';')
    return load_runs(runs_list, root_dir)

def load_stress_ecg_runs(runs_csv, root_dir):
    runs_list = pd.read_csv(runs_csv, sep=';')
    runs_list = runs_list.loc[runs_list.ecg_type == 'Belastungs']
    return load_runs(runs_list, root_dir)

def load_rest_ecg_runs(runs_csv, root_dir):
    runs_list = pd.read_csv(runs_csv, sep=';')
    runs_list = runs_list.loc[runs_list.ecg_type == 'Ruhe']
    return load_runs(runs_list, root_dir)


def to_categorical(y, num_classes=2):
    """ 1-hot encodes a tensor """
    y = y.astype(int)
    return np.eye(num_classes, dtype='uint8')[y]


def flatten_leads(x):
    return x.reshape(x.shape[0], -1)


##########################################################
#               PIPELINE BUILDING BLOCKS                 #
##########################################################

class EcgLeadSelector(BaseEstimator, TransformerMixin):
    def __init__(self, lead_name):
        self.lead_name = lead_name

        LEADS_IDX = {
            'MDC_ECG_LEAD_I': 0,
            'MDC_ECG_LEAD_II': 1,
            'MDC_ECG_LEAD_III': 2,
            'MDC_ECG_LEAD_AVR': 3,
            'MDC_ECG_LEAD_AVL': 4,
            'MDC_ECG_LEAD_AVF': 5,
            'MDC_ECG_LEAD_V1': 6,
            'MDC_ECG_LEAD_V2': 7,
            'MDC_ECG_LEAD_V3': 8,
            'MDC_ECG_LEAD_V4': 9,
            'MDC_ECG_LEAD_V5': 10,
            'MDC_ECG_LEAD_V6': 11
        }
        self.lead_index = LEADS_IDX[self.lead_name]
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        """Select and return single ECG lead from given signal.

        Args:
            x (np.ndarray): Batch of 12-lead ECG signals of shape (batch_size, 12, timesteps)

        Returns:
            np.ndarray: 1-lead ECG signal of shape (batch_size, 1, timesteps)
        """
        lead_signal = x[:, self.lead_index]
        lead_signal = lead_signal[:, np.newaxis]  # restore leads dimension with size 1 (1-lead signal)
        return lead_signal


class EcgSignalCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        cleaned_signals = []
        for lead_i in range(x.shape[1]):
            lead_signals = x[:, lead_i]
            cleaned_lead_signal = [nk.ecg_clean(signal, method='biosppy', sampling_rate=self.sampling_rate) for signal in lead_signals]
            cleaned_signals.append(cleaned_lead_signal)
        cleaned_signals = np.stack(cleaned_signals, axis=1)  # to numpy array
        return cleaned_signals.astype(np.float32)
