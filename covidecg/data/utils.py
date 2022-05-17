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


def load_runs(runs_list, root_dir, min_length=5000, max_length=5000):
    """ Load raw ECG Signals from runs dir """
    
    runs_list = runs_list.loc[runs_list.ecg_length >= min_length]
    runs_list = runs_list.loc[runs_list.ecg_length <= max_length]
    
    signals, targets = [], []
    for i in range(len(runs_list.index)):        
        signal_path = os.path.join(root_dir, runs_list.iloc[i]['run_id'] + '.csv')
        signal = load_signal(signal_path)
        signals.append(signal)
        targets.append(runs_list.iloc[i]['pat_group'])
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
        return x[:, self.lead_index, :]  # x.shape is ( num_samples, num_leads, length_recording )


class EcgSignalCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def fit(self, x, y=None):
        return self

    def transform(self, x):
        cleaned_signals = [nk.ecg_clean(signal, sampling_rate=self.sampling_rate) for signal in x]
        return np.vstack(cleaned_signals)


# def process_signal(ecg_signal):
#     # Do processing
#     ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=1000)
#     instant_peaks, rpeaks, = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000)
#     rate = nk.ecg_rate(rpeaks, sampling_rate=1000, desired_length=len(ecg_cleaned))
#     quality = nk.ecg_quality(ecg_cleaned, sampling_rate=1000)

#     # Prepare output
#     signals = pd.DataFrame({"ECG_Raw": ecg_signal,
#                             "ECG_Clean": ecg_cleaned,
#                             "ECG_Rate": rate,
#                             "ECG_Quality": quality})
#     signals = pd.concat([signals, instant_peaks], axis=1)
#     info = rpeaks

#     return signals, info