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

def load_signal(filepath):
    return pd.read_csv(filepath, index_col=0).to_numpy().T


def clean_signal(signal, sampling_rate=500):
    cleaned_signals = [nk.ecg_clean(lead, method='biosppy', sampling_rate=sampling_rate) for lead in signal]
    cleaned_signals = np.stack(cleaned_signals)
    return cleaned_signals


def load_runs(runs_list, root_dir, min_length=5000, max_length=5000, return_pat_ids=True):
    """ Load raw ECG Signals from runs dir """
    runs_list = pd.read_csv(runs_list, sep=';')
    runs_list.recording_date = pd.to_datetime(runs_list.recording_date)
    runs_list.pat_diagnosis_date = pd.to_datetime(runs_list.pat_diagnosis_date)
    runs_list['date_diff'] = runs_list.recording_date - runs_list.pat_diagnosis_date  # compute time between ECG recording and diagnosis date (at 12am since we don't have wall time for diagnosis)
    runs_list = runs_list.loc[runs_list.date_diff > pd.Timedelta(seconds=0)]  # only use ECGs done _on or after_ the diagnosis date
    
    assert(runs_list.groupby('pat_id').nunique()['session'].max() == 1)
    
    runs_list = runs_list.loc[runs_list.ecg_length >= min_length]
    
    print(f"Counts: {runs_list.shape[0]} total, {runs_list.pat_group.value_counts()}")
    
    # load recordings
    signals, targets, pat_ids = [], [], []
    for i in range(len(runs_list.index)):        
        signal_path = os.path.join(root_dir, runs_list.iloc[i]['recording'] + '.csv')
        signal = load_signal(signal_path)
        signals.append(signal[:, 0:max_length])  # if longer than max_length, cut off everything afterwards
        targets.append(runs_list.iloc[i]['pat_group'])
        pat_ids.append(runs_list.iloc[i]['pat_id'])
        
    # load full sessions
    # signals, targets, pat_ids = [], [], []
    # for session_id in runs_list.session.unique():        
    #     signal_path = os.path.join(root_dir, str(session_id) + '.csv')
    #     signal = load_signal(signal_path)
    #     signals.append(signal[:, 0:max_length])  # if longer than max_length, cut off everything afterwards
    #     pat_group = runs_list.loc[runs_list.session == session_id].iloc[0].pat_group
    #     pat_id = runs_list.loc[runs_list.session == session_id].iloc[0].pat_id
    #     targets.append(pat_group)
    #     pat_ids.append(pat_id)
    
    
    if return_pat_ids:
        return np.stack(signals).astype(np.float32), np.array(targets), np.array(pat_ids)
    else:
        return np.stack(signals).astype(np.float32), np.array(targets)


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


class PretrainedModelApplyTransforms(BaseEstimator, TransformerMixin):
    def __init__(self, transforms_fn):
        self.transforms_fn = transforms_fn
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        transformed_images = [self.transforms_fn(torch.from_numpy(image)).cpu().numpy() for image in tqdm(x, "Apply image transforms for pre-trained model")]
        transformed_images = np.stack(transformed_images)  # to numpy array
        
        plt.figure()
        plt.imshow(transformed_images[0, 0], cmap='binary')
        plt.axis("off")
        plt.gcf().canvas.draw()
        plt.savefig('./data/processed/example_input_for_vgg16.png')
        
        return transformed_images.astype(np.float32)
