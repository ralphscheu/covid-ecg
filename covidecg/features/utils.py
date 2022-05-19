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


##########################################################
#              FEATURE EXTRACTION HELPERS                #
##########################################################

def get_heart_rate_features(rpeak_locs, sampling_rate):
    hr_idx, hr = biosppy.signals.tools.get_heart_rate(
        beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3
    )
    return hr.mean(), hr.std()

def compute_rr_intervals(rpeaks:list) -> np.ndarray:
    if len(rpeaks) < 2:
        return []  # abort
    rr_intervals = []
    for left_peak_i in range(0, len(rpeaks) - 1):
        right_peak_i = left_peak_i + 1
        rr_intervals.append(rpeaks[right_peak_i] - rpeaks[left_peak_i])
    rr_intervals = np.array(rr_intervals)
    return rr_intervals

def compute_pr_intervals(p_onsets:list, q_onsets:list):
    if len(p_onsets) < 1:
        raise Exception("Cannot compute PR Intervals: No P Onsets provided!")
    if len(q_onsets) < 1:
        raise Exception("Cannot compute PR Intervals: No Q Onsets provided!")
    
    if len(p_onsets) != len(q_onsets):
        raise Exception(message=f"Different number of P-Onsets and Q-Onsets, cannot compute PR Intervals! - p_onsets: {len(p_onsets)}, q_onsets:{len(q_onsets)}")
        # TODO handle different lengths -> only use P-Q pairs
    
    pr_intervals = []
    for _p_onset, _q_onset in zip(p_onsets, q_onsets):
        pr_intervals.append(_q_onset - _p_onset)
    return np.array(pr_intervals)

def get_rpeaks_locations(signal, sampling_rate):
    (rpeaks_locations,) = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)
    # correct R-peak locations
    (rpeaks_locations,) = ecg.correct_rpeaks(
        signal=signal, rpeaks=rpeaks_locations, sampling_rate=sampling_rate, tol=0.05
    )
    return rpeaks_locations

def get_peaks_features(signal, sampling_rate):
    """Compute Peaks Features as mean+std of signal at Q, R, S, T peak locations

    Args:
        signal (np.ndarray): filtered signal
        sampling_rate (int): Sampling rate

    Returns:
        list: Peaks features (mean+std signal amplitude at Q, R, S, T peaks)
    """
    peaks_feats = []
    
    # R Peaks
    rpeaks_locations = get_rpeaks_locations(signal, sampling_rate)
    rpeaks_y = signal[rpeaks_locations]
    peaks_feats.append(rpeaks_y.mean())
    peaks_feats.append(rpeaks_y.std())
    
    other_peaks_locations = nk.ecg_delineate(signal, method='dwt', sampling_rate=sampling_rate)[1]

    # Q Peaks
    # print(other_peaks_locations)
    # print("Q Peaks:", other_peaks_locations['ECG_Q_Peaks'])
    qpeaks_y = signal[other_peaks_locations['ECG_Q_Peaks']]
    peaks_feats.append(qpeaks_y.mean())
    peaks_feats.append(qpeaks_y.std())
    
    # S Peaks
    speaks_y = signal[other_peaks_locations['ECG_S_Peaks']]
    peaks_feats.append(speaks_y.mean())
    peaks_feats.append(speaks_y.std())
    
    # T Peaks
    tpeaks_y = signal[other_peaks_locations['ECG_T_Peaks']]
    peaks_feats.append(tpeaks_y.mean())
    peaks_feats.append(tpeaks_y.std())

    return peaks_feats

def get_interval_features(signal, sampling_rate):
    interval_feats = []
    
    # RR Intervals
    rpeaks_locations = nk.ecg_peaks(signal, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
    rr_intervals = compute_rr_intervals(rpeaks_locations)
    interval_feats.append(rr_intervals.mean())
    interval_feats.append(rr_intervals.std())
    
    other_peaks_locations = nk.ecg_delineate(signal, sampling_rate=sampling_rate)[1]
    
    # PR Intervals
    # estimate Q onset locations TODO: find method to actually determine Q onsets
    q_onset_locations = np.array(other_peaks_locations['ECG_Q_Peaks']) - 0.015
    pr_intervals = compute_pr_intervals(other_peaks_locations['ECG_P_Onsets'], list(q_onset_locations))
    interval_feats.append(pr_intervals.mean())
    interval_feats.append(pr_intervals.std())
    
    # estimate S offsets
    s_offset_locations = np.array(other_peaks_locations['ECG_S_Peaks']) + 0.015
    qrs_widths = compute_pr_intervals(q_onset_locations, s_offset_locations)
    interval_feats.append(qrs_widths.mean())
    interval_feats.append(qrs_widths.std())
    
    # QT intervals
    qt_intervals = compute_pr_intervals(q_onset_locations, other_peaks_locations['ECG_T_Offsets'])
    interval_feats.append(qt_intervals.mean())
    interval_feats.append(qt_intervals.std())
    
    return interval_feats

##########################################################
#               PIPELINE BUILDING BLOCKS                 #
##########################################################

class EcgPeaksFeatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        peaks_feats = [get_peaks_features(signal, sampling_rate=self.sampling_rate) for signal in x]
        return np.array(peaks_feats)


class EcgIntervalFeatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        interval_feats = [get_interval_features(signal, sampling_rate=self.sampling_rate) for signal in x]
        return np.array(interval_feats)


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
            lfcc_feats.append(lead_lfcc_feats)
        return np.vstack(lfcc_feats)