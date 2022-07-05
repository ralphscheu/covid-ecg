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
        beats=rpeak_locs, sampling_rate=sampling_rate, smooth=True, size=3
    )
    return hr.mean(), hr.std()

def compute_rr_intervals(rpeaks:list, sampling_rate:int) -> np.ndarray:
    if len(rpeaks) < 2:
        return []  # abort
    rr_intervals = []
    for left_peak_i in range(0, len(rpeaks) - 1):
        right_peak_i = left_peak_i + 1
        rr_intervals.append(rpeaks[right_peak_i] - rpeaks[left_peak_i])
    rr_intervals = np.array(rr_intervals)
    rr_intervals = rr_intervals * 1000 / sampling_rate
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

def get_peaks_locations(signal, sampling_rate, delineate_method='dwt', delineate_check=False):
    print(signal.shape, signal.max() - signal.min())
    peaks_locations = nk.ecg_delineate(signal, sampling_rate=sampling_rate, method=delineate_method, check=delineate_check)[1]

    peaks_locations['ECG_R_Peaks'] = get_rpeaks_locations(signal, sampling_rate)

    # estimate Q onset locations TODO: find method to actually determine Q onsets
    q_onset_correction_in_samples = 10 * 1000 / sampling_rate  # shift 15ms to the left to estimate Q Onset
    peaks_locations['ECG_Q_Onsets'] = np.array(peaks_locations['ECG_Q_Peaks']) - q_onset_correction_in_samples

    # estimate S offsets
    s_offset_correction_in_samples = 10 * 1000 / sampling_rate  # shift 15ms to the left to estimate Q Onset
    peaks_locations['ECG_S_Offsets'] = np.array(peaks_locations['ECG_S_Peaks']) + s_offset_correction_in_samples

    return peaks_locations

def get_peaks_features(signal, sampling_rate):
    """Compute Peaks Features as mean+std of signal at Q, R, S, T peak locations

    Args:
        signal (np.ndarray): filtered signal
        sampling_rate (int): Sampling rate

    Returns:
        list: Peaks features (mean+std signal amplitude at Q, R, S, T peaks)
    """
    peaks_feats = []
    peaks_locations = get_peaks_locations(signal, sampling_rate)
    
    # R Peaks
    rpeaks_y = signal[peaks_locations['ECG_R_Peaks']]
    peaks_feats.append(rpeaks_y.mean())
    peaks_feats.append(rpeaks_y.std())

    # Q Peaks
    qpeaks_y = signal[peaks_locations['ECG_Q_Peaks']]
    peaks_feats.append(qpeaks_y.mean())
    peaks_feats.append(qpeaks_y.std())
    
    # S Peaks
    # remove nan entries
    peaks_locations['ECG_S_Peaks'] = [v for v in peaks_locations['ECG_S_Peaks'] if v == v]
    speaks_y = signal[peaks_locations['ECG_S_Peaks']]
    peaks_feats.append(speaks_y.mean())
    peaks_feats.append(speaks_y.std())
    
    # T Peaks
    # remove nan entries
    peaks_locations['ECG_T_Peaks'] = [v for v in peaks_locations['ECG_T_Peaks'] if v == v]
    tpeaks_y = signal[peaks_locations['ECG_T_Peaks']]
    peaks_feats.append(tpeaks_y.mean())
    peaks_feats.append(tpeaks_y.std())

    return peaks_feats

def get_intervals_features(signal, sampling_rate):
    interval_feats = []
    peaks_locations = get_peaks_locations(signal, sampling_rate)
    
    # RR Intervals
    rpeaks_locations = nk.ecg_peaks(signal, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
    rr_intervals = compute_rr_intervals(peaks_locations['ECG_R_Peaks'], sampling_rate=sampling_rate)
    interval_feats.append(rr_intervals.mean())
    interval_feats.append(rr_intervals.std())
    
    # PR Intervals
    pr_intervals = compute_pr_intervals(peaks_locations['ECG_P_Onsets'], peaks_locations['ECG_Q_Onsets'])
    interval_feats.append(pr_intervals.mean())
    interval_feats.append(pr_intervals.std())
    
    qrs_widths = compute_pr_intervals(peaks_locations['ECG_Q_Onsets'], peaks_locations['ECG_S_Offsets'])
    interval_feats.append(qrs_widths.mean())
    interval_feats.append(qrs_widths.std())
    
    # QT intervals
    qt_intervals = compute_pr_intervals(peaks_locations['ECG_Q_Onsets'], peaks_locations['ECG_T_Offsets'])
    interval_feats.append(qt_intervals.mean())
    interval_feats.append(qt_intervals.std())
    
    interval_feats = np.nan_to_num(interval_feats, nan=0.0)
    # interval_feats[np.isnan(interval_feats)] = 0  # set nan elements to zero
    
    print("intervals_feats:", interval_feats)
    
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
        peaks_feats = []
        for lead_i in range(x.shape[1]):
            lead_signals = x[:, lead_i]
            lead_peaks_feats = [get_peaks_features(signal, sampling_rate=self.sampling_rate) for signal in lead_signals]
            peaks_feats.append(lead_peaks_feats)
        peaks_feats = np.vstack(peaks_feats)
        return peaks_feats


class EcgIntervalsFeatsExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        intervals_feats = []
        for lead_i in range(x.shape[1]):
            lead_signals = x[:, lead_i]
            lead_intervals_feats = [get_intervals_features(signal.squeeze(), sampling_rate=self.sampling_rate) for signal in x]
            intervals_feats.append(lead_intervals_feats)
        intervals_feats = np.vstack(intervals_feats)
        return intervals_feats


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
    def __init__(self, vertical_resolution=200):
        self.vertical_resolution = vertical_resolution

    def fit(self, x:np.ndarray, y=None):
        return self

    def transform(self, x:np.ndarray) -> np.ndarray:
        """Convert signal values in x to image representation of ECG signal

        Args:
            x (np.ndarray): Input array of shape (n_samples, leads, timesteps)
        
        Returns:
            out (np.ndarray): Output array of shape (n_samples, leads, y_resolution, timesteps)
        """
        out = []
        for recording in x:
            # scale signal into fixed value range (= height of output images)
            scaled = np.stack([np.interp(lead, (lead.min(), lead.max()), (0, self.vertical_resolution - 1)) for lead in recording])
            # round values back to integers after scaling
            scaled = np.round(scaled)
            # recordings_scaled.append(scaled)
            # do one-hot encoding along time axis to place black pixels according to given values
            out.append( np.stack([pd.get_dummies(lead).T.reindex(range(self.vertical_resolution), fill_value=0).to_numpy() for lead in scaled]) )
        out = np.stack(out)
        # recordings_scaled = np.stack(recordings_scaled)
        # flip in y direction to fix orientation
        out = np.flip(out, axis=2)
        print("[EcgSignalToImageConverter] out.shape:", out.shape)
        return out    
