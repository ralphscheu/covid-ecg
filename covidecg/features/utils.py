import numpy as np


def compute_rr_intervals(rpeaks:list):
    rr_intervals = []
    for left_peak_i in range(0, len(rpeaks) - 1):
        right_peak_i = left_peak_i + 1
        rr_intervals.append(rpeaks[right_peak_i] - rpeaks[left_peak_i])
    rr_intervals = np.array(rr_intervals)
    return rr_intervals