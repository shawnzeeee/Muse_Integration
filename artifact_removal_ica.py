import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import time
from feature_extraction import bandpass_filter
from sklearn.decomposition import FastICA

import warnings
from sklearn.exceptions import ConvergenceWarning
def process_windows(df, window_sec=2, fs=250):
    """
    Returns a list of (window, t) for each blink (classification==1).
    Each window is shape (samples, 4), t is time axis centered on blink.
    """
    blink_indices = df.index[df.iloc[:, -1] == 1].tolist()
    half_window = int(window_sec * fs // 2)
    windows = []
    for blink_idx in blink_indices:
        blink_idx = blink_idx 
        start = max(0, blink_idx - half_window)
        end = min(len(df), blink_idx + half_window)
        window = df.iloc[start:end, :4].values
        t = np.arange(start, end) / fs
        t = t - t[0] - (window_sec / 2)
        windows.append((window, t))
    return windows

def ica_blink_filter(window, random_state=0):
    """
    Applies FastICA to a 4-channel EEG window, removes the first component,
    and returns the reconstructed signal.
    Suppresses convergence warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)

        try:
            ica = FastICA(n_components=4, random_state=random_state, max_iter=100)
            S_ = ica.fit_transform(window)
            S_removed = S_.copy()
            S_removed[:, 0] = 0
            reconstructed = ica.inverse_transform(S_removed)
            return reconstructed
        except Exception as e:
            print("ICA did not converge, returning original window.")
            return window

csv_path = "blink_data.csv"
window_sec = 2
fs = 250

df = pd.read_csv(csv_path)
windows = process_windows(df, window_sec, fs)

for idx, (window, t) in enumerate(windows):
    filtered_window = bandpass_filter(window)
    ica = FastICA(n_components=4, random_state=0)
    S_ = ica.fit_transform(filtered_window)
    means = np.mean(S_, axis=0)
    print(f"Window {idx}: ICA component means: {means}")




