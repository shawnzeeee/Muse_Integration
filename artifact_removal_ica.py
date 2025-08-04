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

# csv_path = "blink_data.csv"
# window_sec = 2
# fs= 250

# df = pd.read_csv(csv_path)
# windows = process_windows(df, window_sec, fs)
# window, t = windows[3]


# for i in range(4):
#     window[:,i] = bandpass_filter(window[:,i])

# # Measure inference time for ICA and reconstruction using ica_blink_filter
# start_time = time.time()
# reconstructed = ica_blink_filter(window, random_state=0)
# end_time = time.time()
# inference_time = end_time - start_time
# print(f"ICA inference time: {inference_time:.6f} seconds")


# # Apply ICA to get components for plotting
# ica = FastICA(n_components=4, random_state=0)

# S_ = ica.fit_transform(window)
# reconstructed = ica_blink_filter(window)

# Prepare figure for 8 subplots
# plt.figure(figsize=(14, 16))

# # Plot ICA components
# for i in range(4):
#     plt.subplot(4, 2, 2*i+1)
#     plt.plot(S_[:, i])
#     plt.title(f"ICA Component {i+1}")
#     plt.xlabel("Sample")
#     plt.ylabel("Amplitude")


# # Plot reconstructed signals with each component removed (one channel per subplot)
# for i in range(4):
#     plt.subplot(4, 2, 2*i+2)
#     plt.plot(reconstructed[:, i])
#     plt.title(f"Reconstructed EEG Channel {i+1} (Component {i+1} Removed)")
#     plt.xlabel("Sample")
#     plt.ylabel("Amplitude")

# plt.tight_layout()
# plt.show()




