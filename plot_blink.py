
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from feature_extraction import bandpass_filter, notch_filter
from artifact_removal import mca_eye_blink_removal

def process_windows(df, window_sec=2, fs=250):
    """
    Returns a list of (window, t) for each blink (classification==1).
    Each window is shape (samples, 4), t is time axis centered on blink.
    """
    blink_indices = df.index[df.iloc[:, -1] == 1].tolist()
    half_window = int(window_sec * fs // 2)
    windows = []
    for blink_idx in blink_indices:
        start = max(0, blink_idx - half_window)
        end = min(len(df), blink_idx + half_window)
        window = df.iloc[start:end, :4].values
        t = np.arange(start, end) / fs
        t = t - t[0] - (window_sec / 2)
        windows.append((window, t))
    return windows

def plot_blink_window(csv_path, window_sec=2, fs=250, blink_num=0):
    df = pd.read_csv(csv_path)
    windows = process_windows(df, window_sec, fs)
    if not windows:
        print("No blinks found in the data.")
        return
    if blink_num >= len(windows):
        print(f"Only {len(windows)} blinks found. Showing first.")
        blink_num = 0
    window, t = windows[blink_num]
    # Apply filters if desired:
    window = bandpass_filter(window, fs)
    print(window.shape)
    #window = notch_filter(window, fs)
    #window[:,3] = notch_filter(window[:,3], fs)
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for i in range(4):
        window[:, i] = mca_eye_blink_removal(window[:,i])
        axes[i].plot(t, window[:, i], label=f'Electrode {i+1}')
        axes[i].axvline(0, color='r', linestyle='--', label='Blink')
        axes[i].set_ylabel('EEG')
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle('EEG Blink Window')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_blink_window("blink_data.csv", window_sec=2, fs=250)
