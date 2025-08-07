import numpy as np

from scipy.signal import hilbert
from PyEMD import EMD
from feature_extraction import *
def extract_feature_vector(indices, df, window_size, classification, num_windows=5, filter_funcs=[], features=[]):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            #print(window.shape)

            for func in filter_funcs:
                window = func(window)
            
            features = []
            for i in range(4):
                signal = window[:,i]
                for func in feature_funcs:
                    input = func(signal)
                    if hasattr(input, '__iter__') and not isinstance(input, str):
                        features.extend(input)
                    else:
                        features.append(input)
            #actual_class = df.iloc[window_start, 4]
            features.append(classification)
            processed_data.append(features)
    return processed_data

def extract_timeseries(idle_indices, classification, df, window_size, num_windows=5, filter_funcs=[]):
    windows = []
    labels = []
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            for func in filter_funcs:
                window = func(window)
            windows.append(window)  # shape: (window_size, 4)
            labels.append(classification)        # Idle class label
    return windows, labels

def collect_imf_psd(indices, classification, df, window_size=500, num_windows=4, fs=250, filter_funcs=[]):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end]
            for func in filter_funcs:
                window = func(window)
            features = []
            for channel in channel_names:
                signal = window[channel].values
                emd = EMD()
                IMFs = emd(signal)
                inst_freqs = []
                # IMF 1
                if len(IMFs) > 0:
                    imf1 = IMFs[1]
                    analytic_signal1 = hilbert(imf1)
                    instantaneous_phase1 = np.unwrap(np.angle(analytic_signal1))
                    instantaneous_frequency1 = np.diff(instantaneous_phase1) / (2.0 * np.pi) * fs
                    valid_freqs1 = instantaneous_frequency1[(instantaneous_frequency1 >= 13) & (instantaneous_frequency1 <= 30)]
                    inst_freqs.extend(valid_freqs1)
                # IMF 4
                if len(IMFs) > 3:
                    imf4 = IMFs[4]
                    analytic_signal4 = hilbert(imf4)
                    instantaneous_phase4 = np.unwrap(np.angle(analytic_signal4))
                    instantaneous_frequency4 = np.diff(instantaneous_phase4) / (2.0 * np.pi) * fs
                    valid_freqs4 = instantaneous_frequency4[(instantaneous_frequency4 >= 8) & (instantaneous_frequency4 <= 13)]
                    inst_freqs.extend(valid_freqs4)
                # Construct PSD from instantaneous frequencies
                if len(inst_freqs) > 0:
                    psd, freq_bins = np.histogram(inst_freqs, bins=60, range=(0, 60), density=True)
                else:
                    psd = np.zeros(60)
                features.extend(psd)
            features.append(classification)
            processed_data.append(features)
    return processed_data