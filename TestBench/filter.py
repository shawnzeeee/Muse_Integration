import numpy as np
from sklearn.decomposition import FastICA

import warnings
from sklearn.exceptions import ConvergenceWarning


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
        


import numpy as np
from scipy.signal import welch
from scipy.signal import butter, filtfilt, iirnotch

# Function to calculate mobility and complexity (Hjorth parameters)
def calculate_hjorth_parameters(signal):
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    variance = np.var(signal)
    mobility = np.sqrt(np.var(first_derivative) / variance)
    complexity = np.sqrt(np.var(second_derivative) / np.var(first_derivative)) / mobility
    return mobility, complexity

# Function to calculate bandpowers (alpha and beta)
def calculate_bandpowers(signal, fs=250):
    freqs, psd = welch(signal, fs=fs, nperseg=fs)
    alpha_band = np.logical_and(freqs >= 8, freqs <= 13)
    beta_band = np.logical_and(freqs >= 13, freqs <= 30)
    alpha_power = np.sum(psd[alpha_band])
    beta_power = np.sum(psd[beta_band])
    return alpha_power, beta_power

def calculate_rms(signal):
    """
    Returns the root mean square (RMS) value of a 1D signal.
    """
    return np.sqrt(np.mean(np.square(signal)))

def bandpass_filter(data, fs=250, lowcut=3, highcut=50, order=4):
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
def bandpass_filter(window, fs=250, lowcut=3, highcut=50, order=4):
    """
    Apply bandpass filter to each channel (column) in the window.
    window: shape (samples, channels)
    Returns filtered window of same shape.
    """
    b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
    # Filter each channel independently
    filtered = np.zeros_like(window)
    for i in range(window.shape[1]):
        filtered[:, i] = filtfilt(b, a, window[:, i])
    return filtered

def notch_filter(data, fs=250, freq=60, Q=30):
    b, a = iirnotch(freq/(fs/2), Q)
    return filtfilt(b, a, data, axis=0)





    