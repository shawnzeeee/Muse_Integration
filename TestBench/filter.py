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





    