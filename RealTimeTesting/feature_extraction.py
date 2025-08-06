import numpy as np

from scipy.signal import welch
import time
# Function to calculate mobility and complexity (Hjorth parameters)
# Function to calculate mobility and complexity (Hjorth parameters)
def calculate_hjorth_parameters(signal):
    first_derivative = np.diff(signal)
    second_derivative = np.diff(first_derivative)
    var_signal = np.var(signal)
    var_fd = np.var(first_derivative)
    var_sd = np.var(second_derivative)

    if var_signal < 1e-10 or var_fd < 1e-10:
        return 0.0, 0.0

    mobility = np.sqrt(var_fd / var_signal)
    if mobility < 1e-10:
        return mobility, 0.0

    complexity = np.sqrt(var_sd / var_fd) / mobility
    return mobility, complexity

# Function to calculate bandpowers (alpha and beta)
def calculate_bandpowers(signal, fs=250):
    freqs, psd = welch(signal, fs=fs, nperseg=fs)
    alpha_band = np.logical_and(freqs >= 8, freqs <= 13)
    beta_band = np.logical_and(freqs >= 13, freqs <= 30)
    alpha_power = np.sum(psd[alpha_band])
    beta_power = np.sum(psd[beta_band])
    return alpha_power, beta_power


def calculate_mean(signal):
    """Mean value of the signal."""
    return np.mean(signal)

def calculate_std(signal):
    """Standard deviation of the signal."""
    return np.std(signal)

def calculate_rms(signal):
    """Root mean square of the signal."""
    return np.sqrt(np.mean(np.square(signal)))

def calculate_skewness(signal):
    """Skewness of the signal."""
    return np.mean(((signal - np.mean(signal)) / np.std(signal))**3)

def calculate_kurtosis(signal):
    """Kurtosis of the signal."""
    return np.mean(((signal - np.mean(signal)) / np.std(signal))**4)

def calculate_entropy(signal, bins=10):
    """Shannon entropy of the signal."""
    hist, bin_edges = np.histogram(signal, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_peak_to_peak(signal):
    """Peak-to-peak amplitude."""
    return np.ptp(signal)

# function to calculate the log variance
def calculate_log_variance(signal):
    var = np.var(signal)
    if var <= 1e-10:  # very close to 0
        return -10.0  # safe fallback value
    return np.log(var)

__all__ = [
    "calculate_hjorth_parameters",
    "calculate_bandpowers",
    "calculate_mean",
    "calculate_std",
    "calculate_rms",
    "calculate_skewness",
    "calculate_kurtosis",
    "calculate_entropy",
    "calculate_peak_to_peak"
]