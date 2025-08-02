import numpy as np

def stft_basis(signal, fs=250, large_win_sec=2.0, short_win_sec=0.5):
    """
    Computes STFT basis matrices for a signal.
    Returns (large_basis, short_basis) where each is shape (window_samples, freq_bins)
    """
    def stft(sig, win_len, fs):
        nperseg = int(win_len * fs)
        noverlap = nperseg // 2
        # Hann window
        window = np.hanning(nperseg)
        # Number of windows
        step = nperseg - noverlap
        n_windows = (len(sig) - noverlap) // step
        stft_matrix = []
        for i in range(n_windows):
            start = i * step
            end = start + nperseg
            if end > len(sig):
                break
            seg = sig[start:end] * window
            spectrum = np.fft.fft(seg)
            stft_matrix.append(spectrum)
        return np.array(stft_matrix).T  # shape: (freq_bins, windows)
    
    large_basis = stft(signal, large_win_sec, fs)
    short_basis = stft(signal, short_win_sec, fs)
    return large_basis, short_basis