import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def process_windows(df, window_sec=2, fs=250):
    """
    Returns a list of (window, t) for each blink (classification==1).
    Each window is shape (samples, 4), t is time axis centered on blink.
    """
    blink_indices = df.index[df.iloc[:, -1] == 1].tolist()
    half_window = int(window_sec * fs // 2)
    windows = []
    for blink_idx in blink_indices:
        blink_idx = blink_idx + 100
        start = max(0, blink_idx - half_window)
        end = min(len(df), blink_idx + half_window)
        window = df.iloc[start:end, :4].values
        t = np.arange(start, end) / fs
        t = t - t[0] - (window_sec / 2)
        windows.append((window, t))
    return windows


def stft_basis(signal, fs=250, frame_len_sec=2.0, large_win_sec=2.0, short_win_sec=0.5):
    """
    Returns STFT time-domain basis matrices using different window lengths but same frame length.
    Output shapes: (frame_samples, num_atoms)
    """
    def windowed_atoms(sig, win_len_sec, frame_len_samples):
        win_len = int(win_len_sec * fs)
        hop = win_len // 2
        window = np.hanning(win_len)
        atoms = []

        for start in range(0, frame_len_samples - win_len + 1, hop):
            seg = sig[start:start + win_len] * window
            padded_seg = np.zeros(frame_len_samples)
            padded_seg[start:start + win_len] = seg
            atoms.append(padded_seg)

        return np.column_stack(atoms)  # shape: (frame_len_samples, num_atoms)

    frame_len_samples = int(frame_len_sec * fs)
    signal = signal[:frame_len_samples]  # trim to fixed frame length

    large_basis = windowed_atoms(signal, large_win_sec, frame_len_samples)
    short_basis = windowed_atoms(signal, short_win_sec, frame_len_samples)
    return large_basis, short_basis



def soft_thresholding(x, threshold):
    """Soft thresholding operator for L1 norm minimization"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

def salsa_mca(y, phi1, phi2, lamb1=0.1, lamb2=0.1, mu=1.0, max_iter=10000, tol=1e-4):
    """
    Implements the SALSA algorithm to solve:
    min ||lambda1*alpha1||_1 + ||lambda2*alpha2||_1 s.t. y = phi1^T alpha1 + phi2^T alpha2
    
    Parameters:
        y     : observed signal (vector)
        phi1  : EEG basis matrix (columns = atoms)
        phi2  : Blink basis matrix (columns = atoms)
        lamb1 : L1 sparsity penalty for alpha1
        lamb2 : L1 sparsity penalty for alpha2
        mu    : ADMM penalty term
        max_iter : maximum iterations
        tol   : stopping tolerance

    Returns:
        alpha1, alpha2 : sparse coefficients
        y_hat_clean    : reconstructed clean EEG (phi1 @ alpha1)
        y_hat_blink    : reconstructed blink (phi2 @ alpha2)
    """
    A = np.hstack((phi1, phi2))        # Full basis
    lamb = np.concatenate([lamb1 * np.ones(phi1.shape[1]), lamb2 * np.ones(phi2.shape[1])])

    x = np.zeros(A.shape[1])           # Initial x = [alpha1; alpha2]
    v = np.zeros_like(x)               # Initial v
    d = np.zeros_like(x)               # Initial dual variable

    AtA = A.T @ A
    Aty = A.T @ y
    I = np.eye(A.shape[1])

    for _ in range(max_iter):
        # Step 1: update x
        x_new = np.linalg.solve(AtA + mu * I, Aty + mu * (v + d))

        # Step 2: update v via soft thresholding
        v_new = soft_thresholding(x_new - d, lamb / mu)

        # Step 3: update dual variable
        d_new = d - (x_new - v_new)

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            break

        x, v, d = x_new, v_new, d_new

    alpha1 = x[:phi1.shape[1]]
    alpha2 = x[phi1.shape[1]:]
    print(phi1.shape, alpha1.shape)
    y_hat_clean = phi1 @ alpha1
    y_hat_blink = phi2 @ alpha2

    return alpha1, alpha2, y_hat_clean, y_hat_blink

csv_path = "blink_data.csv"
window_sec = 2
fs= 250

df = pd.read_csv(csv_path)
windows = process_windows(df, window_sec, fs)
window, t = windows[0]

channel_1 = window[:,0]
large_basis, short_basis = stft_basis(channel_1)

print(short_basis.shape)
alpha1, alpha2, y_clean, y_blink = salsa_mca(channel_1, large_basis, short_basis)

# Plot original, clean, and blink signals on separate subplots
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(channel_1, label="Original EEG (window)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Original EEG Window")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(y_clean, label="Reconstructed Clean EEG (y_clean)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Reconstructed Clean EEG")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(y_blink, label="Reconstructed Blink (y_blink)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Reconstructed Blink")
plt.legend()

plt.tight_layout()
plt.show()