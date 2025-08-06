from window_extractor import extract_timeseries, collect_imf_psd


import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy.signal import welch
import pandas as pd

from filter import *

window_size = 500
num_windows = 5

csv_path = os.path.join(os.path.dirname(__file__), '../AllData/carina1.csv')
df = pd.read_csv(csv_path)

attention_indices = df.index[df['Class'] == 2].tolist()
idle_indices = df.index[df['Class'] == 1].tolist()

filter_funcs = [bandpass_filter, ica_blink_filter]



# Collect PSDs for idle and attention
psd_attention = collect_imf_psd(attention_indices, 2, df, window_size, num_windows)
psd_idle = collect_imf_psd(idle_indices, 1, df, window_size, num_windows)

# Convert to numpy arrays and remove classification column
psd_attention = np.array([row[:-1] for row in psd_attention])
psd_idle = np.array([row[:-1] for row in psd_idle])

# Average across all windows for each class
mean_psd_attention = np.mean(psd_attention, axis=0)
mean_psd_idle = np.mean(psd_idle, axis=0)

plt.figure(figsize=(14, 6))

# Subplot 1: Idle
plt.subplot(1, 2, 1)

# Only plot between 10 and 50 Hz
freqs = np.arange(60)
freq_mask = (freqs >= 13) & (freqs <= 30)
for ch in range(4):
    plt.plot(freqs[freq_mask], mean_psd_idle[ch*60:(ch+1)*60][freq_mask], label=f'Channel {ch+1}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (normalized)')
plt.title('Idle Windows: Average PSD (IMF1 & IMF4)')
plt.legend()

# Subplot 2: Attention
plt.subplot(1, 2, 2)

# Only plot between 10 and 50 Hz
for ch in range(4):
    plt.plot(freqs[freq_mask], mean_psd_attention[ch*60:(ch+1)*60][freq_mask], label=f'Channel {ch+1}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (normalized)')
plt.title('Attention Windows: Average PSD (IMF1 & IMF4)')
plt.legend()

plt.tight_layout()
plt.show()