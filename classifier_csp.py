import numpy as np
import time
import os
from scipy.signal import welch
import pandas as pd
from sklearn.svm import SVC
import joblib
from muse_stream import get_eeg_buffer
from feature_extraction import bandpass_filter
from artifact_removal_ica import ica_blink_filter

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

# 


from feature_extraction import hjorth_bandpower
def process_windows(idle_indices, classification, df, window_size, num_windows=5):
    windows = []
    labels = []
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            window = bandpass_filter(window)
            window = ica_blink_filter(window)
            windows.append(window)  # shape: (window_size, 4)
            labels.append(classification)        # Idle class label
    return windows, labels


def classify(stop_event):
    all_output_data = []

    window_size = 500  # 2 seconds, 4 channels, 250Hz
    csv_path = os.path.join(os.path.dirname(__file__), 'AllData/nick1.csv')
    df = pd.read_csv(csv_path)

    attention_indices = df.index[df['Class'] == 2].tolist()
    idle_indices = df.index[df['Class'] == 1].tolist()


    # Get windows and labels for both classes
    attention_windows, attention_labels = process_windows(attention_indices, 1, df, window_size)
    idle_windows, idle_labels = process_windows(idle_indices, 2, df, window_size)

    # Combine data
    X = np.array(attention_windows + idle_windows)  # shape: (n_samples, window_size, n_channels)
    y = np.array(attention_labels + idle_labels)    # shape: (n_samples,)

    


    # Train CSP
    csp = CSP(n_components=4)
    X_csp = np.transpose(X, (0, 2, 1))  # shape: (n_samples, n_channels, window_size)
    X_features = csp.fit_transform(X_csp, y)


    # Train LDA
    svm = SVC(kernel="linear")
    svm.fit(X_features, y)




    try:
        attention_threshold = 0
        while not stop_event.is_set():
            data_array = get_eeg_buffer()
            if len(data_array) < window_size * 4:
                time.sleep(0.1)
                continue
            if len(data_array) == window_size * 4:
                eeg_window = np.array(data_array).reshape(window_size, 4)
                eeg_window = bandpass_filter(eeg_window)
                eeg_window = ica_blink_filter(eeg_window)
                # Reshape for CSP: (1, n_channels, window_size)
                eeg_window_csp = eeg_window.T[np.newaxis, :, :]  # shape: (1, 4, window_size)
                # Transform using trained CSP
                features = csp.transform(eeg_window_csp)
                # Predict class
                predicted_class = svm.predict(features)[0]
                adder = -10
                if predicted_class == 2:
                    adder = 1

                attention_threshold += adder
                attention_threshold = max(0, min(attention_threshold, 300))

                gesture = "Open"
                if attention_threshold >= 200:
                    gesture = "Close"
                print(f"Predicted class: {gesture}")
        print("Exiting classification")
        return
    except KeyboardInterrupt:
        print("Exiting...")

import threading
def main():
    stop_event = threading.Event()
    classify(stop_event)

if __name__ == "__main__":
    main()