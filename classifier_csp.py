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


# 


from feature_extraction import hjorth_bandpower
def process_idle_windows(idle_indices, df, window_size, num_windows=5):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            #window = ica_blink_filter(window)
            features = []
            for i in range(4):
                signal = window[:, i]
                input = hjorth_bandpower(signal)
                features.extend(input)
            #actual_class = df.iloc[window_start, 4]
            features.append(1)
            processed_data.append(features)
    return processed_data

def process_attention_windows(attention_indices, df, window_size, num_windows=4):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in attention_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            #print(window.shape)
           #window = ica_blink_filter(window)
            
            features = []
            for i in range(4):
                signal = window[:,i]
                input = hjorth_bandpower(signal)
                features.extend(input)
            #actual_class = df.iloc[window_start, 4]
            features.append(2)
            processed_data.append(features)
    return processed_data

def classify(stop_event):
    all_output_data = []

    window_size = 500  # 2 seconds, 4 channels, 250Hz
    # csv_path = os.path.join(os.path.dirname(__file__), 'shawn1.csv')
    # df = pd.read_csv(csv_path)
    # Combine all CSV files in the AllData folder into a single DataFrame
    all_data_dir = os.path.join(os.path.dirname(__file__), 'AllData')
    csv_files = [f for f in os.listdir(all_data_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(os.path.join(all_data_dir, f)) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # Get indices where class is 2 (attention) and 1 (idle)
    attention_indices = df.index[df['Class'] == 2].tolist()
    idle_indices = df.index[df['Class'] == 1].tolist()

    all_output_data.extend(process_attention_windows(attention_indices, df, window_size))
    all_output_data.extend(process_idle_windows(idle_indices, df, window_size))

    all_output_data = np.array(all_output_data)


    # Features: all columns except last
    X = all_output_data[:, :-1]
    # Labels: last column
    y = all_output_data[:, -1]

    # Split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    svm = SVC(kernel='linear')
    svm.fit(X,y)
    # print("Training")
    # svm.fit(X_train, y_train)

    # # Predict on test set
    # y_pred = svm.predict(X_test)

    # # Plot confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Idle", "Attention"])
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix")
    # plt.show()


    try:
        attention_threshold = 0
        while not stop_event.is_set():
            data_array = get_eeg_buffer()
            if len(data_array) < window_size:
                time.sleep(0.1)
                continue
            if len(data_array) == window_size*4:
                eeg_window = data_array.reshape(window_size, 4)
                features = []
                eeg_window = ica_blink_filter(eeg_window)
                for ch in range(4):
                    signal = eeg_window[:, ch]
                    features.extend(hjorth_bandpower(signal))
                features = np.array(features).reshape(1, -1)
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