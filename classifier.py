
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

# Function to calculate standard deviation of a single-channel signal
def calculate_std(signal):
    """
    Returns the standard deviation of a 1D signal (single channel).
    """
    return np.std(signal)

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
            window = bandpass_filter(window)

            window = ica_blink_filter(window)
            features = []
            for i in range(4):
                signal = window[:, i]
                # if calculate_std(signal) > 100:
                #     print("Unstable signal")
                input = hjorth_bandpower(signal)
                features.extend(input)
            #actual_class = df.iloc[window_start, 4]
            features.append(1)
            processed_data.append(features)
    return processed_data

def process_attention_windows(attention_indices, df, window_size, num_windows=5):
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
            window = bandpass_filter(window)
            window = ica_blink_filter(window)
            
            features = []
            for i in range(4):
                signal = window[:,i]
                if calculate_std(signal) > 100:
                    print("Unstable signal")
                input = hjorth_bandpower(signal)
                features.extend(input)
            #actual_class = df.iloc[window_start, 4]
            features.append(2)
            processed_data.append(features)
    return processed_data

def classify(stop_event):
    all_output_data = []

    window_size = 500  # 2 seconds, 4 channels, 250Hz
    csv_path = os.path.join(os.path.dirname(__file__), 'AllData/shawn1.csv')
    df = pd.read_csv(csv_path)
    # Combine all CSV files in the AllData folder into a single DataFrame
    # all_data_dir = os.path.join(os.path.dirname(__file__), 'AllData')
    # csv_files = [f for f in os.listdir(all_data_dir) if f.endswith('.csv')]
    # df_list = [pd.read_csv(os.path.join(all_data_dir, f)) for f in csv_files]
    # df = pd.concat(df_list, ignore_index=True)

    # Get indices where class is 2 (attention) and 1 (idle)
    attention_indices = df.index[df['Class'] == 2].tolist()
    idle_indices = df.index[df['Class'] == 1].tolist()

    all_output_data.extend(process_attention_windows(attention_indices, df, window_size))
    all_output_data.extend(process_idle_windows(idle_indices, df, window_size))



    all_output_data = np.array(all_output_data)

    # Standardize all features except the last column (classification)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = all_output_data[:, :-1]
    X_scaled = scaler.fit_transform(X)
    # Save the fitted scaler for real-time use

    # Combine scaled features with original classification column
    all_output_data = np.hstack([X_scaled, all_output_data[:, -1].reshape(-1, 1)])

    # Save processed features to CSV
    # processed_data_path = os.path.join(os.path.dirname(__file__), 'processed_data.csv')
    # pd.DataFrame(all_output_data).to_csv(processed_data_path, index=False)
    # print(f"Processed data saved to {processed_data_path}")

    # Features: all columns except last
    X = all_output_data[:, :-1]
    # Labels: last column
    y = all_output_data[:, -1]

    # Split into train and test sets


    svm = SVC(kernel='linear')
    svm.fit(X,y)
    # model_path = os.path.join(os.path.dirname(__file__), 'svm_model.joblib')
    # joblib.dump(svm, model_path)
    # print(f"SVM model saved to {model_path}")
    # print("Training")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    svm_test = SVC(kernel='linear')

    svm_test.fit(X_train, y_train)

    # Predict on test set
    y_pred = svm_test.predict(X_test)


    # Print F1 score
    # from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print(f"F1 Score: {f1:.3f}")

    # # Plot confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap='Blues')
    # plt.title('Confusion Matrix')
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
                eeg_window = bandpass_filter(eeg_window)
                eeg_window = ica_blink_filter(eeg_window)
                for ch in range(4):
                    signal = eeg_window[:, ch]
                    features.extend(hjorth_bandpower(signal))
                features = np.array(features).reshape(1, -1)
                # Load the scaler and scale features
                features_scaled = scaler.transform(features)
                # Predict class
                predicted_class = svm.predict(features_scaled)[0]
                # Distance from decision boundary
                if hasattr(svm, "decision_function"):
                    distance = svm.decision_function(features_scaled)[0]
                    #print(f"Distance from boundary: {distance:.3f}")
                else:
                    distance = None
                adder = -10
                if predicted_class == 2:
                    adder = 1

                attention_threshold += adder
                attention_threshold = max(0, min(attention_threshold, 220))

                gesture = "Open"
                if attention_threshold >= 200:
                    gesture = "Close"
                print(f"Predicted class: {distance}")
                time.sleep(0.1)
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