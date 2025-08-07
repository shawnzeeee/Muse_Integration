

import pandas as pd
import matplotlib.pyplot as plt
import os
# Import classifiers
from classifiers import get_svm, get_lda, get_xgboost, get_random_forest

# Import feature functions
from feature_extraction import *

# Import filter functions
from filter import bandpass_filter, notch_filter, ica_blink_filter, clean_eeg_ica_threshold

# Import feature extraction and training functions
from window_extractor import extract_feature_vector
from feature_extraction import calculate_hjorth_parameters, calculate_bandpowers, calculate_log_variance

import numpy as np

from sklearn.preprocessing import StandardScaler

from muse_stream import get_eeg_buffer
import serial 

def classify(stop_event):
    all_output_data = []

    scaler = StandardScaler()
    # Define classifier
    classifier = get_svm()  # Swap out for get_lda(), get_xgboost(), get_random_forest(), etc.

    # Define features and filters of interest
    feature_funcs = [calculate_hjorth_parameters, calculate_bandpowers, calculate_log_variance]
    filter_funcs = [bandpass_filter, clean_eeg_ica_threshold]

    window_size = 500
    num_windows = 5


    csv_path = os.path.join(os.path.dirname(__file__), '../AllData/nickreal2.csv')
    df = pd.read_csv(csv_path)

    attention_indices = df.index[df['Class'] == 2].tolist()
    idle_indices = df.index[df['Class'] == 1].tolist()

    Xy = []
    Xy.extend(extract_feature_vector(attention_indices, df, window_size, 2, num_windows, filter_funcs, feature_funcs))
    Xy.extend(extract_feature_vector(idle_indices, df, window_size, 1, num_windows, filter_funcs, feature_funcs))
    Xy = pd.DataFrame(Xy).values
    X = Xy[:, :-1]
    y = Xy[:, -1]

    X_scaled = scaler.fit_transform(X)

    classifier.fit(X_scaled,y)
    
    try:
        attention_threshold = 0
        ser = serial.Serial('/dev/serial0', 9600, timeout=1)
        while not stop_event.is_set():
            data_array = get_eeg_buffer()
            #data_array = np.zeros(window_size*4, dtype=np.float32)
            if len(data_array) < window_size:
                time.sleep(0.1)
                continue
            if len(data_array) == window_size*4:
                eeg_window = data_array.reshape(window_size, 4)
                features = []
                for func in filter_funcs:
                    eeg_window = func(eeg_window)
                for ch in range(4):
                    signal = eeg_window[:, ch]
                    for func in feature_funcs:
                        input = func(signal)
                        if hasattr(input, '__iter__') and not isinstance(input, str):
                            features.extend(input)
                            #print(input)
                        else:
                            features.append(input)
                
                features = np.array(features).reshape(1, -1)
                if not np.isfinite(features).all():
                    print("NANANANANANANNANANAN")
                    continue
                features_scaled = scaler.transform(features)

                predicted_class = classifier.predict(features_scaled)[0]

                adder = -5
                if predicted_class == 2:
                    adder = 5

                attention_threshold += adder
                attention_threshold = max(0, min(attention_threshold, 220))

                gesture = 'O'
                if attention_threshold >= 100:
                    gesture = 'C'
                ser.write(gesture.encode())
                print(f"Predicted class: {gesture}      {attention_threshold}")        
        print("Exiting classification")
        return
    except KeyboardInterrupt:
        print("Exiting...")


def main():
    import threading
    stop_event = threading.Event()
    try:
        classify(stop_event)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()