import pandas as pd
import os
import numpy as np
from feature_extraction import *
from filter import *
from window_extractor import extract_timeseries
from sklearn.feature_selection import SelectKBest, f_classif

csv_path = os.path.join(os.path.dirname(__file__), '../shawn1.csv')
df = pd.read_csv(csv_path)

attention_indices = df.index[df['Class'] == 2].tolist()
idle_indices = df.index[df['Class'] == 1].tolist()

filter_funcs = [bandpass_filter, ica_blink_filter]
feature_funcs = [calculate_hjorth_parameters, calculate_bandpowers, calculate_mean, calculate_std, calculate_rms,
                 calculate_skewness, calculate_kurtosis, calculate_entropy, calculate_peak_to_peak, calculate_log_variance]
window_size = 500
num_windows = 5

# Extract windows and labels
windows_a, labels_a = extract_timeseries(attention_indices, 2, df, window_size, num_windows, filter_funcs)
windows_i, labels_i = extract_timeseries(idle_indices, 1, df, window_size, num_windows, filter_funcs)

windows = windows_a + windows_i
labels = labels_a + labels_i

# Build feature matrix
feature_matrix = []
for window in windows:
    features = []
    for ch in range(window.shape[1]):
        signal = window[:, ch]
        for func in feature_funcs:
            result = func(signal)
            if hasattr(result, '__iter__') and not isinstance(result, str):
                features.extend(result)
            else:
                features.append(result)
    feature_matrix.append(features)

X = np.array(feature_matrix)
y = np.array(labels)

# Apply ANOVA feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X, y)
scores = selector.scores_

# Build feature names list
feature_names = []
for ch in range(4):
    for func in feature_funcs:
        test_signal = np.random.randn(window_size)
        result = func(test_signal)
        if hasattr(result, '__iter__') and not isinstance(result, str):
            for i in range(len(result)):
                feature_names.append(f"Ch{ch+1}_{func.__name__}_{i}")
        else:
            feature_names.append(f"Ch{ch+1}_{func.__name__}")

print("ANOVA F-scores for each feature (sorted):")
sorted_features = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
for name, score in sorted_features:
    print(f"{name}: {score:.3f}")