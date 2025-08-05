

import pandas as pd
import matplotlib.pyplot as plt

# Import classifiers
from classifiers import get_svm, get_lda, get_xgboost, get_random_forest

# Import feature functions
from feature_extraction import calculate_hjorth_parameters, calculate_bandpowers

# Import filter functions
from filter import bandpass_filter, notch_filter, ica_blink_filter

# Import feature extraction and training functions
from window_extractor import extract_feature_vector
from train import train


# Define classifier
classifier = get_svm()  # Swap out for get_lda(), get_xgboost(), get_random_forest(), etc.

# Define features and filters of interest
feature_funcs = [calculate_hjorth_parameters, calculate_bandpowers]
filter_funcs = [bandpass_filter, ica_blink_filter]

window_size = 500
num_windows = 5

# Get all CSV files in AllData folder
import os
all_data_dir = os.path.join(os.path.dirname(__file__), '../AllData')
csv_files = [f for f in os.listdir(all_data_dir) if f.endswith('.csv')]
#csv_files = [f for f in os.listdir(all_data_dir) if f.endswith('gabe2.csv')]
f1_scores = []
file_names = []

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(all_data_dir, csv_file))
    attention_indices = df.index[df['Class'] == 2].tolist()
    idle_indices = df.index[df['Class'] == 1].tolist()
    Xy = []
    Xy.extend(extract_feature_vector(attention_indices, df, window_size, 2, num_windows, filter_funcs, feature_funcs))
    Xy.extend(extract_feature_vector(idle_indices, df, window_size, 1, num_windows, filter_funcs, feature_funcs))
    Xy = pd.DataFrame(Xy).values
    X = Xy[:, :-1]
    y = Xy[:, -1]
    f1 = train(X, y, classifier)
    f1_scores.append(f1)
    file_names.append(csv_file)
    print(f"{csv_file}: F1 Score = {f1:.3f}")

# Plot F1 scores in a bar plot
plt.figure(figsize=(10,6))
plt.bar(file_names, f1_scores)
plt.ylabel('F1 Score')
plt.xlabel('CSV File')
plt.title('F1 Scores for Each Subject')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()