

def extract_feature_vector(indices, df, window_size, classification, num_windows=5, filter_funcs=[], feature_funcs=[]):
    processed_data = []
    channel_names = ["Channel 1", "Channel 2", "Channel 3", "Channel 4"]
    for start_idx in indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            #print(window.shape)

            for func in filter_funcs:
                window = func(window)
            
            features = []
            for i in range(4):
                signal = window[:,i]
                for func in feature_funcs:
                    input = func(signal)
                    if hasattr(input, '__iter__') and not isinstance(input, str):
                        features.extend(input)
                    else:
                        features.append(input)
            #actual_class = df.iloc[window_start, 4]
            features.append(classification)
            processed_data.append(features)
    return processed_data

def extract_timeseries(idle_indices, classification, df, window_size, num_windows=5, filter_funcs=[]):
    windows = []
    labels = []
    for start_idx in idle_indices:
        for w in range(num_windows):
            window_start = start_idx + w * window_size
            window_end = window_start + window_size
            if window_end > len(df):
                continue
            window = df.iloc[window_start:window_end, :4].values
            for func in filter_funcs:
                window = func(window)
            windows.append(window)  # shape: (window_size, 4)
            labels.append(classification)        # Idle class label
    return windows, labels