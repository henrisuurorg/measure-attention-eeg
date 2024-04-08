import pywt
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, ttest_ind
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import filtfilt, butter, welch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from detach_rocket.detach_rocket.detach_classes import DetachMatrix
import matplotlib.pyplot as plt


def synchronize_trials(eeg_df, gradcpt_df):
    gradcpt_df = gradcpt_df[3:].reset_index(drop=True) # drop first 2 trials because they can be unstable
    start_diff = eeg_df['timestamps'][0] - gradcpt_df['start_timestamp'][0]

    if start_diff < 0:
        # If eeg_data starts earlier, align it to gradcpt_data
        closest_idx = (eeg_df['timestamps'] - gradcpt_df['start_timestamp'][0]).abs().idxmin()
        eeg_df_aligned = eeg_df[closest_idx-1:].reset_index(drop=True)
        gradcpt_df_aligned = gradcpt_df.copy()
    else:
        # If gradcpt_data starts earlier, find the point where eeg_data starts
        i = next(i for i, ts in enumerate(gradcpt_df['start_timestamp']) if ts > eeg_df['timestamps'][0])
        gradcpt_df_aligned = gradcpt_df[i:].reset_index(drop=True)
        closest_idx = (eeg_df['timestamps'] - gradcpt_df_aligned['start_timestamp'][0]).abs().idxmin()
        eeg_df_aligned = eeg_df[closest_idx-1:].reset_index(drop=True)
    
    # Check if EEG data ends before gradcpt data
    required_eeg_length = len(gradcpt_df_aligned) * 205
    if len(eeg_df_aligned) < required_eeg_length:
        # Calculate the number of gradcpt samples that can be supported by the available eeg samples
        supported_gradcpt_samples = len(eeg_df_aligned) // 205
        gradcpt_df_aligned = gradcpt_df_aligned[:supported_gradcpt_samples].reset_index(drop=True)
        print('Gradcpt data had to be truncated')

    return eeg_df_aligned, gradcpt_df_aligned

def z_normalize_column(column):
    mean = column.mean()
    std = column.std()
    return (column - mean) / std

def remove_artifacts_from_column(column, fs=256, threshold=3, window_ms=450):
    """
    Remove EOG artifacts from a Series representing an EEG signal.

    Parameters:
    - column: Pandas Series containing the EEG data.
    - fs: Sampling frequency in Hz. Default is 256Hz.
    - threshold: The threshold value used to detect artifacts. Default is 3.
    - window_ms: Duration of the EOG event window in milliseconds. Default is 450ms.

    Returns:
    - A Series with the artifacts removed.
    """
    window_samples = int((window_ms / 1000) * fs)  # Convert window duration from ms to samples

    eog_peaks = np.where(np.abs(column) > threshold)[0]
    eog_regions = np.zeros_like(column, dtype=bool)
    for peak in eog_peaks:
        start = max(peak - window_samples // 2, 0)
        end = min(peak + window_samples // 2, len(column))
        eog_regions[start:end] = True

    structuring_element = np.ones(window_samples)
    eog_regions_closed = binary_closing(eog_regions, structure=structuring_element)
    eog_regions_cleaned = binary_opening(eog_regions_closed, structure=structuring_element)

    artifact_removed = column.copy()
    for start in np.where(np.diff(eog_regions_cleaned.astype(int)) == 1)[0] + 1:
        end = start + np.where(eog_regions_cleaned[start:] == False)[0][0]
        replacement_length = end - start
        replacement_start = max(start - replacement_length, 0)

        if start == 0:  # If the artifact is at the very start
            replacement_values = column[end:end+replacement_length]  # Use following clean segment
        else:
            replacement_values = artifact_removed[replacement_start:replacement_start+replacement_length]

        artifact_removed[start:end] = replacement_values.values

    return artifact_removed

def bandpass(column, lowcut=0.5, highcut=50.0, fs=256, order=5):
    # Butterworth filter
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, column)

def segment_column(column, gradcpt_df):
# 256*0.8=204.8 ||| 204*(1/256)=0,7969 and 205*(1/256)=0,8008
# 205 is closer to 800 and gradcpt usually takes a fraction of ms longer than 800ms
    segment_samples = 205
    
    num_segments = len(gradcpt_df['in_the_zone'])
    segments = []
    timestamps = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = column[start:end].values
        segments.append(segment)

    return segments

def decompose_segment(segment, wavelet='sym3', max_level=5):
    if max_level is None:
        max_level = pywt.dwt_max_level(len(segment), wavelet)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(segment, wavelet, level=max_level)

    bands = {'delta': None, 'theta': None, 'alpha': None, 'beta': None, 'gamma': None}

    # Start from gamma to match Zhang et al.
    bands['gamma'] = pywt.upcoef('d', coeffs[1], wavelet, level=max_level, take=len(segment))

    for i in range(2, max_level + 1):
        if i == max_level:
            # Extract delta band from the last level approximation coefficients
            bands['delta'] = pywt.upcoef('a', coeffs[-1], wavelet, level=1, take=len(segment))
        else:
            # Extract other bands by reconstructing from specific detail coefficients
            band_name = {2: 'beta', 3: 'alpha', 4: 'theta'}[i]
            bands[band_name] = pywt.upcoef('d', coeffs[i], wavelet, level=max_level-i+1, take=len(segment))

    return bands

# Features defined below
def extract_features(channel, segments):
    # Define bands and features for clarity and extensibility
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Calculate features for each segment and store them
    features_data = []
    for i in range(len(segments)):
        segment_features = {}
        for band in bands:
            signal = segments[i][band]  # Assuming each segment is a dict with band-labeled keys
            
            # Existing features
            segment_features[f'{channel}_{band}_approx_entropy'] = approximate_entropy(signal)
            segment_features[f'{channel}_{band}_total_variation'] = total_variation(signal)
            segment_features[f'{channel}_{band}_standard_deviation'] = standard_deviation(signal)
            segment_features[f'{channel}_{band}_energy'] = energy(signal)
            segment_features[f'{channel}_{band}_skewness'] = skewness(signal)
            
            # New features
            _, psd = power_spectral_density(signal)
            segment_features[f'{channel}_{band}_psd_mean'] = np.mean(psd)
            segment_features[f'{channel}_{band}_spectral_entropy'] = spectral_entropy(signal)
            segment_features[f'{channel}_{band}_sef'] = spectral_edge_frequency(signal)
            
            # Hjorth parameters (as separate features)
            activity, mobility, complexity = hjorth_parameters(signal)
            segment_features[f'{channel}_{band}_hjorth_activity'] = activity
            segment_features[f'{channel}_{band}_hjorth_mobility'] = mobility
            segment_features[f'{channel}_{band}_hjorth_complexity'] = complexity    
        features_data.append(segment_features)
    
    # Now, for each segment, append features from preceding 9 windows
    augmented_features_data = []
    
    for i in range(len(features_data)):
        augmented_features = {}
        for j in range(max(0, i-9), i+1):
            for key, value in features_data[j].items():
                # Adjust the key to include the window index relative to the current segment
                augmented_key = f'{key}_win{j-i}'
                augmented_features[augmented_key] = value
                
        augmented_features_data.append(augmented_features)
    
    # Create a DataFrame from the augmented features data
    features_df = pd.DataFrame(augmented_features_data)
    features_df.fillna(0, inplace=True)

    return features_df

def SFD(X, y, p=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
    detach_matrix = DetachMatrix(trade_off=p)
    detach_matrix.fit(X_train, y_train)

    # Evaluate Performance on Test Set
    detach_test_score, full_test_score= detach_matrix.score(X_test, y_test)
    print('Test Accuraccy Full Model: {:.2f}%'.format(100*full_test_score))
    print('Test Accuraccy Detach Model: {:.2f}%'.format(100*detach_test_score))

    percentage_vector = detach_matrix._percentage_vector
    acc_curve = detach_matrix._sfd_curve

    c = detach_matrix.trade_off

    x_sfd=(percentage_vector) * 100
    y_sfd=(acc_curve/acc_curve[0]-1) * 100

    point_x = x_sfd[detach_matrix._max_index]
    #point_y = y[DetachMatrixModel._max_index]

    plt.figure(figsize=(8,3.5))
    plt.axvline(x = point_x, color = 'r',label=f'Optimal Model (c={c})')
    plt.plot(x_sfd, y_sfd, label='SFD curve', linewidth=2.5, color='C7', alpha=1)
    #plt.scatter(point_x, point_y, s=50, marker='o', label=f'Optimal point (c={c})')

    plt.grid(True, linestyle='-', alpha=0.5)
    plt.xlim(102,-2)
    plt.xlabel('% of Retained Features')
    plt.ylabel('Relative Validation Set Accuracy (%)')
    plt.legend()
    plt.show()

    print('Optimal Model Size: {:.2f}% of full model'.format(point_x))

    feature_mask = detach_matrix._feature_mask
    
    selected_features_df = X.loc[:, feature_mask]
    selected_features_df = selected_features_df.assign(Label=y)

    print(f'Number of features kept: {len(selected_features_df)-1}')

    return selected_features_df



def top_bot_25(feature_df):
        t_values = []
        feature_names = []

        # Perform t-test for each feature against 'in_the_zone'
        for column in feature_df.columns[:-1]:  # Exclude the last column ('in_the_zone')
            t_stat, p_val = ttest_ind(feature_df[column], feature_df['in_the_zone'], nan_policy='omit')
            t_values.append(t_stat)
            feature_names.append(column)

        # Create a DataFrame to store features and their corresponding t-values
        t_values_df = pd.DataFrame({'Feature': feature_names, 'T-value': t_values})

        # Sort the DataFrame by the absolute t-values
        t_values_df['Abs T-value'] = t_values_df['T-value'].abs()
        t_values_df_sorted = t_values_df.sort_values(by='Abs T-value', ascending=False)

        # Select the 25 most important features
        top_25_features = t_values_df_sorted.head(25)

        # Select the 25 least important features
        bottom_25_features = t_values_df_sorted.tail(25)

        # You can now print or further analyze these subsets
        print("Top 25 Most Important Features:")
        print(top_25_features)

        print("\nBottom 25 Least Important Features:")
        print(bottom_25_features)

def train(runs, num_features, df):
    results = []
    
    for _ in range(runs):
        from scipy.stats import ttest_ind
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.svm import SVC
        
        features = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        
        def select_top_features(X, y, num_features=num_features):
            # Perform a t-test across features
            t_stats, p_values = ttest_ind(X[y == 0], X[y == 1], axis=0)
            # Select indices of top features based on smallest p-values
            top_features_indices = np.argsort(np.abs(t_stats))[-num_features:]
            return top_features_indices
        
        outer_cv = StratifiedKFold(n_splits=10, shuffle=True)
        
        balanced_acc_scores = []
        
        for train_index, test_index in outer_cv.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        
            # Feature selection for the outer fold
            top_features_indices = select_top_features(X_train, y_train)
            X_train_selected = X_train[:, top_features_indices]
            X_test_selected = X_test[:, top_features_indices]
            
            # Initialize the scaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
        
            # Inner CV for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            param_grid = {'C': [0.1, 0.5, 1], 'gamma': ['scale'], 'kernel': ['rbf']}
            grid_search = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='balanced_accuracy')
            grid_search.fit(X_train_scaled, y_train)
        
            best_model = grid_search.best_estimator_
            
            balanced_acc = balanced_accuracy_score(y_test, best_model.predict(X_test_scaled))
            balanced_acc_scores.append(balanced_acc)
        
        final_performance = np.mean(balanced_acc_scores)
        results.append(round(final_performance, 3))
    
    print(f'Avg: {round((sum(results) / len(results)) * 100, 3)}%')

# FEATURES
def approximate_entropy(signal, m=2, r=None):
    def _maxdist(x_i, x_j):
        return np.max(np.abs(x_i - x_j), axis=1)

    def _phi(m, x, N, r):
        C = np.zeros(len(x))
        for i, x_i in enumerate(x):
            # Calculate distance in a vectorized manner
            dists = _maxdist(x_i, x)
            # Count the number of distances less than or equal to r
            C[i] = np.sum(dists <= r) / (N - m + 1.0)
        C += 1e-10  # To avoid log(0)
        
        return np.sum(np.log(C)) / (N - m + 1.0)

    # Ensure 'signal' is a numpy array
    U = np.array(signal)
    N = len(U)

    # Set the similarity criterion 'r' if it is not provided
    if r is None:
        r = 0.2 * np.std(signal)
        
    # Precompute slices of U for different values of m
    x_m = [np.array([U[j:j+m] for j in range(N - m + 1)]) for m in [m, m+1]]
    
    # Calculate approximate entropy using precomputed slices
    return abs(_phi(m + 1, x_m[1], N, r) - _phi(m, x_m[0], N, r))


def total_variation(signal):
    return np.sum(np.abs(np.diff(signal)))

def standard_deviation(signal):
    return np.std(signal)

def energy(signal):
    return np.sum(np.square(signal))

def skewness(signal):
    return skew(signal)

def power_spectral_density(signal, fs=256):
    f, Pxx = welch(signal, fs=fs)
    return f, Pxx

def spectral_entropy(signal, fs=256, method='fft', normalize=False):
    if method == 'welch':
        f, psd = welch(signal, fs=fs)
    else:
        psd = np.abs(np.fft.fft(signal))**2
        psd = psd[:len(psd)//2]
    
    psd /= psd.sum()  # Normalize the PSD
    se = entropy(psd, base=2)
    
    if normalize:
        se /= np.log2(psd.size)
    
    return se

def spectral_edge_frequency(signal, fs=256, edge=0.9):
    f, Pxx = welch(signal, fs=fs)
    cumulative_power = np.cumsum(Pxx)
    total_power = cumulative_power[-1]
    edge_freq = f[np.where(cumulative_power >= total_power * edge)[0][0]]
    
    return edge_freq

def hjorth_parameters(signal):
    activity = np.var(signal)
    gradient = np.diff(signal)
    mobility = np.sqrt(np.var(gradient) / activity)
    gradient2 = np.diff(gradient)
    mobility_derivative = np.sqrt(np.var(gradient2) / np.var(gradient))
    complexity = mobility_derivative / mobility
    
    return activity, mobility, complexity
