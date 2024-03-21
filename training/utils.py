import pywt
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, ttest_ind
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import filtfilt, butter


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

def segment_column(column, gradcpt_df):
# 256*0.8=204.8 ||| 204*(1/256)=0,7969 and 205*(1/256)=0,8008
# 205 is closer to 800 and gradcpt usually takes a fraction of ms longer than 800ms
    segment_samples = 205
    
    num_segments = len(gradcpt_df['in_the_zone'])
    segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = column[start:end].values
        segments.append(segment)

    return segments

def synchronize_trials(eeg_df, gradcpt_df):
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

    return eeg_df_aligned, gradcpt_df_aligned

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

def extract_features(channel, segments, gradcpt_df):
    # Define bands and features for clarity and extensibility
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Calculate features for each segment and store them
    features_data = []
    for i in range(len(segments)):
        segment_features = {}
        for band in bands:
            segment_features[f'{channel}_{band}_approx_entropy'] = approximate_entropy(segments[i][band])
            segment_features[f'{channel}_{band}_total_variation'] = total_variation(segments[i][band])
            segment_features[f'{channel}_{band}_standard_deviation'] = standard_deviation(segments[i][band])
            segment_features[f'{channel}_{band}_energy'] = energy(segments[i][band])
            segment_features[f'{channel}_{band}_skewness'] = skewness(segments[i][band])
    
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
    features_df['in_the_zone'] = gradcpt_df['in_the_zone']
    features_df.fillna(0, inplace=True)

    return features_df

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