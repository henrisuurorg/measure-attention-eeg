import matplotlib.pyplot as plt
import numpy as np

def plot_preprocessing(channels, eeg_df):
    plt.figure(figsize=(15,20))

    for i, channel in enumerate(channels):
        plt.subplot(4, 3, i*3+1)
        plt.plot(eeg_df[f'{channel}_normal'], label=f'{channel} normalized')
        plt.title(f'{channel} normalized')

        plt.subplot(4, 3, i*3+2)
        plt.plot(eeg_df[f'{channel}_artif_removed'], label=f'{channel} artifacts removed')
        plt.title(f'{channel} artifacts removed ')

        plt.subplot(4, 3, i*3+3)
        plt.plot(eeg_df[f'{channel}_bandpassed'], label=f'{channel} bandpassed')
        plt.title(f'{channel} bandpassed')

def plot_sample_durations(eeg_trials, gradcpt_trials):
    assert len(eeg_trials) == len(gradcpt_trials), "Length of GradCPT and EEG trials should match."
    n = len(eeg_trials)

    def trial_length(arr):
        diffs = []
        for i in range(1, len(arr)):
            diffs.append(arr[i] - arr[i-1])
        return diffs
    
    plt.figure(figsize=(n*5,10))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.plot(trial_length(gradcpt_trials[i]['start_timestamp']))
        
        plt.subplot(2, n, i+n+1)
        plt.plot(trial_length(eeg_trials[i]['timestamps']))

def plot_eeg_gradcpt_time_diff(eeg_trials, gradcpt_trials):
    assert len(eeg_trials) == len(gradcpt_trials), "Length of GradCPT and EEG trials should match."
    n = len(eeg_trials)

    def time_diff(eeg_df, gradcpt_df):
        segment_samples = 205

        num_segments = len(gradcpt_df['in_the_zone'])
        diffs = []
        for i in range(num_segments):
            eeg_i = i*segment_samples
            diff = (eeg_df['timestamps'][eeg_i] - gradcpt_df['start_timestamp'][i]) * 1000 # ms
            diffs.append(diff)
        
        return diffs
    
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure(figsize=(15, 5))
    for i in range(n):
        diffs = time_diff(eeg_trials[i], gradcpt_trials[i])
        plt.plot(diffs, label=f'Trial {i+1}', color=plt_colors[i])
        mean = np.mean(diffs)
        plt.axhline(y=mean, linestyle='--', color=plt_colors[i], label=f'Mean of Trial {i+1}', linewidth=1)

    plt.title('Comparison of Trials')
    plt.xlabel('Segment')
    plt.ylabel('Diff. between t_eeg & t_gcpt')
    plt.legend()

    plt.show()
