from typing import Optional, Tuple, List
from time import time 
from psychopy import visual, core, event
from scipy.ndimage import gaussian_filter1d
from pylsl import StreamInfo, StreamOutlet
from eeg import EEG
from utils import eeg_save_fn, save_final_results, save_raw_responses, Trial
import numpy as np
import random

# constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
IMAGE_DISPLAY_TIME = 0.8006  # 800.6 ms (because 256 Hz of the EEG does not allow for 800ms windows)
TARGET_FREQ = 0.9
TRIAL_N = 75 * 10 # 75 trials = 1 minute
EEG_DURATION = 35 + 60 * 10 # 35 seconds of buffer + n minutes
FWHM = 9
SIGMA = FWHM / (2 * np.sqrt(2 * np.log(2)))

# Paths to images
city_images = [f'images/city/city_{i}.jpg' for i in range(10)]
mountain_images = [f'images/mountains/mountain_{i}.jpg' for i in range(10)]

# Setup the window
win = visual.Window(size=(SCREEN_WIDTH, SCREEN_HEIGHT), fullscr=True)

# Preload all images
preloaded_city_images = [visual.ImageStim(win, image=img, size=(0.7, 0.7)) for img in city_images]
preloaded_mountain_images = [visual.ImageStim(win, image=img, size=(0.7, 0.7)) for img in mountain_images]

# Set up marker stream
info = StreamInfo("Markers", "Markers", 1, 0, "int32", "myuidw43536")
outlet = StreamOutlet(info)
markernames = [1, 2]

# EEG device
board_name = "muse2"
mac_addr = "00:55:DA:B5:AB:4C"
eeg = EEG(device=board_name, mac_addr=mac_addr)

# Create save file name
subject_id = 10  # or any identifier for the subject
session_nb = 1  # session number
save_fn = eeg_save_fn(board_name, "GradCPT", subject_id, session_nb)

# Start device
eeg.start(save_fn, EEG_DURATION)

def show_intructions():
    instruction_clock = core.Clock()
    instruction_text = '\nWelcome to the GradCPT experiment!\nStay still, focus on the centre of the screen, and try not to blink. \n The experiment will start in 20 seconds'
    win.setMouseVisible(False)
    text = visual.TextStim(win=win, text=instruction_text, color=[-1, -1, -1])
    text.draw()
    win.flip()

    while instruction_clock.getTime() < 20:
        core.wait(0.05)
    

def transitionImages(from_img: visual.ImageStim, to_img:visual.ImageStim) -> List[float]:
    transition_clock = core.Clock()
    DISPLAY_TIME_WITH_BUFFER = IMAGE_DISPLAY_TIME - 0.05
    
    while transition_clock.getTime() < DISPLAY_TIME_WITH_BUFFER:
        weight = transition_clock.getTime() / DISPLAY_TIME_WITH_BUFFER
        from_img.opacity = 1 - weight
        to_img.opacity = weight
        from_img.draw()
        to_img.draw()
        win.flip()
    
    elapsed_time = transition_clock.getTime()
    remaining_time = IMAGE_DISPLAY_TIME - elapsed_time
    if remaining_time > 0:
        core.wait(remaining_time)
    
    presses = event.getKeys(keyList=['space'], timeStamped=transition_clock)
    timestamps_ms = [timestamp * 1000 for (_, timestamp) in presses]
    
    return timestamps_ms

    

def get_image(last_image:visual.ImageStim = None) -> Tuple[visual.ImageStim, bool]:
    """Returns an image depending on the last image shown."""
    is_mountain = False
    # Initial decision whether to pick from city_images or mountain_images
    if random.random() < TARGET_FREQ or (last_image and last_image in preloaded_mountain_images):
        pool = preloaded_city_images
    else:
        pool = preloaded_mountain_images
        is_mountain = True

    # Now select an image, but ensure it's not the same as last_image
    choice = random.choice(pool)
    while choice == last_image:
        choice = random.choice(pool)

    return choice, is_mountain

def record_responses() -> List[Trial]:
    trials = []
    
    last_image, _ = get_image()
    for _ in range(TRIAL_N):
        trial_clock = core.Clock()
        start_time = time()
        next_image, is_mountain = get_image(last_image)

        eeg.push_sample([1 if is_mountain else 0], start_time)
        responses = transitionImages(last_image, next_image)
        trials.append({'is_mountain': is_mountain, 'responses': responses, 'start_timestamp': start_time})
        last_image = next_image
        
        print(f"Trial duration: {trial_clock.getTime()} s")
    
    return trials

def process_responses(trials: List[Trial]) -> Tuple[List[Optional[float]], List[float], List[bool]]:
    """Calculate response times (RTs) from the trial data."""
    response_times = [float('inf')] * TRIAL_N

    # Edge case: first trial
    if trials[0]['responses']:
        response_times[0] = trials[0]['responses'][0]

    # Loop 0: unamibiguous correct responses
    for i, trial in enumerate(trials[1:], start=1):
        remaining_responses = []
        for rt in trial['responses']:
            if rt < 320 and not trials[i-1]['is_mountain']:
                response_times[i-1] = min(800 + rt, response_times[i-1])
            elif rt > 560 and not trial['is_mountain']:
                response_times[i] = min(rt, response_times[i])
            else:
                remaining_responses.append(rt)
        trial['responses'] = remaining_responses


    # Loop 1: ambigous presses
    for i, trial in enumerate(trials[1:], start=1):
        for rt in trial['responses']:
            if response_times[i-1] == float('inf') and response_times[i] != float('inf'):
                response_times[i-1] = 800 + rt
            elif response_times[i-1] != float('inf') and response_times[i] == float('inf'):
                response_times[i] = rt
            elif response_times[i-1] == float('inf') and response_times[i] == float('inf'):
                if trials[i-1]['is_mountain']:
                    response_times[i] = rt
                elif trial['is_mountain']:
                    response_times[i-1] = 800 + rt
                else:
                    if rt < 400:
                        response_times[i-1] = 800 + rt
                    else:
                        response_times[i] = rt

    # Replace inf with None
    processed = [None if x == float('inf') else x for x in response_times]
    start_timestamps = [trial['start_timestamp'] for trial in trials]
    is_mountains = [trial['is_mountain'] for trial in trials]
    return processed, start_timestamps, is_mountains

def label(response_times: List[Optional[float]], start_timestamps: List[float], is_mountains: List[bool]) -> List[Tuple[float, int, bool]]:
    """Label responses w.r.t RTV aka the trial to trial variation in response time"""
    response_times = np.array(response_times, dtype=float)

    # Remove mountain trials
    mask = np.array(is_mountains, dtype=bool)
    response_times[mask] = np.nan

    # Z-tranform the sequence
    z_normalized_rt = (response_times - np.nanmean(response_times)) / np.nanstd(response_times)

    # Calculate variance time course
    vtc = np.abs(z_normalized_rt - np.nanmean(z_normalized_rt))

    # Linearly interpolate missing values in the vtc
    nans, x = np.isnan(vtc), lambda z: z.nonzero()[0]
    vtc[nans] = np.interp(x(nans), x(~nans), vtc[~nans])

    # Smooth the VTC
    vtc_smoothed = gaussian_filter1d(vtc, sigma=SIGMA)  # sigma derived from FWHM

    # Determine "in the zone" (1) and "out of the zone" (0) labels
    median_vtc = np.median(vtc_smoothed)
    zone_labels = [(start_timestamps[i], 1, is_mountains[i]) if value <= median_vtc else (start_timestamps[i], 0, is_mountains[i]) for i, value in enumerate(vtc_smoothed)]

    return zone_labels



if __name__ == "__main__":
    show_intructions()
    raw_responses = record_responses()
    save_raw_responses(board_name, subject_id, session_nb, raw_responses)
    responses, start_timestamps, is_mountains = process_responses(raw_responses)
    labels = label(responses, start_timestamps, is_mountains)
    save_final_results(board_name, subject_id, session_nb, labels)
    
# Close the window
eeg.stop()
win.close()
core.quit()
