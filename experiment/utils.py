from os import path, makedirs
from time import strftime, gmtime
from typing import TypedDict, List, Tuple
from pathlib import Path
import json
import csv


DATA_DIR = path.join(path.expanduser("~/"), "eeg", "data")

# Types
class Trial(TypedDict):
    """List entry for the `trials` List."""
    is_mountain: bool
    responses: List[float] 
    start_timestamp: float

def get_recording_dir(
    board_name: str,
    experiment: str,
    subject_id: int,
    session_nb: int,
    site="local",
    data_dir=DATA_DIR,
) -> Path:
    # convert subject ID to 4-digit number
    subject_str = f"subject{subject_id:04}"
    session_str = f"session{session_nb:03}"
    return _get_recording_dir(
        board_name, experiment, subject_str, session_str, site, data_dir=data_dir
    )

def _get_recording_dir(
    board_name: str,
    experiment: str,
    subject_str: str,
    session_str: str,
    site: str,
    data_dir=DATA_DIR,
) -> Path:
    """A subroutine of get_recording_dir that accepts subject and session as strings"""
    # folder structure is /DATA_DIR/experiment/site/subject/session/*.csv
    recording_dir = (
        Path(data_dir) / experiment / site / board_name / subject_str / session_str
    )

    # check if directory exists, if not, make the directory
    if not path.exists(recording_dir):
        makedirs(recording_dir)

    return recording_dir

def eeg_save_fn(
    board_name: str,
    experiment: str,
    subject_id: int,
    session_nb: int,
    data_dir=DATA_DIR,
) -> Path:
    """Generates a file name with the proper trial number for the current subject/experiment combo"""
    recording_dir = get_recording_dir(
        board_name, experiment, subject_id, session_nb, data_dir=DATA_DIR
    )

    # generate filename based on recording date-and-timestamp and then append to recording_dir
    return recording_dir / (
        "recording_%s" % strftime("%Y-%m-%d-%H.%M.%S", gmtime()) + ".csv"
    )

def save_final_results(board_name: str, subject_id: int, session_nb:int, labels: List[Tuple[float, int, bool]]):
    headers = ("start_timestamp", 'in_the_zone', 'is_mountain')

    # Define the filename
    subject_str = f"subject{subject_id:04}"
    session_str = f"session{session_nb:03}"
    filename = f"/home/henri/eeg/data/GradCPT/local/{board_name}/{subject_str}/{session_str}/gradcpt/gradcpt_%s" % strftime("%Y-%m-%d-%H.%M.%S", gmtime()) + ".csv"

    directory = path.dirname(filename)
    makedirs(directory, exist_ok=True)

    # Open the file in write mode and write the data
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([headers] + labels)
    
def save_raw_responses(board_name: str, subject_id: int, session_nb:int, responses: List[Trial]):
    trials_json = json.dumps(responses, indent=4)

    # Define the filename
    subject_str = f"subject{subject_id:04}"
    session_str = f"session{session_nb:03}"
    filename = f"/home/henri/eeg/data/GradCPT/local/{board_name}/{subject_str}/{session_str}/gradcpt/raw_%s" % strftime("%Y-%m-%d-%H.%M.%S", gmtime()) + ".json"

    directory = path.dirname(filename)
    makedirs(directory, exist_ok=True)

    # Open the file in write mode and write the data
    with open(filename, "w") as file:
        file.write(trials_json)