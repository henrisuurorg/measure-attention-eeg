def generate_save_fn(
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