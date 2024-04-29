# Measure attention with EEG
## Goal
Implement the GradCPT psychological test as a means to label attention states. Utilize the Muse 2 device to record EEG data from subjects performing the gradCPT task. Preprocess the data and extract relevant features. Develop a predictive model for attention states based on the data.

## How to run experiment
### Download Conda
[Miniconda download](https://docs.anaconda.com/free/miniconda/)

### Create and activate env
```bash
cd experiment

conda env create -n eeg-env

conda activate eeg-env
```

### Run the experiment
`python experiment.py`
