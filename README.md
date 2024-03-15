# Measure attetion with EEG
## Goal
Implement the GradCPT psychological test as a means to label attention states. Utilize the Muse 2 device to record EEG data from subjects performing the gradCPT task. Preprocess the data and extract relevant features. Develop a predictive model for attention states based on the data.

## Setup
### Download Conda if you don't have it 
[Miniconda download](https://docs.anaconda.com/free/miniconda/)

### Create and activate env
`conda env create -n eeg-env` and then
`conda activate eeg-env`

### Run the experiment
`python experiment.py`

### If needed, update env
`conda env update`
