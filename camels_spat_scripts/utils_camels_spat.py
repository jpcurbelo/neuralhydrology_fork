# Description: Utility functions for the CAMELS-CL dataset

## Import necessary libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import yaml
from pathlib import Path
import torch
import random
import subprocess

from neuralhydrology.nh_run import (
    start_run,
    eval_run,
)

# Global variables
generated_configs = set()

## Functions
def validate_basin_data(basin, data_dir, start_date, end_date):
    '''
    Function to validate the data for a given basin - if the data is present for the specified date range
    Args:
    - basin: str, name of the basin
    - data_dir: str, path to the directory containing the basin data
    - start_date: str, start date in the format 'YYYY-MM-DD'
    - end_date: str, end date in the format 'YYYY-MM-DD'
    Returns:
    tuple: A tuple containing basin identifier, processing status, and presence of dates.
        - basin: str, name of the basin
        - valid: bool, True if the data is valid, False otherwise
        - date_presence: pd.Series, series of 1s and 0s indicating the presence of data for each date in the specified range
    '''
    
    df = pd.read_csv(os.path.join(data_dir, basin), index_col=0, parse_dates=True)  

    # print("Basin:", basin)
    # print("Dates in period:", df.index[0] <= pd.to_datetime(start_date) and df.index[-1] >= pd.to_datetime(end_date))
    # display(df.head())
    # aux = input("Press Enter to continue...")


    if df.index[0] <= pd.to_datetime(start_date) and df.index[-1] >= pd.to_datetime(end_date):
        
        # Create a Series of dates between the start and end date: 1 if date is present, 0 if missing
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        # Filter df.index to ensure it's within the specified range
        filtered_index = df.index[(df.index >= start_date) & (df.index <= end_date)]
        # Create present_dates series with index filtered to within the range
        present_dates = pd.Series(1, index=filtered_index)
        # Create missing_dates series with index filtered to within the range
        missing_dates = pd.Series(0, index=date_range[~date_range.isin(filtered_index)])
        # Concatenate present_dates and missing_dates and sort by index
        date_presence = pd.concat([present_dates, missing_dates]).sort_index()
         
        # Check for missing days
        if df.index.freq != 'D':
            # Upsample to daily frequency
            df_daily = df.resample('D').mean()
            
            # Check for NaN values
            missing_days = df_daily.isnull().sum()
            if missing_days.any() > 0:  # Use any() method to check for any True values
                return basin, False, date_presence
        return basin, True, date_presence
    else:
        return basin, False, None

def plot_missing_data_heatmap(df, ds, start_year, end_year, countries_str):
    
    fig, ax = plt.subplots(figsize=(30, 16))
    # Create a heatmap of the missing data with seaborn
    cmap = sns.mpl_palette("Set2", 2)
    sns.heatmap(df, cmap=cmap, cbar=False, ax=ax)
    
    legend_handles = [Patch(color=cmap[True], label='Non Missing Value'),  # red
                    Patch(color=cmap[False], label='Missing Value')]  # green
    plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=20, handlelength=1.2)
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig(f"missing_data_{ds[1]}_{start_year}-{end_year}_{countries_str}.png", dpi=150)

def calculate_total_configs(config_dict):
    """
    Calculate the total number of possible configurations given a dictionary of parameters.
    
    Args:
    - config_dict (dict): A dictionary containing parameter names as keys and lists of possible values as values.
    
    Returns:
    int: The total number of possible configurations.
    """
    total_configs = 1
    for values in config_dict.values():
        total_configs *= len(values)
    return total_configs

def pick_random_config(config_dict):
    '''
    Function to pick a random configuration from a dictionary of parameters
    
    Args:
    - config_dict (dict): A dictionary containing the parameters to be tuned
    
    Returns:
    dict: A dictionary containing the randomly selected parameters
    '''
    global generated_configs
    total_configs = calculate_total_configs(config_dict)
    while len(generated_configs) < total_configs:
        random_config = {key: random.choice(values) for key, values in config_dict.items()}
        config_tuple = tuple(sorted(random_config.items()))
        if config_tuple not in generated_configs:
            generated_configs.add(config_tuple)
            
            # print(len(generated_configs), "configurations generated")
            
            return random_config
    return None  # All configurations have been explored

def generate_config_file(params_dict):
    
    # # Create a ymal file with the parameters to be tuned
    # set = 1
    # fname = f'cudalstm_model2tune{set}.yml'
    # # If the file already exists, increment the set number
    # while os.path.exists(fname):
    #     set += 1
    #     fname = f'cudalstm_params2tune{set}.yml'
    
    fname = f'cudalstm_model2tune.yml'
    with open(fname, 'w') as ymlfile:
        yaml.dump(params_dict, ymlfile, default_flow_style=False)
    
    return fname

def train_model(config_fname):
    """
    Train the model using CUDA if available, otherwise fallback to CPU-only mode.

    Args:
    - config_fname (str): Path to the configuration file.

    Returns:
    None
    """
    if torch.cuda.is_available():
        start_run(config_file=Path(config_fname))
    else:
        start_run(config_file=Path(config_fname), gpu=-1)

def test_model(run_config):
    """
    Test the model using the latest run file from the 'runs' directory.

    Args:
    - run_config (dict): Configuration for the model run.

    Returns:
    None
    """
    run_files = sorted(os.listdir("runs"))
    exp_name = run_config["experiment_name"]
    filtered_run_files = sorted([filename for filename in run_files if exp_name in filename])
    if filtered_run_files:
        run_file = filtered_run_files[-1]
        run_dir = Path(f"runs/{run_file}")
        eval_run(run_dir=run_dir, period="test")

def find_options_file(batch):
    '''
    Function to find the options file for a given batch number
    
    Args:
    - batch_number (int): The batch number
    Returns:
    str: The path to the options file
    '''
    
    options_file = f'cudalstm_params_options_{batch}.yml'
    if os.path.exists(options_file):
        return options_file
    else:
        raise FileNotFoundError(f"Options file for {batch} does not exist.")



if __name__ == '__main__':
    pass