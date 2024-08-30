## Import necessary libraries
import os
import sys
import yaml
from datetime import datetime

# Get the current working directory of the notebook
current_dir = os.getcwd()
# Add the parent directory of the notebook to the Python path
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)

from camels_spat_scripts.utils_camels_spat import (
    train_model,
    test_model,
)

ALL_BASINS_FILE = "569_basin_file.txt"
CONFIG_FNAME = "569_basins_camels_lstm_single.yml"

DEVICE = "cuda:0"

if __name__ == '__main__':

    print("Config file name:", CONFIG_FNAME)
    with open(CONFIG_FNAME, "r") as f:
        run_config = yaml.safe_load(f)

    # Read list of all basins
    with open(ALL_BASINS_FILE, "r") as f:
        all_basins = f.read().splitlines()

    if 'run_folder_name' in run_config:
        run_folder = run_config["run_folder_name"]
        if 'seed' in run_config:
            seed = run_config["seed"]
            # If _{seed} already exists, remove it
            if f"_{seed}" in run_folder:
                run_folder = run_folder.replace(f"_{seed}", "")
            run_config["run_folder_name"] = run_folder + f"_{seed}"
        else:
            run_config["run_folder_name"] = run_folder + "_random-seed"

    # Loop through each basin
    for basin in all_basins[:]:

        print(f"Training and testing model for basin {basin}")

        basin_file = f"{basin}_file.txt"
        # Create file 1_basin_file.txt with only the current basin
        with open(basin_file, "w") as f:
            f.write(basin)

        run_config["experiment_name"] = f"{basin}_lstm"
        run_config['device'] = DEVICE
        if 'train_basin_file' in run_config:
            run_config['train_basin_file'] = basin_file
        if 'test_basin_file' in run_config:
            run_config['test_basin_file'] = basin_file
        if 'validation_basin_file' in run_config:
            run_config['validation_basin_file'] = basin_file

        # Create a temporary configuration file with the current basin
        basin_config_file = f"{basin}_basin_config.yml"
        with open(basin_config_file, "w") as f:
            yaml.dump(run_config, f)

        # Re-write the configuration file with the new experiment name
        with open(basin_config_file, "w") as f:
            yaml.dump(run_config, f)

        ## Train the model
        train_model(basin_config_file)   
    
        ## Test the model
        test_model(run_config)

        # Detete the temporary configuration file and the basin file
        os.remove(basin_config_file)
        os.remove(basin_file)

   