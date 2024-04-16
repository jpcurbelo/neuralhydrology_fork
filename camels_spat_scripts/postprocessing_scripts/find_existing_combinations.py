import os
import glob
import yaml

# Constants
RUN_DIR = '../experiment_scripts/runs'
OUTPUT_FILE = 'existing_param_combinations.txt'

params2look_at = [
    'hidden_size',
    'seq_length',
    'learning_rate',
    'output_dropout',
    'batch_size',
    'loss'
]

# Function to concatenate parameters
def concatenate_parameters(config, params2look_at):
    parameter_values = []
    for param in params2look_at:
        value = config.get(param)
        if param == 'hidden_size':
            if isinstance(value, list):
                value = '[' + '-'.join(str(v) for v in value) + ']'
            else:
                value = '[' + str(value) + ']'
        parameter_values.append(str(value))
    return '-'.join(parameter_values)

# Function to create the parameter combinations file
def create_param_combinations_file():
    parameters_list = []  # List to store parameters
    
    if not os.path.exists(RUN_DIR) or not os.path.isdir(RUN_DIR):
        print(f"{RUN_DIR} does not exists")
        return

    # List of folders in the run directory
    run_folders = sorted(os.listdir(RUN_DIR))

    # Loop through the run folders
    for run_folder in run_folders:

        # Path to the test results file
        inner_folder_path = os.path.join(RUN_DIR, run_folder, 'config.yml')
        inner_folder = glob.glob(inner_folder_path)
        
        # Check if the file exists
        if not inner_folder:
            print(f"No config.yml results file found for run folder {run_folder}.")
            continue

        # Load the config file
        with open(inner_folder[0], 'r') as file:
            config = yaml.safe_load(file)

        # Concatenate parameters and add them to the list
        parameters = concatenate_parameters(config, params2look_at)
        parameters_list.append(parameters)
        
    # print('parameters_list:', parameters_list)
    # aux = input('Press Enter to continue...')

    # Write parameters list to a file
    with open(OUTPUT_FILE, 'w') as output_file:
        for parameters in parameters_list:
            print('Writing parameters:', parameters)
            output_file.write(parameters + '\n')

##################################################
if __name__ == "__main__":
    create_param_combinations_file()