## Import necessary libraries
import os
import sys
import yaml
import argparse

# Get the current working directory of the notebook
current_dir = os.getcwd()
# Add the parent directory of the notebook to the Python path
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)

# sys.path.append('..')
from camels_spat_scripts.utils_camels_spat import (
    find_options_file,
    pick_random_config, 
    generate_config_file,
    train_model,
    test_model,
    calculate_total_configs,
)

## Functions
def main(options_file='cudalstm_params_options.yml', batch=None):
    
    # Load general parameters
    with open('cudalstm_params_general.yml', 'r') as ymlfile:
        general_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    # From basin_id_file, add train_basin_file, validation_basin_file, and test_basin_file.
    basin_id_file = general_config['basin_id_file']
    general_config['train_basin_file'] = basin_id_file
    general_config['validation_basin_file'] = basin_id_file
    general_config['test_basin_file'] = basin_id_file
    # Remove basin_id_file
    general_config.pop('basin_id_file')
    # Update experiment_name if batch is not None
    if batch is not None:
        general_config['experiment_name'] = f"{batch}_{general_config['experiment_name']}"
    
    # Load the parameters to be tuned as a dictionary
    with open(options_file, 'r') as ymlfile:
        cudalstm_params = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    while True:
        random_config = pick_random_config(cudalstm_params)
        if random_config is not None:
            
            # Set hidden_size as an int or a list given n_stack_layers
            if random_config['n_stack_layers'] > 1:
                random_config['hidden_size'] = [random_config['number_units'] for _ in range(random_config['n_stack_layers'])]
            else:
                random_config['hidden_size'] = random_config['number_units']
            # Remove number_units and n_stack_layers
            random_config.pop('number_units')
            random_config.pop('n_stack_layers')
            
            # Create the configuration file with general parameters + the randomly selected parameters
            run_config = {**general_config, **random_config}
            config_fname = generate_config_file(run_config)
            
            ## Train the model
            train_model(config_fname)   
                
            ## Test the model
            test_model(run_config)

        else:
            total_configs = calculate_total_configs(cudalstm_params)
            print(f'All {total_configs} configurations have been explored')
            break
    
if __name__ == '__main__':
    
    ## export LD_LIBRARY_PATH=/home/ame805/neuralhydrology_fork/venv-nh/lib/python3.8/site-packages/nvidia/cudnn/lib
    
    parser = argparse.ArgumentParser(description='Process cudalstm_params_options file.')
    parser.add_argument('--options_batch', type=str, default=None,
                        help='Name of the cudalstm_params_options file without the batch number (e.g., "batchX")')
    args = parser.parse_args()

    if args.options_batch:
        try:
            options_file = find_options_file(args.options_batch)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
    else:
        options_file = 'cudalstm_params_options.yml'
    
    main(options_file, args.options_batch)