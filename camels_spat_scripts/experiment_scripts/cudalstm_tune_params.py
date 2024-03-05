## Import necessary libraries
import os
import sys
import yaml

# Get the current working directory of the notebook
current_dir = os.getcwd()
# Add the parent directory of the notebook to the Python path
root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(root_dir)

# sys.path.append('..')
from camels_spat_scripts.utils_camels_spat import (
    pick_random_config, 
    generate_config_file,
    train_model,
    test_model,
)

## Functions
def main():
    
    # Load general parameters
    with open('cudalstm_params_general.yml', 'r') as ymlfile:
        general_config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    # From basin_id_file, add train_basin_file, validation_basin_file, and test_basin_file.
    basin_id_file = general_config['basin_id_file']
    general_config['train_basin_file'] = basin_id_file
    general_config['validation_basin_file'] = basin_id_file
    general_config['test_basin_file'] = basin_id_file
    general_config.pop('basin_id_file')
    
    # Load the parameters to be tuned as a dictionary
    with open('cudalstm_params_options.yml', 'r') as ymlfile:
        cudalstm_params = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
    while True:
        random_config = pick_random_config(cudalstm_params)
        if random_config is not None:
            
            # Create the configuration file with general parameters + the randomly selected parameters
            run_config = {**general_config, **random_config}
            config_fname = generate_config_file(run_config)
            
            ## Train the model
            train_model(config_fname)   
                
            ## Test the model
            test_model(run_config)

            
        else:
            print('All configurations have been explored')
            break
    


if __name__ == '__main__':
    
    main()