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
    train_model,
    test_model,
)

only_run_these = [
    "416_batch2c_best.yml"
]

seed_list = [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
# seed_list = [7777, 8888]


if __name__ == '__main__':
    
    ## export LD_LIBRARY_PATH=/home/ame805/neuralhydrology_fork/venv-nh/lib/python3.8/site-packages/nvidia/cudnn/lib

    # config_fname = "416_batch2c_best.yml"
    # config_fname = "416_batch2c_best_5inp.yml"
    # config_fname = "426_batch2c_best.yml"
    config_fname = "426_batch2c_best_5inp.yml"
        
    print("Config file name:", config_fname)
    with open(config_fname, "r") as f:
        run_config = yaml.safe_load(f)
        
    # experiment_name = run_config['experiment_name']
    # experiment_name = "426_b2c"
    # experiment_name = "416_b2c_5inp"
    experiment_name = "426_b2c_5inp"
        
    for seed in seed_list:
        
        run_config['seed'] = seed
        run_config['experiment_name'] = experiment_name + f"_{seed}"
        
        # Re-write the configuration file with the new seed
        with open(config_fname, 'w') as ymlfile:
            yaml.dump(run_config, ymlfile, default_flow_style=False)
        
        
        ## Train the model
        train_model(config_fname)   
    
        ## Test the model
        test_model(run_config)