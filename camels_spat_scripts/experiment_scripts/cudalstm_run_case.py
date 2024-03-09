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

if __name__ == '__main__':
    
    ## export LD_LIBRARY_PATH=/home/ame805/neuralhydrology_fork/venv-nh/lib/python3.8/site-packages/nvidia/cudnn/lib
    
<<<<<<< HEAD
    config_fname = "cudalstm_model2tune.yml"
    # config_fname = "426_basin_camels_spat.yml"
=======
    # config_fname = "cudalstm_model2tune.yml"
    config_fname = "426_basin_camels_spat.yml"
    # config_fname = "10_basin_camels_spat.yml"
>>>>>>> 4f8b191... Added postprocessing - find best models and plot basins with neg NSE
    
    with open(config_fname, "r") as f:
        run_config = yaml.safe_load(f)
        
    ## Train the model
    train_model(config_fname)   
        
    ## Test the model
    test_model(run_config)
