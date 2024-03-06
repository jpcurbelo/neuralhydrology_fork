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
    
    # cudnn_version = "8.9.2.26"  # Replace A.B.C with the specific version number needed
    # if setup_cudnn(cudnn_version):
    
    config_fname = "cudalstm_model2tune.yml"
    # config_fname = "426_basin_camels_spat.yml"
    
    with open(config_fname, "r") as f:
        run_config = yaml.safe_load(f)
        
    ## Train the model
    train_model(config_fname)   
        
    ## Test the model
    test_model(run_config)
