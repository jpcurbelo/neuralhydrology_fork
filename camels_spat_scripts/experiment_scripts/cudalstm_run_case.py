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
    # 'batch3_426_camels_spat_USA_2024-04-16_111303.yml',
    # 'batch2_426_camels_spat_USA_2024-04-24_024728.yml',
    # 'batch2_426_camels_spat_USA_2024-04-24_024728.yml',
    "batch2_426_camels_spat_USA_2024-04-24_104522.yml"
]


if __name__ == '__main__':
    
    ## export LD_LIBRARY_PATH=/home/ame805/neuralhydrology_fork/venv-nh/lib/python3.8/site-packages/nvidia/cudnn/lib
    
<<<<<<< HEAD
    # config_fname = "cudalstm_model2tune.yml"
    # config_fname = "426_basin_camels_spat.yml"
    # config_fname = "10_basin_camels_spat.yml"
    # config_fname = "1_basin_camels_spat.yml"
    # config_fname = "465_basin_camels_spat.yml"
    # config_fname = "basin_hcu_01_camels_spat.yml" 
    
    # config_fname = "416_basin_camels_spat.yml"
    
    config_fname = 'best_jobs_from_cedar'
    
    # Check if config_name is a folder
    # Check if config_name is a folder
    if os.path.isdir(config_fname):
        print("Config name is a folder") 
        # All files in the folder .yml
        config_fname_list = sorted([os.path.join(config_fname, f) for f in os.listdir(config_fname) if f.endswith('.yml')], 
                                   reverse=True)
        
    elif os.path.isfile(config_fname) and config_fname.endswith('.yml'):
        print("Config name is a file")
        config_fname_list = [config_fname]
    else:
        print("Config name is neither a file nor a folder")
        sys.exit(1)
        
    for config_fname in config_fname_list:
        
        # if config_fname.split('/')[-1] not in only_run_these:
        #     continue
        
        print("Config file name:", config_fname)
        with open(config_fname, "r") as f:
            run_config = yaml.safe_load(f)
        
        ## Train the model
        train_model(config_fname)   
        
        ## Test the model
        test_model(run_config)
        
    
    # with open(config_fname, "r") as f:
    #     run_config = yaml.safe_load(f)
=======
    config_fname = "cudalstm_model2tune.yml"
    # config_fname = "426_basin_camels_spat.yml"
    
    with open(config_fname, "r") as f:
        run_config = yaml.safe_load(f)
>>>>>>> 72ba5c5a77837d309ef3436192fb6583049cbdcc
        
    # ## Train the model
    # train_model(config_fname)   
        
    # ## Test the model
    # test_model(run_config)
