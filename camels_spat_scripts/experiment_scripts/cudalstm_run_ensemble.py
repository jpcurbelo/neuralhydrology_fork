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

# seed_list = [111, 222, 333, 444, 555, 666, 777, 888]
# seed_list = [111, 222]
# seed_list = [333, 444]
# seed_list = [555, 666]
# seed_list = [666, 777, 888]
# seed_list = [777, 888]
# seed_list = [777]
seed_list = [888]


if __name__ == '__main__':
    
    ## export LD_LIBRARY_PATH=/home/ame805/neuralhydrology_fork/venv-nh/lib/python3.8/site-packages/nvidia/cudnn/lib

    # config_fname = "416_batch2c_best.yml"
    # config_fname = "416_batch2c_best_5inp.yml"
    # config_fname = "426_batch2c_best.yml"
    # config_fname = "426_batch2c_best_5inp.yml"
    # config_fname = '531_nh_paper.yml'
    # config_fname = '505_nh_paper_camelsspat.yml'
    # config_fname = '505_nh_paper_camelsspat_17inp.yml'
    # config_fname = '505_nh_paper_camelsspat_14inp.yml'
    # config_fname = '505_nh_paper_camelsspat_emearth.yml'
    # config_fname = '531_nh_paper_ealstm.yml'
    # config_fname = "505_nh_paper_camelsspat_ealstm15static.yml"
    config_fname = "505_nh_paper_camelsspat_ealstm15static17inp.yml"
        
    print("Config file name:", config_fname)
    with open(config_fname, "r") as f:
        run_config = yaml.safe_load(f)
        
    # experiment_name = run_config['experiment_name']
    # experiment_name = "426_b2c"
    # experiment_name = "416_b2c_5inp"
    # experiment_name = "426_b2c_5inp"
    # experiment_name = "426_b2c_16inp"
    # experiment_name = "416_b2c_16inp"
    # experiment_name = "531_nh_paper"
    # experiment_name = "505_nh_paper_camelsspat"
    # experiment_name = "505_nh_paper_camelsspat_17inp"
    # experiment_name = "505_nh_paper_camelsspat_14inp"
    # experiment_name = "505_nh_paper_camelsspat_emearth"
    # experiment_name = "531_nh_paper_ealstm"
    # experiment_name = "505_nh_paper_camelsspat_ealstm15static"
    experiment_name = "505_nh_paper_camelsspat_ealstm15static17inp"
        
    for seed in seed_list:
        
        run_config['seed'] = seed
        run_config['experiment_name'] = experiment_name + f"_{seed}"

        run_config['device'] = "cuda:1"
        
        # Re-write the configuration file with the new seed
        with open(config_fname, 'w') as ymlfile:
            yaml.dump(run_config, ymlfile, default_flow_style=False)
        
        ## Train the model
        train_model(config_fname)   
    
        ## Test the model
        test_model(run_config)
