from pathlib import Path
import os
import sys
import shutil

# Get the current working directory of the script
current_dir = Path(__file__).resolve().parent
# Add the parent directory of the notebook to the Python path
root_dir = Path(current_dir).resolve().parents[2]
sys.path.append(root_dir)


# JOBS_FOLDER = 'runs_531_nh_paper_1900'
# JOBS_FOLDER = 'runs_b2c_416_ensemble_long_17inp'

# JOBS_FOLDER = 'runs_531_nh_paper'
# JOBS_FOLDER = 'runs_505_nh_paper_camelsspat'
# JOBS_FOLDER = 'runs_505_nh_paper_camelsspat_17inp'
# JOBS_FOLDER = 'runs_505_nh_paper_camelsspat_emearth'
# JOBS_FOLDER = 'runs_531_nh_paper_ealstm'
# JOBS_FOLDER = 'runs_505_nh_paper_camelsspat_14inp'
JOBS_FOLDER = 'runs_505_nh_paper_camelsspat_ealstm15static'

EPOCH = 30

# EPOCH to 3 places string with leading zeros
epoch_str = f"{EPOCH:03d}"

def main():
    # Get the path to the jobs folder in the script's directory
    jobs_path = root_dir / 'camels_spat_scripts' / 'experiment_scripts' / JOBS_FOLDER
    
    # Create the jobs_folder_new folder in the current script directory
    jobs_folder_new = current_dir / JOBS_FOLDER
    jobs_folder_new.mkdir(parents=True, exist_ok=True)
    
    # Copy 'test_ensemble_metrics.csv' to jobs_folder_new
    src_file = jobs_path / 'test_ensemble_metrics.csv'
    dest_file = jobs_folder_new / 'test_ensemble_metrics.csv'
    if src_file.exists():
        shutil.copy(src_file, dest_file)
    else:
        print(f"File {src_file} does not exist.")
    
    # Get list of subdirectories and process them
    for item in os.listdir(jobs_path):
        item_path = jobs_path / item
        if item_path.is_dir():  # and item.startswith('531_nh_paper'):
            # Extract the seed from the directory name
            item_split = item.split('_')
            if len(item_split) == 1:
                seed = item_split[0]
            else:
                seed = item.split('_')[-3]
            # Define the source file path for the test metrics
            if len(item_split) == 1:
                metrics_file = item_path / 'test' / f'model_epoch{epoch_str}' / 'test_metrics.csv'
            else:
                metrics_file = item_path / 'test' / f'model_epoch{epoch_str}' / f'test_metrics_seed{seed}.csv'
            if metrics_file.exists():
                # Define the destination file path in the new folder
                dest_seed_file = jobs_folder_new / f'{seed}.csv'
                # Copy the metrics file to the new folder with the seed name
                shutil.copy(metrics_file, dest_seed_file)
            else:
                print(f"File {metrics_file} does not exist.")
        elif item_path.is_file() and item == 'test_ensemble_metrics.csv':
            continue
        else:
            print(f"Skipping unexpected item {item_path}")
    
    print(f"Folder {jobs_folder_new} created and files processed.")

if __name__ == '__main__':
    main()