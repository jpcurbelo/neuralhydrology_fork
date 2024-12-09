import os
import shutil
import pandas as pd


ROOT_DIR = '../../experiment_scripts'
FOLDER_DIR = ROOT_DIR + '/Aruns_569_hoge22_lstm_single'

BASIN_FILE = '569_basin_file.txt'
JOBS_LIST = [
    'runs_569_basins_lstm_single_111',
    'runs_569_basins_lstm_single_222',
    'runs_569_basins_lstm_single_333',
    'runs_569_basins_lstm_single_444',
    'runs_569_basins_lstm_single_555',
    'runs_569_basins_lstm_single_666',
    'runs_569_basins_lstm_single_777',
    'runs_569_basins_lstm_single_888'
]

RANDOM_SEED_JOB = 'runs_569_basins_lstm_single_random-seed'
EPOCHS = 50

OUTPUT_DIR = 'Aruns_569_hoge22_lstm_single'

def main():

    # Ensure the output directory is fresh
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # Delete the existing directory
    os.makedirs(OUTPUT_DIR)  # Create a fresh directory

    # Load basin file
    basin_file = os.path.join(ROOT_DIR, BASIN_FILE)
    with open(basin_file, 'r') as f:
        basins = f.readlines()
        basins = [basin.strip() for basin in basins]

    output_file = os.path.join(OUTPUT_DIR, 'test_ensemble_metrics.csv')

    # Check if the output file already exists
    if os.path.exists(output_file):
        # Load the existing file
        df_metrics = pd.read_csv(output_file)
        # List of basins already processed
        basins_processed = df_metrics['basin'].unique()
    else:
        df_metrics = pd.DataFrame()
        basins_processed = []

    # Initialize per-seed dataframes
    per_seed_metrics = {job: pd.DataFrame() for job in JOBS_LIST}

    # Initialize a DataFrame for random seed metrics
    random_seed_metrics = pd.DataFrame()

    for basin in basins[:]:

        if basin in basins_processed:
            print(f"Basin '{basin}' already processed. Skipping...")
            continue

        # Create a temporary folder for the job (only if it doesn't already exist)
        temp_folder = f'ensembleSingle_{basin}'

        # Remove the folder if it already exists
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

        # Create a temporary folder for the job (only if it doesn't already exist)
        temp_folder = f'ensembleSingle_{basin}'

        # Remove the folder if it already exists
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        # Create a new temporary folder
        os.makedirs(temp_folder)

        for job in JOBS_LIST + [RANDOM_SEED_JOB]:
            # Construct the path to the job folder
            job_folder = os.path.join(FOLDER_DIR, job)
            
            # List the contents of the job folder
            job_list = os.listdir(job_folder)
            
            # Find the folder that starts with the basin name
            job_basin = next((item for item in job_list if item.startswith(basin)), None)
           
            if job_basin:
                # Copy the job folder to the temporary folder for ensemble jobs only
                if job != RANDOM_SEED_JOB:
                    job_folder_basin = os.path.join(job_folder, job_basin)
                    temp_job_folder = os.path.join(temp_folder, job)
                    shutil.copytree(job_folder_basin, temp_job_folder)
                else:
                    # Handle random seed: Collect metrics for the basin
                    # Find the metrics file for the random seed job: test_metrics*.csv pattern in 
                    # os.path.join(job_folder, job_basin, f'test/model_epoch0{EPOCHS}')
                    testdir = os.path.join(job_folder, job_basin, f'test/model_epoch0{EPOCHS}')
                    random_metrics_file = next((os.path.join(testdir, item) for item in os.listdir(testdir) \
                                                if item.startswith('test_metrics')), None)  
                    if os.path.exists(random_metrics_file):
                        df_random = pd.read_csv(random_metrics_file)
                        df_random['basin'] = basin
                        random_seed_metrics = pd.concat([random_seed_metrics, df_random], ignore_index=True)
                    else:
                        print(f"Warning: Metrics file not found for basin '{basin}' in random seed job.")
            else:
                print(f"Warning: No folder found for basin '{basin}' in job '{job}'")

        # Perform ensemble only for regular jobs
        # Execute the ensemble script: 
        os.system(f'nh-results-ensemble --run-dirs {temp_folder}/* --output-dir {temp_folder}')

        # Load the metrics file
        metrics_file = os.path.join(temp_folder, 'test_ensemble_metrics.csv')
        df = pd.read_csv(metrics_file)

        # Add the basin name to the dataframe
        df['basin'] = basin
        df_metrics = pd.concat([df_metrics, df], ignore_index=True)   

        # Add to the per-seed dataframes
        for job in JOBS_LIST:
            # Find the metrics file for the random seed job: test_metrics*.csv pattern in 
            # os.path.join(temp_folder, job, f'test/model_epoch0{EPOCHS}')
            testdir = os.path.join(temp_folder, job, f'test/model_epoch0{EPOCHS}')
            seed_file = next((os.path.join(testdir, item) for item in os.listdir(testdir) \
                                        if item.startswith('test_metrics')), None)
            if os.path.exists(seed_file):
                df_seed = pd.read_csv(seed_file)
                df_seed['basin'] = basin
                per_seed_metrics[job] = pd.concat([per_seed_metrics[job], df_seed], ignore_index=True)

        # Remove the temporary folder and its contents
        shutil.rmtree(temp_folder)

        # Save per-seed metrics files
        df_metrics.to_csv(output_file, index=False)

    # Save per-seed metrics files
    for job, df_seed in per_seed_metrics.items():
        seed_output_file = os.path.join(OUTPUT_DIR, f"{job.split('_')[-1]}.csv")
        df_seed.to_csv(seed_output_file, index=False)
        print(f"Saved seed metrics for {job} to {seed_output_file}")

    # Save random seed metrics
    random_seed_output_file = os.path.join(OUTPUT_DIR, 'random_seed_metrics.csv')
    random_seed_metrics.to_csv(random_seed_output_file, index=False)
    print(f"Saved random seed metrics to {random_seed_output_file}")


if __name__ == '__main__':

    main()