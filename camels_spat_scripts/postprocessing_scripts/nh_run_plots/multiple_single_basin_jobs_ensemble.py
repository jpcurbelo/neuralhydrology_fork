import os
import shutil
import pandas as pd


FOLDER_DIR = '../../experiment_scripts'

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

OUTPUT_DIR = 'runs_569_basins_lstm_single_ensemble_8seeds'

def main():

    # Load basin file
    basin_file = os.path.join(FOLDER_DIR, BASIN_FILE)
    with open(basin_file, 'r') as f:
        basins = f.readlines()
        basins = [basin.strip() for basin in basins]

    output_file = os.path.join(OUTPUT_DIR, 'test_all-single_ensemble_8seeds.csv')

    # Check if the output file already exists
    if os.path.exists(output_file):
        # Load the existing file
        df_metrics = pd.read_csv(output_file)
        # List of basins already processed
        basins_processed = df_metrics['basin'].unique()
    else:
        df_metrics = pd.DataFrame()
        basins_processed = []


    for basin in basins:

        if basin in basins_processed:
            print(f"Basin '{basin}' already processed. Skipping...")
            continue

        # Create a temporary folder for the job (only if it doesn't already exist)
        temp_folder = f'ensembleSingle_{basin}'

        # Remove the folder if it already exists
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)
        
        # Create a new temporary folder
        os.makedirs(temp_folder)

        for job in JOBS_LIST:
            # Construct the path to the job folder
            job_folder = os.path.join(FOLDER_DIR, job)
            
            # List the contents of the job folder
            job_list = os.listdir(job_folder)
            
            # Find the folder that starts with the basin name
            job_basin = next((item for item in job_list if item.startswith(basin)), None)
            
            if job_basin:
                # Perform further operations with job_basin here
                pass
            else:
                print(f"Warning: No folder found for basin '{basin}' in job '{job}'")


            # Copy the job folder to the temporary folder
            job_folder_basin = os.path.join(FOLDER_DIR, job, job_basin)

            temp_job_folder = os.path.join(temp_folder, job)
            os.system(f'cp -r {job_folder_basin} {temp_job_folder}')

        print("Ensemble script will be executed for basin:", basin)

        # Execute the ensemble script: 
        os.system(f'nh-results-ensemble --run-dirs {temp_folder}/* --output-dir {temp_folder}')

        # Load the metrics file
        metrics_file = os.path.join(temp_folder, 'test_ensemble_metrics.csv')
        df = pd.read_csv(metrics_file)

        # Add the basin name to the dataframe
        df_metrics = pd.concat([df_metrics, df], ignore_index=True)      

        # Remove the temporary folder and its contents
        shutil.rmtree(temp_folder)

        # Save the metrics dataframe to a CSV file
        # Create the output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        df_metrics.to_csv(output_file, index=False)



if __name__ == '__main__':

    main()