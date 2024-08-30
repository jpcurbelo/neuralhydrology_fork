import os
import pandas as pd
import glob
import re

# JOBS_FOLDER = '../../experiment_scripts/runs_569_basins_lstm_single_random-seed'
JOBS_FOLDER = '../../experiment_scripts/runs_569_basins_lstm_single_111'


def bundle_single_results():
    
    job_list = sorted(os.listdir(JOBS_FOLDER))
    print("job_list:", len(job_list))

    # Create dataframe
    df_metrics = pd.DataFrame()

    for job in job_list:

        # Extract basin
        basin = job.split('_')[0]
        
        # Look for test/model_epoch*/test_metrics*.csv file
        test_folder = os.path.join(JOBS_FOLDER, job, 'test')
        
        if os.path.exists(test_folder):
            # Find all model_epoch* folders
            model_epochs_folders = glob.glob(os.path.join(test_folder, 'model_epoch*'))
            
            if model_epochs_folders:
                # Extract the numeric part of the folder name using regex and sort by it
                # This assumes the numeric part is at the end of the folder name, e.g., model_epoch050
                def extract_number(folder_name):
                    match = re.search(r'epoch(\d+)', folder_name)
                    return int(match.group(1)) if match else -1  # -1 if no match found
                
                # Sort the folders by the extracted number in descending order
                model_epochs_folders.sort(key=lambda x: extract_number(x), reverse=True)
                
                # Take the folder with the highest suffix
                latest_model_epochs_folder = model_epochs_folders[0]
                
                # Now search for test_metrics*.csv in the latest model_epoch* folder
                csv_files = glob.glob(os.path.join(latest_model_epochs_folder, 'test_metrics*.csv'))
                
                # If there are any matching files, process them
                if csv_files:
                    for csv_file in csv_files:
                        # Here you can add the code to process each csv_file
                        # print(f"Processing file: {csv_file}")
                        print(f"Processing basin: {basin}")

                        # Read the csv file
                        df = pd.read_csv(csv_file)

                        # Ensure that the 'basin' column is of string type. 8 places with leading zeros
                        df['basin'] = basin.zfill(8)

                        # Add df data to the main dataframe
                        df_metrics = pd.concat([df_metrics, df], ignore_index=True)

                else:
                    print(f"No test_metrics*.csv files found in {latest_model_epochs_folder}")
            else:
                print(f"No model_epoch* folders found in {test_folder}")
        else:
            print(f"Folder does not exist: {test_folder}")

    return df_metrics



if __name__ == '__main__':

    # Process the single basin jobs
    df_metrics = bundle_single_results() 

    # Create folder and save the dataframe
    output_folder = JOBS_FOLDER.split('/')[-1]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    seed = output_folder.split('_')[-1]
    output_file = os.path.join(output_folder, f'test_all-single_{seed}.csv')
    df_metrics.to_csv(output_file, index=False)