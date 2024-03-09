# Import necessary libraries
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Constants
RUN_DIR = '../experiment_scripts/runs'
TOPN = 20
BOTTOMN = 426

# Functions
def main(top_n=TOPN, bottom_n=BOTTOMN):
    # List of folders in the run directory
    run_folders = os.listdir(RUN_DIR)
    
    # Initialize an empty DataFrame to store the results
    all_results = None
    
    # Initialize a dictionary to track the count of NSE<0 for each basin
    basin_nse_count = {}

    # Loop through the run folders
    for run_folder in run_folders:
        results_df, basin_counts = extract_test_results(run_folder)
        
        # Update basin_nse_count dictionary
        if basin_counts is not None:
            for basin, count in basin_counts.items():
                basin_nse_count[basin] = basin_nse_count.get(basin, 0) + count
        
        if results_df is not None:
            if all_results is None:
                all_results = results_df
            else:
                all_results = pd.concat([all_results, results_df], ignore_index=True)
    
    # Sort by 'NSE<0' in ascending order and 'NSE' in descending order
    sorted_results = all_results.sort_values(by=['NSE<0', 'NSE'], ascending=[True, False])
    
    # Save the DataFrame with the top N results
    top_results = sorted_results.head(top_n)
    top_results.to_csv(f'top{top_n}_models.csv', index=False)
    
    # Sort the basin_nse_count dictionary by values in descending order
    sorted_basin_nse_count = dict(sorted(basin_nse_count.items(), key=lambda item: item[1], reverse=True))

    # Select the top N items
    worst_n_basins = dict(list(sorted_basin_nse_count.items())[:bottom_n])
    worst_n_basins_df = pd.DataFrame(worst_n_basins.items(), columns=['basin', 'count'])
    worst_n_basins_df.to_csv(f'worst{bottom_n}_basins.csv', index=False)
    
    # Create histograms of the NSE<0 counts and save as images
    create_histogram(worst_n_basins_df, bottom_n)
    

    
    return sorted_results, worst_n_basins

def extract_test_results(run_folder):
    # Path to the test results file
    inner_folder_path = os.path.join(RUN_DIR, run_folder, 'test', '*', 'test_metrics.csv')
    inner_folder = glob.glob(inner_folder_path)
    
    # Check if the file exists
    if not inner_folder:
        print(f"No test results file found for run folder {run_folder}.")
        return None, None
    
    results_file = inner_folder[0]
    
    # Load as a pandas dataframe
    results = pd.read_csv(results_file)
    
    # Count the number of NSE values less than 0
    nse_values = results['NSE']
    nse_values_neg = nse_values[nse_values < 0].count()
    
    # Count the occurrences of NSE<0 for each basin
    basin_counts = results[results['NSE'] < 0]['basin'].value_counts().to_dict()
    
    # Drop rows with NSE values less than 0
    results = results[results['NSE'] > 0]    
    
    # Calculate the mean of the test results for each column 
    mean_results = results.iloc[:, 1:].mean(axis=0)
    
    # Create a DataFrame with the extracted results
    results_df = pd.DataFrame({
        'run_folder': [run_folder],
        'NSE<0': [nse_values_neg],
        **mean_results.to_dict()
    })
    
    return results_df, basin_counts

def create_histogram(worst_n_basins_df, bottom_n):
    # Plot histogram of NSE<0 counts
    plt.figure(figsize=(BOTTOMN // 10 * 2, BOTTOMN // 15))  # Adjust figure size if needed
    plt.bar(worst_n_basins_df['basin'], worst_n_basins_df['count'])
    plt.xlabel('Basin')
    plt.ylabel('NSE<0 Count')
    plt.title(f'Worst Basins (NSE<0) Counts - {len(worst_n_basins_df)} basins')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    
    # Set y-axis tick locator to integer values
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig(f'worst{bottom_n}_basins.png')
    plt.show()



if __name__ == '__main__':
    
    best_results, basin_nse_count = main()
    # print(best_results)
    # print(basin_nse_count)
