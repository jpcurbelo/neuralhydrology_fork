import os
import pandas as pd

from functools import partial
import concurrent.futures

dataset = "camels_spat"
# countries = ["CAN", "USA"]
countries = ["CAN"]

NUM_BASINS = 2000
StartDate = "1975-10-01"
EndDate = "2019-09-30"

def process_basin(basin, data_dir):
    df = pd.read_csv(os.path.join(data_dir, basin), index_col=0, parse_dates=True)
    if df.index[0] <= pd.to_datetime(StartDate) and df.index[-1] >= pd.to_datetime(EndDate):
        # Check for missing days
        if df.index.freq != 'D':
            # Upsample to daily frequency
            df_daily = df.resample('D').mean()
            # Check for NaN values
            if df_daily.isnull().values.any():
                print(f"Missing days found in {basin}")
                return None
        return basin
    else:
        return None


selected_basins = []
counter = 0
not_to_stop = True
for country in countries:
    if not_to_stop:
        data_dir = f"../../../../../gladwell/hydrology/SUMMA/summa-ml-models/CAMELS_spat_NH/CAMELS_spat_{country}"
        basins = sorted(os.listdir(data_dir))
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Define a partial function to pass the country
            process_basin_partial = partial(process_basin, data_dir=data_dir)
            
            # Process each basin concurrently
            results = [executor.submit(process_basin_partial, basin) for basin in basins]
            
            # Wait for all results to be available
            for result in concurrent.futures.as_completed(results):
                selected_basin = result.result()
                if selected_basin:
                    selected_basins.append(selected_basin)
                    counter += 1
                    if counter == NUM_BASINS:
                        not_to_stop = False
                        break
                    
        print(country, len([res for res in results if res.result() is not None]), '->', len(selected_basins))

# Write all selected basin names to a file named after the counter
with open(f"{len(selected_basins)}_basin_{dataset}.txt", "w") as f:
    for basin_name in selected_basins:
        f.write(basin_name.split('.')[0] + "\n")
    
    # Remove last newline character
    f.seek(f.tell() - 1)
    f.truncate()