import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def contains_cyrils_data(folder4cdf_dir_list):
    """
    Function to check if any of the plots contain Cyril's data (928 basins).
    
    Parameters:
    - folder4cdf_dir_list: List of dictionaries containing experiment data.
    
    Returns:
    - Boolean: True if Cyril's data is found, False otherwise.
    """
    # Keywords to identify Cyril's data
    cyril_directory_keywords = ['runs_928_cyril']
    # cyril_experiment_symbols = ['^\diamond$', '^\star$', '^\star \star$']  # Special symbols in experiment names

    # Check if any of the folder dictionaries contain Cyril's data
    for folder in folder4cdf_dir_list:
        # Check if the directory contains Cyril's keyword
        if any(keyword in folder['directory'] for keyword in cyril_directory_keywords):
            return True
        
        # # Check if the experiment name contains Cyril's symbols
        # if any(symbol in folder['experiment'] for symbol in cyril_experiment_symbols):
        #     return True

    # If no Cyril's data is found
    return False

def plot_cdf_with_zoom(folder4cdf_dir_list, zoom_ranges_x=None, zoom_ranges_y=None):
    """
    Function to generate CDF plots with both the main plot and optional zoomed-in subplots.
    
    Parameters:
    - folder4cdf_dir_list: List of dictionaries containing experiment data.
    - zoom_ranges_x: List of x-axis ranges for zoomed-in subplots.
    - zoom_ranges_y: List of y-axis ranges for zoomed-in subplots.
    - plot_basins_flag: Boolean flag to include or exclude 928 basins in the legend.
    - output_file_main: Filename to save the main plot.
    - output_file_zoomed: Filename to save the plot with zoomed-in subplots.
    """

    # # output_file_main="cdf_nse_usa.png", output_file_zoomed="cdf_nse_zoomed_usa.png"

    # Step 1: Create and save the main plot without zoomed-in subplots
    fig, ax_main = plt.subplots(figsize=(10, 8))

    # Add an empty plot for the title in the legend
    ax_main.plot([], [], ' ', label=r'$\it{Experiment\ (model: dynamic+static)}$')

    # Main plot loop
    for folder in folder4cdf_dir_list:
        folder_dir = Path(folder['directory'])
        folder_exp = folder['experiment']
        folder_color = folder['color']
        folder_marker = folder['marker']
        folder_line = folder['linestyle']

        # Load the result dataframe
        if os.path.exists(folder_dir / 'test_ensemble_metrics.csv'):
            df = pd.read_csv(folder_dir / 'test_ensemble_metrics.csv')
        else:
            csv_files = [f for f in os.listdir(folder_dir) if f.endswith('.csv')]
            df = pd.read_csv(folder_dir / csv_files[0])

        # Make negative NSE values zero
        df['NSE'] = df['NSE'].apply(lambda x: 0 if x < 0 else x)

        # Select basin and NSE columns
        df = df[['basin', 'NSE']].copy()
        # Sort NSE values
        nse_values = np.sort(df['NSE'])
        # Calculate cumulative probabilities
        cdf = np.arange(1, len(nse_values) + 1) / len(nse_values)

        # Plot the main CDF curve
        ax_main.plot(nse_values, cdf, color=folder_color,
                     alpha=0.8,
                     linewidth=2.5,
                     marker=folder_marker,
                     markevery=20,
                     ms=7,
                     linestyle=folder_line,
                     label=folder_exp)

    # Add footnotes for basins based on the flag
    ax_main.plot([], [], ' ', label=r'$^*$ 531 basins (CAMELS-US)')
    ax_main.plot([], [], ' ', label=r'$^\dagger$ 505 basins (CAMELS-SPAT)')

    if contains_cyrils_data(folder4cdf_dir_list):
        ax_main.plot([], [], ' ', label=r'$^{\star\star}$ 928 basins (CAMELS-SPAT)')
        output_file_main = 'cdf_nse_usa-can.png'
    else:
        output_file_main = 'cdf_nse_usa.png'


    # Set labels and limits for the main plot
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel('NSE', fontsize=14)
    ax_main.set_ylabel('CDF', fontsize=14)
    ax_main.grid(True)
    ax_main.legend(fontsize=12)

    # Save the main plot
    plt.tight_layout()
    plt.savefig(output_file_main, dpi=150, bbox_inches='tight')

    # Show the main plot (optional, you can remove this if not needed)
    plt.show()

    # Check if zoom ranges are provided for the zoomed-in subplots
    if zoom_ranges_x is None or zoom_ranges_y is None:
        return

    # Step 2: Create and save the zoomed-in subplot version
    fig_zoomed = plt.figure(figsize=(14, 8))
    gs = fig_zoomed.add_gridspec(3, 2, width_ratios=[2, 1])  # Main plot spans two columns, three rows for zoomed plots

    # Main plot on the left
    ax_main_zoomed = fig_zoomed.add_subplot(gs[:, 0])  # This spans all rows on the left

    # Zoomed-in plots on the right (stacked vertically)
    ax_inset_top = fig_zoomed.add_subplot(gs[0, 1])  # Top zoom
    ax_inset_middle = fig_zoomed.add_subplot(gs[1, 1])  # Middle zoom
    ax_inset_bottom = fig_zoomed.add_subplot(gs[2, 1])  # Bottom zoom

    # List of axes for zoomed-in plots
    inset_axes = [ax_inset_top, ax_inset_middle, ax_inset_bottom]

    # Add an empty plot for the title in the legend
    ax_main_zoomed.plot([], [], ' ', label=r'$\it{Experiment\ (model: dynamic+static)}$')

    # Main plot loop
    for folder in folder4cdf_dir_list:
        folder_dir = Path(folder['directory'])
        folder_exp = folder['experiment']
        folder_color = folder['color']
        folder_marker = folder['marker']
        folder_line = folder['linestyle']

        # Load the result dataframe
        if os.path.exists(folder_dir / 'test_ensemble_metrics.csv'):
            df = pd.read_csv(folder_dir / 'test_ensemble_metrics.csv')
        else:
            csv_files = [f for f in os.listdir(folder_dir) if f.endswith('.csv')]
            df = pd.read_csv(folder_dir / csv_files[0])

        # Make negative NSE values zero
        df['NSE'] = df['NSE'].apply(lambda x: 0 if x < 0 else x)

        # Select basin and NSE columns
        df = df[['basin', 'NSE']].copy()
        # Sort NSE values
        nse_values = np.sort(df['NSE'])
        # Calculate cumulative probabilities
        cdf = np.arange(1, len(nse_values) + 1) / len(nse_values)

        # Plot the main CDF curve
        ax_main_zoomed.plot(nse_values, cdf, color=folder_color,
                    alpha=0.8,
                    linewidth=2.5,
                    marker=folder_marker,
                    markevery=20,
                    ms=7,
                    linestyle=folder_line,
                    label=folder_exp)

        # Plot zoomed-in sections for each inset plot
        for ax_inset, zoom_range in zip(inset_axes, zoom_ranges_x):
            # Filter NSE values for the zoomed range
            cdf_inset = cdf[(nse_values >= zoom_range[0]) & (nse_values <= zoom_range[1])]
            nse_values_inset = nse_values[(nse_values >= zoom_range[0]) & (nse_values <= zoom_range[1])]

            # Plot on the respective inset
            ax_inset.plot(nse_values_inset, cdf_inset, color=folder_color,
                        alpha=0.8,
                        linewidth=2.5,
                        marker=folder_marker,
                        markevery=2,
                        ms=5,
                        linestyle=folder_line)

    # Add footnotes for basins based on the flag in zoomed version
    ax_main_zoomed.plot([], [], ' ', label=r'$^*$ 531 basins (CAMELS-US)')
    ax_main_zoomed.plot([], [], ' ', label=r'$^\dagger$ 505 basins (CAMELS-SPAT)')

    if contains_cyrils_data(folder4cdf_dir_list):
        ax_main_zoomed.plot([], [], ' ', label=r'$^{\star\star}$ 928 basins (CAMELS-SPAT)')
        output_file_zoomed = 'cdf_nse_zoomed_usa-can.png'
    else:
        output_file_zoomed = 'cdf_nse_zoomed_usa.png'

    # Set labels and limits for the main plot
    ax_main_zoomed.set_xlim(0, 1)
    ax_main_zoomed.set_ylim(0, 1)
    ax_main_zoomed.set_xlabel('NSE', fontsize=14)
    ax_main_zoomed.set_ylabel('CDF', fontsize=14)
    ax_main_zoomed.grid(True)
    ax_main_zoomed.legend(fontsize=12)

    # Customize each inset plot (zoomed-in subplots)
    for ax_inset, zoom_range_x, zoom_range_y in zip(inset_axes, zoom_ranges_x, zoom_ranges_y):
        # Set the x and y limits based on the provided ranges
        ax_inset.set_xlim(zoom_range_x[0], zoom_range_x[1])
        ax_inset.set_ylim(zoom_range_y[0], zoom_range_y[1])

        # Set coarser grid spacing for X and Y ticks and round to 1 decimal place
        ax_inset.set_xticks(np.round(np.linspace(zoom_range_x[0], zoom_range_x[1], num=3), 2))  # 3 major ticks on X-axis
        ax_inset.set_yticks(np.round(np.linspace(zoom_range_y[0], zoom_range_y[1], num=3), 2))  # 3 major ticks on Y-axis

        # Set the grid with a coarser appearance
        ax_inset.grid(True, which='major', linestyle='--', linewidth=0.7)  # Coarser major grid
        ax_inset.minorticks_on()
        ax_inset.grid(True, which='minor', linestyle=':', linewidth=0.5)  # Minor grid as dotted lines

    # Save the zoomed-in plot
    plt.tight_layout()
    plt.savefig(output_file_zoomed, dpi=150, bbox_inches='tight')

    # Show the zoomed-in plot (optional, you can remove this if not needed)
    plt.show()



if __name__ == '__main__':
    
    pass