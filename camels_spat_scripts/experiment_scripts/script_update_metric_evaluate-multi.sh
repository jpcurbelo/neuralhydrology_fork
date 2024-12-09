#!/bin/bash

# Check if a folder name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <folder_name>"
    exit 1
fi

# Set the folder name
base_folder=$1

# Define the seed patterns
# seeds=("_111_" "_222_" "_333_" "_444_" "_555_" "_666_" "_777_" "_888_")
seeds=("_222_")

# Iterate through the seed folders
for seed in "${seeds[@]}"; do
    # Find the subfolder containing the seed pattern
    seed_folder=$(find "$base_folder" -type d -name "*${seed}*" | sort | tail -n 1)
    echo "Seed folder1 for $seed: $seed_folder"

    # Check if the folder exists
    if [ -z "$seed_folder" ]; then
        echo "Seed folder for $seed not found in $base_folder"
        continue
    fi

    # echo "Processing seed folder: $seed_folder"
    # Locate the config.yml file within the seed folder
    config_file="$seed_folder/config.yml"
    if [ ! -f "$config_file" ]; then
        echo "config.yml not found in $seed_folder"
        continue
    fi

    # Update the `run_dir` in config.yml
    echo "Updating run_dir in $config_file..."
    if grep -q "^run_dir:" "$config_file"; then
        # Extract the current `run_dir` value, handling multi-line definitions
        current_run_dir=$(awk '/^run_dir:/ {getline; print $1}' "$config_file")

        echo "Current run_dir: $current_run_dir"
        
        # Check if it ends with "runs/something"
        if [[ $current_run_dir =~ runs/[^/]+$ ]]; then
            # Replace "runs" and everything after it with the seed folder path
            new_run_dir="${current_run_dir/runs*/$(basename "$(dirname "$seed_folder")")}/$(basename "$seed_folder")"

            # Clear all lines under `run_dir` and replace with the new value
            sed -i -E "/^run_dir:/,/^[^-]/c\run_dir:\n  $new_run_dir" "$config_file"
            echo "Updated run_dir to $new_run_dir in $config_file."
        else
            echo "run_dir does not match expected pattern in $config_file. Skipping update."
        fi
    else
        echo "run_dir not found in $config_file."
    fi

    # Add `- mNSE` to the metrics list in the config.yml
    echo "Updating metrics in $config_file..."
    # Check if `- mNSE` is already in the list
    if grep -q "mNSE" "$config_file"; then
        echo "mNSE already exists in metrics list for $config_file. Skipping."
    else
        # Append `- mNSE` to the `metrics` list
        sed -i '/metrics:/a\- mNSE' "$config_file"
        echo "Added mNSE to metrics in $config_file."
    fi

    # Check if `- FHV_1%` is already in the list
    if grep -q "FHV_1%" "$config_file"; then
        echo "FHV_1% already exists in metrics list for $config_file. Skipping."
    else
        # Append `- FHV_1%` to the `metrics` list
        sed -i '/metrics:/a\- FHV_1%' "$config_file"
        echo "Added FHV_1% to metrics in $config_file."
    fi

    # Run the `nh-run evaluate` command for the current seed folder
    echo "Running nh-run evaluate for $seed_folder..."
    nh-run evaluate --run-dir "$seed_folder"
    if [ $? -eq 0 ]; then
        echo "nh-run evaluate completed successfully for $seed_folder."
    else
        echo "nh-run evaluate failed for $seed_folder."
    fi
done

echo "Script completed."
