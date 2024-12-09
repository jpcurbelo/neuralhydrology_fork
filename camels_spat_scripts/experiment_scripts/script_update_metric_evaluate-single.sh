#!/bin/bash

# Check if a folder name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_folder>"
    exit 1
fi

# Set the base folder name
base_folder=$1

# Define the seed patterns (e.g., "_111_", "_222_", ...)
# seeds=("_111" "_222" "_333" "_444" "_555" "_666" "_777" "_888" "random-seed")
# seeds=("_111")
seeds=("_777" "_888" "random-seed")

# Iterate through each seed folder
for seed in "${seeds[@]}"; do

    # Find the seed folder at the correct depth (one level inside the base folder)
    seed_folder=$(find "$base_folder" -maxdepth 1 -type d -name "*${seed}*" | sort | tail -n 1)

    echo "Seed folder for $seed: $seed_folder"

    # Check if the seed folder exists
    if [ -z "$seed_folder" ]; then
        echo "Seed folder for $seed not found in $base_folder."
        continue
    fi

    # Iterate through each basin folder inside the seed folder
    for basin_folder in "$seed_folder"/*; do
        if [ -d "$basin_folder" ]; then
            echo "Processing basin folder: $basin_folder"

            # Locate the config.yml file within the basin folder
            config_file="$basin_folder/config.yml"
            if [ ! -f "$config_file" ]; then
                echo "config.yml not found in $basin_folder. Skipping."
                continue
            fi

            # Update the `run_dir` in config.yml
            echo "Updating run_dir in $config_file..."
            if grep -q "^run_dir:" "$config_file"; then
                # Extract the current `run_dir` value
                current_run_dir=$(awk '/^run_dir:/ {getline; print $1}' "$config_file")

                # Dynamically determine the base header (current working directory where the script is run)
                base_header=$(pwd)

                # Construct the new `run_dir` with the correct base folder and basin path
                new_run_dir="${base_header}/${base_folder}$(basename "$(dirname "$current_run_dir")")/$(basename "$current_run_dir")"
                
                # Check if the `run_dir` is already correct
                if [ "$current_run_dir" == "$new_run_dir" ]; then
                    echo "run_dir is already correct. Skipping update."
                else
                    # Update the config.yml file
                    sed -i -E "/^run_dir:/,/^[^-]/c\run_dir:\n  $new_run_dir" "$config_file"
                    echo "Updated run_dir to $new_run_dir in $config_file."
                fi
            else
                echo "run_dir not found in $config_file."
            fi

            # Update metrics in config.yml
            echo "Updating metrics in $config_file..."
            if ! grep -q "mNSE" "$config_file"; then
                sed -i '/metrics:/a\- mNSE' "$config_file"
                echo "Added mNSE to metrics in $config_file."
            fi
            if ! grep -q "FHV_1%" "$config_file"; then
                sed -i '/metrics:/a\- FHV_1%' "$config_file"
                echo "Added FHV_1% to metrics in $config_file."
            fi

            # Extract the basin ID from the folder name
            basin_id=$(basename "$basin_folder" | cut -d '_' -f 1)

            # Create a file in base_header with the basin ID as content
            basin_file="${base_header}/${basin_id}_file.txt"
            echo "$basin_id" > "$basin_file"

            # Run the `nh-run evaluate` command for the basin folder
            echo "Running nh-run evaluate for $basin_folder..."
            nh-run evaluate --run-dir "$basin_folder"
            if [ $? -eq 0 ]; then
                echo "nh-run evaluate completed successfully for $basin_folder."
            else
                echo "nh-run evaluate failed for $basin_folder."
            fi

            # Remove the basin file
            rm "$basin_file"
        fi
    done
done

echo "Script completed."
