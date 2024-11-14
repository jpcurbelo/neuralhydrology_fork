#!/bin/bash

# Global variables
RUNS_PATH="runs"
# NAME_PATTERN="559_cyril_camelsus_ealstm_*"  # Updated to include wildcard
NAME_PATTERN="559_cyril_camelsus_lstm_lrok_*"  # Updated to include wildcard

# Function to get the list of folders that match the pattern, sorted alphabetically
get_matched_folders() {
    local folder_name_pattern="$1"
    local matched_folders

    # Get and sort the list of folders that match the pattern
    if ! matched_folders=$(find "$RUNS_PATH" -type d -name "$folder_name_pattern" | sort); then
        printf "Error: Failed to retrieve folders with pattern %s\n" "$folder_name_pattern" >&2
        return 1
    fi

    # Output the sorted list of matched folders
    echo "$matched_folders"
}

# Function to evaluate each folder in parallel
evaluate_folders() {
    local folders=("$@")

    # Run `nh-run evaluate` on each folder in parallel using xargs
    printf "%s\n" "${folders[@]}" | xargs -n 1 -P 0 -I {} bash -c 'nh-run evaluate --run-dir "{}"'

    # # Run `nh-run evaluate` on each folder in serial
    # for folder in "${folders[@]}"; do
    #     nh-run evaluate --run-dir "$folder"
    #     exit 1
    # done
}

# Main function to orchestrate tasks
main() {
    # Get the list of folders that match the pattern
    local matched_folders
    matched_folders=($(get_matched_folders "$NAME_PATTERN"))

    # Evaluate each folder in parallel
    if [[ ${#matched_folders[@]} -gt 0 ]]; then
        evaluate_folders "${matched_folders[@]}"
    else
        printf "No folders matched the pattern %s\n" "$NAME_PATTERN" >&2
    fi
}

main "$@"