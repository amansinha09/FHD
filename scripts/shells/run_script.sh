#!/bin/bash

# Define the directory containing the job scripts
script_dir="shells/"  # Change this to your script directory

# Iterate over each file that starts with the specified prefix
for script in "${script_dir}bert"*; do
    # Check if the file exists and is a regular file
    if [ -f "$script" ]; then
        # Submit the job script using sbatch
        sbatch "$script"
        echo "Submitted: $script"
    else
        echo "No files found with prefix ${prefix}."
    fi
done

