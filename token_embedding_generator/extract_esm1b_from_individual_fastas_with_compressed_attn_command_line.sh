#!/bin/bash

conda activate conda activate poolparti
ml gcc/10.1.0
ml cuda/12.1.1

nvidia-smi

# Check if an argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <DIRECTORY_PATH>. You're calling this function wrong!"
    exit 1
fi

# Path to the Python script
PYTHON_SCRIPT="extract_esm1b_with_all_layer_attentions.py"

# Define paths for clarity and maintainability
embeddings_output_dir="representation_matrices"
attentions_output_dir="attention_matrices"
contacts_output_dir="contact_matrices"
output_dir="./"

DIRECTORY_PATH="$1"  # Use the first argument as the directory path

process_files() {
    # Find all .fa files and sort them by size in ascending order
    mapfile -t files < <(find "$1" -maxdepth 1 -name '*.fa' -printf "%s\t%p\n" | sort -n | cut -f2)

    total_files=${#files[@]}
    processed_files=0

    for file in "${files[@]}"; do
        basename=$(basename "$file" .fa)

        output_file_embeddings="$embeddings_output_dir/${basename}.pt"
        output_file_attentions="$attentions_output_dir/${basename}.pt"
        output_file_contacts="$contacts_output_dir/${basename}.pt"

        # Check if neither attention file exist
        if [[ ! -f "$output_file_attentions" ]]; then
            echo "Processing $basename at $output_dir"  # Print the name of the fasta file before processing
            python "$PYTHON_SCRIPT" "$file" "$output_dir"
            echo "Processed $basename at $output_dir"  # Print the name of the fasta file after processing
        fi

        # Update and print progress
        ((processed_files++))
        echo -ne "Processing: $processed_files/$total_files\r"
    done
    echo -e "\nAll files in $1 processed."
}

# Call process_files with the directory path provided as an argument
process_files "$DIRECTORY_PATH"


## how you call this
# sbatch extract_esm_from_individual_fastas_with_compressed_attn_command_line.sh <path-to-fasta-file-directory>
