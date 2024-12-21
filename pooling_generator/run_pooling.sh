#!/bin/bash
# run_pooling.sh: A script to process and generate post-pooling sequence embeddings
# Usage:
#   ./run_pooling.sh [--path_token_emb PATH] [--path_attention_matrices PATH] [--output_dir PATH] [--accession_list FILE]
#
# Arguments:
#   --path_token_emb         (Optional) Path to the token embedding directory. Default: "../token_embedding_generator/representation_matrices"
#   --path_attention_matrices (Optional) Path to the attention matrices directory. Default: "./token_embedding_generator/attention_matrices"
#   --output_dir             (Optional) Path to the output directory for post-pooling sequence embeddings. Default: "./post_pooling_sequence_embeddings"
#   --accession_list         (Optional) File containing a list of accession codes. If not provided, defaults to .pt files in path_attention_matrices.
#
# Examples:
# 1. Use default paths and process all .pt files in the attention matrices directory:
#    ./run_pooling.sh
#
# 2. Specify custom token embedding and attention matrices paths:
#    ./run_pooling.sh --path_token_emb /custom/token_emb --path_attention_matrices /custom/attention_matrices
#
# 3. Specify all paths including output directory:
#    ./run_pooling.sh --path_token_emb /custom/token_emb --path_attention_matrices /custom/attention_matrices --output_dir /custom/output_dir
#
# 4. Provide an accession list file for processing specific files:
#    ./run_pooling.sh --path_token_emb /custom/token_emb --path_attention_matrices /custom/attention_matrices --output_dir /custom/output_dir --accession_list accession_list.txt
#
# Notes:
# - The script ensures that required files (token embedding and attention matrix files) exist before processing.
# - Processed and skipped file counts are tracked and reported at the end.


# Default values for the input arguments
path_token_emb="../token_embedding_generator/representation_matrices"
path_attention_matrices="./token_embedding_generator/attention_matrices"
output_dir="./post_pooling_sequence_embeddings"
accession_list=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path_token_emb)
            path_token_emb="$2"
            shift 2
            ;;
        --path_attention_matrices)
            path_attention_matrices="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --accession_list)
            accession_list="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--path_token_emb PATH] [--path_attention_matrices PATH] [--output_dir PATH] [--accession_list FILE]"
            exit 1
            ;;
    esac
done

# Print the directories being used
echo "Token Embedding Directory: $path_token_emb"
echo "Attention Matrices Directory: $path_attention_matrices"
echo "Output Directory: $output_dir"
if [[ -n "$accession_list" ]]; then
    echo "Accession List File: $accession_list"
else
    echo "No accession list file provided, defaulting to .pt files in $path_attention_matrices"
fi

# Path to the Python script
PYTHON_SCRIPT="./pooled_sequence_generator.py"

# Load accession codes from file or derive from directory
if [[ -n "$accession_list" ]]; then
    echo "Reading accession codes from file: $accession_list"
    mapfile -t accessions < "$accession_list"
else
    total_files=$(find "$path_attention_matrices" -type f -name "*.pt" | wc -l)
    echo "Total .pt files to go through: $total_files"
    files_to_process=$(find "$path_attention_matrices" -name "*.pt" -type f -printf "%s\t%p\n" | sort -n | cut -f2)
    accessions=($(for attention_file in $files_to_process; do basename -- "$attention_file" .pt; done))
fi

# Create the output directory if it does not exist
mkdir -p "$output_dir"

# Processing loop
processed_count=0
skipped_count=0

for accession in "${accessions[@]}"; do
    token_emb_file="$path_token_emb/$accession.pt"
    attention_matrix_file="$path_attention_matrices/$accession.pt"
    output_file="$output_dir/$accession.pt"

    if [[ -f "$token_emb_file" ]] && [[ -f "$attention_matrix_file" ]]; then
        echo "Processing $accession..."
        python -u "$PYTHON_SCRIPT" "$token_emb_file" "$attention_matrix_file" "$output_file"
        if [[ $? -eq 0 ]]; then
            ((processed_count++))
            echo "Processed $processed_count files..."
        else
            echo "Error processing $accession."
        fi
    else
        echo "Skipping $accession, required files not found."
        ((skipped_count++))
        echo "Skipped $skipped_count files..."
    fi
done

echo "Processing complete."
echo "Total processed files: $processed_count"
echo "Total skipped files: $skipped_count"
