#!/bin/bash

# Start time of the script
start_time=$(date +%s)

# Maximum duration the script should run (46 hours in seconds)
max_duration=$((46 * 3600))

source /home/groups/rbaltman/tartici/miniconda3/etc/profile.d/conda.sh
conda activate substrateSpec1
ml gcc/10.1.0
ml cuda/12.1.1

nvidia-smi

# Path to the Python script
PYTHON_SCRIPT="./pooled_sequence_generator.py"

embeddings_output_dir="../token_embedding_generator/representation_matrices"
attentions_output_dir="./token_embedding_generator/attention_matrices"
contacts_output_dir="./token_embedding_generator/contact_matrices"

# Time tracking
start_time=$(date +%s)
max_duration=$((46 * 3600))  # 46 hours in seconds


# Check if an accession list file is provided as an argument
if [ "$#" -eq 1 ]; then
    echo "Reading accession codes from file: $1"
    mapfile -t accessions < "$1"
else
    echo "No accession list file provided, defaulting to .pt files in $attentions_output_dir"
    # Generate accession codes from .pt files
    # Count total .pt files in attentions_output_dir
    total_files=$(find "$attentions_output_dir" -type f -name "*.pt" | wc -l)
    echo "Total .pt files to go through: $total_files"
    
    # Count and sort .pt files by size
    files_to_process=$(find "$attentions_output_dir" -name "*.pt" -type f -printf "%s\t%p\n" | sort -n | cut -f2)
    accessions=($(for attention_file in $files_to_process; do basename -- "$attention_file" .pt; done))
fi

# Processing loop
processed_count=0
skipped_count=0

for accession in "${accessions[@]}"; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    # Check if maximum duration has been reached
    if [ $elapsed_time -ge $max_duration ]; then
        echo "Maximum duration reached, scheduling the next job."
        sbatch --dependency=afterany:$SLURM_JOB_ID $0  # Reschedule this script
        exit 0
    fi

    # Extract filename without extension
    filename="$accession.pt"

    # Check for corresponding files in other directories and if it has already been processed
    if [ -f "$embeddings_output_dir/$filename" ] && [ -f "$contacts_output_dir/$filename" ] ; then
        echo "Processing $accession..."
        python -u "$PYTHON_SCRIPT" "$accession"
        ((processed_count++))
        echo "Processed $processed_count..."
    else
        echo "Skipping $accession, corresponding files not found in all directories or already processed."
        ((skipped_count++))
        echo "Skipped $skipped_count..."
    fi

    # Optional sleep to prevent potential system overload
    sleep 1
done



echo "Completed processing $processed_count files. Skipped $skipped_count files."