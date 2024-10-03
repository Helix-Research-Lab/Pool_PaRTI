# fetch_uniprot_sequences_individual.py

"""
Script to fetch protein sequences from UniProt given a list of accession numbers
and save each sequence into a separate FASTA file within a specified directory.

Usage:
    python fetch_uniprot_sequences_individual.py --accessions ACCESSIONS_FILE --output_dir OUTPUT_DIRECTORY

Example:
    python fetch_uniprot_sequences_individual.py --accessions accessions.txt --output_dir ./sequences

Requirements:
    - requests
    - argparse
    - os

You can install the required packages using:
    pip install requests
"""

import requests
import argparse
import os
import time

def fetch_sequence(accession):
    """
    Fetches the protein sequence for a given UniProt accession number.

    Args:
        accession (str): The UniProt accession number.

    Returns:
        tuple: A tuple containing the accession number and the protein sequence,
               or (accession, None) if not found.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        # The response text is in FASTA format; extract the sequence
        lines = response.text.strip().split('\n')
        sequence = ''.join(lines[1:])  # Skip the header line
        return accession, sequence
    else:
        print(f"Error fetching data for accession '{accession}'. Status code: {response.status_code}")
        return accession, None

def write_fasta_file(accession, sequence, output_directory):
    """
    Writes the sequence to a separate FASTA file named after the accession number.

    Args:
        accession (str): The UniProt accession number.
        sequence (str): The protein sequence.
        output_directory (str): The directory where the FASTA file will be saved.
    """
    filename = f"{accession}.fasta"
    filepath = os.path.join(output_directory, filename)

    with open(filepath, 'w') as fasta_file:
        fasta_file.write(f">{accession}\n")
        # Wrap the sequence every 60 characters
        for i in range(0, len(sequence), 60):
            fasta_file.write(sequence[i:i+60] + '\n')

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Fetch UniProt sequences for a list of accession numbers.')
    parser.add_argument('--accessions', type=str, required=True,
                        help='Path to the file containing UniProt accession numbers (one per line).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory where FASTA files will be saved.')
    args = parser.parse_args()

    accessions_file = args.accessions
    output_directory = args.output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Read accession numbers from the input file
    with open(accessions_file, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]

    total_accessions = len(accessions)
    print(f"Found {total_accessions} accession numbers.")

    # Fetch and save each sequence individually
    for idx, accession in enumerate(accessions):
        print(f"Processing {idx+1}/{total_accessions}: {accession}")
        accession, sequence = fetch_sequence(accession)
        if sequence:
            write_fasta_file(accession, sequence, output_directory)
            print(f"Saved sequence to {accession}.fasta")
        else:
            print(f"Skipping accession '{accession}' due to missing sequence.")

        time.sleep(0.5)

    print(f"\nAll sequences have been processed. FASTA files are saved in '{output_directory}'.")

if __name__ == "__main__":
    main()


## to run this script:
# python fetch_uniprot_sequences_individual.py --accessions accessions.txt --output_dir ./sequences
