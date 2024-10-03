# create_token_embeddings_esm_protbert.py

"""
Script to process multiple FASTA files in a directory using either the ESM-2 or ProtBERT model,
and save the token representations, attention matrices, and contact predictions (for ESM-2).

Usage:
    python process_fasta_files.py --model [esm|protbert] --input_dir INPUT_DIR --output_dir OUTPUT_DIR

Example:
    python process_fasta_files.py --model esm --input_dir ./fasta_files --output_dir ./output

Requirements:
    - torch
    - transformers
    - esm (for ESM model)
    - biopython

You can install the required packages using:
    pip install torch transformers biopython fair-esm

Note: The 'esm' package can be installed via pip with 'fair-esm'.
"""

import os
import sys
import argparse
import glob
import torch
from Bio import SeqIO

def parse_fasta(fasta_file):
    """
    Parses a FASTA file and yields tuples of (identifier, sequence).

    Args:
        fasta_file (str): Path to the FASTA file.

    Yields:
        tuple: (identifier, sequence)
    """
    with open(fasta_file, 'r') as file:
        identifier = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if identifier is not None:
                    yield (identifier, ''.join(sequence))
                identifier = line[1:]  # Remove the '>' character
                sequence = []
            else:
                sequence.append(line)
        # Yield the last entry
        if identifier is not None:
            yield (identifier, ''.join(sequence))

def process_fasta_with_esm(fasta_file, output_dir):
    """
    Processes the FASTA file and extracts representations, attention matrices,
    and contact predictions using the ESM-2 model.

    Args:
        fasta_file (str): Path to the input FASTA file.
        output_dir (str): Path to the output directory.
    """
    import esm  # Import esm here since it's only needed for ESM model

    # Load the ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disable dropout for deterministic results

    # Read sequences from the FASTA file using the custom parser
    data = list(parse_fasta(fasta_file))

    # Create output directories if they do not exist
    attention_dir = os.path.join(output_dir, 'attention_matrices_mean_max_perLayer')
    representation_dir = os.path.join(output_dir, 'representation_matrices')
    contact_dir = os.path.join(output_dir, 'contact_matrices')
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(representation_dir, exist_ok=True)
    os.makedirs(contact_dir, exist_ok=True)

    # Convert batch data
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations, contacts, and attention heads
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # Extract, process, and save data for each sequence
    for i, (label, sequence) in enumerate(data):
        print(f'Processing sequence {i+1}/{len(data)}: {label}')

        # Process and save attention heads
        attn_mean_pooled_layers = []
        attn_max_pooled_layers = []
        for layer in range(33):
            attn_raw = results["attentions"][i, layer]  # Attention from each layer

            # Compress attention data
            attn_mean_pooled = torch.mean(attn_raw, dim=0)  # Mean over all attention heads
            attn_max_pooled = torch.max(attn_raw, dim=0).values  # Max over all attention heads

            attn_mean_pooled_layers.append(attn_mean_pooled)
            attn_max_pooled_layers.append(attn_max_pooled)

        # Stack the pooled attention matrices
        attn_mean_pooled_stacked = torch.stack(attn_mean_pooled_layers)
        attn_max_pooled_stacked = torch.stack(attn_max_pooled_layers)

        # Combine the mean and max pooled attention matrices
        combined_attention = torch.stack(
            [attn_mean_pooled_stacked, attn_max_pooled_stacked]
        ).unsqueeze(1)

        # Use sequence identifier as filename
        safe_label = label.replace("/", "_").replace("\\", "_")
        basename = os.path.splitext(os.path.basename(fasta_file))[0]
        filename_prefix = f"{basename}_{safe_label}"

        # Save the combined attention matrix
        attention_file_path = os.path.join(attention_dir, f'{filename_prefix}.pt')
        if not os.path.exists(attention_file_path):
            try:
                torch.save(combined_attention, attention_file_path)
            except Exception as e:
                print(f"Error saving attention data for {label}: {e}")
                continue
            print(f"Saved attention data: {attention_file_path}")

        # Save representations
        representations_file_path = os.path.join(representation_dir, f'{filename_prefix}.pt')
        if not os.path.exists(representations_file_path):
            representations = results["representations"][33][i]  # Layer 33 representations
            try:
                torch.save(representations, representations_file_path)
            except Exception as e:
                print(f"Error saving representations for {label}: {e}")
                continue
            print(f"Saved representations: {representations_file_path}")

        # Save contacts if available
        if 'contacts' in results:
            contacts_file_path = os.path.join(contact_dir, f'{filename_prefix}.pt')
            if not os.path.exists(contacts_file_path):
                contacts = results["contacts"][i]  # Contact predictions
                try:
                    torch.save(contacts, contacts_file_path)
                except Exception as e:
                    print(f"Error saving contacts for {label}: {e}")
                    continue
                print(f"Saved contacts: {contacts_file_path}")

def process_fasta_with_protbert(fasta_file, output_dir):
    """
    Processes the FASTA file and extracts representations and attention matrices
    using the ProtBERT model.

    Args:
        fasta_file (str): Path to the input FASTA file.
        output_dir (str): Path to the output directory.
    """
    from transformers import BertModel, BertTokenizer

    # Load the ProtBERT model and tokenizer
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    # Read sequences from the FASTA file
    data = list(parse_fasta(fasta_file))

    # Create output directories if they do not exist
    attention_dir = os.path.join(output_dir, 'attention_matrices_mean_max_perLayer')
    representation_dir = os.path.join(output_dir, 'representation_matrices')
    os.makedirs(attention_dir, exist_ok=True)
    os.makedirs(representation_dir, exist_ok=True)

    # Process each sequence
    for i, (label, sequence) in enumerate(data):
        print(f'Processing sequence {i+1}/{len(data)}: {label}')

        # Tokenize the sequence
        inputs = tokenizer(' '.join(sequence), return_tensors="pt")
        try:
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            print(f"Error processing {label}: {e}")
            continue

        # Extract token representations and attention matrices
        token_representations = outputs.last_hidden_state  # Embeddings for each token
        attention_matrices = outputs.attentions  # Attention matrices from each layer

        # Process attention matrices to obtain mean and max per layer
        attention_matrices_converted_mean = []
        attention_matrices_converted_max = []
        for layer_attention in attention_matrices:
            mean_attention = layer_attention.mean(dim=1)  # Mean over all heads
            max_attention = layer_attention.max(dim=1).values  # Max over all heads
            attention_matrices_converted_mean.append(mean_attention)
            attention_matrices_converted_max.append(max_attention)
        attention_matrices_converted_mean = torch.stack(attention_matrices_converted_mean, dim=1)
        attention_matrices_converted_max = torch.stack(attention_matrices_converted_max, dim=1)

        # Combine mean and max attention tensors
        combined_attention = torch.stack(
            [attention_matrices_converted_mean.squeeze(2), attention_matrices_converted_max.squeeze(2)],
            dim=0
        )

        # Use sequence identifier as filename
        safe_label = label.replace("/", "_").replace("\\", "_")
        basename = os.path.splitext(os.path.basename(fasta_file))[0]
        filename_prefix = f"{basename}_{safe_label}"

        # Save the attention matrices and token representations
        attention_file_path = os.path.join(attention_dir, f'{filename_prefix}.pt')
        if not os.path.exists(attention_file_path):
            try:
                torch.save(combined_attention, attention_file_path)
            except Exception as e:
                print(f"Error saving attention data for {label}: {e}")
                continue
            print(f"Saved attention data: {attention_file_path}")

        representations_file_path = os.path.join(representation_dir, f'{filename_prefix}.pt')
        if not os.path.exists(representations_file_path):
            try:
                torch.save(token_representations, representations_file_path)
            except Exception as e:
                print(f"Error saving representations for {label}: {e}")
                continue
            print(f"Saved representations: {representations_file_path}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process multiple FASTA files using ESM or ProtBERT model.')
    parser.add_argument('--model', type=str, choices=['esm', 'protbert'], required=True,
                        help='Model to use: "esm" or "protbert".')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing FASTA files.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the directory where output files will be saved.')
    args = parser.parse_args()

    model_choice = args.model.lower()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # Get list of FASTA files in the input directory
    fasta_files = glob.glob(os.path.join(input_dir, '*.fa')) + \
                  glob.glob(os.path.join(input_dir, '*.fasta')) + \
                  glob.glob(os.path.join(input_dir, '*.faa')) + \
                  glob.glob(os.path.join(input_dir, '*.fna'))

    if not fasta_files:
        print(f"No FASTA files found in directory: {input_dir}")
        sys.exit(1)

    total_files = len(fasta_files)
    print(f"Found {total_files} FASTA files in directory: {input_dir}")

    # Process each FASTA file
    for idx, fasta_file in enumerate(fasta_files):
        print(f"\nProcessing file {idx+1}/{total_files}: {fasta_file}")
        if model_choice == 'esm':
            process_fasta_with_esm(fasta_file, output_dir)
        elif model_choice == 'protbert':
            process_fasta_with_protbert(fasta_file, output_dir)
        else:
            print(f"Invalid model choice: {model_choice}")
            sys.exit(1)

    print("\nAll files processed.")

if __name__ == "__main__":
    main()
