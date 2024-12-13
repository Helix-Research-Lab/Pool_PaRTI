import torch
import esm
import os
import sys

def parse_fasta(fasta_file):
    """Simple FASTA parser."""
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

def process_fasta_and_extract_data(fasta_file, output_dir):
    # Load the ESM-2 model
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # Disables dropout for deterministic results

    # Read sequences from the FASTA file using the custom parser
    data = list(parse_fasta(fasta_file))

    # Convert batch data
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations, contacts, and attention heads
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # Extract, process, and save data for each sequence
    for i, (label, _) in enumerate(data):

        # Process and save attention heads across layers
        attn_mean_pooled_layers = []
        attn_max_pooled_layers = []
        for layer in range(33):
            attn_raw = results["attentions"][i, layer]  # Attention from each layer

            # Compress attention data
            attn_mean_pooled = torch.mean(attn_raw, dim=0)
            attn_max_pooled = torch.max(attn_raw, dim=0).values

            attn_mean_pooled_layers.append(attn_mean_pooled)
            attn_max_pooled_layers.append(attn_max_pooled)

        # Stack the pooled attention matrices
        attn_mean_pooled_stacked = torch.stack(attn_mean_pooled_layers)
        attn_max_pooled_stacked = torch.stack(attn_max_pooled_layers)

        # Combine the mean and max pooled attention matrices
        combined_attention = torch.stack([attn_mean_pooled_stacked, attn_max_pooled_stacked]).unsqueeze(1)

        # Save the combined attention matrix
        attention_file_path = f"{output_dir}/attention_matrices_mean_max_perLayer/{fasta_file.split('.fa')[0].split('/')[-1]}.pt"
        if not os.path.exists(attention_file_path):
            try:
                torch.save(combined_attention, attention_file_path)
            except Exception as e:
                print(f"output dir {output_dir}", flush=True)
                print(f"attention_file_path {attention_file_path}", flush=True)
                print(f"fasta_file {fasta_file}", flush=True)
                print(f"fasta_file.split('.fa')[0].split('/')[-1] {fasta_file.split('.fa')[0].split('/')[-1]}", flush=True)
                raise Exception(e)

            print(f"Processed and saved attention data: {attention_file_path}", flush=True)

        # Save representations
        representations_file_path = f"{output_dir}/representation_matrices/{fasta_file.split('.fa')[0].split('/')[-1]}.pt"
        if not os.path.exists(representations_file_path):
            representations = results["representations"][33][i]  # Layer 33 representations
            torch.save(representations, representations_file_path)
            print(f"Processed and saved representations: {representations_file_path}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <FASTA_FILE> <OUTPUT_DIR>")
        print("You're not prodiving the input correctly!")
        sys.exit(1)

    fasta_file = sys.argv[1]
    output_dir = sys.argv[2]
    print(f"fasta_file {fasta_file} -- output_dir {output_dir}", flush=True)
    process_fasta_and_extract_data(fasta_file, output_dir)
