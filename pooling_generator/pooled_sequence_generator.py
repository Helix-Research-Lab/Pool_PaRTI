# pooled_sequence_generator.py

"""
Script to generate sequence embeddings from token embeddings and attention matrices using Pool PaRTI.

By default, this script generates only the Pool PaRTI embedding. Optionally, it can also generate
CLS-pooled, mean-pooled, and max-pooled embeddings if specified by additional arguments.

Usage:
    python pooled_sequence_generator.py --path_token_emb PATH_TO_TOKEN_EMBEDDINGS \
                                        --path_attention_layers PATH_TO_ATTENTION_MATRICES \
                                        [--output_dir OUTPUT_DIRECTORY] \
                                        [--generate_all]

Example:
    python pooled_sequence_generator.py --path_token_emb embeddings.pt \
                                        --path_attention_layers attentions.pt \
                                        --output_dir ./sequence_embeddings \
                                        --generate_all

Requirements:
    - torch
    - numpy

You can install the required packages using:
    pip install torch numpy
"""

import argparse
import os
import torch
from token_to_sequence_pooler import TokenToSequencePooler

def main(path_token_emb, path_attention_layers, output_dir, generate_all):
    """
    Main function to perform pooling operations on protein sequence data.

    Args:
        path_token_emb (str): Path to the token embeddings file.
        path_attention_layers (str): Path to the attention matrices file.
        output_dir (str): Directory where the output embeddings will be saved.
        generate_all (bool): If True, generates all pooling embeddings (CLS, mean, max, Pool PaRTI).
                             If False, only generates the Pool PaRTI embedding.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(path_token_emb)
    
    # Instantiate the TokenToSequencePooler
    pooler = TokenToSequencePooler(path_token_emb=path_token_emb, 
                                   path_attention_layers=path_attention_layers)

    rep_w_cls = pooler.representations_with_cls
    attn = pooler.attn_all_layers

    # Check if the shapes of representations and attentions match
    if not rep_w_cls.shape[0] == attn.shape[-1]:
        if len(rep_w_cls.shape) == 3 and not rep_w_cls.shape[1] == attn.shape[-1]:
            print(f"The attention and representation shapes don't match for {file_name}", flush=True)
            return

    # Perform Pool PaRTI pooling
    pool_parti_dir = os.path.join(output_dir, "pool_parti")
    os.makedirs(pool_parti_dir, exist_ok=True)
    address = os.path.join(pool_parti_dir, file_name)
    if not os.path.exists(address):
        pooled = pooler.pool_parti(verbose=False, return_importance=False)
        torch.save(pooled, address)
        print(f"Pool PaRTI embedding saved at {address}")
    else:
        print(f"Pool PaRTI embedding already exists at {address}")

    # If generate_all is True, perform additional pooling methods
    if generate_all:
        # CLS Pooling
        cls_pooled_dir = os.path.join(output_dir, "cls_pooled")
        os.makedirs(cls_pooled_dir, exist_ok=True)
        address = os.path.join(cls_pooled_dir, file_name)
        if not os.path.exists(address):
            cls_pooled = pooler.cls_pooling()
            torch.save(cls_pooled, address)
            print(f"CLS-pooled embedding saved at {address}")
        else:
            print(f"CLS-pooled embedding already exists at {address}")

        # Mean Pooling
        mean_pooled_dir = os.path.join(output_dir, "mean_pooled")
        os.makedirs(mean_pooled_dir, exist_ok=True)
        address = os.path.join(mean_pooled_dir, file_name)
        if not os.path.exists(address):
            mean_pooled = pooler.mean_pooling()
            torch.save(mean_pooled, address)
            print(f"Mean-pooled embedding saved at {address}")
        else:
            print(f"Mean-pooled embedding already exists at {address}")

        # Max Pooling
        max_pooled_dir = os.path.join(output_dir, "max_pooled")
        os.makedirs(max_pooled_dir, exist_ok=True)
        address = os.path.join(max_pooled_dir, file_name)
        if not os.path.exists(address):
            max_pooled = pooler.max_pooling()
            torch.save(max_pooled, address)
            print(f"Max-pooled embedding saved at {address}")
        else:
            print(f"Max-pooled embedding already exists at {address}")

    print(f"Pooling operations completed for {file_name}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequence embeddings using Pool PaRTI pooling method.")
    parser.add_argument("--path_token_emb", type=str, required=True, help="Path to the token embeddings file.")
    parser.add_argument("--path_attention_layers", type=str, required=True, help="Path to the attention matrices file.")
    parser.add_argument("--output_dir", type=str, default="./post_pooling_seq_vectors", help="Output directory for the sequence embeddings.")
    parser.add_argument("--generate_all", action='store_true', help="Generate all pooling embeddings (CLS, mean, max) in addition to Pool PaRTI.")

    args = parser.parse_args()
    main(args.path_token_emb, args.path_attention_layers, args.output_dir, args.generate_all)
