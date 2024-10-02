import argparse
import numpy as np
import torch
from token_to_sequence_pooler import TokenToSequencePooler
import os

def main(uniprot_accession):
    parent_dir = "./post_pooling_seq_vectors"
    os.makedirs(parent_dir, exist_ok=True)
    # Instantiate the TokenToSequencePooler with the given UniProt accession
    pooler = TokenToSequencePooler(uniprot_accession=uniprot_accession)

    rep_w_cls = pooler.representations_with_cls
    attn = pooler.attention_matrices
    if not rep_w_cls.shape[0] == attn.shape[1]:
        if len(rep_w_cls.shape) == 3 and not rep_w_cls.shape[1] == attn.shape[1]:
            print(f"The attention and representation shapes don't match for {uniprot_accession}", flush=True)
            return

    # Perform the pooling operations
    os.makedirs(f"{parent_dir}/cls_pooled/", exist_ok=True)
    address = f"{parent_dir}/cls_pooled/{uniprot_accession}.pt"
    if not os.path.exists(address):
        cls_pooled = pooler.cls_pooling()
        torch.save(cls_pooled, address)
        #print(f"saved at {address}")

            

    ##############
    os.makedirs(f"{parent_dir}/mean_pooled/", exist_ok=True)
    address = f"{parent_dir}/mean_pooled/{uniprot_accession}.pt"
    if not os.path.exists(address):
        mean_pooled = pooler.mean_pooling()
        torch.save(mean_pooled, address)
    
    ##############
    os.makedirs(f"{parent_dir}/max_pooled/", exist_ok=True)
    address = f"{parent_dir}/max_pooled/{uniprot_accession}.pt"
    if not os.path.exists(address):
        max_pooled = pooler.max_pooling()
        torch.save(max_pooled, address)

    

    ##############
    os.makedirs(f"{parent_dir}/Pool_PaRTI/", exist_ok=True)
    address = f"{parent_dir}/Pool_PaRTI/{uniprot_accession}.pt"
    if not os.path.exists(address):
        pooled = pooler.pageRank_pooling(which_matrix = "contact",
                                         sigmoid_enhancement = False,
                                         prune_type="top_k_outdegree",
                                        include_cls = False)
        torch.save(pooled, address)

   
    ##############
    

    print(f"Pooling operations completed for UniProt accession {uniprot_accession}.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pooling operations on protein sequence data.")
    parser.add_argument("uniprot_accession", type=str, help="UniProt accession for the protein.")
    
    args = parser.parse_args()
    main(args.uniprot_accession)
