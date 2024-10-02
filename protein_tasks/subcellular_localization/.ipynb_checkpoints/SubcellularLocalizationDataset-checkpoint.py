import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

class SubcellularLocalizationDataset(Dataset):
    def __init__(self, dataframe, embedding_dict):
        self.dataframe = dataframe
        self.embedding_dict = embedding_dict
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        acc = self.dataframe.iloc[idx]['ACC']
        if acc in self.embedding_dict:
            embedding = self.embedding_dict[acc]
        else:
            #raise Exception(f"The following protein doesn't seem to have pooled embeddings generated {acc}")
            print(f"The following protein doesn't seem to have pooled embeddings generated {acc}")
            return None, None
        labels = self.dataframe.iloc[idx][1:].to_numpy().astype(float)  # Assuming labels start from 3rd column
        #labels = self.dataframe.iloc[idx][2:].to_numpy().astype(float)  # Assuming labels start from 3rd column
        labels = torch.tensor(labels, dtype=torch.float32)
        return embedding, labels

def preload_embeddings(pooling_method, parent_dir_poolings=None, rapid_debug_mode=False):
    embedding_dict = {}
    if parent_dir_poolings is None:
        parent_dir_poolings = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors"
    embedding_dir = os.path.join(parent_dir_poolings, pooling_method)
    #file_list = os.listdir(embedding_dir)
    proteins_to_load = np.load('proteins_to_load.npy', allow_pickle=True)

    if rapid_debug_mode:
        print("rapid_debug_mode active")
        
        #file_list = file_list[0:20]
        #print(f"file_list {file_list[0:20]}", flush=True)
        for file_index in tqdm(range(len(proteins_to_load))):
           
            acc = proteins_to_load[file_index]
            if acc not in proteins_to_load:
                print(f"avoided including {acc} because its absent in some pooling methods")
                continue
            file_name  = f"{acc}.pt"
            if os.path.exists(f"{embedding_dir}/{file_name}"):
                accession = file_name[:-3]  # Remove file extension
                embedding_path = os.path.join(embedding_dir, file_name)
                embedding = torch.load(embedding_path)
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.from_numpy(embedding).to(torch.float32).squeeze()
                
                #print("embedding", embedding, flush=True)
                embedding_dict[accession] = embedding
        print(f"len(embedding_dict) {len(embedding_dict)}")

        #print(f"embedding dict\n{embedding_dict}", flush=True)
        
    else:
        for file_index in range(len(proteins_to_load)):
            acc = proteins_to_load[file_index]
            if acc not in proteins_to_load:
                continue
            file_name  = f"{acc}.pt"
            if os.path.exists(f"{embedding_dir}/{file_name}"):
                accession = file_name[:-3]  # Remove file extension
                embedding_path = os.path.join(embedding_dir, file_name)
                embedding = torch.load(embedding_path)
                if not isinstance(embedding, torch.Tensor):
                    embedding = torch.from_numpy(embedding).to(torch.float32).squeeze()
                
                #print("embedding", embedding, flush=True)
                embedding_dict[accession] = embedding
    
    
    return embedding_dict 
