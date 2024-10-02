import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pandas as pd
from tqdm import tqdm
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, pos_file_path, neg_file_path, pooling_method, rapid_debug_mode, embedding_dict, exponent_class_weight=1):
        self.pooling_method = pooling_method
        self.rapid_debug_mode = rapid_debug_mode
        pos_pairs = self._load_data_pairs(pos_file_path, label=1)
        neg_pairs = self._load_data_pairs(neg_file_path, label=0)
        self.data_pairs = pos_pairs + neg_pairs
        if self.rapid_debug_mode:
            print(f"{pos_file_path} data size is ", len(self.data_pairs))
        random.shuffle(self.data_pairs)  # Shuffle to mix positive and negative examples
        self.exponent_class_weight = exponent_class_weight
        self.class_weights = self._calculate_class_weights()
        self.embedding_dict = embedding_dict
        

    def _calculate_class_weights(self):
        label_count = torch.tensor([pair[2] for pair in self.data_pairs])
        positive = torch.sum(label_count).float()
        total = torch.tensor(len(label_count)).float()
        negative = total - positive
        weight_for_0 = (total / negative).item()
        weight_for_1 = (total / positive).item()
        return torch.pow(torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32), self.exponent_class_weight)


    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        acc1, acc2, label = self.data_pairs[idx]

        try:
            embedding1 = self.embedding_dict.get(acc1)
            embedding2 = self.embedding_dict.get(acc2)
        except:
            embedding1 = None
            embedding2 = None
        if embedding1 is None or embedding2 is None:
            if self.rapid_debug_mode:
                print("None encountered")
                print(f"acc1 {acc1} -- embedding1\n{embedding1}")
                print(f"acc2 {acc2} -- embedding1\n{embedding2}")
            return None  # We will handle None in the custom collate_fn
        if self.rapid_debug_mode:
            pass
            #print("successful pair", flush=True)
            # Checking and printing the type of embeddings if they're not None

        # Convert numpy arrays to torch tensors if not already in that format
        if not isinstance(embedding1, torch.Tensor):
            embedding1 = torch.from_numpy(embedding1)
        if not isinstance(embedding2, torch.Tensor):
            embedding2 = torch.from_numpy(embedding2)
    
        # Ensure the tensors are of float type for torch.cat operation
        embedding1 = embedding1.float()
        embedding2 = embedding2.float()

        # Squeeze only if necessary
        if embedding1.dim() > 1:
            embedding1 = embedding1.squeeze()
        if embedding2.dim() > 1:
            embedding2 = embedding2.squeeze()

        if self.rapid_debug_mode:
            pass
            #print(f"shape of embedding1: {embedding1.shape}")
            #print(f"shape of embedding2: {embedding2.shape}")

        if embedding1 is None or embedding2 is None: 
            if self.rapid_debug_mode:
                #print("None encountered")
                #print(f"acc1 {acc1} -- embedding1\n{embedding1}")
                #print(f"acc2 {acc2} -- embedding1\n{embedding2}")
                pass
            return None  # We will handle None in the custom collate_fn
        elif torch.isnan(embedding1).any() or torch.isnan(embedding2).any():
            if self.rapid_debug_mode:
                #print("None encountered")
                #print(f"acc1 {acc1} -- embedding1\n{embedding1}")
                #print(f"acc2 {acc2} -- embedding1\n{embedding2}")
                pass
            return None  # We will handle None in the custom collate_fn

        
        try:   
            return torch.cat([embedding1, embedding2], dim=0), torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"{e} for acc1 {acc1} and for acc2 {acc2}", flush=True)
            print(f"shape of embedding1: {embedding1.shape} for acc {acc1}", flush=True)
            print(f"shape of embedding2: {embedding2.shape} for acc {acc2}", flush=True)
            if self.rapid_debug_mode:
                print(f"shape of embedding1: {embedding1.shape} for acc {acc1}", flush=True)
                print(f"shape of embedding2: {embedding2.shape} for acc {acc2}", flush=True)
            return None
            
            

    def _load_data_pairs(self, file_path, label):
        df = pd.read_csv(file_path)
        if self.rapid_debug_mode:
            #df = df.head(400)  # Limit rows for debugging
            pass
        accessions = df[['Protein1', 'Protein2']].values.tolist()
        data_pairs = [(acc1, acc2, label) for acc1, acc2 in accessions]
        return data_pairs

    


def preload_embeddings(pooling_method, parent_dir_poolings=None, rapid_debug_mode=False):
    embedding_dict = {}
    if parent_dir_poolings is None:
        parent_dir_poolings = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors"
    embedding_dir = os.path.join(parent_dir_poolings, pooling_method)
    #file_list = os.listdir(embedding_dir)
    file_list = np.load("PPI_protein_list.npy")

    proteins_to_avoid = np.load('proteins_to_avoid.npy')

    if rapid_debug_mode:
        print("rapid_debug_mode active")
        
        #file_list = file_list[0:20]
        #print(f"file_list {file_list[0:20]}", flush=True)
        for file_index in tqdm(range(len(file_list))):
            acc = file_list[file_index]
            if acc in proteins_to_avoid:
                print(f"avoided including {acc} because its absent in some pooling methods")
                continue
            file_name  = f"{acc}.pt"
            if os.path.exists(f"{embedding_dir}/{file_name}"):
                accession = file_name[:-3]  # Remove file extension
                embedding_path = os.path.join(embedding_dir, file_name)
                embedding = torch.load(embedding_path)
                #print("embedding", embedding, flush=True)
                embedding_dict[accession] = embedding
        print(f"len(embedding_dict) {len(embedding_dict)}")

        #print(f"embedding dict\n{embedding_dict}", flush=True)
        
    else:
        for file_index in range(len(file_list)):
            acc = file_list[file_index]
            if acc in proteins_to_avoid:
                continue
            file_name  = f"{acc}.pt"
            if os.path.exists(f"{embedding_dir}/{file_name}"):
                accession = file_name[:-3]  # Remove file extension
                embedding_path = os.path.join(embedding_dir, file_name)
                embedding = torch.load(embedding_path)
                #print("embedding", embedding, flush=True)
                embedding_dict[accession] = embedding
    
    
    return embedding_dict       



def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    data, labels = zip(*batch)
    #print(f"after filtering len(data) {len(data)}")
    return torch.stack(data), torch.stack(labels)

def get_loader(pos_file_path, neg_file_path, pooling_method, batch_size, shuffle, embedding_dict, exponent_class_weight=1, rapid_debug_mode = False):
    dataset = ProteinDataset(pos_file_path = pos_file_path, neg_file_path = neg_file_path, 
                             pooling_method = pooling_method, exponent_class_weight = exponent_class_weight, 
                             embedding_dict=embedding_dict,
                             rapid_debug_mode = rapid_debug_mode)
    if rapid_debug_mode:
        print(f"{pos_file_path } dataset size is {dataset.__len__()}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn, drop_last=True)
