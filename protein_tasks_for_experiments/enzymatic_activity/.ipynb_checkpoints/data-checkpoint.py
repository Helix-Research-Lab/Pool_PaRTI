import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ProteinDataset(Dataset):
    def __init__(self, file_path, pooling_method, rapid_debug_mode, exponent_class_weight=1, vector_dim=1280):
        self.pooling_method = pooling_method
        self.file_path = file_path
        self.vector_dim = vector_dim
        self.rapid_debug_mode = rapid_debug_mode
        if self.rapid_debug_mode:
            data_frame = pd.read_csv(file_path)
            self.data_frame = data_frame.iloc[np.random.randint(0, len(data_frame), 50)]
        else:
            self.data_frame = pd.read_csv(file_path)
        self.exponent_class_weight = exponent_class_weight
        self.class_weights = self._calculate_class_weights()

    def _calculate_class_weights(self):
        positive = self.data_frame['isEnzyme'].sum()
        total = len(self.data_frame)
        negative = total - positive
        if negative < 0.5:
            print(f"division by zero for {self.file_path}", flush=True)
            print(f"self.data_frame['isEnzyme'].sum() {self.data_frame['isEnzyme'].sum()}", flush=True)
            print(f"len(self.data_frame) {len(self.data_frame)}", flush=True)
        weight_for_0 = (total / negative).item()
        weight_for_1 = (total / positive).item()
        return torch.pow(torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32), self.exponent_class_weight)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        accession = self.data_frame.iloc[idx]['Protein UniProt Acc.']
        label = self.data_frame.iloc[idx]['isEnzyme']
        embedding = self._load_embedding(accession)
        if embedding is None:
            if self.rapid_debug_mode:
                print(f"None encountered for accession {accession}")
            return None
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.float()
        else:
            embedding = torch.from_numpy(embedding).float()
        return embedding, torch.tensor(label, dtype=torch.float32)

    def _load_embedding(self, accession):
        parent_dir_poolings = "../../post_pooling_seq_vectors"
        embedding_path = f"{parent_dir_poolings}/{self.pooling_method}/{accession}.pt"
        representation_path = f"../../token_embedding_generator/representation_matrices/{accession}.pt"
        
        
        # Apply cls or sep logic based on embeddings_path
        if self.pooling_method == "cls":
            if not os.path.exists(representation_path):
                if self.rapid_debug_mode:
                    print(f"{embedding_path} doesn't exist")
                return None
            embedding = torch.load(representation_path)
            if len(embedding.shape) == 3:
                embedding = embedding[:,0,:].squeeze()
            else:
                embedding = embedding[0]
        elif self.pooling_method == "sep":
            if not os.path.exists(representation_path):
                if self.rapid_debug_mode:
                    print(f"{embedding_path} doesn't exist")
                return None
            embedding = torch.load(representation_path)
            if len(embedding.shape) == 3:
                embedding = embedding[:,-1,:].squeeze()
            else:
                embedding = embedding[-1]
        elif not os.path.exists(embedding_path):
            if self.rapid_debug_mode:
                print(f"{embedding_path} doesn't exist")
            return None
        else:
            embedding = torch.load(embedding_path)

        if embedding is None:
            return None

        # Universal NaN check for both Torch tensors and NumPy arrays
        try:
            if isinstance(embedding, torch.Tensor):
                if torch.isnan(embedding).any():
                    if self.rapid_debug_mode:
                        print("NaN values found in Torch tensor.")
                    return None
            elif isinstance(embedding, np.ndarray):
                if np.isnan(embedding).any():
                    if self.rapid_debug_mode:
                        print("NaN values found in NumPy array.")
                    return None
        except Exception as e:
            print(f"Error checking for NaNs: {e}", flush=True)
            return None

        try:
            if len(embedding.shape) > 1:
                embedding = embedding.squeeze()  # Ensure the tensor is 1D
        except Exception as e:
            print(f"after trying to squeeze in data.py {e}", flush=True)
            return None


        # Check if embedding has the exact size of 1280
        if embedding.shape[0] == self.vector_dim and len(embedding.shape) == 1:
            return embedding
        else:
            if self.rapid_debug_mode:
                print(f"Embedding size mismatch for accession: {accession}, expected 1280, got {embedding.shape[0]}")
            return None

        

        return embedding

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    data, labels = zip(*batch)
    return torch.stack(data), torch.stack(labels)

def get_loader(file_path, pooling_method, batch_size, shuffle, exponent_class_weight=1, rapid_debug_mode=False):
    dataset = ProteinDataset(file_path=file_path, pooling_method=pooling_method, exponent_class_weight=exponent_class_weight, rapid_debug_mode=rapid_debug_mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)
