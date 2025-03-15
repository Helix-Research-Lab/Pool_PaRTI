import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import h5py

class ProteinDataset(Dataset):
    def __init__(self, dataframe, args, data_fraction=1):
        self.pooling_method = args.pooling_method
        self.PLM = args.PLM
        if self.PLM == "esm2":
            self.vector_dim = 1280
        elif self.PLM == "protbert":
            self.vector_dim = 1024
        self.rapid_debug_mode = args.rapid_debug_mode
        self.data_frame = dataframe
        if data_fraction < 1:
            self.data_frame = stratified_sampling(self.data_frame, data_fraction)
            if self.rapid_debug_mode:
                print(f"Since data_fraction = {data_fraction}", flush=True)
                print(f"New dataset size is {len(self.data_frame)}", flush=True)
        self.class_weights = self._calculate_class_weights()


    def _calculate_class_weights(self):
        weights = np.zeros(6)
        for i in range(1, 7):
            percentage = len(self.data_frame[self.data_frame['EC_Level_1'] == i])/len(self.data_frame)
            if percentage > 0:
                weights[i-1] = 1/percentage
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        accession = self.data_frame.iloc[idx]['Protein UniProt Acc.']
        label = self.data_frame.iloc[idx]['EC_Level_1'] - 1
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
        #parent_dir_poolings = "../../post_pooling_seq_vectors"
        #parent_dir_poolings = "../../../../../esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/"
        #for protbert

        ## CHANGE THIS PART BASED ON THE LOCATION OF THE EMBEDDINGS

        if self.PLM == 'protbert':
            parent_dir_poolings = "../../../../../protbert_embeddings/post_pooling_seq_vectors/"
        elif self.PLM == 'esm2':
            parent_dir_poolings = "../../../../../esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/"
            

        
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

def get_loader(file_path, shuffle, args, data_fraction = 1):
    dataframe = pd.read_csv(file_path)
    dataset = ProteinDataset(dataframe=dataframe, args=args, data_fraction = data_fraction)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)

def get_loader_with_split(file_path,
                          args,
                          val_fold_index,
                          data_fraction=1
                         ):
    dataframe = pd.read_csv(file_path)

    # Assign subsets reproducibly if total_fold_no > 1
    if args.total_fold_no > 1:
        rng = np.random.default_rng(args.seed)
        subset_indices = rng.integers(1, args.total_fold_no+1, len(dataframe))
        dataframe["cv_index"] = subset_indices

    df_train = dataframe[dataframe['cv_index'] != val_fold_index]
    df_val = dataframe[dataframe['cv_index'] == val_fold_index]

    if args.rapid_debug_mode:
        print(f"len(df_train) {len(df_train)}")
        print(f"len(df_val) {len(df_val)}")
    
    
    dataset_train = ProteinDataset(df_train, args, data_fraction=data_fraction)

    dataset_val = ProteinDataset(df_val, args, data_fraction=data_fraction)

    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  collate_fn=custom_collate_fn,
                                 drop_last=True)
    dataloader_val = DataLoader(dataset_val, 
                                  batch_size=args.batch_size, 
                                  shuffle=False, 
                                  collate_fn=custom_collate_fn,
                               drop_last=True)

    print(f"number of batches in dataloader_train is {len(dataloader_train)}", flush=True)
    print(f"number of batches in dataloader_val is {len(dataloader_val)}", flush=True)
    
    return dataloader_train, dataloader_val






class ProteinDatasetLightAttention(Dataset):
    def __init__(self, dataframe, h5_path, args, data_fraction=1):
        """
        Dataset that loads protein embeddings from an HDF5 (.h5) file.

        Args:
            dataframe (pd.DataFrame): Dataframe containing protein accessions and labels.
            h5_path (str): Path to the HDF5 file storing embeddings.
            rapid_debug_mode (bool): Whether to print debug information.
            PLM (str): Pre-trained language model type ('esm2' or 'protbert').
        """
        self.PLM = args.PLM
        self.vector_dim = 1280 if self.PLM == "esm2" else 1024
        self.rapid_debug_mode = args.rapid_debug_mode
        self.data_frame = dataframe
        self.h5_file = h5py.File(h5_path, 'r')  # Keep the HDF5 file open to minimize I/O overhead
        if data_fraction < 1:
            self.data_frame = stratified_sampling(self.data_frame, data_fraction)
            if self.rapid_debug_mode:
                print(f"New dataset size is {len(self.data_frame)}", flush=True)
        self.class_weights = self._calculate_class_weights()
        

    def _calculate_class_weights(self):
        weights = np.zeros(6)
        for i in range(1, 7):
            percentage = len(self.data_frame[self.data_frame['EC_Level_1'] == i])/len(self.data_frame)
            if percentage == 0:
                continue
            weights[i-1] = 1/percentage
        return torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        accession = self.data_frame.iloc[idx]['Protein UniProt Acc.']
        label = self.data_frame.iloc[idx]['EC_Level_1'] - 1
        embedding = self._load_embedding(accession)
        if embedding is None:
            if self.rapid_debug_mode:
                print(f"None encountered for accession {accession}")
            return None
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.float()
        else:
            embedding = torch.from_numpy(embedding).float()
        return embedding, torch.tensor(label, dtype=torch.long)

    def _load_embedding(self, accession):
        """Load embedding from HDF5 file and apply pooling method."""
        if accession not in self.h5_file:
            if self.rapid_debug_mode:
                print(f"Embedding for {accession} not found in HDF5 file")
            return None

        embedding = self.h5_file[accession][:]  # Load embedding as NumPy array

        if embedding is None:
            return None
        embedding = torch.tensor(embedding, dtype=torch.float32)  # Convert to PyTorch tensor


        # Check for NaNs
        if torch.isnan(embedding).any():
            if self.rapid_debug_mode:
                print(f"NaN values found in embedding for {accession}")
            return None

        # Ensure tensor is 1D and matches expected size
        if embedding.shape[-1] != self.vector_dim or embedding.shape[-2] < 20 :
            if self.rapid_debug_mode:
                print(f"Embedding size mismatch for {accession}: expected {self.vector_dim}, got {embedding.shape}")
            return None

        return embedding.squeeze()

    def __del__(self):
        """Ensure HDF5 file is closed when the dataset is deleted."""
        self.h5_file.close()



def get_loader_LA(file_path, h5_path, args, shuffle, drop_last=False, data_fraction=1):
    """
    Loads the dataset from a CSV file and initializes a DataLoader.

    Args:
        file_path (str): Path to the CSV file containing protein data.
        h5_path (str): Path to the HDF5 file storing embeddings.
        PLM (str): Pre-trained language model ('esm2' or 'protbert').
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        rapid_debug_mode (bool): Whether to print debug information.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataframe = pd.read_csv(file_path)

    dataset = ProteinDatasetLightAttention(
        dataframe=dataframe,
        h5_path=h5_path,
        args=args,
        data_fraction=data_fraction
    )



    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=pad_collate_fn, drop_last=drop_last)


def get_loader_with_split_LA(file_path, h5_path, val_fold_index, args, data_fraction=1):
    """
    Loads the dataset from a CSV file, splits into training and validation, and initializes DataLoaders.

    Args:
        file_path (str): Path to the CSV file containing protein data.
        h5_path (str): Path to the HDF5 file storing embeddings.
        PLM (str): Pre-trained language model ('esm2' or 'protbert').
        batch_size (int): Batch size for DataLoader.
        val_fold_index (int): Fold index for validation split.
        total_fold_no (int): Number of total folds for cross-validation.
        random_seed (int): Random seed for reproducibility.
        rapid_debug_mode (bool): Whether to print debug information.

    Returns:
        tuple: (train DataLoader, validation DataLoader)
    """
    dataframe = pd.read_csv(file_path)

    # Assign cross-validation indices if more than 1 fold
    if args.total_fold_no > 1:
        rng = np.random.default_rng(args.seed)
        subset_indices = rng.integers(1, args.total_fold_no+1, len(dataframe))
        dataframe["cv_index"] = subset_indices

    # Split dataset into training and validation sets
    df_train = dataframe[dataframe['cv_index'] != val_fold_index]
    df_val = dataframe[dataframe['cv_index'] == val_fold_index]

    if args.rapid_debug_mode:
        print(f"Training set size: {len(df_train)}")
        print(f"Validation set size: {len(df_val)}")

    # Initialize datasets
    dataset_train = ProteinDatasetLightAttention(
        dataframe=df_train,
        h5_path=h5_path,
        args=args,
        data_fraction=data_fraction
    )

    dataset_val = ProteinDatasetLightAttention(
        dataframe=df_val,
        h5_path=h5_path,
        args=args,
        data_fraction=data_fraction
    )

    # Create DataLoaders
    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  collate_fn=pad_collate_fn,
                                  drop_last=True,
                                  num_workers=0, 
                                  pin_memory=True)

    dataloader_val = DataLoader(dataset_val, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                collate_fn=pad_collate_fn,
                                drop_last=True,
                                num_workers=0, 
                                pin_memory=True)

    print(f"Number of batches in training DataLoader: {len(dataloader_train)}", flush=True)
    print(f"Number of batches in validation DataLoader: {len(dataloader_val)}", flush=True)
    
    return dataloader_train, dataloader_val


def pad_collate_fn(batch):
    batch = [b for b in batch if b is not None]  # Remove None entries

    embeddings, labels = zip(*batch)
    embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]

    # Pad sequences to the max length in batch
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)

    # Correctly create a mask: [batch_size, sequence_length]
    mask = (padded_embeddings.sum(dim=-1) != 0).float()  # Sum across embedding dimension

    # Ensure correct shape for broadcasting: [batch_size, 1, sequence_length]
    mask = mask.unsqueeze(1)

    #print(f"Mask shape that comes out of padding function is {mask.shape}")

    #return padded_embeddings, mask, torch.tensor(labels, dtype=torch.long)
    return padded_embeddings, mask, torch.tensor(np.array(labels), dtype=torch.long)



def pad_collate_fn_old(batch):
    """
    Pads protein embeddings in a batch and generates a mask.
    """
    batch = [b for b in batch if b is not None]  # Remove None entries

    embeddings, labels = zip(*batch)
    embeddings = [torch.tensor(e) for e in embeddings]

    # Pad sequences to the max length in batch
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0)

    # Create a mask (1 for real values, 0 for padding)
    mask = (padded_embeddings != 0).float()

    # Ensure mask has shape [batch_size, 1, sequence_length]
    mask = mask.permute(0, 2, 1)  # Swap dimensions to match attention shape

    return padded_embeddings, mask, torch.tensor(labels)


def stratified_sampling(df, sample_fraction, class_column = 'EC_Level_1', min_samples_per_class = 1):
    """
    Perform stratified sampling to retain a specified percentage of rows while preserving class distribution.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        class_column (str): The column representing class labels (e.g., 'EC_Level_1').
        sample_fraction (float): The fraction of rows to retain (between 0 and 1).
        min_samples_per_class (int, optional): The minimum number of samples to retain per class. Defaults to 1.
    
    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    sampled_dfs = []
    
    # Get unique classes
    unique_classes = df[class_column].unique()
    
    for cls in unique_classes:
        # Subset data for the current class
        class_df = df[df[class_column] == cls]
        
        # Determine the number of samples to retain
        n_samples = max(min_samples_per_class, int(len(class_df) * sample_fraction))
        
        # Sample the rows randomly
        sampled_class_df = class_df.sample(n=n_samples, random_state=42, replace=False)
        
        sampled_dfs.append(sampled_class_df)
    
    # Concatenate sampled data
    sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
    
    return sampled_df
