import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder


import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances


def evaluate_embeddings_for_dist_ratio(df, uniprot_col, label_col, plm_type, pooling_method, distance_metric="euclidean", 
                        verbose=False, sampling_mode=False, num_iterations=10, sample_size=1000, min_samples_per_class=5):
    def load_embeddings(dataframe):
        embeddings = []
        labels = []
        for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Processing UniProt Accessions", leave=False):
            uniprot_accession = row[uniprot_col]
            label = encoded_labels[index]
            
            if plm_type == "esm":
                embedding_path = f"/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
            else:
                embedding_path = f"/oak/stanford/groups/rbaltman/{plm_type}_embeddings/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
            
            if os.path.exists(embedding_path):
                embedding = torch.load(embedding_path)
                if embedding is not None:
                    try:
                        #embedding = embedding.squeeze().numpy()
                        try:
                            embedding = embedding.squeeze()
                        except:
                            pass
                        try:
                            embedding = embedding.numpy()
                        except:
                            pass
                        if not embedding.shape[0] == 0:
                            embeddings.append(embedding)
                            labels.append(label)  # Only append label if embedding is successfully processed
                    except Exception as e:
                        if verbose:
                            print(f"Error processing embedding for {uniprot_accession}: {e}")
                elif verbose:
                    print(f"Embedding is None for {uniprot_accession}")
            elif verbose:
                print(f"Skipping {uniprot_accession}: Embedding file not found.")

        try:
            embeddings = np.array(embeddings)
        except Exception as e:
            print(f"len(embeddings) {len(embeddings)}")
            for i in range(len(embeddings)):
                print(f"{i} has shape {embeddings[i].shape}")
            raise Exception(e)
            
        return embeddings, np.array(labels)

    def calculate_distances(embeddings, labels):
        if len(embeddings) != len(labels):
            print(f"Mismatch: {len(embeddings)} embeddings but {len(labels)} labels")
            return None
    
        distances = pairwise_distances(embeddings, metric=distance_metric)
        class_weights = {label: 1.0 / np.sum(labels == label) for label in np.unique(labels)}
        
        intra_class_distances = []
        inter_class_distances = []
        
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if labels[i] == labels[j]:
                    intra_class_distances.append(distances[i, j] * class_weights[labels[i]])
                else:
                    inter_class_distances.append(distances[i, j] * (class_weights[labels[i]] + class_weights[labels[j]]) / 2)
        
        if not intra_class_distances or not inter_class_distances:
            print("No intra-class or inter-class distances computed")
            return None
        
        intra_class_mean = np.mean(intra_class_distances)
        inter_class_mean = np.mean(inter_class_distances)
        ratio_of_dist = inter_class_mean / intra_class_mean
        
        return ratio_of_dist

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_col])

    if not sampling_mode:
        embeddings, labels = load_embeddings(df)
        if len(embeddings) < 2:
            print("Not enough valid embeddings found. Aborting metric calculations.")
            return None
        ratio = calculate_distances(embeddings, labels)
        return {"ratio_of_inter_to_intra": ratio}
    else:
        ratios = []
        for _ in range(num_iterations):
            sampled_df = pd.DataFrame()
            for label in np.unique(encoded_labels):
                class_df = df[encoded_labels == label]
                class_size = len(class_df)
                sample_count = max(min(int(class_size * sample_size / len(df)), class_size), min_samples_per_class)
                if class_size < min_samples_per_class:
                    sample_count = class_size
                sampled_class = class_df.sample(n=sample_count, replace=False)
                sampled_df = pd.concat([sampled_df, sampled_class])
            
            embeddings, labels = load_embeddings(sampled_df)
            if len(embeddings) < 2:
                print("Not enough valid embeddings in sample. Skipping this iteration.")
                continue
            ratio = calculate_distances(embeddings, labels)
            ratios.append(ratio)
        
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        return {
            "ratios": ratios,
            "ratio_of_inter_to_intra_mean": mean_ratio,
            "ratio_of_inter_to_intra_std": std_ratio
        }

import os
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pandas as pd

def load_embeddings_and_save_distances(df, uniprot_col, plm_type, pooling_method, output_path, distance_metric="euclidean", verbose=False):
    """
    Loads embeddings for the given UniProt accessions, calculates pairwise distances,
    and saves the resulting distance matrix.

    Parameters:
        df (pd.DataFrame): DataFrame containing UniProt accessions.
        uniprot_col (str): Column name for UniProt accessions.
        plm_type (str): Type of protein language model (e.g., "esm").
        pooling_method (str): Pooling method for embeddings.
        output_path (str): File path to save the pairwise distance matrix.
        distance_metric (str, optional): Metric for distance calculation (default is "euclidean").
        verbose (bool, optional): Print additional information for debugging (default is False).

    Returns:
        None
    """
    def load_embeddings(dataframe):
        embeddings = []
        valid_accessions = []
        for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Loading Embeddings", leave=False):
            uniprot_accession = row[uniprot_col]

            if plm_type == "esm":
                embedding_path = f"/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
            else:
                embedding_path = f"/oak/stanford/groups/rbaltman/{plm_type}_embeddings/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"

            if os.path.exists(embedding_path):
                try:
                    embedding = torch.load(embedding_path).squeeze().numpy()
                    embeddings.append(embedding)
                    valid_accessions.append(uniprot_accession)
                except Exception as e:
                    if verbose:
                        print(f"Error processing {uniprot_accession}: {e}")
            elif verbose:
                print(f"Embedding not found for {uniprot_accession}")

        return np.array(embeddings), valid_accessions

    # Load embeddings
    embeddings, valid_accessions = load_embeddings(df)

    if len(embeddings) < 2:
        raise ValueError("Not enough valid embeddings found for distance calculation.")

    # Calculate pairwise distances
    distance_matrix = pairwise_distances(embeddings, metric=distance_metric)

    # Save distance matrix as a CSV file
    distance_df = pd.DataFrame(distance_matrix, index=valid_accessions, columns=valid_accessions)
    distance_df.to_csv(output_path)

    if verbose:
        print(f"Pairwise distance matrix saved to {output_path}")


import os
import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import pandas as pd

def load_embeddings_to_dict(plm_type, pooling_method, verbose=False, protein_list = None):
    """
    Loads embeddings for the given UniProt accessions into a dictionary.

    Parameters:
        df (pd.DataFrame): DataFrame containing UniProt accessions.
        uniprot_col (str): Column name for UniProt accessions.
        plm_type (str): Type of protein language model (e.g., "esm").
        pooling_method (str): Pooling method for embeddings.
        verbose (bool, optional): Print additional information for debugging (default is False).

    Returns:
        dict: A dictionary with UniProt accessions as keys and embeddings as values.
    """
    embeddings_dict = {}

    if plm_type == "esm":
        embedding_dir = f"/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/{pooling_method}"
    else:
        embedding_dir = f"/oak/stanford/groups/rbaltman/{plm_type}_embeddings/post_pooling_seq_vectors/{pooling_method}"

    if protein_list is None:
        protein_list = os.listdir(embedding_dir)

    print(f"len protein list is {len(protein_list)}")

    for protein_file in tqdm(protein_list, total=len(protein_list)):
        if '.pt' not in protein_file:
            continue

        if 'mut' in protein_file:
            continue
            
        uniprot_accession = protein_file.split('.pt')[0]

        embedding_path = f"{embedding_dir}/{protein_file}"
        
        if os.path.exists(embedding_path):
            try:
                embedding = torch.load(embedding_path).squeeze()
                if embedding.shape[0] != 1280 and plm_type == 'esm':
                    raise Exception(f"{protein_file} has the wrong dimensions of {embedding.shape}")

                if embedding.shape[0] != 1024 and plm_type == 'protbert':
                    raise Exception(f"{protein_file} has the wrong dimensions of {embedding.shape}")
                # Check if the embedding is a PyTorch tensor and convert it to a NumPy array
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.numpy()  # Ensure it's on the CPU before calling .numpy()
                embeddings_dict[uniprot_accession] = embedding
            except Exception as e:
                if verbose:
                    print(f"Error processing {uniprot_accession}: {e}")
        elif verbose:
            print(f"Embedding not found for {uniprot_accession}")

    return embeddings_dict

def compute_and_save_distances(embeddings_dict, output_path_distances, output_path_accessions, distance_metric="euclidean", verbose=False):
    """
    Computes pairwise distances between embeddings and saves the result as numpy objects.

    Parameters:
        embeddings_dict (dict): Dictionary with file names as keys and embeddings as values.
        output_path_distances (str): File path to save the pairwise distance matrix as a numpy object.
        output_path_accessions (str): File path to save the accessions in the correct order as a numpy object.
        distance_metric (str, optional): Metric for distance calculation (default is "euclidean").
        verbose (bool, optional): Print additional information for debugging (default is False).

    Returns:
        None
    """
    if len(embeddings_dict) < 2:
        raise ValueError("Not enough embeddings to calculate distances.")

    accessions = list(embeddings_dict.keys())
    embeddings = np.array(list(embeddings_dict.values()))

    # Calculate pairwise distances
    distance_matrix = pairwise_distances(embeddings, metric=distance_metric)

    # Save distance matrix and accessions as numpy objects
    np.save(output_path_distances, distance_matrix)
    np.save(output_path_accessions, np.array(accessions))

    if verbose:
        print(f"Pairwise distance matrix saved to {output_path_distances}")
        print(f"Accessions saved to {output_path_accessions}")


from scipy.stats import ks_2samp, mannwhitneyu
import numpy as np
import pandas as pd

def compute_group_distance_stats(dist_matrix, accessions, df, accession_col, label_col):
    """
    Computes statistics for intergroup vs intragroup distances.

    Parameters:
        dist_matrix (np.ndarray): Pairwise distance matrix (NumPy array).
        accessions (list): List of accessions corresponding to the indices of the distance matrix.
        df (pd.DataFrame): DataFrame with accessions and their corresponding group labels.
        accession_col (str): Column name for accessions in the DataFrame.
        label_col (str): Column name for group labels in the DataFrame.

    Returns:
        dict: Results of the Kolmogorov-Smirnov and Mann-Whitney U tests.
    """
    # Map accessions to their labels
    accession_to_label = dict(zip(df[accession_col], df[label_col]))

    # Preallocate memory by estimating sizes
    num_entries = len(accession_to_label)
    max_distances = num_entries * (num_entries - 1) // 2  # Upper triangular matrix entries

    intra_group_distances = np.zeros(max_distances, dtype=np.float32)
    inter_group_distances = np.zeros(max_distances, dtype=np.float32)

    intra_index = 0
    inter_index = 0

    # Process upper triangle of the distance matrix
    for i in range(num_entries):
        for j in range(i + 1, num_entries):  # Upper triangular part
            acc_i, acc_j = accessions[i], accessions[j]
            label_i = accession_to_label.get(acc_i)
            label_j = accession_to_label.get(acc_j)

            if label_i is None or label_j is None:
                continue  # Skip if any accession is not found in the DataFrame

            distance = dist_matrix[i, j]
            if label_i == label_j:
                intra_group_distances[intra_index] = distance
                intra_index += 1
            else:
                inter_group_distances[inter_index] = distance
                inter_index += 1

    # Trim unused parts of the arrays
    intra_group_distances = intra_group_distances[:intra_index]
    inter_group_distances = inter_group_distances[:inter_index]

    # Perform statistical tests
    ks_stat, ks_p_value = ks_2samp(intra_group_distances, inter_group_distances, alternative='less')
    mw_stat, mw_p_value = mannwhitneyu(intra_group_distances, inter_group_distances, alternative='less')

    # Return results
    return {
        "ks_test": {"statistic": ks_stat, "p_value": ks_p_value},
        "mann_whitney_u_test": {"statistic": mw_stat, "p_value": mw_p_value}
    }


import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
import random

def compute_subsample_metrics(dist_matrix, accessions, df, accession_col, label_col, subsample_size, subsample_time, K):
    """
    Computes mean reciprocal rank, precision@K, and recall@K for subsampled proteins.

    Parameters:
        dist_matrix (np.ndarray): Pairwise distance matrix (NumPy array).
        accessions (list): List of accessions corresponding to the indices of the distance matrix.
        df (pd.DataFrame): DataFrame with accessions and their corresponding group labels.
        accession_col (str): Column name for accessions in the DataFrame.
        label_col (str): Column name for group labels in the DataFrame.
        subsample_size (int): Number of proteins to subsample from each label category.
        subsample_time (int): Number of subsampling iterations.
        K (int): Top K value for precision and recall.

    Returns:
        dict: Dictionary with averages and standard deviations of MRR, precision@K, and recall@K.
    """
    # Map accessions to their labels
    accession_to_label = dict(zip(df[accession_col], df[label_col]))
    labels = df[label_col].unique()
    
    # Convert accessions to indices for matrix lookup
    accession_to_index = {acc: i for i, acc in enumerate(accessions)}

    # Results storage
    mrr_values = []
    precision_at_k_values = []

    for _ in range(subsample_time):
        # Subsample proteins from each label
        subsampled_indices = []
        for label in labels:
            label_accessions = df[df[label_col] == label][accession_col].tolist()
            subsampled_accessions = random.sample(label_accessions, min(subsample_size, len(label_accessions)))
            subsampled_indices.extend([accession_to_index[acc] for acc in subsampled_accessions])

        # Filter the distance matrix for the subsample
        subsample_dist_matrix = dist_matrix[np.ix_(subsampled_indices, subsampled_indices)]
        subsample_labels = [accession_to_label[accessions[idx]] for idx in subsampled_indices]

        # Compute metrics
        mrr = 0.0
        precision_at_k = 0.0
        
        total_queries = len(subsampled_indices)

        for i in range(total_queries):
            # Get distances and labels for the current query
            distances = subsample_dist_matrix[i]
            true_label = subsample_labels[i]

            # Sort indices by distance (ascending)
            sorted_indices = np.argsort(distances)
            sorted_labels = [subsample_labels[idx] for idx in sorted_indices if idx != i]

            # Calculate MRR
            for rank, label in enumerate(sorted_labels, start=1):
                if label == true_label:
                    mrr += 1 / rank
                    break

            # Calculate Precision@K and Recall@K
            top_k_labels = sorted_labels[:K]
            relevant_count = top_k_labels.count(true_label)
            total_relevant = subsample_labels.count(true_label) - 1  # Exclude the query itself

            precision_at_k += relevant_count / K

        # Normalize metrics by the number of queries
        mrr /= total_queries
        precision_at_k /= total_queries

        # Store metrics for this iteration
        mrr_values.append(mrr)
        precision_at_k_values.append(precision_at_k)

    # Compute averages and standard deviations
    results = {
        "mean_reciprocal_rank": {
            "mean": np.mean(mrr_values),
            "ste": np.std(mrr_values)/len(mrr_values)
        },
        "precision_at_k": {
            "mean": np.mean(precision_at_k_values),
            "ste": np.std(precision_at_k_values)/len(precision_at_k_values)
        }
    }

    return results


import numpy as np
import pandas as pd
import random

def compute_subsample_metrics_multilabel(dist_matrix, accessions, df, accession_col, label_col, subsample_size, subsample_time, K):
    """
    Computes mean reciprocal rank, precision@K for subsampled proteins, 
    considering multi-label overlap for correctness.

    Parameters:
        dist_matrix (np.ndarray): Pairwise distance matrix (NumPy array).
        accessions (list): List of accessions corresponding to the indices of the distance matrix.
        df (pd.DataFrame): DataFrame with accessions and their corresponding group labels (lists).
        accession_col (str): Column name for accessions in the DataFrame.
        label_col (str): Column name for group labels in the DataFrame.
        subsample_size (int): Number of proteins to subsample from each label category.
        subsample_time (int): Number of subsampling iterations.
        K (int): Top K value for precision.

    Returns:
        dict: Dictionary with averages and standard errors of MRR and precision@K.
    """
    # Convert label column into lists if they are stored as strings
    df[label_col] = df[label_col].apply(lambda x: x if isinstance(x, list) else eval(x))  # Ensure it's a list

    # Map accessions to their list of labels
    accession_to_labels = dict(zip(df[accession_col], df[label_col]))

    # Convert accessions to indices for matrix lookup
    accession_to_index = {acc: i for i, acc in enumerate(accessions)}

    # Unique labels
    unique_labels = set(label for labels in df[label_col] for label in labels)

    # Results storage
    mrr_values = []
    precision_at_k_values = []

    for _ in range(subsample_time):
        # Subsample proteins from each label
        subsampled_indices = []
        sampled_accessions = set()
        
        for label in unique_labels:
            # Get all accessions that contain this label
            label_accessions = df[df[label_col].apply(lambda x: label in x)][accession_col].tolist()
            sampled = random.sample(label_accessions, min(subsample_size, len(label_accessions)))
            sampled_accessions.update(sampled)
        
        # Convert sampled accessions to indices
        subsampled_indices = [accession_to_index[acc] for acc in sampled_accessions]

        # Filter the distance matrix for the subsample
        subsample_dist_matrix = dist_matrix[np.ix_(subsampled_indices, subsampled_indices)]
        subsample_labels = [accession_to_labels[accessions[idx]] for idx in subsampled_indices]

        # Compute metrics
        mrr = 0.0
        precision_at_k = 0.0
        
        total_queries = len(subsampled_indices)

        for i in range(total_queries):
            # Get distances and labels for the current query
            distances = subsample_dist_matrix[i]
            true_labels = set(subsample_labels[i])

            # Sort indices by distance (ascending)
            sorted_indices = np.argsort(distances)
            sorted_labels = [set(subsample_labels[idx]) for idx in sorted_indices if idx != i]  # Exclude self

            # Calculate MRR
            for rank, labels in enumerate(sorted_labels, start=1):
                if true_labels & labels:  # Check if there is an overlap in labels
                    mrr += 1 / rank
                    break

            # Calculate Precision@K
            top_k_labels = sorted_labels[:K]
            relevant_count = sum(1 for labels in top_k_labels if true_labels & labels)  # Count intersections

            precision_at_k += relevant_count / K

        # Normalize metrics by the number of queries
        mrr /= total_queries
        precision_at_k /= total_queries

        # Store metrics for this iteration
        mrr_values.append(mrr)
        precision_at_k_values.append(precision_at_k)

    # Compute averages and standard errors
    results = {
        "mean_reciprocal_rank": {
            "mean": np.mean(mrr_values),
            "ste": np.std(mrr_values) / np.sqrt(len(mrr_values))
        },
        "precision_at_k": {
            "mean": np.mean(precision_at_k_values),
            "ste": np.std(precision_at_k_values) / np.sqrt(len(precision_at_k_values))
        }
    }

    return results

