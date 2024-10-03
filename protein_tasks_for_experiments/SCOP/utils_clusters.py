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

def evaluate_embeddings_for_dist_ratio_old(df, 
                                       uniprot_col, 
                                       label_col, 
                                       plm_type, 
                                       pooling_method, 
                                       distance_metric="euclidean", 
                                       verbose=False, 
                                       sampling_mode=False, 
                                       num_iterations=10, 
                                       sample_size=1000, 
                                       min_samples_per_class=5):
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
            

                # Check if the file exists
            if os.path.exists(embedding_path):
                # Load the embedding
                embedding = torch.load(embedding_path)
                labels.append(label)
                if embedding is None:
                    continue
                try:
                    embedding = embedding.squeeze()
                except:
                    pass
                try:
                    embedding = embedding.numpy()
                except:
                    pass
                embeddings.append(embedding)
            elif verbose:
                print(f"Skipping {uniprot_accession}: Embedding file not found.")
                
        
        return np.array(embeddings), np.array(labels)

    def calculate_distances(embeddings, labels):
        distances = pairwise_distances(embeddings, metric=distance_metric)
        class_weights = {label: 1.0 / np.sum(labels == label) for label in np.unique(labels)}
        
        intra_class_distances = []
        inter_class_distances = []
        
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    intra_class_distances.append(distances[i, j] * class_weights[labels[i]])
                else:
                    inter_class_distances.append(distances[i, j] * (class_weights[labels[i]] + class_weights[labels[j]]) / 2)
        
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
            "ratio_of_inter_to_intra_mean": mean_ratio,
            "ratio_of_inter_to_intra_std": std_ratio
        }


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


def evaluate_embeddings_old(df, uniprot_col, label_col, plm_type, pooling_method, distance_metric= "euclidian", verbose=False):
    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    # Encode labels if they are in string form
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_col])
    
    # Iterate over each row in the DataFrame with tqdm for progress tracking
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing UniProt Accessions", leave=False):
        uniprot_accession = row[uniprot_col]
        label = encoded_labels[index]
        
        # Construct the file path
        if plm_type == "esm":
            embedding_path = f"/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
        else:
            embedding_path = f"/oak/stanford/groups/rbaltman/{plm_type}_embeddings/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
        
        # Check if the file exists
        if os.path.exists(embedding_path):
            # Load the embedding
            embedding = torch.load(embedding_path)
            if embedding is None:
                continue
            try:
                embedding = embedding.squeeze()
            except:
                pass
            try:
                embedding = embedding.numpy()
            except:
                pass
            embeddings.append(embedding)
            labels.append(label)
        else:
            if verbose: 
                print(f"Skipping {uniprot_accession}: Embedding file not found.")
    
    # Convert lists to numpy arrays for metric calculations
    try:
        embeddings = np.array(embeddings)
    except:
        print(f"len(embeddings) {len(embeddings)}")
        print(f"len(embeddings[-1]) {len(embeddings[-1])}")
        print(f"len(embeddings[-2]) {len(embeddings[-2])}")
    labels = np.array(labels)
    
    # Check if there are enough embeddings to proceed
    if len(embeddings) < 2:
        print("Not enough valid embeddings found. Aborting metric calculations.")
        return

    # Compute Silhouette Score
    #print("Calculating Silhouette Score...")
    silhouette = silhouette_score(embeddings, labels)
    #print(f"Silhouette Score: {silhouette}")

    # Perform K-means clustering to generate cluster labels
    #print("Clustering embeddings with K-means...")
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute Adjusted Rand Index (ARI)
    #print("Calculating Adjusted Rand Index (ARI)...")
    ari = adjusted_rand_score(labels, cluster_labels)
    #print(f"Adjusted Rand Index: {ari}")

    # Compute Pairwise Distances
    #print("Calculating Pairwise Distances...")
    distances = pairwise_distances(embeddings, metric = distance_metric)
    
    # Initialize lists for intra-class and inter-class distances
    intra_class_distances = []
    inter_class_distances = []
    
    class_weights = {label: 1.0 / np.sum(labels == label) for label in np.unique(labels)}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_class_distances.append(distances[i, j] * class_weights[labels[i]])
            else:
                inter_class_distances.append(distances[i, j] * (class_weights[labels[i]] + class_weights[labels[j]]) / 2)
    
    # Convert lists to numpy arrays
    intra_class_distances = np.array(intra_class_distances)
    inter_class_distances = np.array(inter_class_distances)
    
    # Calculate Weighted Mean Distances
    #print("Calculating Weighted Mean Distances...")
    intra_class_mean = np.mean(intra_class_distances)
    inter_class_mean = np.mean(inter_class_distances)
    ratio_of_dist = inter_class_mean/intra_class_mean
    
    #print(f"Weighted Intra-class Mean Distance: {intra_class_mean}")
    #print(f"Weighted Inter-class Mean Distance: {inter_class_mean}")

    # Return the computed metrics
    return {
        "Silhouette Score": silhouette,
        "Adjusted Rand Index": ari,
        "Weighted Intra-class Mean Distance": intra_class_mean,
        "Weighted Inter-class Mean Distance": inter_class_mean,
        "ratio_of_inter_to_intra": ratio_of_dist
    }



#import os
#import torch
#import numpy as np
#import pandas as pd
#from tqdm import tqdm
#from sklearn.metrics import silhouette_score, adjusted_rand_score, pairwise_distances
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import LabelEncoder

def evaluate_embeddings_old(df, uniprot_col, label_col, plm_type, pooling_method, verbose=False):
    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    # Encode labels if they are in string form
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df[label_col])
    
    # Iterate over each row in the DataFrame with tqdm for progress tracking
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing UniProt Accessions"):
        uniprot_accession = row[uniprot_col]
        label = encoded_labels[index]
        
        # Construct the file path
        if plm_type == "esm":
            embedding_path = f"/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
        else:
            embedding_path = f"/oak/stanford/groups/rbaltman/{plm_type}_embeddings/post_pooling_seq_vectors/{pooling_method}/{uniprot_accession}.pt"
        
        # Check if the file exists
        if os.path.exists(embedding_path):
            # Load the embedding
            embedding = torch.load(embedding_path)
            if embedding is None:
                continue
            try:
                embedding = embedding.squeeze()
            except:
                pass
            try:
                embedding = embedding.numpy()
            except:
                pass
            embeddings.append(embedding)
            labels.append(label)
        else:
            if verbose: 
                print(f"Skipping {uniprot_accession}: Embedding file not found.")
    
    # Convert lists to numpy arrays for metric calculations
    try:
        embeddings = np.array(embeddings)
    except:
        print(f"len(embeddings) {len(embeddings)}")
        print(f"len(embeddings[-1]) {len(embeddings[-1])}")
        print(f"len(embeddings[-2]) {len(embeddings[-2])}")
    labels = np.array(labels)
    
    # Check if there are enough embeddings to proceed
    if len(embeddings) < 2:
        print("Not enough valid embeddings found. Aborting metric calculations.")
        return

    # Compute Silhouette Score
    print("Calculating Silhouette Score...")
    silhouette = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {silhouette}")

    # Perform K-means clustering to generate cluster labels
    print("Clustering embeddings with K-means...")
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Compute Adjusted Rand Index (ARI)
    print("Calculating Adjusted Rand Index (ARI)...")
    ari = adjusted_rand_score(labels, cluster_labels)
    print(f"Adjusted Rand Index: {ari}")

    # Compute Pairwise Distances
    print("Calculating Pairwise Distances...")
    distances = pairwise_distances(embeddings)
    
    # Initialize lists for intra-class and inter-class distances
    intra_class_distances = []
    inter_class_distances = []
    
    class_weights = {label: 1.0 / np.sum(labels == label) for label in np.unique(labels)}

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_class_distances.append(distances[i, j] * class_weights[labels[i]])
            else:
                inter_class_distances.append(distances[i, j] * (class_weights[labels[i]] + class_weights[labels[j]]) / 2)
    
    # Convert lists to numpy arrays
    intra_class_distances = np.array(intra_class_distances)
    inter_class_distances = np.array(inter_class_distances)
    
    # Calculate Weighted Mean Distances
    print("Calculating Weighted Mean Distances...")
    intra_class_mean = np.mean(intra_class_distances)
    inter_class_mean = np.mean(inter_class_distances)
    
    print(f"Weighted Intra-class Mean Distance: {intra_class_mean}")
    print(f"Weighted Inter-class Mean Distance: {inter_class_mean}")

    # Return the computed metrics
    return {
        "Silhouette Score": silhouette,
        "Adjusted Rand Index": ari,
        "Weighted Intra-class Mean Distance": intra_class_mean,
        "Weighted Inter-class Mean Distance": inter_class_mean
    }