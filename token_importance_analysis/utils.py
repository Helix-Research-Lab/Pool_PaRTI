import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Import colormap library
import requests
import networkx as nx
from scipy.stats import hypergeom
import torch
import math


def plot_importance_distribution(matrix_of_importances, labels, title=None, fig_file_name=None):
    plt.figure(figsize=(12, 6))

    num_categories = len(labels)
    array_length = len(matrix_of_importances[0])
    for i in range(num_categories):
        x_vals = np.random.normal(i+1, 0.05, size=array_length)
        imp_array = matrix_of_importances[i]
        plt.scatter(x_vals, imp_array, alpha=0.6, label=labels[i])
    
    # Set font sizes
    axes_label_fontsize = 16
    title_fontsize = 16
    legend_fontsize = 16
    ticks_fontsize = 16
    
    # Set axes labels with increased font size
    plt.xlabel('\nImportance Type', fontsize=axes_label_fontsize)
    plt.ylabel('Importance Weights', fontsize=axes_label_fontsize)
    
    # Set legend with increased font size
    #plt.legend(fontsize=legend_fontsize)
    plt.xticks(np.arange(1, num_categories + 1, 1), labels, fontsize=ticks_fontsize, rotation=80)
    # Set tick labels with increased font size
    plt.yticks(fontsize=ticks_fontsize)


    # Set title with increased font size
    if not title == None:
        plt.title(title, fontsize=title_fontsize)
    
    # Save the figure
    if not fig_file_name == None:
        plt.savefig(fig_file_name, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()




def get_ranked_importance_dataframe(importance_array, k=10, highest=True, features_dict = None):
    """
    Returns a DataFrame with the rankings of the highest or lowest values in a NumPy array,
    along with their corresponding indices.

    Args:
        importance_array: The NumPy array containing the importance values.
        num_top_values: The number of top values to include in the DataFrame.
            Defaults to 10.
        highest: A boolean flag indicating whether to return the highest (True) or lowest (False) values.

    Returns:
        A Pandas DataFrame with columns "rank", "value", and "index".
    """

    if highest:
        top_indices = np.argsort(-importance_array)[:k]
    else:
        top_indices = np.argsort(importance_array)[:k]

    df = pd.DataFrame({
        "rank": np.arange(1, k + 1),
        "value": importance_array[top_indices],
        "index": top_indices + 1
    })

    if not highest:
        df = df.iloc[::-1]  # Reverse the order for lowest values

    return df

def plot_attention_mtx(matrix, title=None, fig_file_name=None, xlims=None):
    # Set font sizes
    axes_label_fontsize = 16
    title_fontsize = 16
    legend_fontsize = 16
    ticks_fontsize = 16
    
    plt.figure(figsize=(12, 12))
    # Choose a colormap (explore options at https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    cmap = cm.magma  # Example: Coolwarm colormap
    
    # Add the colormap to imshow
    plt.imshow(matrix, cmap=cmap)
    # Create a colorbar
    colorbar = plt.colorbar()
    
    # Set axes labels with increased font size
    plt.xlabel('\nResidue position', fontsize=axes_label_fontsize)
    plt.ylabel('Residue position', fontsize=axes_label_fontsize)
    
    
    
    # Set legend with increased font size
    #plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    # Set tick labels with increased font size
    plt.yticks(fontsize=ticks_fontsize)
    if xlims is not None:
        plt.xlim(xlims)

    # Set title with increased font size
    if not title == None:
        plt.title(title, fontsize=title_fontsize)
    
    # Save the figure
    if not fig_file_name == None:
        plt.savefig(fig_file_name, dpi=300, bbox_inches='tight')

def get_top_k_in_attn_matrix(array, k, highest=True):
    """
    Extracts the top k highest or lowest elements from a 2D numpy array and returns them as a DataFrame.

    Args:
        array: A 2D numpy array.
        k: The number of elements to extract.
        highest: A boolean flag indicating whether to extract the highest (True) or lowest (False) elements.

    Returns:
        A pandas DataFrame containing the following columns:
            value: The element value.
            row: The row index of the element.
            col: The column index of the element.
    """

    mask = np.triu(np.ones_like(array))
    masked_array = array * mask

    flat_array = masked_array.flatten()
    if highest:
        sorted_indices = np.argsort(flat_array)[::-1][:k]
    else:
        sorted_indices = np.argsort(flat_array)[:k]

    rows, cols = np.unravel_index(sorted_indices, masked_array.shape)
    values = flat_array[sorted_indices]

    return pd.DataFrame({"value": values, "res 1": rows+1, "res 2": cols+1})

def mask_array(array, masking_range):
    """
    Masks a 2D numpy array by setting to zero all entries whose indices are within masking_range from one another.
    
    Args:
    array: A 2D numpy array.
    masking_range: The distance from the diagonal to zero out entries.
    
    Returns:
    A 2D numpy array with the masked entries set to zero.
    """
    mask = np.ones_like(array)
    rows, cols = array.shape
    for i in range(rows):
        for j in range(cols):
          if abs(i - j) <= masking_range:
            mask[i, j] = 0
    return array * mask


def get_protein_features(uniprot_accession):
    """
    Fetches the features data of a protein for a given UniProt accession code.

    Args:
    uniprot_accession (str): The UniProt accession code.

    Returns:
    dict: A dictionary containing the features data of the protein.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_accession}.json"
    #url = f"https://rest.uniprot.org/uniprotkb/v2//features"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for UniProt accession {uniprot_accession}. Status code: {response.status_code}")
        return None

def extract_tuples_by_type(array_of_dicts):
    """
    Extracts tuples of (start_value, end_value) for each type of associated value
    from an array of dictionaries.

    Args:
        array_of_dicts (list): A list of dictionaries, where each dictionary has
            a "type" key and an associated value.

    Returns:
        dict: A dictionary where keys are associated values and values are lists
            of tuples (start_value, end_value).
    """

    result = {}
    for item in array_of_dicts:
        item_type = item["type"]
        try:
            start_value = item["location"]["start"]["value"]
            end_value = item["location"]["end"]["value"]
            result.setdefault(item_type, []).append((start_value, end_value))
        except KeyError:
            # Handle missing keys gracefully
            pass

    return result

def compile_importances_by_feature_types(importance_array, dict_feature_tuple):
    """
    Compiles sub-arrays from an importance array based on inclusive start and end indices
    provided in a dictionary of tuples.
    
    Args:
      importance_array: The array containing the importance scores.
      dict_feature_tuple: A dictionary where keys are strings and values are lists of tuples
          of inclusive start and end indices (e.g., [(4, 9), (12, 15)]).
    
    Returns:
      A dictionary where keys are the same as the input dictionary and values are
      sub-arrays compiled from the importance array based on the corresponding indices.
    """
    
    compiled_data = {}
    for key, index_tuples in dict_feature_tuple.items():
        compiled_data[key] = []
        for indices in index_tuples:
            start, end = indices
            # Ensure inclusive indexing by adding 1 to the end index
            compiled_data[key].extend(importance_array[start - 1:end])
    return compiled_data


import numpy as np
from scipy.stats import rankdata

def importances_to_percentiles(importance_array):
    # Compute the ranks in descending order (highest value gets lowest rank)
    ranks = rankdata(-importance_array, method='average')  # Negate for descending order
    
    # Convert ranks to percentiles (0-100 scale)
    percentile_array = (ranks - 1) / (len(importance_array) - 1) * 100
    
    return percentile_array





def compile_percentiles_by_feature_types(percentile_array, dict_feature_tuple, dict_features_percentiles, verbose=False):
    """
    Compiles sub-arrays from an importance array based on inclusive start and end indices
    provided in a dictionary of tuples.
    
    Args:
      importance_array: The array containing the importance scores.
      dict_feature_tuple: A dictionary where keys are strings and values are lists of tuples
          of inclusive start and end indices (e.g., [(4, 9), (12, 15)]).
    
    Returns:
      A dictionary where keys are the same as the input dictionary and values are
      sub-arrays compiled from the importance array based on the corresponding indices.
    """

    irrelevant_features = ['Chain', 'Region', ]
    
    for key, index_tuples in dict_feature_tuple.items():
        if key in irrelevant_features:
            continue
        if key not in dict_features_percentiles:
            dict_features_percentiles[key] = []
        for indices in index_tuples:
            start, end = indices
            # Ensure inclusive indexing by adding 1 to the end index
            dict_features_percentiles[key].extend(percentile_array[start - 1:end])
    return dict_features_percentiles


def plot_imp_distribution_by_feature(data_dict, title=None, file_name_for_plot=None):
    """
    Plots the distribution of values in a scatter plot.

    Parameters:
    data_dict (dict): A dictionary where each key is associated with an array of values.
    """
    plt.figure(figsize=(12, 6))
    # Set font sizes
    axes_label_fontsize = 16
    title_fontsize = 16
    legend_fontsize = 16
    ticks_fontsize = 16
    #print(f"data_dict\n {data_dict}\n")
    # For each key in the dictionary
    for i, (key, values) in enumerate(data_dict.items(), 1):
        # Creating an x-coordinate for each value in the array
        x_values = np.random.normal(i, 0.1, size=len(values))
        #print(f'x_values  {x_values}, \nvalues  {values}')
        # Scatter plot for each category
        plt.scatter(x_values, values, alpha=0.6, label=key)

    # Set the x-ticks to correspond to the dictionary keys
    plt.xticks(range(1, len(data_dict) + 1), data_dict.keys(), fontsize=ticks_fontsize, rotation=80)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('\nCategories', fontsize=axes_label_fontsize)
    plt.ylabel('Values', fontsize=axes_label_fontsize)
    #plt.legend()
    plt.grid(True)
    if not title == None:
        plt.title(title, fontsize=title_fontsize)
    # Save the figure
    if not file_name_for_plot == None:
        plt.savefig(file_name_for_plot, dpi=300, bbox_inches='tight')
    plt.show()


def plot_imp_distribution_by_feature_with_boxplot(data_dict, title=None, file_name_for_plot=None, rotation=45, y_axis="Values"):
    """
    Enhances the scatter plot with box plots to show the distribution of values more clearly.
    The box plot interiors are made almost transparent to ensure visibility of the scatter dots.

    Parameters:
    data_dict (dict): A dictionary where each key is associated with an array of values.
    """
    plt.figure(figsize=(12, 6))
    # Set font sizes
    axes_label_fontsize = 16
    title_fontsize = 20
    legend_fontsize = 14
    ticks_fontsize = 14

    # Create box plot data
    box_plot_data = [values for values in data_dict.values()]

    # Create box plots with almost transparent interiors
    plt.boxplot(box_plot_data, positions=range(1, len(data_dict) + 1), widths=0.6, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='blue', alpha=0.2),  # Adjust alpha for transparency
                medianprops=dict(color='red'), whiskerprops=dict(color='blue'), 
                capprops=dict(color='blue'), flierprops=dict(marker='o', color='blue', alpha=0.5))

    # Overlay scatter plots
    for i, (key, values) in enumerate(data_dict.items(), 1):
        x_values = np.random.normal(i, 0.1, size=len(values))
        plt.scatter(x_values, values, alpha=0.6, label=key)

    # Set the x-ticks to correspond to the dictionary keys
    plt.xticks(range(1, len(data_dict) + 1), data_dict.keys(), fontsize=ticks_fontsize, rotation=rotation)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlabel('\nCategories', fontsize=axes_label_fontsize)
    plt.ylabel(y_axis, fontsize=axes_label_fontsize)
    plt.grid(True, linestyle='--', linewidth=0.5, axis='y', alpha=0.7)
    if title:
        plt.title(title, fontsize=title_fontsize)
    # Save the figure
    if file_name_for_plot:
        plt.savefig(file_name_for_plot, dpi=300, bbox_inches='tight')
    plt.show()



def _prune_by_min_one_in_out_degree(arr):
    # Step 2: Find the threshold
    row_max = arr.max(axis=1)
    col_max = arr.max(axis=0)
    threshold = min(row_max.min(), col_max.min())
    #threshold = 0

    # Step 3: Set values below the threshold to zero
    arr_thresholded = np.where(arr < threshold, 0, arr)
    return arr_thresholded

def _prune_by_top_k_outdegree(arr, k=None):
    n = arr.shape[1]  # Assuming a square matrix
    if k is None:
        kept = math.floor(0.1 * n)  # Default k is 10% of n, rounded up
    else:
        kept = math.floor(k * n)  # Default k is 10% of n, rounded up

    pruned_matrix = np.zeros_like(arr)
    for i in range(n):
        top_k_indices = np.argsort(arr[i])[-kept:]
        pruned_matrix[i, top_k_indices] = arr[i, top_k_indices]
    return pruned_matrix


def array_to_pagerank(arr, pruning = "min-1-in-out-degree"):
    # Ensure the input is a 2D array
    if arr.ndim != 2:
        raise ValueError("Input array must be 2D")

    # Step 3: Set values below the threshold to zero
    if pruning == "min-1-in-out-degree":
        arr_thresholded = _prune_by_min_one_in_out_degree(arr)
    else:
        arr_thresholded = _prune_by_top_k_outdegree(arr)

    # Step 5: Create a NetworkX graph from the thresholded array
    G = nx.from_numpy_array(arr_thresholded, create_using=nx.DiGraph)

    # Step 6: Run PageRank
    pagerank_dict = nx.pagerank(G)

    # Step 7: Convert the PageRank result to a numpy array
    pagerank_array = np.array([pagerank_dict[i] for i in range(len(pagerank_dict))])

    return pagerank_array



def calculate_domain_enrichment_score_and_p_value(df_features_imps, category_of_choice):
    all_values = np.array(df_features_imps["Chain"])
    category_values = np.array(df_features_imps[category_of_choice])  # Ensure this matches your data structure

    # Ensure there are category values to compare
    if len(category_values) == 0:
        return 0, 1  # No enrichment, p-value of 1 indicates no significance

    # Rank all values (higher is better)
    ranked_indices = np.argsort(-all_values)

    # Initialize the running sum and enrichment score
    running_sum = 0
    enrichment_score = 0

    # Total number of items
    N = len(all_values)

    # Initialize counters for hits and total
    hits = 0
    total = 0

    # Calculate hits in the entire dataset for the category of choice
    total_hits = np.sum(np.isin(all_values, category_values))

    if total_hits == 0:
        return 0, 1  # No hits, return no enrichment and non-significant p-value

    # Increment for hits and misses
    hit_increment = 1 / total_hits
    miss_decrement = 1 / (N - total_hits)

    for idx in ranked_indices:
        total += 1
        if all_values[idx] in category_values:
            hits += 1
            running_sum += hit_increment
        else:
            running_sum -= miss_decrement

        enrichment_score = max(enrichment_score, running_sum)

    # Cap the enrichment score at 1
    enrichment_score = min(enrichment_score, 1)

    # Calculate p-value using hypergeometric test
    p_value = hypergeom.sf(hits-1, N, total_hits, total)

    # Ensure p-value is within [0, 1]
    p_value = max(min(p_value, 1), 0)

    return enrichment_score, p_value

def calculate_domain_enrichment_score(df_features_imps, category_of_choice):
    all_values = np.array(df_features_imps["Chain"])
    category_values_set = set(np.array(df_features_imps[category_of_choice]))  # Use a set for efficient look-up

    # Ensure there are category values to compare
    if len(category_values_set) == 0:
        return 0, 1  # No enrichment, p-value of 1 indicates no significance

    # Rank all values (higher is better)
    ranked_indices = np.argsort(-all_values)

    # Initialize the running sum and enrichment score
    running_sum = 0
    enrichment_score = 0

    # Total number of items
    N = len(all_values)

    # Calculate hits in the entire dataset for the category of choice
    total_hits = sum(value in category_values_set for value in all_values)

    if total_hits == 0:
        return 0, 1  # No hits, return no enrichment and non-significant p-value

    # Increment for hits and misses
    hit_increment = 1
    miss_decrement = 1 / (N - total_hits)

    for idx in ranked_indices:
        if all_values[idx] in category_values_set:
            running_sum += hit_increment
        else:
            running_sum -= miss_decrement

        # Update enrichment score based on the proportion of hits so far
        current_score = running_sum / total_hits
        enrichment_score = max(enrichment_score, current_score)

    # Normalization or capping of the enrichment score if necessary
    enrichment_score = max(min(enrichment_score, 1), -1)

    return enrichment_score

def calculate_average_rank_enrichment(df_features_imps, category_of_choice):
    all_values = np.array(df_features_imps["Chain"])
    category_values_set = set(np.array(df_features_imps[category_of_choice]))

    # Rank all values (higher is better)
    ranked_indices = np.argsort(-all_values)

    # Find ranks of category values
    category_ranks = [idx for idx, value in enumerate(all_values[ranked_indices], start=1) if value in category_values_set]

    # Calculate the average rank of category values
    if category_ranks:
        average_rank = np.mean(category_ranks)
        # Compare to the expected average rank (N+1)/2 if uniformly distributed
        expected_average_rank = (len(all_values) + 1) / 2
        enrichment = (expected_average_rank - average_rank) / expected_average_rank
    else:
        # No category values found
        enrichment = 0

    return enrichment


def calculate_site_enrichments_for_attention_layers(protein_identifier):
    parent_path = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/attention_matrices"
    file_path = f"{parent_path}/{protein_identifier}.pt"
    attn_raw = torch.load(file_path)
    uniprot_accession = protein_identifier  # Replace with the UniProt accession code of your interest
    features_data = get_protein_features(uniprot_accession)
    df_features_tuples = extract_tuples_by_type(features_data['features'])
    active_site_enrichment_scores = np.zeros(20)
    binding_site_enrichment_scores = np.zeros(20)
    
    for head in range(20):
        imp_array = array_to_pagerank(attn_raw[head].numpy()[1:-1, 1:-1])
        df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
        df_top_layers_for_active_site = None
        df_top_layers_for_binding_site = None
        if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
            enrichment_active_site = calculate_domain_enrichment_score(df_features_imps, "Active site")
            #print(f'Active site enrichment: {enrichment_active_site}')
            active_site_enrichment_scores[head] = enrichment_active_site
            df_top_layers_for_active_site = get_ranked_importance_dataframe(active_site_enrichment_scores, k=5)
        if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
            enrichment_binding_site = calculate_domain_enrichment_score(df_features_imps, "Binding site")
            #print(f"Binding site enrichment: {enrichment_binding_site}")
            binding_site_enrichment_scores[head] = enrichment_binding_site
            df_top_layers_for_binding_site = get_ranked_importance_dataframe(binding_site_enrichment_scores, k=5)

    return df_top_layers_for_active_site, df_top_layers_for_binding_site


def calculate_site_enrichments_for_attention_poolings(protein_identifier):
    parent_path = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/attention_matrices"
    parent_path_cont = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/contact_matrices"
    file_path = f"{parent_path}/{protein_identifier}.pt"
    file_path_contact = f"{parent_path_cont}/{protein_identifier}.pt"
    attn_raw = torch.load(file_path)
    if attn_raw.shape[-1] > 400:
        #print("sequence too long")
        raise Exception("sequence too long")
    contact_map = torch.load(file_path_contact)
    uniprot_accession = protein_identifier  # Replace with the UniProt accession code of your interest
    features_data = get_protein_features(uniprot_accession)
    df_features_tuples = extract_tuples_by_type(features_data['features'])
    active_site_enrichment_scores = np.empty(8)
    binding_site_enrichment_scores = np.empty(8)

    attn_mean_pooled = torch.mean(attn_raw, dim=0)
    attn_max_pooled = torch.max(attn_raw, dim=0)[0]
    max_values = torch.max(torch.cat([attn_raw.max(dim=1, keepdim=True)[0].unsqueeze(-1),  # Add a singleton dimension
                                       attn_raw.max(dim=2, keepdim=True)[0].unsqueeze(1)],  # Add a singleton dimension
                                      dim=-1), dim=-1)[0]
    
    normalized_attn_raw = torch.div(attn_raw, max_values)
    attn_max_pooled_norm = normalized_attn_raw.max(dim=0)[0]

    pooled_tensors = [attn_mean_pooled, attn_max_pooled, attn_max_pooled_norm]
    
    #print("got to the end of pooling")

    if (not "Active site" in df_features_tuples.keys()) or (not "Binding site" in df_features_tuples.keys()):

        raise Exception("We don't have either site")
    
    for index_pool in range(3):
        imp_array = array_to_pagerank(pooled_tensors[index_pool].numpy()[1:-1, 1:-1], pruning = "min-1-in-out-degree")
        df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
        
        if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
            enrichment_active_site = calculate_average_rank_enrichment(df_features_imps, "Active site")
            #enrichment_active_site = calculate_domain_enrichment_score(df_features_imps, "Active site")
            #print(f'Active site enrichment: {enrichment_active_site}')
            active_site_enrichment_scores[index_pool] = enrichment_active_site
    
        if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
            enrichment_binding_site = calculate_average_rank_enrichment(df_features_imps, "Binding site")
            #enrichment_binding_site = calculate_domain_enrichment_score(df_features_imps, "Binding site")
            #print(f"Binding site enrichment: {enrichment_binding_site}")
            binding_site_enrichment_scores[index_pool] = enrichment_binding_site

    for index_pool in range(3,6):
        imp_array = array_to_pagerank(pooled_tensors[index_pool-3].numpy()[1:-1, 1:-1], pruning = "k-outdegrees")
        df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
        
        if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
            enrichment_active_site = calculate_average_rank_enrichment(df_features_imps, "Active site")
            #enrichment_active_site = calculate_domain_enrichment_score(df_features_imps, "Active site")
            #print(f'Active site enrichment: {enrichment_active_site}')
            active_site_enrichment_scores[index_pool] = enrichment_active_site
    
        if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
            enrichment_binding_site = calculate_average_rank_enrichment(df_features_imps, "Binding site")
            #enrichment_binding_site = calculate_domain_enrichment_score(df_features_imps, "Binding site")
            #print(f"Binding site enrichment: {enrichment_binding_site}")
            binding_site_enrichment_scores[index_pool] = enrichment_binding_site

    imp_array = array_to_pagerank(contact_map.numpy(), pruning = "min-1-in-out-degree")
    df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
    
    if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
        enrichment_active_site = calculate_average_rank_enrichment(df_features_imps, "Active site")
        #enrichment_active_site = calculate_domain_enrichment_score(df_features_imps, "Active site")
        #print(f'Active site enrichment: {enrichment_active_site}')
        active_site_enrichment_scores[3] = enrichment_active_site

    if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
        enrichment_binding_site = calculate_average_rank_enrichment(df_features_imps, "Binding site")
        #enrichment_binding_site = calculate_domain_enrichment_score(df_features_imps, "Binding site")
        #print(f"Binding site enrichment: {enrichment_binding_site}")
        binding_site_enrichment_scores[3] = enrichment_binding_site

    imp_array = array_to_pagerank(contact_map.numpy(), pruning = "k-outdegrees")
    df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
    
    if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
        enrichment_active_site = calculate_average_rank_enrichment(df_features_imps, "Active site")
        #enrichment_active_site = calculate_domain_enrichment_score(df_features_imps, "Active site")
        #print(f'Active site enrichment: {enrichment_active_site}')
        active_site_enrichment_scores[3] = enrichment_active_site

    if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
        enrichment_binding_site = calculate_average_rank_enrichment(df_features_imps, "Binding site")
        #enrichment_binding_site = calculate_domain_enrichment_score(df_features_imps, "Binding site")
        #print(f"Binding site enrichment: {enrichment_binding_site}")
        binding_site_enrichment_scores[3] = enrichment_binding_site
            
    df_top_layers_for_active_site = get_ranked_importance_dataframe(active_site_enrichment_scores, k=8)
    df_top_layers_for_binding_site = get_ranked_importance_dataframe(binding_site_enrichment_scores, k=8)
    dict_index_to_method = {
        1:"mean pool prune 1 in/outdegree",
        2:"max pool prune 1 in/outdegree",
        3:"max pool - head norm prune 1 in/outdegree",
        4:"mean pool - head norm prune k outdegrees",
        5:"max pool prune k outdegrees",
        6:"max pool - head norm prune k outdegrees",
        7:"contact map prune 1 in/outdegree",
        8:"contact map prune k outdegrees"
    }
    df_top_layers_for_active_site["index"] = df_top_layers_for_active_site["index"].map(dict_index_to_method)
    df_top_layers_for_binding_site["index"] = df_top_layers_for_binding_site["index"].map(dict_index_to_method)
    return df_top_layers_for_active_site, df_top_layers_for_binding_site


def calculate_site_enrichments_for_attention_poolings_no_contact(protein_identifier):
    parent_path = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/attention_matrices"
    file_path = f"{parent_path}/{protein_identifier}.pt"
    attn_raw = torch.load(file_path)
    if attn_raw.shape[-1] > 400:
        #print("sequence too long")
        raise Exception("sequence too long")
    
    uniprot_accession = protein_identifier  # Replace with the UniProt accession code of your interest
    features_data = get_protein_features(uniprot_accession)
    df_features_tuples = extract_tuples_by_type(features_data['features'])
    active_site_enrichment_scores = np.empty(6)
    binding_site_enrichment_scores = np.empty(6)

    attn_mean_pooled = torch.mean(attn_raw, dim=0)
    attn_max_pooled = torch.max(attn_raw, dim=0)[0]
    max_values = torch.max(torch.cat([attn_raw.max(dim=1, keepdim=True)[0].unsqueeze(-1),  # Add a singleton dimension
                                       attn_raw.max(dim=2, keepdim=True)[0].unsqueeze(1)],  # Add a singleton dimension
                                      dim=-1), dim=-1)[0]
    
    normalized_attn_raw = torch.div(attn_raw, max_values)
    attn_max_pooled_norm = normalized_attn_raw.max(dim=0)[0]

    pooled_tensors = [attn_mean_pooled, attn_max_pooled, attn_max_pooled_norm]
    
    #print("got to the end of pooling")

    if (not "Active site" in df_features_tuples.keys()) or (not "Binding site" in df_features_tuples.keys()):

        raise Exception("We don't have any of the sites")
    
    for index_pool in range(3):
        imp_array = array_to_pagerank(pooled_tensors[index_pool].numpy()[1:-1, 1:-1], pruning = "min-1-in-out-degree")
        df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
        
        if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
            enrichment_active_site = calculate_average_rank_enrichment(df_features_imps, "Active site")
            #print(f'Active site enrichment: {enrichment_active_site}')
            active_site_enrichment_scores[index_pool] = enrichment_active_site
    
        if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
            enrichment_binding_site = calculate_average_rank_enrichment(df_features_imps, "Binding site")
            #print(f"Binding site enrichment: {enrichment_binding_site}")
            binding_site_enrichment_scores[index_pool] = enrichment_binding_site

    for index_pool in range(3,6):
        imp_array = array_to_pagerank(pooled_tensors[index_pool-3].numpy()[1:-1, 1:-1], pruning = "k-outdegrees")
        df_features_imps = compile_importances_by_feature_types(imp_array, df_features_tuples)
        
        if "Active site" in df_features_imps.keys() and len(df_features_imps["Active site"]) > 0:
            enrichment_active_site = calculate_average_rank_enrichment(df_features_imps, "Active site")
            #print(f'Active site enrichment: {enrichment_active_site}')
            active_site_enrichment_scores[index_pool] = enrichment_active_site
    
        if "Binding site" in df_features_imps.keys() and len(df_features_imps["Binding site"]) > 0:
            enrichment_binding_site = calculate_average_rank_enrichment(df_features_imps, "Binding site")
            #print(f"Binding site enrichment: {enrichment_binding_site}")
            binding_site_enrichment_scores[index_pool] = enrichment_binding_site


            
    df_top_layers_for_active_site = get_ranked_importance_dataframe(active_site_enrichment_scores, k=6)
    df_top_layers_for_binding_site = get_ranked_importance_dataframe(binding_site_enrichment_scores, k=6)
    dict_index_to_method = {
        1:"mean pool prune 1 in/outdegree",
        2:"max pool prune 1 in/outdegree",
        3:"max pool - head norm prune 1 in/outdegree",
        4:"mean pool - head norm prune k outdegrees",
        5:"max pool prune k outdegrees",
        6:"max pool - head norm prune k outdegrees",
    }
    df_top_layers_for_active_site["index"] = df_top_layers_for_active_site["index"].map(dict_index_to_method)
    df_top_layers_for_binding_site["index"] = df_top_layers_for_binding_site["index"].map(dict_index_to_method)
    return df_top_layers_for_active_site, df_top_layers_for_binding_site


def visualize_attention_heads(protein_identifier, filepath=None):
    parent_path = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/attention_matrices"
    file_path = f"{parent_path}/{protein_identifier}.pt"
    
    # Load attention heads
    attn_raw = torch.load(file_path)
    
    # Calculate mean and max pooled attention
    attn_mean_pooled = torch.mean(attn_raw, dim=0).numpy()[1:-1, 1:-1]
    attn_max_pooled = torch.max(attn_raw, dim=0)[0].numpy()[1:-1, 1:-1]
    
    # Normalize attention heads
    max_values = torch.max(torch.cat([attn_raw.max(dim=1, keepdim=True)[0].unsqueeze(-1),
                                     attn_raw.max(dim=2, keepdim=True)[0].unsqueeze(1)],
                                    dim=-1), dim=-1)[0]
    normalized_attn_raw = torch.div(attn_raw, max_values)
    attn_max_pooled_norm = normalized_attn_raw.max(dim=0)[0].numpy()[1:-1, 1:-1]
    
    # Create a figure with subplots for individual heads and pooled attention
    fig, axes = plt.subplots(figsize=(15, 15))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.3)
    plt.box(False)
    
    # Plot individual attention heads
    for head in range(20):
        ax = fig.add_subplot(5, 5, 1 + head, xlabel=f"", title=f'Head {head + 1}')
        im = ax.imshow(attn_raw[head].numpy()[1:-1, 1:-1])
    
    # Plot mean pooled attention
    #ax = fig.add_subplot(5, 5, 21, xlabel='Residue i', ylabel='Residue j', title='Mean Pooled')
    ax = fig.add_subplot(5, 5, 21, title='Mean Pooled')
    im = ax.imshow(attn_mean_pooled)
    
    # Plot max pooled attention
    #ax = fig.add_subplot(5, 5, 22, xlabel='Residue i', ylabel='Residue j', title='Max Pooled')
    ax = fig.add_subplot(5, 5, 22, title='Max Pooled')
    im = ax.imshow(attn_max_pooled)
    
    # Plot normalized max pooled attention
    #ax = fig.add_subplot(5, 5, 23, xlabel='Residue i', ylabel='Residue j', title='Normalized Max Pooled')
    ax = fig.add_subplot(5, 5, 23, title='Normalized Max Pooled')
    im = ax.imshow(attn_max_pooled_norm)

    
    # Adjust spacing and colorbars
    fig.suptitle(f"The 20 Attention Heads from the last layer of ESM2-650M for {protein_identifier}")
    plt.tight_layout()
    plt.show()
    # Save the figure at high resolution with specified filepath
    if not filepath == None:
        fig.savefig(filepath, dpi=600)  # Adjust dpi as needed


def visualize_attention_head_poolings_and_prunings(protein_identifier, filepath=None, title=None):
    parent_path = "/oak/stanford/groups/rbaltman/esm_embeddings/esm2_t33_650M_uniprot/attention_matrices_uncompressed"
    file_path = f"{parent_path}/{protein_identifier}.pt"
    
    # Load attention heads
    attn_raw = torch.load(file_path)
    
    # Calculate mean and max pooled attention
    attn_mean_pooled = torch.mean(attn_raw, dim=0).numpy()[1:-1, 1:-1]
    attn_max_pooled = torch.max(attn_raw, dim=0)[0].numpy()[1:-1, 1:-1]
    
    # Normalize attention heads
    max_values = torch.max(torch.cat([attn_raw.max(dim=1, keepdim=True)[0].unsqueeze(-1),
                                     attn_raw.max(dim=2, keepdim=True)[0].unsqueeze(1)],
                                    dim=-1), dim=-1)[0]
    normalized_attn_raw = torch.div(attn_raw, max_values)
    attn_max_pooled_norm = normalized_attn_raw.max(dim=0)[0].numpy()[1:-1, 1:-1]

    attn_mean_pooled_1inout_prune = _prune_by_min_one_in_out_degree(attn_mean_pooled)
    attn_max_pooled_1inout_prune = _prune_by_min_one_in_out_degree(attn_max_pooled)
    attn_max_pooled_norm_1inout_prune = _prune_by_min_one_in_out_degree(attn_max_pooled_norm)

    attn_mean_pooled_k_out_prune = _prune_by_top_k_outdegree(attn_mean_pooled)
    attn_max_pooled_k_out_prune = _prune_by_top_k_outdegree(attn_max_pooled)
    attn_max_pooled_norm_k_out_prune = _prune_by_top_k_outdegree(attn_max_pooled_norm)
    
    # Create a figure with subplots for individual heads and pooled attention
    fig, axes = plt.subplots(figsize=(15, 15))
    plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.3)
    plt.box(False)

    print(attn_mean_pooled)
    
    
    # Plot mean pooled attention
    #ax = fig.add_subplot(5, 5, 21, xlabel='Residue i', ylabel='Residue j', title='Mean Pooled')
    description = "Mean Pooled"
    nonzeros = np.count_nonzero(attn_mean_pooled)
    ax = fig.add_subplot(3, 3, 1, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_mean_pooled)
    
    # Plot max pooled attention
    #ax = fig.add_subplot(5, 5, 22, xlabel='Residue i', ylabel='Residue j', title='Max Pooled')
    description = 'Max Pooled'
    nonzeros = np.count_nonzero(attn_max_pooled)
    ax = fig.add_subplot(3, 3, 2, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_max_pooled)
    
    # Plot normalized max pooled attention
    #ax = fig.add_subplot(5, 5, 23, xlabel='Residue i', ylabel='Residue j', title='Normalized Max Pooled')
    description = 'Normalized Max Pooled'
    nonzeros = np.count_nonzero(attn_max_pooled_norm)
    ax = fig.add_subplot(3, 3, 3, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_max_pooled_norm)

     # Plot mean pooled attention
    #ax = fig.add_subplot(5, 5, 21, xlabel='Residue i', ylabel='Residue j', title='Mean Pooled')
    description = 'Mean Pooled - min in/out degree pooling'
    nonzeros = np.count_nonzero(attn_mean_pooled_1inout_prune)
    ax = fig.add_subplot(3, 3, 4, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_mean_pooled_1inout_prune)
    
    # Plot max pooled attention
    #ax = fig.add_subplot(5, 5, 22, xlabel='Residue i', ylabel='Residue j', title='Max Pooled')
    description = 'Max Pooled - min in/out degree pooling'
    nonzeros = np.count_nonzero(attn_max_pooled_1inout_prune)
    ax = fig.add_subplot(3, 3, 5, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_max_pooled_1inout_prune)
    
    # Plot normalized max pooled attention
    #ax = fig.add_subplot(5, 5, 23, xlabel='Residue i', ylabel='Residue j', title='Normalized Max Pooled')
    description = 'Normalized Max Pooled - min in/out degree pooling'
    nonzeros = np.count_nonzero(attn_max_pooled_norm_1inout_prune)
    ax = fig.add_subplot(3, 3, 6, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_max_pooled_norm_1inout_prune)
    

      # Plot mean pooled attention
    #ax = fig.add_subplot(5, 5, 21, xlabel='Residue i', ylabel='Residue j', title='Mean Pooled')
    description = 'Mean Pooled - k outdegree pooling'
    nonzeros = np.count_nonzero(attn_mean_pooled_k_out_prune)
    ax = fig.add_subplot(3, 3, 7, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_mean_pooled_k_out_prune)
    
    
    # Plot max pooled attention
    #ax = fig.add_subplot(5, 5, 22, xlabel='Residue i', ylabel='Residue j', title='Max Pooled')
    nonzeros = np.count_nonzero(attn_max_pooled_k_out_prune)
    description = f'Max Pooled - k outdegree pooling'
    ax = fig.add_subplot(3, 3, 8, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_max_pooled_k_out_prune)
    
    # Plot normalized max pooled attention
    #ax = fig.add_subplot(5, 5, 23, xlabel='Residue i', ylabel='Residue j', title='Normalized Max Pooled')
    description = 'Normalized Max Pooled - k outdegree pooling'
    nonzeros = np.count_nonzero(attn_max_pooled_norm_k_out_prune)
    ax = fig.add_subplot(3, 3, 9, title=f"{description} ({nonzeros} nze)")
    im = ax.imshow(attn_max_pooled_norm_k_out_prune)
    

    
    # Adjust spacing and colorbars
    if not title == None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    # Save the figure at high resolution with specified filepath
    if not filepath == None:
        fig.savefig(filepath, dpi=600)  # Adjust dpi as needed


def convert_importances_to_txt(values, file_name):
    """
    Convert an array of values to a text file in the format:
    residue_number value

    Parameters:
    values (list or numpy array): Array of values corresponding to residues.
    file_name (str): Name of the output text file.
    """
    base_dir = "interpretability/importance_value_files"
    with open(f"{base_dir}/{file_name}", 'w') as f:
        for i, value in enumerate(values, start=1):
            f.write(f"{i} {value}\n") 


