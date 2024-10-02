import numpy as np
import networkx as nx
import math
import torch


class TokenToSequencePooler:
    def __init__(self, uniprot_accession=None, last_layer=33):
        self.uniprot_accession = uniprot_accession
        self.ESM_size = ESM_size
        self.representations = self._load_torch_data("representation").squeeze()
        self.representations_with_cls = self._load_torch_data("representation", with_cls=True).squeeze()
        self.contact_matrices = self._load_torch_data("contact")
        self.num_tokens = self.contact_matrices.shape[1]
        self.last_layer = last_layer
        


    def _load_torch_data(self, data_type, with_cls = False, verbose=False):
        file_path = f"../token_embedding_generator/{data_type}_matrices/{self.uniprot_accession}.pt"
        tensors = torch.load(file_path)
        if verbose:
            print(tensors)
            print("tensors.shape",tensors.shape)
            print(f"data_type {data_type}")
            print(f"with_cls {with_cls}")
        if data_type == "representation" and not with_cls:
            try: 
                if verbose:
                    print(f"tensors[1:-1].numpy() \n{tensors[1:-1].numpy()}")
                if len(tensors.shape) == 2:
                    return tensors[1:-1].numpy()
                elif len(tensors.shape) == 3:
                    return tensors[:,1:-1, :].numpy()
            except:
                if verbose:
                    print("tensors[1:-1].numpy() didn't work")
                return tensors["representations"][self.last_layer].numpy()
        elif data_type == "representation" and with_cls:
            try:
                return tensors.numpy()
            except Exception as e:
                # Print the error message and the problematic UniProt accession
                print(f"Error loading data for {self.uniprot_accession}: {e}", flush=True)
                
                # Define the filename where you want to store the UniProt accessions
                filename = "uniprot_accessions_with_noCLSrep.txt"
                
                # Open the file in append mode (creates the file if it doesn't exist) and write the accession
                with open(filename, "a") as file:
                    file.write(f"{self.uniprot_accession}\n")
                
                # You may want to handle the exception differently here, e.g., by returning None or raising another exception
                return None
        else:
            try:
                return tensors.numpy()
            except:
                print(f"Error loading data for {self.uniprot_accession}", flush=True)
                return tensors["contacts"].numpy()

    def cls_pooling(self, save_path=None):
        if self.representations_with_cls is not None:
            cls_token = self.representations_with_cls[0]
            if save_path:
                torch.save(save_path, cls_token)
            return cls_token.squeeze()
        print(f"representations_with_cls was None for sequence {self.uniprot_accession}", flush=True)
        return None  # Handle cases where CLS token is not available or representations are not loaded properly
            
            

    def mean_pooling(self, apply_to_wt=False):
        if apply_to_wt:
            return np.mean(self.representations, axis=0), np.mean(self.wt_representations, axis=0)
        else:
            return np.mean(self.representations, axis=0)

    def max_pooling(self, apply_to_wt=False):
        if apply_to_wt:
            return np.max(self.representations, axis=0), np.max(self.wt_representations, axis=0)
        else:
            if len(self.representations.shape) == 2:
                return np.max(self.representations, axis=0)
            else:
                return np.max(self.representations, axis=1)
            
    
    
    def pool_parti(self, 
                   which_matrix,
                   top_k_vals=None, 
                   verbose=False, 
                   return_importance=False, 
                   apply_to_wt=False, 
                   softmax_enhancement=False, 
                   sigmoid_enhancement=True, 
                   k_sigmoid=10, 
                   prune_type="top_k_outdegree",
                   include_cls = False
                   ):

        
        matrix_to_pool = self.contact_matrices
        
        dict_importance = self._page_rank(matrix_to_pool, top_k_vals, prune_type=prune_type)
        
        importance_weights = np.array(list(self._calculate_importance_weights(dict_importance, 
                                                                              include_cls=include_cls).values()))
        if return_importance:
            return importance_weights
        if verbose:
            print(f'pagerank direct outcome is {dict_importance}\n')
            print(f'importance_weights dict of length {len(importance_weights)} looks like\n {importance_weights}')
            print(f'shape of the importance matrix is {len(importance_weights)} and for repr, its {self.representations.shape}')
            print(f"importance weights look like {sorted(importance_weights, reverse=True)[0:5]}")
            
        
        
        if include_cls:
            
            try:
                if len(self.representations_with_cls.shape) == 2: 
                    return torch.tensor(np.average(self.representations_with_cls, weights=importance_weights, axis=0))
                elif len(self.representations_with_cls.shape) == 3:
                    return torch.tensor(np.average(self.representations_with_cls, weights=importance_weights, axis=1))
                else:
                    raise Exception(f"In pageRank, the shape of the representations_with_cls is {self.representations_with_cls.shape}")
            except Exception as e:
                print(f"{e} in PageRank with cls", flush=True)
                print(f"This happened for protein {self.uniprot_accession}")
                print(f"shape of representations_with_cls {self.representations_with_cls.shape}\n", flush=True)
                print(f"shape of importance_weights {len(importance_weights)}\n", flush=True)
                
        else:
            try:
                return torch.tensor(np.average(self.representations, weights=importance_weights, axis=0))
            except Exception as e:
                print(f"{e} in PageRank without cls", flush=True)
                print(f"This happened for protein {self.uniprot_accession}")
                print(f"self.representations shape {self.representations.shape}", flush=True)
                print(f"importance_weights {len(importance_weights)}", flush=True)
                    
                    

    def _prune_matrix(self, matrix, k=None, prune_type="top_k_outdegree"):
        if prune_type == "top_k_outdegree":
            n = matrix.shape[-1]  # Assuming a square matrix
            if k is None:
                k = math.floor(0.1 * n)  # Default k is 10% of n, rounded up
                #print(f"keeping {k} outdegrees for {self.num_tokens} tokens")
            pruned_matrix = np.zeros_like(matrix)
            for i in range(n):
                top_k_indices = np.argsort(matrix[i])[-k:]
                pruned_matrix[i, top_k_indices] = matrix[i, top_k_indices]
        elif prune_type == "min_one_in_out_degree":
            # ensures that all nodes have at least one indegree and one outdegree and prunes everything else
            # and the indegree and outdegree won't be self loop

            no_diag_matrix = matrix.copy()
            np.fill_diagonal(no_diag_matrix, 0)
            # Find the maximum of each row and column
            row_max = np.max(no_diag_matrix, axis=1)
            col_max = np.max(no_diag_matrix, axis=0)
            
            # Find the minimum element among the maximums
            threshold = np.min(np.concatenate((row_max, col_max)))
            
            # Set elements less than the threshold to zero
            pruned_matrix = np.where(matrix < threshold, 0, matrix)
        
        return pruned_matrix

    def _page_rank(self, matrix, top_k_vals, personalization=None, nstart=None, prune_type="top_k_outdegree"):
        # Implement pruning here
        # Convert to graph and apply PageRank
        pruned_mtx = self._prune_matrix(matrix, top_k_vals, prune_type=prune_type)
        
        G = self._convert_to_graph(pruned_mtx)
        #print("Number of nodes in the graph:", G.number_of_nodes())
        #print("Number of edges in the graph:", G.number_of_edges())

        if G.number_of_nodes() != matrix.shape[0]:
            raise Exception(f"The number of nodes in the graph should be equal to the number of tokens in sequence! You have {G.number_of_nodes()} nodes for {matrix.shape[0]} tokens.") 
        if G.number_of_edges() == 0:
            raise Exception(f"You don't seem to have any edges left in the graph.") 
        
        return nx.pagerank(G, weight='weight', personalization=personalization, nstart=nstart, max_iter=10000)
        #return nx.pagerank(G)

    def _convert_to_graph(self, matrix):
        # Convert the matrix to a graph
        # Placeholder for actual implementation
        #G = nx.DiGraph(matrix)
        # Create a directed graph from the adjacency matrix
        G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
        # Print the number of nodes and edges
        #print(f"Number of nodes: {G.number_of_nodes()}")
        #print(f"Number of edges: {G.number_of_edges()}")
        
        # Add edges and weights based on the matrix
        return G

    def _calculate_importance_weights(self, dict_importance, include_cls=False):
        # Normalize PageRank scores to sum to 1
        if not include_cls:
            # Get the highest integer key
            highest_key = max(dict_importance.keys())
            
            # Remove the entry with the highest key (EOT) and the entry with key 0 (CLS token)
            del dict_importance[highest_key]
            del dict_importance[0]
        total = sum(dict_importance.values())
        return {k: v / total for k, v in dict_importance.items()}


    
  






