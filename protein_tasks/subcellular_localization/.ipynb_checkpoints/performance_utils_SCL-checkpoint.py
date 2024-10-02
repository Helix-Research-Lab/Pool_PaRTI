import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



def extract_global_metrics(pooling, metric, remove_corrupted_runs=True):
    # List of log files within the specified directory
    base_loc = f"logs_efficient/test_logs/{pooling}"
    log_file_names = os.listdir(base_loc)
    log_file_names = [file for file in log_file_names if "txt" in file and not "999" in file]
    
    # Initialize an empty list to store each row of results
    data_list = []
    
    # Process each log file
    for file in log_file_names:
        run_no = int(file.split(".")[0])
        human = mammalian = vertebrate = animal = eukaryote = -2  # default value if metric is not found
        path = f"{base_loc}/{file}"
        if remove_corrupted(run_no) and remove_corrupted_runs:
            continue
        
        with open(path, "r") as log_file:
            line = log_file.readlines()[-1]
        splitLine = line.split('- ')[1].split(', ')
        loss = float(splitLine[0].split(": ")[-1])
        f1_micro = float(splitLine[1].split(": ")[-1])
        f1_macro = float(splitLine[2].split(": ")[-1])
        mcc = float(splitLine[3].split(": ")[-1])
        hamming_loss = float(splitLine[4].split(": ")[-1])
        hamming_acc = float(splitLine[5].split(": ")[-1])
        jaccard_ind = float(splitLine[6].split(": ")[-1])
            
        
        # Check if all values are updated from the default
        data_list.append({
            'run_no': run_no,
            'loss': loss,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'mcc': mcc,
            'hamming_loss': hamming_loss,
            'hamming_acc': hamming_acc,
            'jaccard_ind': jaccard_ind
        })
    
    # Create a DataFrame from the list of dictionaries
    results = pd.DataFrame(data_list)

    # Sort the DataFrame by the 'eukaryote' column in descending order
    results_sorted = results.sort_values(by=metric, ascending=False)

    return results_sorted


def get_label_specific_auprc(pooling, metric="avg", remove_corrupted_runs=True):
    base_loc = f"logs_efficient/metrics/{pooling}"
    log_file_names = os.listdir(base_loc)
    log_file_names = [file for file in log_file_names if "csv" in file and not "999" in file]
    #print(f"log_file_names {log_file_names}")

    dict_auprc = {}
    data_list = []
    for fileName in log_file_names:
        run_no = int(fileName.split(".")[0])
        if remove_corrupted(run_no) and remove_corrupted_runs:
            continue
        df_metrics = pd.read_csv(f"{base_loc}/{fileName}")
        auprcs = df_metrics["auprc"].values
        
        #auprcs
        dict_auprc[run_no] = auprcs
        data_list.append({
            'run_no': run_no,
            'cell_mem': auprcs[0],
            'cytoplasm': auprcs[1],
            'ER': auprcs[2],
            'golgi': auprcs[3],
            'lysosome': auprcs[4],
            'mitochondrion': auprcs[5],
            'nucleus': auprcs[6],
            'peroxisome': auprcs[7],
            'global': auprcs[8],
            'avg': np.mean(auprcs[0:-1])
        })

    results = pd.DataFrame(data_list)

    # Sort the DataFrame by the 'eukaryote' column in descending order
    results_sorted = results.sort_values(by=metric, ascending=False)

    return results_sorted


def get_label_specific_mccs(pooling, metric='avg', remove_corrupted_runs=True):
    base_loc = f"logs_efficient/confusion/{pooling}"
    log_file_names = os.listdir(base_loc)
    log_file_names = [file for file in log_file_names if "txt" in file and not "999" in file]
    #print(f"log_file_names {log_file_names}")

    data_list = []
    for fileName in log_file_names:
        run_no = int(fileName.split(".")[0])
        #print(run_no)
        if remove_corrupted(run_no) and remove_corrupted_runs:
            continue
        #print('unremoved for run_no', run_no)
        with open(f"{base_loc}/{fileName}", "r") as file:
            lines = file.readlines()
        mccs = []
        TN, FP, FN, TP = -1, -1, -1, -1
        for line in lines:
            if "[[" in line:
                line = line.replace('[', '').replace(']', '').replace('\n', '')
                nums = line.strip().split(" ")
                TN = int(nums[0])
                FP = int(nums[-1])
            elif "]]" in line:
                line = line.replace('[', '').replace(']', '').replace('\n', '')
                nums = line.strip().split(" ")
                FN = int(nums[0])
                TP = int(nums[-1])
            
        
            if TN >= 0 and FP >= 0 and FN >= 0 and TP >= 0:
                mcc = calculate_mcc(tp = TP, 
                            tn = TN, 
                            fp = FP, 
                            fn = FN)
                #print(mcc)
                TN, FP, FN, TP = -1, -1, -1, -1
                mccs.append(mcc)
        #mccs
        
        #dict_mccs[run_no] = mccs
        data_list.append({
            'run_no': run_no,
            'cell_mem': mccs[0],
            'cytoplasm': mccs[1],
            'ER': mccs[2],
            'golgi': mccs[3],
            'lysosome': mccs[4],
            'mitochondrion': mccs[5],
            'nucleus': mccs[6],
            'peroxisome': mccs[7],
            'avg': np.mean(mccs)
        })

    results = pd.DataFrame(data_list)

    # Sort the DataFrame by the 'eukaryote' column in descending order
    results_sorted = results.sort_values(by=metric, ascending=False)

    return results_sorted



###########################################################################################################

def analyze_and_plot_single(pooling_methods, metric, top_k=1, remove_corrupted_runs=True):
    """
    Analyzes and plots the average of the top k results for each pooling method.

    Parameters:
    - pooling_methods: list of pooling method names to analyze.
    - metric: the metric to compare across pooling methods.
    - color_dict: dictionary mapping pooling method names to colors.
    - top_k: the number of top results to average. Defaults to 1 (taking the top result only).
    - save_path: path to save the plot image (optional).
    - efficient: whether to use efficient extraction methods.

    Returns:
    - DataFrame of the top results averaged per pooling method, including standard deviation if top_k > 1.
    """
    # List to hold the best/average results for one category
    results_by_category = []

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Extract and sort results by the provided metric

        if metric == "hamming_acc":
            sorted_results = extract_global_metrics(pooling, 'hamming_acc', remove_corrupted_runs=remove_corrupted_runs)
            sort_metric = 'hamming_acc'
        elif metric == "mcc-label-avg":
            sorted_results = extract_global_metrics(pooling, 'mcc', remove_corrupted_runs=remove_corrupted_runs)
            sort_metric = 'mcc'
        elif metric == "auprc-label-avg":
            sorted_results = get_label_specific_auprc(pooling, 'avg', remove_corrupted_runs=remove_corrupted_runs)
            sort_metric = 'avg'

        # Determine the average and standard deviation of the top k results if available
        if not sorted_results.empty:
            top_k_results = sorted_results.sort_values(by=sort_metric, ascending=False).head(top_k)
            avg_result = top_k_results.mean(numeric_only=True)
            if top_k > 1:
                std_result = top_k_results[sort_metric].std()
                avg_result[f"{metric}_std"] = std_result
            avg_result['method'] = pooling
            results_by_category.append(avg_result)

    # Convert list to DataFrame
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=sort_metric, ascending=False, inplace=True)  # Highest value at the top


    return df_top_results

###########################################################################################################

def calculate_mcc(tp, tn, fp, fn):
    """
    Calculates the Matthews Correlation Coefficient (MCC)

    Args:
        tp (int): True positives.
        tn (int): True negatives.
        fp (int): False positives.
        fn (int): False negatives.

    Returns:
        float: The MCC value.
    """

    #print(tp, tn, fp, fn)

    under_root = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    # Avoid division by zero
    if under_root == 0:
        return 0

    mcc = (tp * tn - fp * fn) / math.sqrt(under_root)
    return mcc

###########################################################################################################

def remove_corrupted(run_no):
    df_params = pd.read_csv('hyperparams3.csv', index_col='hyperparam_group')
    if df_params.iloc[run_no]['dropout'] == 0.05 or df_params.iloc[run_no]['lr'] == 0.001:
        return True
    