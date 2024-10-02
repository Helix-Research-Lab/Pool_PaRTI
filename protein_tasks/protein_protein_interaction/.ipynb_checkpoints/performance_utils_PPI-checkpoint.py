import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_accuracy(predicted, actual):
    correct_predictions = (predicted == actual).sum()
    total_predictions = len(actual)
    accuracy = correct_predictions / total_predictions
    return accuracy

def extract_accuracy(pooling):
    # List of CSV files within the specified directory
    csv_file_names = os.listdir(f"predictions_efficient/{pooling}")
    csv_file_names = [file for file in csv_file_names if file.endswith(".csv") and not "999" in file]
    
    # Initialize an empty list to store each row of results
    data_list = []
    
    # Process each CSV file
    for file in csv_file_names:
        run_no = int(file.split("_")[0])
        path = f"predictions_efficient/{pooling}/{file}"
        
        # Read the CSV file
        df = pd.read_csv(path)
        accuracy = -1
        # Calculate accuracy
        accuracy = calculate_accuracy(df["Predicted Label"], df["Actual Label"])
        
    
        
        # Check if all values are updated from the default
        if accuracy > 0:
            data_list.append({
                'run_no': run_no,
                'accuracy': accuracy,
            })
    
    # Create a DataFrame from the list of dictionaries
    results = pd.DataFrame(data_list)

    # Sort the DataFrame by the specified column in descending order
    results_sorted = results.sort_values(by='accuracy', ascending=False)

    return results_sorted

def extract_metrics(pooling, metric, verbose=False, efficient=False):
    # List of log files within the specified directory
    if efficient:
        log_file_names = os.listdir(f"logs_efficient/{pooling}")
    else:
        log_file_names = os.listdir(f"logs/{pooling}")
    log_file_names = [file for file in log_file_names if "log" in file and not "999" in file]
    value = -2
    # Initialize an empty list to store each row of results
    data_list = []
    
    # Process each log file
    for file in log_file_names:
        run_no = int(file.split("_")[0])
        mcc = -2  # default value if metric is not found

        if efficient:
            path = f"logs_efficient/{pooling}/{file}"
        else:
            path = f"logs/{pooling}/{file}"
        
        
        with open(path, "r") as log_file:
            lines = log_file.readlines()
        
        for line in lines:
            if "Test Metrics" in line:
                metrics_side = line.split("{")[1][:-2].split(", ")
                for data in metrics_side:
                    if metric in data:
                        value = float(data.split(": ")[1])
        
        # Check if all values are updated from the default
        if value > -1.5:
            data_list.append({
                    'run_no': run_no,
                    f'{metric}': value,
            })
            if verbose:
                print(f"run_no {run_no}, {metric} {value}")
    
    # Create a DataFrame from the list of dictionaries
    results = pd.DataFrame(data_list)
    # Sort the DataFrame by the 'eukaryote' column in descending order
    results_sorted = results.sort_values(by=metric, ascending=False)

    return results_sorted

################################################################################################



def analyze_and_plot_single_old1(pooling_methods, metric, color_dict, save_path=None, efficient=False):
    # Dictionary to hold the best results for one category
    results_by_category = []

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Initial extraction sorted by the provided category
        sorted_results = extract_metrics(pooling, metric, efficient=efficient)
        
        # Find the best performance for the category
        if not sorted_results.empty:
            top_result = sorted_results.sort_values(by=metric, ascending=False).iloc[0]
            top_result['method'] = pooling
            results_by_category.append(top_result)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=metric, ascending=False, inplace=True)  # Highest value at the top

    # Plotting each method's performance using the passed color dictionary
    for i, row in df_top_results.iterrows():
        #print(i, row)
        ax.barh(row['method'], row[metric], color=color_dict[row['method']])

    # Individual x-limits
    min_val, max_val = df_top_results[metric].min(), df_top_results[metric].max()
    ax.set_xlim([min_val * 0.95, max_val * 1.03])

    ax.set_title(f"Pooling PPI Performance Comparison")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel('Methods')
    ax.invert_yaxis()  # Invert y-axis to have the highest value at the top

    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return df_top_results

################################################################################################


def analyze_and_plot_single_old1_withoutSTD(pooling_methods, metric, color_dict, top_k=1, save_path=None, efficient=False):
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
    - DataFrame of the top results averaged per pooling method.
    """
    # List to hold the best/average results for one category
    results_by_category = []

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Extract and sort results by the provided metric
        sorted_results = extract_metrics(pooling, metric, efficient=efficient)

        # Determine the average of the top k results if available
        if not sorted_results.empty:
            top_k_results = sorted_results.sort_values(by=metric, ascending=False).head(top_k)
            avg_result = top_k_results.mean(numeric_only=True)
            avg_result['method'] = pooling
            results_by_category.append(avg_result)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=metric, ascending=False, inplace=True)  # Highest value at the top

    # Plot each method's performance using the passed color dictionary
    for i, row in df_top_results.iterrows():
        ax.barh(row['method'], row[metric], color=color_dict.get(row['method'], 'gray'))

    # Individual x-limits
    min_val, max_val = df_top_results[metric].min(), df_top_results[metric].max()
    ax.set_xlim([min_val * 0.95, max_val * 1.03])

    ax.set_title(f"Pooling PPI Performance Comparison (Top {top_k} Average)")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel('Methods')
    ax.invert_yaxis()  # Invert y-axis to have the highest value at the top

    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return df_top_results

#######################

def analyze_and_plot_single(pooling_methods, metric, color_dict, top_k=1, save_path=None, efficient=False):
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

        if metric == "accuracy":
            sorted_results = extract_accuracy(pooling)
        else:
            sorted_results = extract_metrics(pooling, metric, efficient=efficient)

        # Determine the average and standard deviation of the top k results if available
        if not sorted_results.empty:
            top_k_results = sorted_results.sort_values(by=metric, ascending=False).head(top_k)
            avg_result = top_k_results.mean(numeric_only=True)
            if top_k > 1:
                std_result = top_k_results[metric].std()
                avg_result[f"{metric}_std"] = std_result
            avg_result['method'] = pooling
            results_by_category.append(avg_result)

    # Convert list to DataFrame
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=metric, ascending=False, inplace=True)  # Highest value at the top

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot each method's performance using the passed color dictionary
    for i, row in df_top_results.iterrows():
        ax.barh(row['method'], row[metric], color=color_dict.get(row['method'], 'gray'))
        if top_k > 1:
            # Add error bars for standard deviation
            ax.errorbar(row[metric], row['method'], xerr=row[f"{metric}_std"], fmt='o', color='black')
    
    # Individual x-limits
    min_val, max_val = df_top_results[metric].min(), df_top_results[metric].max()
    ax.set_xlim([min_val * 0.95, max_val * 1.03])

    ax.set_title(f"Pooling PPI Performance Comparison (Top {top_k} Average)")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel('Methods')
    ax.invert_yaxis()  # Invert y-axis to have the highest value at the top

    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return df_top_results
