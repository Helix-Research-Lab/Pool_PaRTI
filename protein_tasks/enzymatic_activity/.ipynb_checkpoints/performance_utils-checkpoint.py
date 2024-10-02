import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculate_accuracy(predicted, actual):
    correct_predictions = (predicted == actual).sum()
    total_predictions = len(actual)
    accuracy = correct_predictions / total_predictions
    return accuracy

def extract_accuracy(pooling, category):
    # List of CSV files within the specified directory
    csv_file_names = os.listdir(f"predictions/{pooling}")
    csv_file_names = [file for file in csv_file_names if file.endswith(".csv") and category in file and not "999" in file]
    
    # Initialize an empty list to store each row of results
    data_list = []
    
    # Process each CSV file
    for file in csv_file_names:
        run_no = int(file.split("_")[0][3:])
        path = f"predictions/{pooling}/{file}"
        
        # Read the CSV file
        df = pd.read_csv(path)
        accuracy = -1
        # Calculate accuracy
        accuracy = calculate_accuracy(df["Predicted Label"], df["Actual Label"])
        
    
        
        # Check if all values are updated from the default
        if accuracy > 0:
            data_list.append({
                'run_no': run_no,
                category: accuracy,
            })
    
    # Create a DataFrame from the list of dictionaries
    results = pd.DataFrame(data_list)

    # Sort the DataFrame by the specified column in descending order
    results_sorted = results.sort_values(by=category, ascending=False)

    return results_sorted
    

def extract_metrics(pooling, metric, sort_by="eukaryote"):
    # List of log files within the specified directory
    log_file_names = os.listdir(f"logs/{pooling}")
    log_file_names = [file for file in log_file_names if "log" in file and not "999" in file]
    
    # Initialize an empty list to store each row of results
    data_list = []
    
    # Process each log file
    for file in log_file_names:
        run_no = int(file.split("_")[0])
        human = mammalian = vertebrate = animal = eukaryote = -2  # default value if metric is not found
        path = f"logs/{pooling}/{file}"
        
        with open(path, "r") as log_file:
            lines = log_file.readlines()
        
        for line in lines:
            if any(category in line for category in ["human", "mammalian", "vertebrate", "animal", "eukaryote"]):
                metrics_side = line.split("{")[1][:-2].split(", ")
                for data in metrics_side:
                    if metric in data:
                        value = float(data.split(": ")[1])
                        if "human" in line:
                            human = value
                        elif "mammalian" in line:
                            mammalian = value
                        elif "vertebrate" in line:
                            vertebrate = value
                        elif "animal" in line:
                            animal = value
                        elif "eukaryote" in line:
                            eukaryote = value
        
        # Check if all values are updated from the default
        if all([v > -2 for v in [human, mammalian, vertebrate, animal, eukaryote]]):
            data_list.append({
                'run_no': run_no,
                'human': human,
                'mammalian': mammalian,
                'vertebrate': vertebrate,
                'animal': animal,
                'eukaryote': eukaryote
            })
    
    # Create a DataFrame from the list of dictionaries
    results = pd.DataFrame(data_list)

    # Sort the DataFrame by the 'eukaryote' column in descending order
    results_sorted = results.sort_values(by=sort_by, ascending=False)

    return results_sorted

################################################################################################

def analyze_and_plot_one_row(pooling_methods, metric, sort_by="eukaryote"):
    top_results = []

    # Iterate over each pooling method
    for pooling in pooling_methods:
        # Extract the metrics
        sorted_results = extract_metrics(pooling, metric, sort_by=sort_by)
        
        # Get the top result
        if not sorted_results.empty:
            top_result = sorted_results.iloc[0]
            top_result['method'] = pooling  # Add the pooling method to the result
            top_results.append(top_result)

    # Convert the list of top results to a DataFrame
    if top_results:
        df_top_results = pd.DataFrame(top_results)

        # Sort DataFrame based on the 'sort_by' category in descending order
        df_top_results.sort_values(by=sort_by, ascending=False, inplace=True)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 7))
        categories = ['human', 'mammalian', 'vertebrate', 'animal', 'eukaryote']
        n_methods = len(pooling_methods)
        bar_width = 0.1  # bar width

        # Plot horizontal bars for each category and each method
        for i, category in enumerate(categories):
            offsets = [j * (bar_width * len(categories) + 0.05) for j in range(n_methods)]
            ax.barh([x + i * bar_width for x in offsets], df_top_results[category], height=bar_width, label=category)

        # Label formatting
        ax.set_yticks([i + 2 * bar_width for i in offsets])
        ax.set_yticklabels(df_top_results['method'])
        ax.set_xlabel(metric)
        ax.set_title(f'Top {metric} Scores by Pooling Method Sorted by {sort_by}')
        ax.legend(loc="upper left")
        plt.show()

################################################################################################


def analyze_and_plot_old(pooling_methods, metric, sort_by="eukaryote"):
    results_by_category = {category: [] for category in ['human', 'mammalian', 'vertebrate', 'animal', 'eukaryote']}

    # Iterate over each pooling method and extract the metrics
    for pooling in pooling_methods:
        sorted_results = extract_metrics(pooling, metric, sort_by=sort_by)
        
        # For each category, find the best performance and store it
        if not sorted_results.empty:
            for category in results_by_category:
                top_result = sorted_results.sort_values(by=category, ascending=False).iloc[0]
                top_result['method'] = pooling
                results_by_category[category].append(top_result)

    # Plotting
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    categories = ['human', 'mammalian', 'vertebrate', 'animal', 'eukaryote']

    for i, category in enumerate(categories):
        df_top_results = pd.DataFrame(results_by_category[category])
        df_top_results.sort_values(by=category, ascending=True, inplace=True)

        axes[i].barh(df_top_results['method'], df_top_results[category], color=plt.cm.Paired(np.arange(len(df_top_results))))
        axes[i].set_title(category.capitalize())
        axes[i].invert_yaxis()  # Invert the y-axis to have the highest value at the top

    plt.xlabel(metric)
    plt.tight_layout()
    plt.show()

################################################################################################


def analyze_and_plot(pooling_methods, metric, sort_by="eukaryote", save_path=None):
    # Dictionary to hold the best results for each category
    results_by_category = {category: [] for category in ['human', 'mammalian', 'vertebrate', 'animal', 'eukaryote']}

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Initial extraction sorted by a default category for initial filtering
        initial_results = extract_metrics(pooling, metric, sort_by=sort_by)
        
        # For each category, re-sort the results and find the best performance
        if not initial_results.empty:
            for category in results_by_category:
                # Re-sort by the current category
                sorted_results = initial_results.sort_values(by=category, ascending=False)
                if not sorted_results.empty:
                    top_result = sorted_results.iloc[0]  # Get the top result
                    top_result['method'] = pooling
                    results_by_category[category].append(top_result)

    # Create a set of all methods across all categories to maintain color consistency
    all_methods = set()
    for results in results_by_category.values():
        for result in results:
            all_methods.add(result['method'])
    all_methods = list(all_methods)  # Convert set to list to maintain order

    # Color map based on all methods
    color_map = plt.cm.Paired(np.arange(len(all_methods)))
    color_dict = {method: color_map[i] for i, method in enumerate(all_methods)}

    # Plotting
    fig, axes = plt.subplots(len(results_by_category), 1, figsize=(10, 15))

    for i, (category, results) in enumerate(results_by_category.items()):
        df_top_results = pd.DataFrame(results)
        df_top_results.sort_values(by=category, ascending=False, inplace=True)  # Ensure highest value is at top

        # Plotting each method's performance
        for j, row in df_top_results.iterrows():
            axes[i].barh(row['method'], row[category], color=color_dict[row['method']])

        # Individual x-limits for each subplot
        min_val, max_val = df_top_results[category].min(), df_top_results[category].max()
        axes[i].set_xlim([min_val * 0.95, max_val * 1.03])

        axes[i].set_title(category.capitalize())
        axes[i].invert_yaxis()  # Invert y-axis to have the highest value at the top

    fig.text(0.5, 0.04, metric, ha='center', fontsize=12)  # Common x-label
    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()




def analyze_and_plot_single_old1(pooling_methods, metric, category, color_dict, k=1, save_path=None):
    # Dictionary to hold the best results for one category
    results_by_category = []

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Initial extraction sorted by the provided category
        sorted_results = extract_metrics(pooling, metric, sort_by=category)
        
        # Find the best performance for the category
        if not sorted_results.empty:
            top_result = sorted_results.sort_values(by=category, ascending=False).iloc[0]
            top_result['method'] = pooling
            results_by_category.append(top_result)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=category, ascending=False, inplace=True)  # Highest value at the top

    # Plotting each method's performance using the passed color dictionary
    for i, row in df_top_results.iterrows():
        ax.barh(row['method'], row[category], color=color_dict[row['method']])

    # Individual x-limits
    min_val, max_val = df_top_results[category].min(), df_top_results[category].max()
    ax.set_xlim([min_val * 0.95, max_val * 1.03])

    ax.set_title(f"{category.capitalize()} Performance Comparison")
    ax.set_xlabel(metric)
    ax.set_ylabel('Methods')
    ax.invert_yaxis()  # Invert y-axis to have the highest value at the top

    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return df_top_results


def analyze_and_plot_single_old2_withoutSTD(pooling_methods, metric, category, color_dict, k=1, save_path=None):
    """
    Analyzes and plots the average of the top k results for each pooling method.
    
    Parameters:
    - pooling_methods: list of pooling method names to analyze.
    - metric: the metric to compare across pooling methods.
    - category: the category by which to sort and compare results.
    - color_dict: dictionary mapping pooling method names to colors.
    - k: the number of top results to average. Defaults to 1 (taking the top result only).
    - save_path: path to save the plot image (optional).
    
    Returns:
    - DataFrame of the top results averaged per pooling method.
    """
    # List to hold the best/average results for one category
    results_by_category = []

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Extract and sort results by the provided category
        sorted_results = extract_metrics(pooling, metric, sort_by=category)
        
        # Determine the average of the top k results if available
        if not sorted_results.empty:
            top_k_results = sorted_results.sort_values(by=category, ascending=False).head(k)
            avg_result = top_k_results.mean(numeric_only=True)
            avg_result['method'] = pooling
            results_by_category.append(avg_result)
            

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=category, ascending=False, inplace=True)  # Highest value at the top

    # Plot each method's performance using the passed color dictionary
    for i, row in df_top_results.iterrows():
        ax.barh(row['method'], row[category], color=color_dict.get(row['method'], 'gray'))

    # Individual x-limits
    min_val, max_val = df_top_results[category].min(), df_top_results[category].max()
    ax.set_xlim([min_val * 0.95, max_val * 1.03])

    ax.set_title(f"{category.capitalize()} Performance Comparison (Top {k} Average)")
    ax.set_xlabel(metric)
    ax.set_ylabel('Methods')
    ax.invert_yaxis()  # Invert y-axis to have the highest value at the top

    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return df_top_results

#########

def analyze_and_plot_single(pooling_methods, metric, category, color_dict, k=1, save_path=None):
    """
    Analyzes and plots the average of the top k results for each pooling method.
    
    Parameters:
    - pooling_methods: list of pooling method names to analyze.
    - metric: the metric to compare across pooling methods.
    - category: the category by which to sort and compare results.
    - color_dict: dictionary mapping pooling method names to colors.
    - k: the number of top results to average. Defaults to 1 (taking the top result only).
    - save_path: path to save the plot image (optional).
    
    Returns:
    - DataFrame of the top results averaged per pooling method.
    """
    # List to hold the best/average results for one category
    results_by_category = []

    # Iterate over each pooling method and extract metrics
    for pooling in pooling_methods:
        # Extract and sort results by the provided category
        if metric == "accuracy":
            sorted_results = extract_accuracy(pooling, category)
        else:
            sorted_results = extract_metrics(pooling, metric, sort_by=category)

        
        # Determine the average and standard deviation of the top k results if available
        if not sorted_results.empty:
            
            top_k_results = sorted_results.sort_values(by=category, ascending=False).head(k)
            avg_result = top_k_results.mean(numeric_only=True)
            if k > 1:
                std_result = top_k_results[category].std()
                avg_result[f"{category}_std"] = std_result
            avg_result['method'] = pooling
            results_by_category.append(avg_result)
            
    # Convert list to DataFrame
    df_top_results = pd.DataFrame(results_by_category)
    df_top_results.sort_values(by=category, ascending=False, inplace=True)  # Highest value at the top

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot each method's performance using the passed color dictionary
    for i, row in df_top_results.iterrows():
        ax.barh(row['method'], row[category], color=color_dict.get(row['method'], 'gray'))
        if k > 1:
            # Add error bars for standard deviation
            ax.errorbar(row[category], row['method'], xerr=row[f"{category}_std"], fmt='o', color='black')
    
    # Individual x-limits
    min_val, max_val = df_top_results[category].min(), df_top_results[category].max()
    ax.set_xlim([min_val * 0.95, max_val * 1.03])

    ax.set_title(f"{category.capitalize()} Performance Comparison (Top {k} Average)")
    ax.set_xlabel(metric)
    ax.set_ylabel('Methods')
    ax.invert_yaxis()  # Invert y-axis to have the highest value at the top

    plt.tight_layout()

    # Save the figure if a filepath is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

    return df_top_results
