# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import sys

# # Function to load and parse JSON data from file
# def load_data(file_path):
#     """
#     Load model configurations from a JSON file
#     """
#     try:
#         with open(file_path, 'r') as f:
#             data = json.load(f)
#         return data
#     except FileNotFoundError:
#         print(f"Error: File {file_path} not found.")
#         sys.exit(1)
#     except json.JSONDecodeError:
#         print(f"Error: File {file_path} contains invalid JSON.")
#         sys.exit(1)

# # Function to convert data to DataFrame
# def create_dataframe(data):
#     """
#     Convert JSON data to a pandas DataFrame
#     """
#     # Check if data is a list of dictionaries
#     if isinstance(data, list):
#         df = pd.DataFrame(data)
#     # If it's a dictionary of dictionaries, convert to list first
#     elif isinstance(data, dict):
#         df = pd.DataFrame(list(data.values()))
#     else:
#         print("Error: Unexpected data format. Expected a list or dictionary of model configurations.")
#         sys.exit(1)
#     return df

# # Function to create comparative histograms
# def plot_parameter_comparison(df, selected_hyperparams, output_file='parameter_comparison.png'):
#     """
#     Create side-by-side histograms comparing LSTM and Mamba for selected hyperparameters
#     """
#     # Set style
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(20, 15))
    
#     # Filter data for LSTM and Mamba
#     lstm_data = df[df['RNN_TYPE'] == 'lstm']
#     mamba_data = df[df['RNN_TYPE'] == 'mamba']
    
#     if lstm_data.empty or mamba_data.empty:
#         print("Error: Data does not contain both 'lstm' and 'mamba' RNN types.")
#         sys.exit(1)
    
#     # Plot histograms for each hyperparameter
#     for i, param in enumerate(selected_hyperparams):
#         if param not in df.columns:
#             print(f"Warning: Hyperparameter '{param}' not found in data. Skipping.")
#             continue
            
#         plt.subplot(3, 3, i+1)
        
#         # Get unique values for this hyperparameter
#         unique_values = sorted(df[param].unique())
        
#         # Prepare data for the bar plot
#         labels = []
#         lstm_params = []
#         mamba_params = []
        
#         for val in unique_values:
#             labels.append(str(val))
            
#             # Get mean parameter count for LSTM with this value
#             lstm_vals = lstm_data[lstm_data[param] == val]['NUM_PARAMETERS']
#             if not lstm_vals.empty:
#                 lstm_params.append(lstm_vals.mean())
#             else:
#                 lstm_params.append(0)
            
#             # Get mean parameter count for Mamba with this value
#             mamba_vals = mamba_data[mamba_data[param] == val]['NUM_PARAMETERS']
#             if not mamba_vals.empty:
#                 mamba_params.append(mamba_vals.mean())
#             else:
#                 mamba_params.append(0)
        
#         # Create positions for bars
#         x = np.arange(len(labels))
#         width = 0.35
        
#         # Create bars
#         plt.bar(x - width/2, lstm_params, width, label='LSTM', color='blue', alpha=0.7)
#         plt.bar(x + width/2, mamba_params, width, label='SSM (Mamba)', color='red', alpha=0.7)
        
#         # Add labels and title
#         plt.xlabel(param)
#         plt.ylabel('Number of Parameters')
#         plt.title(f'Parameter Count by {param}')
#         plt.xticks(x, labels)
#         plt.legend()
        
#         # Add value labels on top of bars
#         for j, v in enumerate(lstm_params):
#             if v > 0:  # Only add labels for non-zero values
#                 plt.text(j - width/2, v + 0.01*max(max(lstm_params), max(mamba_params)), 
#                          f'{int(v):,}', ha='center', va='bottom', rotation=45, fontsize=8)
#         for j, v in enumerate(mamba_params):
#             if v > 0:  # Only add labels for non-zero values
#                 plt.text(j + width/2, v + 0.01*max(max(lstm_params), max(mamba_params)), 
#                          f'{int(v):,}', ha='center', va='bottom', rotation=45, fontsize=8)
    
#     # Add an overall title
#     plt.suptitle('Comparison of Parameter Count: LSTM vs SSM (Mamba)', fontsize=16, y=0.98)
    
#     # Adjust layout
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
    
#     # Save or show
#     plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     print(f"Plot saved to {output_file}")
#     plt.show()

# # Function to calculate and display parameter difference statistics
# def display_parameter_stats(df):
#     """
#     Calculate and display statistics about parameter differences between LSTM and Mamba
#     """
#     # Group by all parameters except RNN_TYPE and NUM_PARAMETERS
#     group_cols = [col for col in df.columns if col not in ['RNN_TYPE', 'NUM_PARAMETERS']]
    
#     results = []
    
#     for _, group in df.groupby(group_cols):
#         if len(group) >= 2 and 'lstm' in group['RNN_TYPE'].values and 'mamba' in group['RNN_TYPE'].values:
#             lstm_params = group[group['RNN_TYPE'] == 'lstm']['NUM_PARAMETERS'].values[0]
#             mamba_params = group[group['RNN_TYPE'] == 'mamba']['NUM_PARAMETERS'].values[0]
#             diff = lstm_params - mamba_params
#             diff_percent = (diff / lstm_params) * 100 if lstm_params != 0 else 0
            
#             row = {col: group[col].values[0] for col in group_cols}
#             row['LSTM_PARAMS'] = lstm_params
#             row['MAMBA_PARAMS'] = mamba_params
#             row['DIFF'] = diff
#             row['DIFF_PERCENT'] = diff_percent
            
#             results.append(row)
    
#     if not results:
#         print("No matching configurations found for both LSTM and Mamba.")
#         return None
    
#     stats_df = pd.DataFrame(results)
    
#     print("\nParameter Difference Statistics:")
#     print(f"Average difference: {stats_df['DIFF'].mean():,.2f} parameters")
#     print(f"Average percentage difference: {stats_df['DIFF_PERCENT'].mean():.2f}%")
#     print(f"Maximum difference: {stats_df['DIFF'].max():,.2f} parameters")
#     print(f"Minimum difference: {stats_df['DIFF'].min():,.2f} parameters")
    
#     return stats_df

# # Main function
# def main():
#     # Check for command line arguments
#     if len(sys.argv) < 2:
#         print("Usage: python script_name.py path_to_json_file [output_image_path]")
#         sys.exit(1)
    
#     # Get file paths from command line arguments
#     json_file_path = sys.argv[1]
#     output_file = sys.argv[2] if len(sys.argv) > 2 else 'parameter_comparison.png'
    
#     # Load data
#     data = load_data(json_file_path)
    
#     # Convert to DataFrame
#     df = create_dataframe(data)
    
#     # Check required columns
#     required_cols = ['RNN_TYPE', 'NUM_PARAMETERS']
#     for col in required_cols:
#         if col not in df.columns:
#             print(f"Error: Required column '{col}' not found in data.")
#             sys.exit(1)
    
#     # Select 8 hyperparameters to compare
#     all_hyperparams = [col for col in df.columns if col not in ['RNN_TYPE', 'NUM_PARAMETERS']]
#     selected_hyperparams = all_hyperparams[:8] if len(all_hyperparams) > 8 else all_hyperparams
    
#     print(f"Comparing parameters across these hyperparameters: {', '.join(selected_hyperparams)}")
    
#     # Create plots
#     plot_parameter_comparison(df, selected_hyperparams, output_file)
    
#     # Display statistics
#     stats_df = display_parameter_stats(df)
#     if stats_df is not None:
#         print("\nDetailed statistics by configuration:")
#         display_cols = [col for col in stats_df.columns if col not in ['DIFF', 'DIFF_PERCENT']]
#         print(stats_df[display_cols + ['DIFF', 'DIFF_PERCENT']])

# if __name__ == "__main__":
#     main()

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

# Function to load and parse JSON data from file
def load_data(file_path):
    """
    Load model configurations from a JSON file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {file_path} contains invalid JSON.")
        sys.exit(1)

# Function to convert data to DataFrame
def create_dataframe(data):
    """
    Convert JSON data to a pandas DataFrame
    """
    # Check if data is a list of dictionaries
    if isinstance(data, list):
        df = pd.DataFrame(data)
    # If it's a dictionary of dictionaries, convert to list first
    elif isinstance(data, dict):
        df = pd.DataFrame(list(data.values()))
    else:
        print("Error: Unexpected data format. Expected a list or dictionary of model configurations.")
        sys.exit(1)
    return df

# Function to create comparative histograms for RNN_LAYER and HIDDEN_SIZE
def plot_layer_size_comparison(df, output_file='parameter_comparison.png'):
    """
    Create side-by-side histograms comparing LSTM and Mamba for RNN_LAYER and HIDDEN_SIZE
    """
    # Set style
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Filter data for LSTM and Mamba
    lstm_data = df[df['RNN_TYPE'] == 'lstm']
    mamba_data = df[df['RNN_TYPE'] == 'mamba']
    
    if lstm_data.empty or mamba_data.empty:
        print("Error: Data does not contain both 'lstm' and 'mamba' RNN types.")
        sys.exit(1)
    
    # Plot for RNN_LAYER
    param = 'RNN_LAYER'
    ax = axes[0]
    
    # Get unique values for RNN_LAYER
    unique_values = sorted(df[param].unique())
    
    # Prepare data for the bar plot
    labels = []
    lstm_params = []
    mamba_params = []
    
    for val in unique_values:
        labels.append(str(val))
        
        # Get mean parameter count for LSTM with this value
        lstm_vals = lstm_data[lstm_data[param] == val]['NUM_PARAMETERS']
        if not lstm_vals.empty:
            lstm_params.append(lstm_vals.mean())
        else:
            lstm_params.append(0)
        
        # Get mean parameter count for Mamba with this value
        mamba_vals = mamba_data[mamba_data[param] == val]['NUM_PARAMETERS']
        if not mamba_vals.empty:
            mamba_params.append(mamba_vals.mean())
        else:
            mamba_params.append(0)
    
    # Create positions for bars
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, lstm_params, width, label='LSTM', color='blue', alpha=0.7)
    ax.bar(x + width/2, mamba_params, width, label='SSM (Mamba)', color='red', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Number of RNN Layers')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Parameter Count by RNN Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on top of bars
    for j, v in enumerate(lstm_params):
        if v > 0:
            ax.text(j - width/2, v + 0.01*max(max(lstm_params), max(mamba_params)), 
                     f'{int(v):,}', ha='center', va='bottom', rotation=45, fontsize=8)
    for j, v in enumerate(mamba_params):
        if v > 0:
            ax.text(j + width/2, v + 0.01*max(max(lstm_params), max(mamba_params)), 
                     f'{int(v):,}', ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Plot for HIDDEN_SIZE
    param = 'HIDDEN_SIZE'
    ax = axes[1]
    
    # Get unique values for HIDDEN_SIZE
    unique_values = sorted(df[param].unique())
    
    # Prepare data for the bar plot
    labels = []
    lstm_params = []
    mamba_params = []
    
    for val in unique_values:
        labels.append(str(val))
        
        # Get mean parameter count for LSTM with this value
        lstm_vals = lstm_data[lstm_data[param] == val]['NUM_PARAMETERS']
        if not lstm_vals.empty:
            lstm_params.append(lstm_vals.mean())
        else:
            lstm_params.append(0)
        
        # Get mean parameter count for Mamba with this value
        mamba_vals = mamba_data[mamba_data[param] == val]['NUM_PARAMETERS']
        if not mamba_vals.empty:
            mamba_params.append(mamba_vals.mean())
        else:
            mamba_params.append(0)
    
    # Create positions for bars
    x = np.arange(len(labels))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, lstm_params, width, label='LSTM', color='blue', alpha=0.7)
    ax.bar(x + width/2, mamba_params, width, label='SSM', color='red', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Hidden Size')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Parameter Count by Hidden Size')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Add value labels on top of bars
    for j, v in enumerate(lstm_params):
        if v > 0:
            ax.text(j - width/2, v + 0.01*max(max(lstm_params), max(mamba_params)), 
                     f'{int(v):,}', ha='center', va='bottom', rotation=45, fontsize=8)
    for j, v in enumerate(mamba_params):
        if v > 0:
            ax.text(j + width/2, v + 0.01*max(max(lstm_params), max(mamba_params)), 
                     f'{int(v):,}', ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Add an overall title
    fig.suptitle('Comparison of Parameter Count: LSTM vs SSM', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save or show
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.show()

# Function to create a combined comparison of RNN_LAYER and HIDDEN_SIZE
def plot_combined_comparison(df, output_file='combined_comparison.png'):
    """
    Create a matrix plot showing parameter counts for combinations of RNN_LAYER and HIDDEN_SIZE
    """
    # Filter data for LSTM and Mamba
    lstm_data = df[df['RNN_TYPE'] == 'lstm']
    mamba_data = df[df['RNN_TYPE'] == 'mamba']
    
    # Get unique values
    layers = sorted(df['RNN_LAYER'].unique())
    sizes = sorted(df['HIDDEN_SIZE'].unique())
    
    # Compute parameter difference percentages
    results = []
    
    for layer in layers:
        for size in sizes:
            lstm_vals = lstm_data[(lstm_data['RNN_LAYER'] == layer) & 
                                  (lstm_data['HIDDEN_SIZE'] == size)]['NUM_PARAMETERS']
            
            mamba_vals = mamba_data[(mamba_data['RNN_LAYER'] == layer) & 
                                    (mamba_data['HIDDEN_SIZE'] == size)]['NUM_PARAMETERS']
            
            if not lstm_vals.empty and not mamba_vals.empty:
                lstm_params = lstm_vals.mean()
                mamba_params = mamba_vals.mean()
                diff = lstm_params - mamba_params
                diff_percent = (diff / lstm_params) * 100
                
                results.append({
                    'RNN_LAYER': layer,
                    'HIDDEN_SIZE': size,
                    'LSTM_PARAMS': lstm_params,
                    'MAMBA_PARAMS': mamba_params,
                    'DIFF': diff,
                    'DIFF_PERCENT': diff_percent
                })
    
    if not results:
        print("No matching configurations found for both LSTM and Mamba.")
        return
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Create a pivot table for the heatmap
    pivot_df = result_df.pivot(index='RNN_LAYER', columns='HIDDEN_SIZE', values='DIFF_PERCENT')
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with diverging colormap (blue for negative, red for positive)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    ax = sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap=cmap, center=0,
                cbar_kws={'label': 'Parameter Difference (% fewer in Mamba)'})
    
    # Add labels and title
    plt.title('Parameter Reduction in Mamba vs LSTM (%)', fontsize=16)
    plt.xlabel('Hidden Size', fontsize=12)
    plt.ylabel('RNN Layers', fontsize=12)
    
    # Save or show
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Combined comparison plot saved to {output_file}")
    plt.show()

# Function to calculate and display parameter difference statistics
def display_parameter_stats_by_layer_size(df):
    """
    Calculate and display statistics about parameter differences between LSTM and Mamba
    grouped by RNN_LAYER and HIDDEN_SIZE
    """
    # Group by RNN_LAYER and HIDDEN_SIZE
    group_cols = ['RNN_LAYER', 'HIDDEN_SIZE']
    
    results = []
    
    for _, group in df.groupby(group_cols):
        if len(group) >= 2 and 'lstm' in group['RNN_TYPE'].values and 'mamba' in group['RNN_TYPE'].values:
            lstm_params = group[group['RNN_TYPE'] == 'lstm']['NUM_PARAMETERS'].values[0]
            mamba_params = group[group['RNN_TYPE'] == 'mamba']['NUM_PARAMETERS'].values[0]
            diff = lstm_params - mamba_params
            diff_percent = (diff / lstm_params) * 100 if lstm_params != 0 else 0
            
            row = {col: group[col].values[0] for col in group_cols}
            row['LSTM_PARAMS'] = lstm_params
            row['MAMBA_PARAMS'] = mamba_params
            row['DIFF'] = diff
            row['DIFF_PERCENT'] = diff_percent
            
            results.append(row)
    
    if not results:
        print("No matching configurations found for both LSTM and Mamba.")
        return None
    
    stats_df = pd.DataFrame(results)
    
    print("\nParameter Difference Statistics:")
    print(f"Average difference: {stats_df['DIFF'].mean():,.2f} parameters")
    print(f"Average percentage difference: {stats_df['DIFF_PERCENT'].mean():.2f}%")
    print(f"Maximum difference: {stats_df['DIFF'].max():,.2f} parameters")
    print(f"Minimum difference: {stats_df['DIFF'].min():,.2f} parameters")
    
    # Sort by layer and size for better readability
    stats_df = stats_df.sort_values(['RNN_LAYER', 'HIDDEN_SIZE'])
    
    return stats_df

# Main function
def main():
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path_to_json_file [output_base_name]")
        sys.exit(1)
    
    # Get file paths from command line arguments
    json_file_path = sys.argv[1]
    output_base = sys.argv[2] if len(sys.argv) > 2 else 'parameter_comparison'
    
    # Load data
    data = load_data(json_file_path)
    
    # Convert to DataFrame
    df = create_dataframe(data)
    
    # Check required columns
    required_cols = ['RNN_TYPE', 'NUM_PARAMETERS', 'RNN_LAYER', 'HIDDEN_SIZE']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in data.")
            sys.exit(1)
    
    # Create individual plots for RNN_LAYER and HIDDEN_SIZE
    plot_layer_size_comparison(df, f"{output_base}_individual.png")
    
    # Create combined comparison plot
    plot_combined_comparison(df, f"{output_base}_combined.png")
    
    # Display statistics
    stats_df = display_parameter_stats_by_layer_size(df)
    if stats_df is not None:
        print("\nDetailed statistics by RNN_LAYER and HIDDEN_SIZE:")
        pd.set_option('display.float_format', '{:.2f}'.format)
        print(stats_df[['RNN_LAYER', 'HIDDEN_SIZE', 'LSTM_PARAMS', 'MAMBA_PARAMS', 'DIFF', 'DIFF_PERCENT']])

if __name__ == "__main__":
    main()