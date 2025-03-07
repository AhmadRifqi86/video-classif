import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
import numpy as np

# Load JSON data from file
def load_json_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def modify_rnn_type(data):
    """
    Replace "mamba" with "ssm" to maintain consistency.
    """
    data["RNN_TYPE"] = data["RNN_TYPE"].replace("mamba", "ssm")
    return data

# Parse the JSON data into a DataFrame
def parse_json_to_dataframe(json_data):
    rows = []
    for entry in json_data:
        config = entry["config"]
        metrics = entry["metrics"]
        rows.append({
            "CNN_BACKBONE": config["CNN_BACKBONE"],
            "RNN_TYPE": config["RNN_TYPE"],
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "training_duration": metrics["training duration"],
        })
    return pd.DataFrame(rows)

# Compute KL divergence between two distributions
def compute_kl_divergence(data, cnn, rnn1, rnn2, metric):
    subset1 = data[(data["CNN_BACKBONE"] == cnn) & (data["RNN_TYPE"] == rnn1)][metric]
    subset2 = data[(data["CNN_BACKBONE"] == cnn) & (data["RNN_TYPE"] == rnn2)][metric]
    
    # Create histograms (probability distributions)
    hist1, bins = np.histogram(subset1, bins=20, density=True)
    hist2, _ = np.histogram(subset2, bins=bins, density=True)
    
    # Normalize to ensure valid probability distributions
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Compute KL divergence
    kl_div = entropy(hist1 + 1e-10, hist2 + 1e-10)  # Add small value to avoid division by zero
    return kl_div

# Plot the violin chart
def plot_violin(data):
    """
    Generate violin plots for Accuracy, F1-Score, and Training Duration.
    """
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=data,
        x="CNN_BACKBONE",
        y="accuracy",
        hue="RNN_TYPE",
        split=True,
        inner="quart",
        palette="muted"
    )
    plt.title("Accuracy Distribution (SSM vs Transformer)")
    plt.show()
    
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=data,
        x="CNN_BACKBONE",
        y="f1_score",
        hue="RNN_TYPE",
        split=True,
        inner="quart",
        palette="muted"
    )
    plt.title("F1-Score Distribution (SSM vs Transformer)")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=data,
        x="CNN_BACKBONE",
        y="training_duration",
        hue="RNN_TYPE",
        split=True,
        inner="quart",
        palette="muted"
    )
    plt.title("Training Duration Distribution (SSM vs Transformer)")
    plt.show()

# Display KL divergence
def display_kl_divergence(data):
    """
    Compute and display KL divergence only for SSM vs Transformer.
    """
    for cnn in data["CNN_BACKBONE"].unique():
        for metric in ["accuracy", "f1_score"]:
            kl_ssm_transformer = compute_kl_divergence(data, cnn, "ssm", "transformer", metric)
            print(f"KL Divergence for {cnn} (SSM vs Transformer) on {metric}: {kl_ssm_transformer:.4f}")

# Main function
def main():
    file_path1 = "grid_medsos_checkpoint.json"  # Path to first JSON file
    file_path2 = "grid_medsos_checkpoint_transformer.json"  # Path to second JSON file
    
    # Load both JSON files
    json_data1 = load_json_from_file(file_path1)
    json_data2 = load_json_from_file(file_path2)
    
    # Combine both datasets
    combined_json_data = json_data1 + json_data2
    
    # Convert JSON data to DataFrame
    data = parse_json_to_dataframe(combined_json_data)
    
    # Modify "mamba" to "ssm"
    data = modify_rnn_type(data)

    # **Filter only SSM and Transformer (Remove LSTM)**
    data = data[data["RNN_TYPE"].isin(["ssm", "transformer"])]

    # **Plot results**
    plot_violin(data)

    # **Display KL divergence**
    #display_kl_divergence(data)

if __name__ == "__main__":
    main()






# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.stats import entropy
# import numpy as np

# # Load JSON data from file
# def load_json_from_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# def modify_rnn_type(data):
#     data["RNN_TYPE"] = data["RNN_TYPE"].replace("mamba", "ssm")
#     return data

# # Parse the JSON data into a DataFrame
# def parse_json_to_dataframe(json_data):
#     rows = []
#     for entry in json_data:
#         config = entry["config"]
#         metrics = entry["metrics"]
#         rows.append({
#             "CNN_BACKBONE": config["CNN_BACKBONE"],
#             "RNN_TYPE": config["RNN_TYPE"],
#             "accuracy": metrics["accuracy"],
#             "f1_score": metrics["f1_score"],
#             "training_duration":metrics["training duration"],
#         })
#     return pd.DataFrame(rows)

# # Compute KL divergence between two distributions
# def compute_kl_divergence(data, cnn, rnn1, rnn2, metric):
#     subset1 = data[(data["CNN_BACKBONE"] == cnn) & (data["RNN_TYPE"] == rnn1)][metric]
#     subset2 = data[(data["CNN_BACKBONE"] == cnn) & (data["RNN_TYPE"] == rnn2)][metric]
    
#     # Create histograms (probability distributions)
#     hist1, bins = np.histogram(subset1, bins=20, density=True)
#     hist2, _ = np.histogram(subset2, bins=bins, density=True)
    
#     # Normalize to ensure valid probability distributions
#     hist1 = hist1 / np.sum(hist1)
#     hist2 = hist2 / np.sum(hist2)
    
#     # Compute KL divergence
#     kl_div = entropy(hist1 + 1e-10, hist2 + 1e-10)  # Add small value to avoid division by zero
#     return kl_div

# # Plot the violin chart
# def plot_violin(data):
#     plt.figure(figsize=(12, 6))
#     sns.violinplot(
#         data=data,
#         x="CNN_BACKBONE",
#         y="accuracy",
#         hue="RNN_TYPE",
#         split=True,
#         inner="quart",
#         palette="muted"
#     )
#     plt.title("Accuracy Distribution by CNN Backbone and RNN Type")
#     plt.show()
    
#     plt.figure(figsize=(12, 6))
#     sns.violinplot(
#         data=data,
#         x="CNN_BACKBONE", #CNN_BACKBONE, RNN_TYPE
#         y="f1_score",
#         hue="RNN_TYPE",
#         split=True,
#         inner="quart",
#         palette="muted"
#     )
#     plt.title("F1-Score Distribution by CNN Backbone and RNN Type")
#     plt.show()

#     plt.figure(figsize=(12, 6))
#     sns.violinplot(
#         data=data,
#         x="CNN_BACKBONE", #CNN_BACKBONE, RNN_TYPE
#         y="training_duration",
#         hue="RNN_TYPE",
#         split=True,
#         inner="quart",
#         palette="muted"
#     )
#     plt.title("F1-Score Distribution by CNN Backbone and RNN Type")
#     plt.show()

# # Display KL divergence
# def display_kl_divergence(data):
#     for cnn in data["CNN_BACKBONE"].unique():
#         for metric in ["accuracy", "f1_score"]:
#             kl_ssm_lstm = compute_kl_divergence(data, cnn, "ssm", "lstm", metric)
#             kl_ssm_transformer = compute_kl_divergence(data, cnn, "ssm", "transformer", metric)
#             kl_lstm_transformer = compute_kl_divergence(data, cnn, "lstm", "transformer", metric)

#             print(f"KL Divergence for {cnn} (SSM vs. LSTM) on {metric}: {kl_ssm_lstm:.4f}")
#             print(f"KL Divergence for {cnn} (SSM vs. Transformer) on {metric}: {kl_ssm_transformer:.4f}")
#             print(f"KL Divergence for {cnn} (LSTM vs. Transformer) on {metric}: {kl_lstm_transformer:.4f}")

# # Main function
# def main():
#     file_path1 = "grid_medsos_checkpoint.json"  # Path to first JSON file
#     file_path2 = "grid_medsos_checkpoint_transformer.json"  # Path to second JSON file
    
#     # Load first JSON file
#     json_data1 = load_json_from_file(file_path1)
    
#     # Load second JSON file
#     json_data2 = load_json_from_file(file_path2)
    
#     # Append second JSON data to first JSON data
#     combined_json_data = json_data1 + json_data2
    
#     # Convert JSON data to DataFrame
#     data = parse_json_to_dataframe(combined_json_data)
    
#     # Modify "mamba" to "ssm"
#     data = modify_rnn_type(data)
    
#     # Plot the violin distribution
#     plot_violin(data)

# if __name__ == "__main__":
#     main()

