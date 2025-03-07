import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Parse JSON into a DataFrame
def parse_json_to_dataframe(json_data, source):
    """
    Convert JSON data to DataFrame and add a column indicating the source file.
    """
    rows = []
    for entry in json_data:
        config = entry["config"]
        metrics = entry["metrics"]
        if config["CNN_BACKBONE"] == "mobilenet_v2":  # Filter only mobilenet_v2 models
            rows.append({
                    "Source": source,  # Identify the file
                    "CNN_BACKBONE": config["CNN_BACKBONE"],
                    "RNN_TYPE": config["RNN_TYPE"],
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"]
            })
    return pd.DataFrame(rows)

# Plot violin chart for Accuracy and F1-score
def plot_violin(data1, data2):
    """
    Plot accuracy and F1-score distribution for both files in a single plot.
    """
    # Combine both datasets for visualization
    combined_data = pd.concat([data1, data2], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=combined_data,
        x="Source",  # Compare by source
        y="accuracy",
        hue="RNN_TYPE",
        split=True,
        inner="quart",
        palette="muted"
    )
    plt.title("Accuracy Distribution for SSM vs LSTM (With Adapt Layer vs No Adapt Layer)")
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=combined_data,
        x="Source",
        y="f1_score",
        hue="RNN_TYPE",
        split=True,
        inner="quart",
        palette="muted"
    )
    plt.title("F1-Score Distribution for SSM vs LSTM (With Adapt Layer vs No Adapt Layer)")
    plt.show()

# Main function
def main():
    file_path1 = "grid_medsos_checkpoint.json"  # Path to first JSON file
    file_path2 = "grid_medsos_checkpoint_noadapt.json"  # Path to second JSON file
    
    # Load and process data separately
    json_data1 = load_json_from_file(file_path1)
    json_data2 = load_json_from_file(file_path2)

    
    data1 = parse_json_to_dataframe(json_data1, "With Adapt Layer")
    data2 = parse_json_to_dataframe(json_data2, "No Adapt Layer")

    data1 = modify_rnn_type(data1)
    data2 = modify_rnn_type(data2)

    # Plot both datasets in the same image
    plot_violin(data1, data2)

if __name__ == "__main__":
    main()




# import json
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load JSON data from file
# def load_json_from_file(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return data

# def modify_rnn_type(data):
#     """
#     Replace "mamba" with "ssm" to maintain consistency.
#     """
#     data["RNN_TYPE"] = data["RNN_TYPE"].replace("mamba", "ssm")
#     return data

# # Parse JSON into a DataFrame
# def parse_json_to_dataframe(json_data, source):
#     """
#     Convert JSON data to DataFrame and add a column indicating the source file.
#     """
#     rows = []
#     for entry in json_data:
#         config = entry["config"]
#         metrics = entry["metrics"]
#         rows.append({
#             "Source": source,  # Identify the file
#             "CNN_BACKBONE": config["CNN_BACKBONE"],
#             "RNN_TYPE": config["RNN_TYPE"],
#             "accuracy": metrics["accuracy"],
#             "f1_score": metrics["f1_score"]
#         })
#     return pd.DataFrame(rows)

# # Plot violin chart for Accuracy and F1-score
# def plot_violin(data1, data2):
#     """
#     Plot accuracy and F1-score distribution for both files in a single plot.
#     """
#     # Combine both datasets for visualization
#     combined_data = pd.concat([data1, data2], ignore_index=True)

#     # Create a new column to differentiate CNN_BACKBONE and Source
#     combined_data["Category"] = combined_data["CNN_BACKBONE"] + " (" + combined_data["Source"] + ")"

#     plt.figure(figsize=(12, 6))
#     sns.violinplot(
#         data=combined_data,
#         x="Category",  # X-axis has CNN_BACKBONE + Source
#         y="accuracy",
#         hue="RNN_TYPE",
#         split=True,
#         inner="quart",
#         palette="muted"
#     )
#     plt.xticks(rotation=30)  # Rotate for better visibility
#     plt.title("Accuracy Distribution for SSM vs LSTM (With Adapt Layer vs No Adapt Layer)")
#     plt.show()

#     plt.figure(figsize=(12, 6))
#     sns.violinplot(
#         data=combined_data,
#         x="Category",  # X-axis has CNN_BACKBONE + Source
#         y="f1_score",
#         hue="RNN_TYPE",
#         split=True,
#         inner="quart",
#         palette="muted"
#     )
#     plt.xticks(rotation=30)  # Rotate for better visibility
#     plt.title("F1-Score Distribution for SSM vs LSTM (With Adapt Layer vs No Adapt Layer)")
#     plt.show()

# # Main function
# def main():
#     file_path1 = "grid_medsos_checkpoint.json"  # Path to first JSON file
#     file_path2 = "grid_medsos_checkpoint_noadapt.json"  # Path to second JSON file
    
#     # Load and process data separately
#     json_data1 = load_json_from_file(file_path1)
#     json_data2 = load_json_from_file(file_path2)

#     # Convert to DataFrame
#     data1 = parse_json_to_dataframe(json_data1, "With Adapt Layer")
#     data2 = parse_json_to_dataframe(json_data2, "No Adapt Layer")

#     # Modify RNN_TYPE if needed
#     data1 = modify_rnn_type(data1)
#     data2 = modify_rnn_type(data2)

#     # Plot both datasets in the same image
#     plot_violin(data1, data2)

# if __name__ == "__main__":
#     main()

