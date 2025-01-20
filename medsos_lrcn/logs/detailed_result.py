import json
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load JSON data from file
def load_json_from_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
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
            "hidden_size": config["RNN_INPUT_SIZE"] * config["MULT_FACTOR"],
            "batch_size": config["BATCH_SIZE"],
            "dropout": config["DROPOUT"],
            "accuracy": metrics["accuracy"]
        })
    return pd.DataFrame(rows)

# Plot violin chart for each CNN backbone separately
def plot_violin_by_cnn(data):
    cnn_backbones = data["CNN_BACKBONE"].unique()
    for cnn in cnn_backbones:
        subset_data = data[data["CNN_BACKBONE"] == cnn]
        
        # Create a combined configuration label for grouping
        subset_data['config_label'] = subset_data.apply(
            lambda x: f"hidden:{x['hidden_size']}-batch:{x['batch_size']}-drop:{x['dropout']}",
            axis=1
        )
        
        plt.figure(figsize=(20, 10))
        sns.violinplot(
            data=subset_data,
            x="config_label",
            y="accuracy",
            hue="RNN_TYPE",  # Split based on RNN_TYPE (LSTM vs Mamba)
            split=True,      # Enables side-by-side comparison in each violin
            inner="quart",
            palette="muted"
        )
        plt.title(f"Accuracy Distribution by Configurations (LSTM vs Mamba) for {cnn}")
        plt.xlabel("Configurations")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=90)  # Rotate x-axis labels for readability
        plt.legend(title="RNN Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

# Main function
def main():
    file_path = "grid_medsos_checkpoint.json"  # Replace with your JSON file path
    json_data = load_json_from_file(file_path)
    data = parse_json_to_dataframe(json_data)
    plot_violin_by_cnn(data)

if __name__ == "__main__":
    main()
