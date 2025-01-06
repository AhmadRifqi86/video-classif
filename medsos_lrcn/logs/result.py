import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load all JSON files into a DataFrame
def load_jsons(json_folder):
    data = []
    for file in os.listdir(json_folder):
        if file.endswith(".json"):
            with open(os.path.join(json_folder, file), 'r') as f:
                content = json.load(f)
                config = content["config"]
                metrics = content["metrics"]
                row = {**config, **metrics}
                # Calculate derived fields like hidden_size
                row["HIDDEN_SIZE"] = row["MULT_FACTOR"] * row["RNN_INPUT_SIZE"]
                data.append(row)
    return pd.DataFrame(data)

# Generate distribution charts for each combination
def create_distribution_charts(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Unique combinations of CNN_BACKBONE and RNN_TYPE
    combinations = df[["CNN_BACKBONE", "RNN_TYPE"]].drop_duplicates()

    for _, combo in combinations.iterrows():
        backbone = combo["CNN_BACKBONE"]
        rnn_type = combo["RNN_TYPE"]

        # Filter data for this combination
        subset = df[(df["CNN_BACKBONE"] == backbone) & (df["RNN_TYPE"] == rnn_type)]

        # Distribution charts for each indicator
        indicators = ["BATCH_SIZE", "RNN_INPUT_SIZE", "HIDDEN_SIZE", "DROPOUT", "BIDIR"]

        for indicator in indicators:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=subset, x=indicator, y="accuracy", palette="Set2")
            plt.title(f"Accuracy Distribution for {backbone}-{rnn_type} by {indicator}")
            plt.ylabel("Accuracy")
            plt.xlabel(indicator)
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            filename = f"{backbone}_{rnn_type}_{indicator}_accuracy_distribution.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

            # Add F1-score distribution
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=subset, x=indicator, y="f1_score", palette="Set3")
            plt.title(f"F1-Score Distribution for {backbone}-{rnn_type} by {indicator}")
            plt.ylabel("F1-Score")
            plt.xlabel(indicator)
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            filename = f"{backbone}_{rnn_type}_{indicator}_f1_distribution.png"
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()

    print(f"Distribution charts saved in {output_dir}")

# Main function
if __name__ == "__main__":
    # Specify the folder containing the JSON files
    json_folder = "/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/medsos_lrcn/logs/grid_medsos_checkpoint.json"
    output_dir = "/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/medsos_lrcn/charts"

    # Load the data
    df = load_jsons(json_folder)

    # Create distribution charts
    create_distribution_charts(df, output_dir)
