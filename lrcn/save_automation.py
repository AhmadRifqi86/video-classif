import os
import subprocess
import re
import itertools
import json
from datetime import datetime

# Configuration dictionary
CONFIG = {
    "EARLY_STOP": [0.0,0.42],
    "SEQUENCE_LENGTH": [40, 60],
    "BATCH_SIZE": [8],
    "HIDDEN_SIZE": [48, 56],
    "CNN_BACKBONE": ["densenet121", "resnet50"],
    "RNN_INPUT_SIZE": [512, 768],
    "RNN_LAYER": [4, 6, 2],
    "RNN_TYPE": ["lstm","gru"],
    "SAMPLING_METHOD": ["uniform", "ssim", "optiflow"],
    "RNN_OUT": ["all", "last"],
    "MAX_VIDEOS": [300, 400],
    "EPOCH": [10],
    "FINETUNE": [True],
    "CLASSIF_MODE": ["multiple_binary", "multiclass"]
}

MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/model.pth'
SOURCE_CODE_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/lrcn/dump_lrcn.py'
LOG_FILE_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/lrcn/train_log.txt'
BEST_MODEL_DIR = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/best_models/'
TEST_RUNS = 2  # Number of times to test each configuration
CHECKPOINT_FILE = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/lrcn/fix_checkpoint.json'  # File to track best results

if not os.path.exists(BEST_MODEL_DIR):
    os.makedirs(BEST_MODEL_DIR)

# Load checkpoint file (if exists)
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = f.read()
            try:
                return json.loads(data)  # Load JSON data
            except json.JSONDecodeError:
                print("Error loading checkpoint. Invalid JSON format.")
                return []
    return []

# Save checkpoint (best results) to file
def save_checkpoint(best_results):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(best_results, f, indent=4)  # Save with indentation for better readability

# Function to run the command and log the results
def run_training(config, test_runs, best_results):
    best_f1 = None
    best_model_filename = None

    for run in range(test_runs):
        # Prepare sed commands to update the source code with the current config values
        sed_commands = [
            "sed -i '/^{key} =/ s|=.*|= {value}|' {source}".format(
                key=key, value=value if isinstance(value, (int, float, list)) else f'"{value}"', source=SOURCE_CODE_PATH
            )
            for key, value in config.items()
        ]

        # Apply the sed commands
        print("Applying config")
        print(config)
        for command in sed_commands:
            subprocess.run(command, shell=True)

        # Run the training script
        print("Perform training")
        process = subprocess.Popen(f'python3 {SOURCE_CODE_PATH}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        # Process the output
        print("Training done, recording result")
        result = stdout.decode('utf-8')
        try:
            # Extract metrics from the output
            accuracy, precision, recall, f1 = extract_metrics(result)
            print("extracted f1: ", f1)
        except Exception as e:
            with open(LOG_FILE_PATH, 'a') as log_file:
                log_file.write(f"Error extracting metrics: {e}\n")
                log_file.write(f"Run {run} output: {result}\n")
            continue

        # Save the best model based on F1 score
        if best_f1 is None or f1 > best_f1:
            best_f1 = f1
            best_acc, best_precision, best_recal = accuracy,precision,recall
            best_model_filename = f"best_model_seq{config['SEQUENCE_LENGTH']}_batch{config['BATCH_SIZE']}_hidden{config['HIDDEN_SIZE']}_cnn{config['CNN_BACKBONE']}_rnn{config['RNN_INPUT_SIZE']}_layer{config['RNN_LAYER']}_rnnType{config['RNN_TYPE']}_method{config['SAMPLING_METHOD']}_out{config['RNN_OUT']}_max{config['MAX_VIDEOS']}_epochs{config['EPOCH']}_finetune{config['FINETUNE']}_f1{f1:.4f}.pth"
            best_model_path = os.path.join(BEST_MODEL_DIR, best_model_filename)

        with open(LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"Config (Run {run+1}/{test_runs}): {config}, ACCURACY={accuracy}, F1={f1}\n")
            log_file.write(result)
            if stderr:
                log_file.write(f"Error: {stderr.decode('utf-8')}\n")
            log_file.write("\n\n")

    if best_model_path:
        # Save the best model
        print(f"Saving best model for configuration: {best_model_filename}")
        subprocess.run(f"cp {MODEL_PATH} {best_model_path}", shell=True)

        best_results.append({
            "config": config,
            "metrics": {
                "accuracy": best_acc,
                "precision": best_precision,
                "recall": best_recal,
                "f1_score": best_f1
            },
            "best_model_filename": best_model_filename
        })

    return best_f1, best_model_filename


# Function to extract accuracy, precision, recall, and f1 score from the stdout
def extract_metrics(output):
    # Patterns to match precision, recall, f1, and accuracy (allowing different decimal places)
    overall_accuracy_pattern = r"Overall Accuracy: (\d\.\d+|\d\.\d)"
    precision_pattern = r"Overall Precision: (\d\.\d+|\d\.\d)"
    recall_pattern = r"Overall Recall: (\d\.\d+|\d\.\d)"
    f1_pattern = r"Overall F1-Score: (\d\.\d+|\d\.\d)"

    # Search for metrics in the output
    accuracy = re.search(overall_accuracy_pattern, output)
    precision = re.search(precision_pattern, output)
    recall = re.search(recall_pattern, output)
    f1 = re.search(f1_pattern, output)

    # Ensure all metrics are found
    if accuracy and precision and recall and f1:
        return float(accuracy.group(1)), float(precision.group(1)), float(recall.group(1)), float(f1.group(1))
    else:
        raise ValueError("Could not extract metrics from output")

# Load the checkpoint (best results) to get completed configurations
best_results = load_checkpoint()
#print(best_results)

# Convert best_results to a set of completed configurations
completed_configs = {json.dumps(result['config'], sort_keys=True) for result in best_results}

# Dynamic iteration using itertools.product
keys, values = zip(*CONFIG.items())
for value_combination in itertools.product(*values):
    # Create the config dictionary for each combination of values
    config = dict(zip(keys, value_combination))

    # Skip already completed configurations
    if json.dumps(config, sort_keys=True) in completed_configs:
        print(f"Skipping already completed configuration: {config}")
        continue

    # Run the training with the current configuration
    best_f1, best_model_filename = run_training(config, TEST_RUNS, best_results)
    print(f"Best F1 score for this configuration: {best_f1}, Model saved as: {best_model_filename}")

    # Save the checkpoint (best results) after every run
    save_checkpoint(best_results)
