import subprocess
import all_config
import os
import time
import re


def run_training(config, test_runs, best_results):
    best_f1 = None
    best_model_filename = None

    for run in range(test_runs):
        sed_commands = []

        # Apply configuration to source code with proper handling of string values
        for key, value in config.items():
            if isinstance(value, str):
                sed_command = f"sed -i '/^{key} =/ s|=.*|= \"{value}\"|' {all_config.CONFIG_PATH}"  # Quote strings
            else:
                sed_command = f"sed -i '/^{key} =/ s|=.*|= {value}|' {all_config.CONFIG_PATH}"  # Leave non-strings as-is
            sed_commands.append(sed_command)

        print("Applying config:")
        print(config)

        # Execute sed commands
        for command in sed_commands:
            subprocess.run(command, shell=True)

        # Run training
        print("Starting training...")
        process = subprocess.Popen(
            f'python3 {all_config.SOURCE_PATH}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        result = stdout.decode('utf-8')
        error_output = stderr.decode('utf-8')

        print("Training completed.")
        try:
            # Extract metrics
            accuracy, precision, recall, f1, train_dur, inf_dur = extract_metrics(result)
            print(f"Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, "
                  f"Train Duration={train_dur}s, Inference Duration={inf_dur}s")
        except Exception as e:
            # Log errors during metric extraction
            with open(all_config.LOG_FILE_PATH, 'a') as log_file:
                log_file.write(f"Error extracting metrics: {e}\n")
                log_file.write(f"Run {run} output:\n{result}\n")
                log_file.write(f"Error Output:\n{error_output}\n\n")
            print(f"Error extracting metrics: {e}")
            continue

        # Save the best model
        if best_f1 is None or (f1 > best_f1 and f1 > 0.73):
            best_f1 = f1
            best_model_filename = (
                f"best_model_seq{config['SEQUENCE_LENGTH']}_batch{config['BATCH_SIZE']}_hidden{config['HIDDEN_SIZE']}_"
                f"cnn{config['CNN_BACKBONE']}_rnn{config['RNN_INPUT_SIZE']}_layer{config['RNN_LAYER']}_"
                f"rnnType{config['RNN_TYPE']}_method{config['SAMPLING_METHOD']}_out{config['RNN_OUT']}_"
                f"max{config['MAX_VIDEOS']}_epochs{config['EPOCH']}_classifmode{config['CLASSIF_MODE']}_"
                f"f1{f1:.4f}.pth"
            )
            best_model_path = os.path.join(all_config.BEST_MODEL_DIR, best_model_filename)

            # Save the best model file
            print(f"Saving best model: {best_model_filename}")
            subprocess.run(f"cp {all_config.MODEL_PATH} {best_model_path}", shell=True)

            # Update best results
            best_results.append({
                "config": config,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "training_duration": train_dur,
                    "inference_duration": inf_dur
                },
                "best_model_filename": best_model_filename
            })

        # Log results for the current run
        with open(all_config.LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"Run {run + 1}/{test_runs}\n")
            log_file.write(f"Config: {config}\n")
            log_file.write(f"Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, "
                           f"Train Duration={train_dur}s, Inference Duration={inf_dur}s\n")
            if error_output:
                log_file.write(f"Error Output:\n{error_output}\n\n")
    time.sleep(all_config.SLEEP)
    return best_f1, best_model_filename


# Extract metrics
def extract_metrics(output):
    patterns = {
        "accuracy": r"Overall Accuracy: (\d\.\d+|\d\.\d)",
        "precision": r"Overall Precision: (\d\.\d+|\d\.\d)",
        "recall": r"Overall Recall: (\d\.\d+|\d\.\d)",
        "f1": r"Overall F1-Score: (\d\.\d+|\d\.\d)",
        "train_duration": r"training_duration:\s+([\d.]+)",
        "inf_duration": r"inference_duration:\s+([\d.]+)"
    }

    metrics = {key: float(re.search(pattern, output).group(1)) for key, pattern in patterns.items()}
    return metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"], metrics["train_duration"], metrics["inf_duration"]