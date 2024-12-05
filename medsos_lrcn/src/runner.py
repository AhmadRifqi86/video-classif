import subprocess
import all_config
import os
import time
import re
from loader_data import save_checkpoint


def run_training(config, test_runs, best_results):
    #best_f1 = None
    best_model_filename = None
    best_f1 = -float("inf")  # Use negative infinity to represent the initial state

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

        # Run training and capture real-time logs
        print("Starting training...")
        process = subprocess.Popen(
            f'python3 {all_config.SOURCE_PATH}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        result = []
        # Log training progression in real-time
        with open(all_config.LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"Run {run + 1}/{test_runs}\n")
            log_file.write(f"Config: {config}\n")
            log_file.write("Training logs:\n")
            for line in process.stdout:
                log_file.write(line)
                result.append(line)
                print(line, end="")  # Print to console as well

        stdout, stderr = process.communicate()
        result = ''.join(result)  # Concatenate list of lines into a single string
        error_output = stderr
        print("Training completed.")
        try:
            # Extract metrics
            accuracy, precision, recall, f1, train_dur, inf_dur,trainable = extract_metrics(result)
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
        if f1 > best_f1 and f1 > 0.71:  # Only update if the new result is better and exceeds the threshold
            best_f1 = f1
            best_model_filename = (
                f"best_model_seq{all_config.SEQUENCE_LENGTH}_batch{all_config.CONF_BATCH_SIZE}_hidden{all_config.HIDDEN_SIZE}_"
                f"cnn{all_config.CONF_CNN_BACKBONE}_rnn{all_config.RNN_INPUT_SIZE}_layer{all_config.RNN_LAYER}_"
                f"rnnType{all_config.RNN_TYPE}_method{all_config.SAMPLING_METHOD}_out{all_config.RNN_OUT}_"
                f"max{all_config.MAX_VIDEOS}_epochs{all_config.EPOCH}_classifmode{all_config.CLASSIF_MODE}.pth"
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
                    "inference_duration": inf_dur,
                    "trainable_param":trainable
                },
                "best_model_filename": best_model_filename
            })
            #save the best result after append
            save_checkpoint(best_results)
        # Log results for the current run
        with open(all_config.LOG_FILE_PATH, 'a') as log_file:
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
        "inf_duration": r"inference_duration:\s+([\d.]+)",
        "trainable_params": r"'Trainable parameters':\s+(\d+)"
    }

    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1)) if key not in ["trainable_params"] else int(match.group(1))
        else:
            raise ValueError(f"Could not find a match for {key} in the output.")
    print("extracted metrics: ", metrics)
    return (
        metrics["accuracy"], 
        metrics["precision"], 
        metrics["recall"], 
        metrics["f1"], 
        metrics["train_duration"], 
        metrics["inf_duration"], 
        metrics["trainable_params"]
    )