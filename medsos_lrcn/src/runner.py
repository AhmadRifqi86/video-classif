# import subprocess
# import all_config
# import os
# import time
# import re
# from loader_data import save_checkpoint


# def run_training(config, test_runs, best_results):
#     #best_f1 = None
#     best_model_filename = None
#     best_f1 = -float("inf")  # Use negative infinity to represent the initial state

#     for run in range(test_runs):
#         sed_commands = []

#         # Apply configuration to source code with proper handling of string values
#         for key, value in config.items():
#             if isinstance(value, str):
#                 sed_command = f"sed -i '/^{key} =/ s|=.*|= \"{value}\"|' {all_config.CONFIG_PATH}"  # Quote strings
#             else:
#                 sed_command = f"sed -i '/^{key} =/ s|=.*|= {value}|' {all_config.CONFIG_PATH}"  # Leave non-strings as-is
#             sed_commands.append(sed_command)

#         print("Applying config:")
#         print(config)

#         # Execute sed commands
#         for command in sed_commands:
#             subprocess.run(command, shell=True)

#         # Run training and capture real-time logs
#         print("Starting training...")
#         process = subprocess.Popen(
#             f'python3 {all_config.SOURCE_PATH}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         result = []
#         # Log training progression in real-time
#         with open(all_config.LOG_FILE_PATH, 'a') as log_file:
#             log_file.write(f"Run {run + 1}/{test_runs}\n")
#             log_file.write(f"Config: {config}\n")
#             log_file.write("Training logs:\n")
#             for line in process.stdout:
#                 log_file.write(line)
#                 result.append(line)
#                 print(line, end="")  # Print to console as well

#         stdout, stderr = process.communicate()
#         result = ''.join(result)  # Concatenate list of lines into a single string
#         error_output = stderr
#         print("Training completed.")
#         try:
#             # Extract metrics
#             accuracy, precision, recall, f1, train_dur, inf_dur,trainable = extract_metrics(result)
#             print(f"Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, "
#                 f"Train Duration={train_dur}s, Inference Duration={inf_dur}s")
#         except Exception as e:
#             # Log errors during metric extraction
#             with open(all_config.LOG_FILE_PATH, 'a') as log_file:
#                 log_file.write(f"Error extracting metrics: {e}\n")
#                 log_file.write(f"Run {run} output:\n{result}\n")
#                 log_file.write(f"Error Output:\n{error_output}\n\n")
#             print(f"Error extracting metrics: {e}")
#             continue

#         # Save the best model
#         if f1 > best_f1 and f1 > 0.71:  # Only update if the new result is better and exceeds the threshold
#             best_f1 = f1
#             best_model_filename = (
#                 f"best_model_seq{all_config.SEQUENCE_LENGTH}_batch{all_config.CONF_BATCH_SIZE}_hidden{all_config.HIDDEN_SIZE}_"
#                 f"cnn{all_config.CONF_CNN_BACKBONE}_rnn{all_config.RNN_INPUT_SIZE}_layer{all_config.RNN_LAYER}_"
#                 f"rnnType{all_config.RNN_TYPE}_method{all_config.SAMPLING_METHOD}_out{all_config.RNN_OUT}_"
#                 f"max{all_config.MAX_VIDEOS}_epochs{all_config.EPOCH}_classifmode{all_config.CLASSIF_MODE}.pth"
#             )
#             best_model_path = os.path.join(all_config.BEST_MODEL_DIR, best_model_filename)

#             # Save the best model file
#             print(f"Saving best model: {best_model_filename}")
#             subprocess.run(f"cp {all_config.MODEL_PATH} {best_model_path}", shell=True)

#             # Update best results
#             best_results.append({
#                 "config": config,
#                 "metrics": {
#                     "accuracy": accuracy,
#                     "precision": precision,
#                     "recall": recall,
#                     "f1_score": f1,
#                     "training_duration": train_dur,
#                     "inference_duration": inf_dur,
#                     "trainable_param":trainable
#                 },
#                 "best_model_filename": best_model_filename
#             })
#             #save the best result after append
#             save_checkpoint(best_results)
#         # Log results for the current run
#         with open(all_config.LOG_FILE_PATH, 'a') as log_file:
#             log_file.write(f"Metrics: Accuracy={accuracy}, Precision={precision}, Recall={recall}, F1={f1}, "
#                         f"Train Duration={train_dur}s, Inference Duration={inf_dur}s\n")
#             if error_output:
#                 log_file.write(f"Error Output:\n{error_output}\n\n")
#     time.sleep(all_config.SLEEP)
#     return best_f1, best_model_filename


# # Extract metrics
# def extract_metrics(output):
#     patterns = {
#         "accuracy": r"Overall Accuracy: (\d\.\d+|\d\.\d)",
#         "precision": r"Overall Precision: (\d\.\d+|\d\.\d)",
#         "recall": r"Overall Recall: (\d\.\d+|\d\.\d)",
#         "f1": r"Overall F1-Score: (\d\.\d+|\d\.\d)",
#         "train_duration": r"training_duration:\s+([\d.]+)",
#         "inf_duration": r"inference_duration:\s+([\d.]+)",
#         "trainable_params": r"'Trainable parameters':\s+(\d+)"
#     }

#     metrics = {}
#     for key, pattern in patterns.items():
#         match = re.search(pattern, output)
#         if match:
#             metrics[key] = float(match.group(1)) if key not in ["trainable_params"] else int(match.group(1))
#         else:
#             raise ValueError(f"Could not find a match for {key} in the output.")
#     print("extracted metrics: ", metrics)
#     return (
#         metrics["accuracy"], 
#         metrics["precision"], 
#         metrics["recall"], 
#         metrics["f1"], 
#         metrics["train_duration"], 
#         metrics["inf_duration"], 
#         metrics["trainable_params"]
#     )




#### Disini bikin buat genetic algo atau bayesian optim
import all_config
import subprocess
import os
import re
import time


def run_training(config, test_runs, best_results):
    best_f1 = None
    best_acc = None
    best_model_filename = None

    for run in range(test_runs):
        # Prepare sed commands to update the source code with the current config values
        sed_commands = [
            "sed -i '/^{key} =/ s|=.*|= {value}|' {source}".format(
                key=key, value=value if isinstance(value, (int, float, list)) else f'"{value}"', source=all_config.CONFIG_PATH
            )
            for key, value in config.items()
        ]

        # Apply the sed commands
        print("Applying config")
        print(config)
        for command in sed_commands:
            subprocess.run(command, shell=True)

        # Run the training script
        # print("Perform training")
        # process = subprocess.Popen(f'python3 {all_config.SOURCE_PATH}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # stdout, stderr = process.communicate()

        # # Process the output
        # print("Training done, recording result")
        # result = stdout.decode('utf-8')

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
        accuracy, precision, recall, f1, train_dur, inf_dur = extract_metrics(result)
        print("Extracted F1: ", f1)
        print("Extracted Accuracy: ", accuracy)
        try:
            # Extract metrics from the output
            accuracy, precision, recall, f1, train_dur, inf_dur = extract_metrics(result)
            
        except Exception as e:
            with open(all_config.LOG_FILE_PATH, 'a') as log_file:
                log_file.write(f"Error extracting metrics: {e}\n")
                log_file.write(f"Run {run} output: {result}\n")
            continue
        
        # Save the best model for this configuration based on F1 score
        if best_f1 is None or f1 > best_f1:
            print("Assign best f1")
            best_f1 = f1
            best_acc = accuracy
            best_prec = precision
            best_rec = recall
            best_train_dur = train_dur
            best_inf_dur = inf_dur
        
        # Save the model only if accuracy > 0.76
        # if accuracy > 0.79:
        #     # Construct best model filename using config dictionary, ini nanti diganti
        #     best_model_filename_parts = [
        #         f"seq{all_config.SEQUENCE_LENGTH}",
        #         f"batch{config['BATCH_SIZE']}",
        #         f"hidden{config['MULT_FACTOR']*config['RNN_INPUT_SIZE']}",
        #         f"ssmexpand{config['RNN_INPUT_SIZE']*config['SSM_EXPAND']}"
        #         f"delta{config['SSM_DELTA']*config['RNN_INPUT_SIZE']*config['MULT_FACTOR']}"
        #         f"cnn{all_config.CONF_CNN_BACKBONE}",
        #         f"rnn{config['RNN_INPUT_SIZE']}",
        #         f"layer{config['RNN_LAYER']}",
        #         f"rnnType{all_config.CONF_RNN_TYPE}",
        #         f"drop{config['DROPOUT']}",
        #         f"bidir{all_config.CONF_BIDIR}",
        #         f"acc{accuracy:.4f}",
        #         f"f1{f1:.4f}.pth"
        #     ]
        #     # Join filename parts with underscores
        #     best_model_filename = "_".join(best_model_filename_parts)
        #     # Construct full path for the best model
        #     best_model_path = os.path.join(all_config.BEST_MODEL_DIR, best_model_filename)

        #     # best_model_filename = f"seq{all_config.SEQUENCE_LENGTH}_batch{all_config.CONF_BATCH_SIZE}_hidden{all_config.CONF_HIDDEN_SIZE}_cnn{all_config.CONF_CNN_BACKBONE}_rnn{all_config.CONF_RNN_INPUT_SIZE}_layer{all_config.CONF_RNN_LAYER}_rnnType{all_config.CONF_RNN_TYPE}_drop{all_config.DROPOUT}_bidir{all_config.BIDIR}_acc{accuracy:.4f}_f1{f1:.4f}.pth"
        #     # best_model_path = os.path.join(all_config.BEST_MODEL_DIR, best_model_filename)

        #     print(f"Saving model with accuracy > 0.76 for configuration: {best_model_filename}")
        #     subprocess.run(f"cp {all_config.MODEL_PATH} {best_model_path}", shell=True)

        with open(all_config.LOG_FILE_PATH, 'a') as log_file:
            log_file.write(f"Config (Run {run+1}/{test_runs}): {config}, ACCURACY={accuracy}, F1={f1}\n")
            log_file.write(result)
            if stderr:
                log_file.write(f"Error: {stderr}\n")
            log_file.write("\n\n")

    # Record the best result for this configuration
    if best_f1 is not None:
        print("best runner f1: ",best_f1)
        best_results.append({
            "config": config,
            "metrics": {
                "accuracy": best_acc,
                "precision": best_prec,
                "recall": best_rec,
                "f1_score": best_f1,
                "training duration": best_train_dur,
                "inference duration": best_inf_dur
            },
            "best_model_filename": best_model_filename if best_acc > 0.79 else None
        })
    print("Cooling down GPU")
    time.sleep(all_config.SLEEP)
    print("returned value from run_training(): ",best_f1)
    return best_f1

# Function to extract accuracy, precision, recall, and f1 score from the stdout
def extract_metrics(output):
    overall_accuracy_pattern = r"Overall Accuracy: (\d\.\d+|\d\.\d)"
    precision_pattern = r"Overall Precision: (\d\.\d+|\d\.\d)"
    recall_pattern = r"Overall Recall: (\d\.\d+|\d\.\d)"
    f1_pattern = r"Overall F1-Score: (\d\.\d+|\d\.\d)"
    train_dur = r"training_duration:\s+([\d.]+)"
    inf_dur = r"inference_duration:\s+([\d.]+)"

    accuracy = re.search(overall_accuracy_pattern, output)
    precision = re.search(precision_pattern, output)
    recall = re.search(recall_pattern, output)
    f1 = re.search(f1_pattern, output)
    train_time = re.search(train_dur, output)
    inf_time = re.search(inf_dur, output)
    #print("inside extractor, f1: ",f1)

    if accuracy and precision and recall and f1 and train_time and inf_time:
        print("[1] - Metrics Extracted, f1: ",f1)
        return float(accuracy.group(1)), float(precision.group(1)), float(recall.group(1)), float(f1.group(1)), float(train_time.group(1)), float(inf_time.group(1))
    else:
        raise ValueError("Could not extract metrics from output")