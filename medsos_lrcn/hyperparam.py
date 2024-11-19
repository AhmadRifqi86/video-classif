import os
import subprocess
import re
import itertools
import json
import time
import all_config
from datetime import datetime
import optuna
from deap import base, creator, tools, algorithms
import random

# Configuration dictionary
CONFIG = {
    "EPOCH": [8, 10, 12],
    "SEQUENCE_LENGTH": [40],
    "RNN_TYPE": ["mamba", "lstm"],
    "CNN_BACKBONE": ["resnet34", "resnet50", "mobilenet_v2"],
    "BATCH_SIZE": [8, 16],
    "HIDDEN_SIZE": [8, 16, 24, 32],
    "RNN_INPUT_SIZE": [16, 24, 8, 32],
    "RNN_LAYER": [2, 3, 4],
    "SAMPLING_METHOD": ["uniform"],
    "RNN_OUT": ["all"],
    "MAX_VIDEOS": [700],
    "CLASSIF_MODE": ["multiclass"],

}

# Ensure directories exist
if not os.path.exists(all_config.BEST_MODEL_DIR):
    os.makedirs(all_config.BEST_MODEL_DIR)

# Checkpointing functions
def load_checkpoint():
    if os.path.exists(all_config.CHECKPOINT_FILE):
        with open(all_config.CHECKPOINT_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Error loading checkpoint. Invalid JSON format.")
                return []
    return []

def save_checkpoint(best_results):
    with open(all_config.CHECKPOINT_FILE, 'w') as f:
        json.dump(best_results, f, indent=4)

# Function to run the training and process results
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
        if best_f1 is None or f1 > best_f1:
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

# Optimization strategies
def grid_search():
    completed_configs = {json.dumps(result['config'], sort_keys=True) for result in best_results}
    keys, values = zip(*CONFIG.items())
    for value_combination in itertools.product(*values):
        config = dict(zip(keys, value_combination))
        if json.dumps(config, sort_keys=True) in completed_configs:
            print(f"Skipping already completed configuration: {config}")
            continue
        best_f1, best_model_filename = run_training(config, all_config.TEST_RUNS, best_results)
        save_checkpoint(best_results)
        print(f"Completed Grid Search: {config}, Best F1: {best_f1}")

def bayesian_optimization():
    def objective(trial):
        config = {}
        for param, values in CONFIG.items():
            print("param: ",param)
            print("values: ",values[0])
            if isinstance(values[0], int):
                config[param] = trial.suggest_int(param, min(values), max(values))
                print("config[param]: ",config[param])
            elif isinstance(values[0], float):
                config[param] = trial.suggest_float(param, min(values), max(values))
                print("config[param]: ",config[param])
            elif isinstance(values[0], bool):
                config[param] = trial.suggest_categorical(param, [True, False])
                print("config[param]: ",config[param])
            else:  # Categorical values (e.g., strings)
                config[param] = trial.suggest_categorical(param, values)
                print("config[param]: ",config[param])
        
        best_f1, _ = run_training(config, all_config.TEST_RUNS, best_results)
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    save_checkpoint(best_results)

def genetic_algorithm():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    # Dynamically register attributes based on CONFIG
    for param, values in CONFIG.items():
        if isinstance(values[0], int):
            toolbox.register(param, random.randint, min(values), max(values))
        elif isinstance(values[0], float):
            toolbox.register(param, random.uniform, min(values), max(values))
        elif isinstance(values[0], bool):
            toolbox.register(param, random.choice, [True, False])
        else:  # Categorical values
            toolbox.register(param, random.choice, values)
    
    # Dynamically create individuals based on attributes
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     tuple(getattr(toolbox, param) for param in CONFIG.keys()), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (run_training(dict(zip(CONFIG.keys(), ind)), all_config.TEST_RUNS, best_results)[0],))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=5, up=100, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
    save_checkpoint(best_results)

# Main function to choose the strategy
if __name__ == "__main__":
    best_results = load_checkpoint()
    strategy = "genetic"
    if strategy == "grid":
        grid_search()
    elif strategy == "bayesian":
        bayesian_optimization()
    elif strategy == "genetic":
        genetic_algorithm()
    else:
        print("Invalid strategy selected.")
