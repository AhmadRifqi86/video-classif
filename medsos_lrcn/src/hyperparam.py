import os
import runner
import itertools
import json
import all_config
from datetime import datetime
import optuna
from deap import base, creator, tools, algorithms
import random
import loader_data

# Configuration dictionary
CONFIG = {
    "EPOCH": [8],
    "SEQUENCE_LENGTH": [60],
    "RNN_TYPE": ["mamba"],
    "CNN_BACKBONE": ["resnet34", "resnet50", "mobilenet_v2"],
    "BATCH_SIZE": [8, 16],
    "HIDDEN_SIZE": [8, 16, 24, 32],
    "RNN_INPUT_SIZE": [16, 24, 8, 32],
    "RNN_LAYER": [2, 3, 4],
    "SAMPLING_METHOD": ["uniform"],
    "RNN_OUT": ["all"],
    "MAX_VIDEOS": [1000],
    "CLASSIF_MODE": ["multiclass"],
}

# Ensure directories exist
if not os.path.exists(all_config.BEST_MODEL_DIR):
    os.makedirs(all_config.BEST_MODEL_DIR)


# Optimization strategies
def grid_search(configs):
    completed_configs = {json.dumps(result['config'], sort_keys=True) for result in best_results}
    keys, values = zip(*configs.items())
    for value_combination in itertools.product(*values):
        config = dict(zip(keys, value_combination))
        if json.dumps(config, sort_keys=True) in completed_configs:
            print(f"Skipping already completed configuration: {config}")
            continue
        best_f1, best_model_filename = runner.run_training(config, all_config.TEST_RUNS, best_results)
        loader_data.save_checkpoint(best_results)
        print(f"Completed Grid Search: {config}, Best F1: {best_f1}")

def bayesian_optimization(configs):
    def objective(trial):
        config = {}
        for param, values in configs.items():
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
        
        best_f1, _ = runner.run_training(config, all_config.TEST_RUNS, best_results)
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    loader_data.save_checkpoint(best_results)

def genetic_algorithm(configs):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    # Dynamically register attributes based on CONFIG
    for param, values in configs.items():
        print("param: ",param)
        print("values: ",values[0])
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
                     tuple(getattr(toolbox, param) for param in configs.keys()), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (runner.run_training(dict(zip(configs.keys(), ind)), all_config.TEST_RUNS, best_results)[0],))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=5, up=100, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=20)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
    loader_data.save_checkpoint(best_results)

# Main function to choose the strategy
if __name__ == "__main__":
    best_results = loader_data.load_checkpoint()
    strategy = "bayesian"
    if strategy == "grid":
        grid_search(CONFIG)
    elif strategy == "bayesian":
        bayesian_optimization(CONFIG)
    elif strategy == "genetic":
        genetic_algorithm(CONFIG)
    else:
        print("Invalid strategy selected.")
