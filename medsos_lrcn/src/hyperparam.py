import os
import runner
import itertools
import json
import all_config
import numpy as np
import pickle
import optuna
from deap import base, creator, tools, algorithms
import random
import loader_data

# Configuration dictionary
CONFIG = {
    "CNN_BACKBONE": ["resnet50", "mobilenet_v2"],
    "BATCH_SIZE": [2,3,4],
    "MULT_FACTOR": [2, 3, 4],
    "RNN_INPUT_SIZE": [6,8,12,16],
    "RNN_LAYER": [2, 3, 4]
}

# Ensure directories exist
if not os.path.exists(all_config.BEST_MODEL_DIR):
    os.makedirs(all_config.BEST_MODEL_DIR)

def is_config_duplicate(config, best_results):
    """Check if a configuration is already present in best_results."""
    return any(config == result["config"] for result in best_results)

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

def bayesian_optimization_old(configs):
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
        
        if is_config_duplicate(config, best_results):
            print(f"Skipping duplicate configuration: {config}")
            return -float("inf") 
        
        best_f1, _ = runner.run_training(config, all_config.TEST_RUNS, best_results)
        return best_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    loader_data.save_checkpoint(best_results)


def bayesian_optimization(configs):
    def objective(trial):
        config = {}
        for param, values in configs.items():
            if isinstance(values[0], int):
                config[param] = trial.suggest_int(param, min(values), max(values))
            elif isinstance(values[0], float):
                config[param] = trial.suggest_float(param, min(values), max(values))
            elif isinstance(values[0], bool):
                config[param] = trial.suggest_categorical(param, [True, False])
            else:  # Categorical values (e.g., strings)
                config[param] = trial.suggest_categorical(param, values)
        
        if is_config_duplicate(config, best_results):
            print(f"Skipping duplicate configuration: {config}")
            return -float("inf") 
        
        best_f1, _ = runner.run_training(config, all_config.TEST_RUNS, best_results)
        return best_f1

    # Load or create the study
    storage = "sqlite:///optuna_study.db"  # SQLite database for saving study state
    study_name = "bayesian_optimization_study"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        print("Loaded existing study.")
    except KeyError:
        study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage)
        print("Created new study.")

    # Run optimization and automatically save the state
    study.optimize(objective, n_trials=50)
    loader_data.save_checkpoint(best_results)


def genetic_algorithm_old(configs):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    # Dynamically register attributes based on CONFIG
    for param, values in configs.items():
        if isinstance(values[0], int):
            toolbox.register(param, random.randint, min(values), max(values))
        elif isinstance(values[0], float):
            toolbox.register(param, random.uniform, min(values), max(values))
        elif isinstance(values[0], bool):
            toolbox.register(param, random.choice, [True, False])
        else:  # Categorical values
            toolbox.register(param, random.choice, values)
    
    # Define evaluate function with duplicate check
    def evaluate_individual(ind):
        config = dict(zip(configs.keys(), ind))
        if is_config_duplicate(config, best_results):
            print(f"Skipping duplicate configuration: {config}")
            return -float("inf"),  # Assign the lowest possible score
        
        best_f1, _ = runner.run_training(config, all_config.TEST_RUNS, best_results)
        return best_f1,

    # Register functions
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     tuple(getattr(toolbox, param) for param in configs.keys()), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=5, up=100, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Run Genetic Algorithm
    population = toolbox.population(n=20)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=True)
    loader_data.save_checkpoint(best_results)


def genetic_algorithm(configs):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    
    # Dynamically register attributes based on CONFIG
    for param, values in configs.items():
        if isinstance(values[0], int):
            toolbox.register(param, random.randint, min(values), max(values))
        elif isinstance(values[0], float):
            toolbox.register(param, random.uniform, min(values), max(values))
        elif isinstance(values[0], bool):
            toolbox.register(param, random.choice, [True, False])
        else:  # Categorical values
            toolbox.register(param, random.choice, values)
    
    # Define evaluate function with duplicate check
    def evaluate_individual(ind):
        config = dict(zip(configs.keys(), ind))
        if is_config_duplicate(config, best_results):
            print(f"Skipping duplicate configuration: {config}")
            return -float("inf"),  # Assign the lowest possible score
        
        best_f1, _ = runner.run_training(config, all_config.TEST_RUNS, best_results)
        return best_f1,

    # Register functions
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     tuple(getattr(toolbox, param) for param in configs.keys()), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=2, up=8, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Load or initialize population
    checkpoint_file = "deap_checkpoint.pkl"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
        population = checkpoint["population"]
        start_gen = checkpoint["generation"]
        hall_of_fame = checkpoint["hall_of_fame"]
        print(f"Resuming from generation {start_gen}")
    else:
        population = toolbox.population(n=20)
        start_gen = 0
        hall_of_fame = tools.HallOfFame(1)

    # Define statistics and logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()

    # Run Genetic Algorithm
    for gen in range(start_gen, 10):  # Replace 10 with desired number of generations
        population = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fits):
            ind.fitness.values = fit

        # Update hall of fame and logbook
        hall_of_fame.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        print(logbook.stream)

        # Save checkpoint after each generation
        with open(checkpoint_file, "wb") as f:
            pickle.dump({"population": population, "generation": gen + 1, "hall_of_fame": hall_of_fame}, f)

    loader_data.save_checkpoint(best_results)

# Main function to choose the strategy
if __name__ == "__main__":
    best_results = loader_data.load_checkpoint()
    strategy = "bayesian"
    if strategy == "grid":
        grid_search(CONFIG)
    elif strategy == "bayesian":
        bayesian_optimization_old(CONFIG)
    elif strategy == "genetic":
        genetic_algorithm(CONFIG)
    else:
        print("Invalid strategy selected.")
