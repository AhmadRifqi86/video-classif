# import runner
# import all_config
# import itertools
# import json
# import os
# from skopt import gp_minimize, load, dump
# from skopt.space import Real, Integer, Categorical
# from skopt.utils import use_named_args
# from deap import base, creator, tools, algorithms
# import numpy as np

# # Bayesian Optimization space
# BO_SPACE = [
#     Integer(2, 6, name="SSM_HIDDEN"),
#     Categorical([0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0], name="SSM_DELTA"),
#     Integer(8, 16, name="RNN_INPUT_SIZE"),
#     Categorical([16, 32], name="BATCH_SIZE"),
#     Integer(2, 4, name="RNN_LAYER"),
#     Categorical([0.25,0.3,0.4,0.5],name="DROPOUT"),
#     Integer(2,4,name="MULT_FACTOR")
# ]

# # Define fitness function for Bayesian Optimization
# @use_named_args(BO_SPACE)
# def fitness_bayesian(**params):
#     config = {key: value for key, value in params.items()}
#     print(f"Running Bayesian Optimization with config: {config}")
#     try:
#         best_f1, _ = runner.run_training(config,1,None)  #nanti angka 1 jadi 4
#         return -best_f1  # Negative because Bayesian Optimization minimizes
#     except Exception as e:
#         print(f"Error in training: {e}")
#         return 1e6  # Large penalty for errors

# # Initialize DEAP for Genetic Algorithm
# creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize F1-score
# creator.create("Individual", list, fitness=creator.FitnessMax)
# toolbox = base.Toolbox()

# # Register attributes for the Genetic Algorithm
# toolbox.register("attr_hidden", lambda: np.random.randint(2, 7))  # Integer in range [2, 6]
# toolbox.register("attr_delta", lambda: np.random.choice([0.25,0.5,0.75,1.0]))  # Real in range [0.25, 2]
# toolbox.register("attr_rnn_input", lambda: np.random.randint(8, 17))  # Integer in range [8, 16]
# toolbox.register("attr_batch", lambda: np.random.choice([16, 32]))  # Categorical with fixed values [16, 32]
# toolbox.register("attr_rnn_layer", lambda: np.random.randint(2, 5))  # Integer in range [2, 4]
# toolbox.register("attr_dropout", lambda: np.random.choice([0.25,0.3,0.4,0.5]))  # Real in range [0.3, 0.5]
# toolbox.register("attr_mult_factore", lambda: np.random.randint(2, 4))

# # Create individual and population
# toolbox.register(
#     "individual",
#     tools.initCycle,
#     creator.Individual,
#     (toolbox.attr_hidden, toolbox.attr_delta, toolbox.attr_rnn_input, toolbox.attr_batch,
#      toolbox.attr_rnn_layer, toolbox.attr_dropout),
# )
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # Define evaluation function
# def evaluate_genetic(individual):
#     config = {
#         "SSM_HIDDEN": individual[0],
#         "SSM_DELTA": individual[1],
#         "RNN_INPUT_SIZE": individual[2],
#         "BATCH_SIZE": individual[3],
#         "RNN_LAYER": individual[4],
#         "DROPOUT": individual[5],
#         "MULT_FACTOR": individual[6]
#     }
#     print(f"Running Genetic Algorithm with config: {config}")
#     try:
#         best_f1, _ = runner.run_training(config)
#         return (best_f1,)
#     except Exception as e:
#         print(f"Error in training: {e}")
#         return (0.0,)

# toolbox.register("evaluate", evaluate_genetic)
# toolbox.register("mate", tools.cxTwoPoint)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)  # Gaussian mutation
# toolbox.register("select", tools.selTournament, tournsize=3)

# # Custom mutation for categorical values
# def mutate_categorical(individual):
#     if np.random.random() < 0.2:  # 20% chance to mutate
#         idx = np.random.randint(len(individual))
#         if idx == 3:  # BATCH_SIZE is categorical
#             individual[idx] = 16 if individual[idx] == 32 else 32
#         elif idx == 0:  # SSM_HIDDEN
#             individual[idx] = np.random.randint(2, 7)
#         elif idx == 1:  # SSM_DELTA
#             individual[idx] = np.random.uniform(0.25, 2)
#     return individual,

# toolbox.register("mutate_categorical", mutate_categorical)

# def save_checkpoint(result):
#     """
#     Callback function to save the optimization checkpoint.
#     """
#     dump(result, all_config.BO_CHECKPOINT, store_objective=False)
#     print(f"Checkpoint saved to {all_config.BO_CHECKPOINT}")

# # Main driver function
# if __name__ == "__main__":

#     # Bayesian Optimization
#     print("Running Bayesian Optimization...")
#     res = gp_minimize(
#         fitness_bayesian,
#         BO_SPACE,
#         n_calls=20,  # Total calls
#         random_state=None,  # Allow randomness for exploration
#         callback=[save_checkpoint],  # Use the named function here
#     )

#     print(f"Best config from Bayesian Optimization: {res.x}, F1: {-res.fun}")

#     # Save the final result
#     dump(res,all_config.BO_CHECKPOINT, store_objective=False)
#     print(f"Optimization results saved to {all_config.BO_CHECKPOINT}")

#     # # Genetic Algorithm
#     # print("Running Genetic Algorithm...")
#     # pop = toolbox.population(n=20)
#     # hof = tools.HallOfFame(1)
#     # stats = tools.Statistics(lambda ind: ind.fitness.values)
#     # stats.register("avg", np.mean)
#     # stats.register("min", np.min)
#     # stats.register("max", np.max)

#     # algorithms.eaSimple(
#     #     pop,
#     #     toolbox,
#     #     cxpb=0.5,  # Crossover probability
#     #     mutpb=0.2,  # Mutation probability
#     #     ngen=10,  # Number of generations
#     #     stats=stats,
#     #     halloffame=hof,
#     #     verbose=True,
#     # )
#     # print(f"Best config from Genetic Algorithm: {hof[0]}, F1: {hof[0].fitness.values[0]}")



import runner
import all_config
import os
import numpy as np
from skopt import gp_minimize, load, dump
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import time

# Bayesian Optimization space
BO_SPACE = [
    Integer(2, 4, name="SSM_HIDDEN"),
    Categorical([0.25,0.5,0.75,1.0], name="SSM_DELTA"),
    Categorical([6,8,10,12,14,16,18,20], name="RNN_INPUT_SIZE"),
    Categorical([16, 24,32], name="BATCH_SIZE"),
    Integer(2, 4, name="RNN_LAYER"),
    Categorical([0.25,0.3,0.4,0.5],name="DROPOUT"),
    Integer(2,4,name="MULT_FACTOR")
]

def save_checkpoint(result):
    """
    Callback function to save the optimization checkpoint.
    """
    dump(result, all_config.BO_CHECKPOINT, store_objective=False)
    print(f"Checkpoint saved to {all_config.BO_CHECKPOINT}")

@use_named_args(BO_SPACE)
def fitness_bayesian(**params):
    config = {key: value for key, value in params.items()}
    print(f"Running Bayesian Optimization with config: {config}")
    try:
        best_f1, _ = runner.run_training(config, 3, None)  # Example: Change `1` to `4` for multiple runs
        print("fitness best f1: ",best_f1)
        if best_f1 is None:
            return 1e6
        return -best_f1  # Return the negative F1 score
    except Exception as e:
        print(f"Error in training: {e}")
        return 1e6  # Return `None` for failed runs


def main():
    # Check if checkpoint exists
    checkpoint_path = all_config.BO_CHECKPOINT
    print("Starting new Bayesian Optimization...")
    res = gp_minimize(
            fitness_bayesian,
            BO_SPACE,
            n_calls=50,  # Total calls
            random_state=None,
            callback=[save_checkpoint]
    )


    # if os.path.exists(checkpoint_path):
    #     print(f"Loading existing checkpoint from {checkpoint_path}")
    #     try:
    #         checkpoint = load(checkpoint_path)
    #         completed_iterations = len(checkpoint.x_iters)
    #         total_calls = 50
    #         remaining_calls = total_calls - completed_iterations

    #         # Validate the results in the checkpoint
    #         valid_indices = [
    #             i for i, f_val in enumerate(checkpoint.func_vals) if f_val < 1e6
    #         ]  # Only consider valid runs (exclude penalty values like 1e6)

    #         # Filter out invalid runs
    #         valid_x_iters = [checkpoint.x_iters[i] for i in valid_indices]
    #         valid_func_vals = [checkpoint.func_vals[i] for i in valid_indices]

    #         if len(valid_x_iters) == 0:
    #             print("No valid runs found in the checkpoint. Starting fresh optimization.")
    #             remaining_calls = total_calls
    #             checkpoint = None
    #         else:
    #             print(f"Found {len(valid_x_iters)} valid runs in the checkpoint.")
    #             completed_iterations = len(valid_x_iters)
    #             remaining_calls = total_calls - completed_iterations

    #         if remaining_calls <= 0:
    #             print("Optimization already completed.")
    #             print(f"Best config: {checkpoint.x}")
    #             print(f"Best F1 Score: {-checkpoint.fun}")
    #             return checkpoint

    #         # Resume optimization with valid data
    #         res = gp_minimize(  
    #             fitness_bayesian,
    #             BO_SPACE,
    #             n_calls=remaining_calls,
    #             x0=valid_x_iters,
    #             y0=valid_func_vals,
    #             random_state=checkpoint.random_state,
    #             callback=[save_checkpoint],
    #         )
    #     except Exception as e:
    #         print(f"Error loading checkpoint: {e}. Starting new optimization.")
    #         checkpoint = None
    # else:
    #     # Start fresh optimization
    #     print("Starting new Bayesian Optimization...")
    #     res = gp_minimize(
    #         fitness_bayesian,
    #         BO_SPACE,
    #         n_calls=50,  # Total calls
    #         random_state=None,
    #         callback=[save_checkpoint]
    #     )

    # Filter out unsuccessful results after optimization
    valid_results = [
        (x, y) for x, y in zip(res.x_iters, res.func_vals) if y is not None and y < 1e6
    ]

    # Find the best result among valid ones
    if valid_results:
        best_config, best_neg_f1 = min(valid_results, key=lambda pair: pair[1])  # Minimize
        best_f1 = -best_neg_f1  # Negate to get the F1 score
        print(f"Best config: {best_config}")
        print(f"Best F1 Score: {best_f1}")
    else:
        print("No valid results found!")

  
    return res


if __name__ == "__main__":
    main()