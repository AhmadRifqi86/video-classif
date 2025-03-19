import os
import torch
import all_config
import itertools
from model2 import LRCN
import time

def count_parameters(model):
    """Counts the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define hyperparameter configurations to test
CONFIGS = { #SSM-LSTM
    "CNN_BACKBONE": ["mobilenet_v2"],
    "RNN_TYPE": ["mamba", "lstm"],
    "BATCH_SIZE": [16, 32],
    "MULT_FACTOR": [2, 3, 4],
    "RNN_INPUT_SIZE": [8, 16, 32],
    "RNN_LAYER": [2, 3, 4],
    "DROPOUT": [0.3, 0.4, 0.5],
    "BIDIR": [True, False],
}

# CONFIGS = { #transformer
#     "CNN_BACKBONE": ["mobilenet_v2"],
#     "RNN_TYPE": ["transformer"],
#     "MULT_FACTOR": [2, 3, 4],
#     "NUM_HEAD": [4, 8],
#     "RNN_INPUT_SIZE": [16, 32],
#     "RNN_LAYER": [2, 3, 4],
#     "DROPOUT": [0.3, 0.4, 0.5],
#     #"BIDIR": [True, False],
# }

# Iterate over all configurations
keys, values = zip(*CONFIGS.items())
results = []

for value_combination in itertools.product(*values):
    config = dict(zip(keys, value_combination))
    print(f"\nTesting Configuration: {config}")
    
    seq_len = all_config.CONF_SEQUENCE_LENGTH
    rnn_inp_size = config['RNN_INPUT_SIZE']
    hidden_size = config['MULT_FACTOR'] * config['RNN_INPUT_SIZE']
    
    # Initialize LRCN model with the given config - FIXED to pass all relevant parameters
    model = LRCN(
        num_classes=4,
        sequence_length=all_config.CONF_SEQUENCE_LENGTH,
        hidden_size=hidden_size,
        rnn_input_size=rnn_inp_size,
        cnn_backbone=config["CNN_BACKBONE"],
        rnn_type=config["RNN_TYPE"],
        rnn_layers=config["RNN_LAYER"],  # FIXED: Pass the RNN_LAYER parameter
        dropout=config["DROPOUT"],       # FIXED: Pass the DROPOUT parameter
        rnn_out=all_config.CONF_RNN_OUT,
        bidirectional=config['BIDIR'],
    ).to(all_config.CONF_DEVICE)
    
    # Count parameters
    num_params = count_parameters(model)
    
    # Store result with full configuration details
    result_entry = config.copy()  # Copy all config parameters
    result_entry["NUM_PARAMETERS"] = num_params
    result_entry["HIDDEN_SIZE"] = hidden_size
    results.append(result_entry)
    
    print(f"{config['RNN_TYPE']}, Bidir={config['BIDIR']}, Layers={config['RNN_LAYER']}, "
          f"InputSize={config['RNN_INPUT_SIZE']}, DropOut={config['DROPOUT']}, "
          f"HiddenSize={hidden_size}: {num_params:,} parameters")
    
    # Uncomment if you need a delay between tests
    # time.sleep(all_config.SLEEP)

# Save results to a file with timestamp to avoid overwriting
import json
import datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"param_count_results_rnn_{timestamp}.json"

with open(filename, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nCompleted parameter counting! Results saved to '{filename}'.")

# Optional: Sort and print summary of results by parameter count
print("\nSummary of results (sorted by parameter count):")
sorted_results = sorted(results, key=lambda x: x["NUM_PARAMETERS"])
for idx, res in enumerate(sorted_results):
    print(f"{idx+1}. {res['RNN_TYPE']}, Layers={res['RNN_LAYER']}, "
          f"InputSize={res['RNN_INPUT_SIZE']}, HiddenSize={res['HIDDEN_SIZE']}, "
          f"Bidir={res['BIDIR']}: {res['NUM_PARAMETERS']:,} parameters")