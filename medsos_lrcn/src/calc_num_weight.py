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
CONFIGS = {    #SSM-LSTM
    "CNN_BACKBONE": ["mobilenet_v2"],
    "RNN_TYPE": ["mamba", "lstm"],
    "BATCH_SIZE": [16, 32],
    "MULT_FACTOR": [2,3,4],
    "RNN_INPUT_SIZE": [8,16,32],
    "RNN_LAYER": [2, 3, 4],
    "DROPOUT":[0.3,0.4,0.5],
    "BIDIR": [True, False],
}

# CONFIGS = {    #transformer
#     "CNN_BACKBONE": ["mobilenet_v2"],
#     "RNN_TYPE": ["transformer"],
#     "MULT_FACTOR": [2,3,4],
#     "NUM_HEAD": [4,8],
#     "RNN_INPUT_SIZE": [16,32],
#     "RNN_LAYER": [2, 3, 4],
#     "DROPOUT":[0.3,0.4,0.5],
#     #"BIDIR": [True, False],
# }

# Iterate over all configurations
keys, values = zip(*CONFIGS.items())
results = []

for value_combination in itertools.product(*values):
    config = dict(zip(keys, value_combination))
    
    print(f"\nTesting Configuration: {config}")
    seq_len = all_config.CONF_SEQUENCE_LENGTH
    rnn_inp_size=config['RNN_INPUT_SIZE']
    hidden_size=config['MULT_FACTOR']*config['RNN_INPUT_SIZE']

    # Initialize LRCN model with the given config
    model = LRCN(
        num_classes=4,  
        sequence_length=all_config.CONF_SEQUENCE_LENGTH,
        hidden_size=hidden_size,  
        rnn_input_size=rnn_inp_size,
        cnn_backbone=config["CNN_BACKBONE"],
        rnn_type=config["RNN_TYPE"],  
        rnn_out=all_config.CONF_RNN_OUT,
        bidirectional=config['BIDIR'],

    ).to(all_config.CONF_DEVICE)

    # Count parameters
    num_params = count_parameters(model)
    
    # Store result
    results.append({
        "RNN_TYPE": config["RNN_TYPE"],
        "RNN_INPUT_SIZE": config["RNN_INPUT_SIZE"],
        "RNN_LAYER": config["RNN_LAYER"],
        "BIDIR": config["BIDIR"],
        "NUM_PARAMETERS": num_params
    })
    
    print(f"{config['RNN_TYPE']}, Bidir={config['BIDIR']}, Layers={config['RNN_LAYER']}, InputSize={config['RNN_INPUT_SIZE']},DropOut={config['DROPOUT']} HiddenSize={config['MULT_FACTOR']*config['RNN_INPUT_SIZE']}: {num_params:,} parameters")
    #time.sleep(all_config.SLEEP)

# Save results to a file (optional)
import json
with open("param_count_results_rnn.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nCompleted parameter counting! Results saved to 'param_count_results_rnn.json'.")
