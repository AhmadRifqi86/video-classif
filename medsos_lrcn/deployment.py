import os
import torch
import cv2
import numpy as np
import argparse
import all_config
from datetime import datetime
from loader_data import load_dataset_inference
from collections import Counter
from torchvision import transforms
from loader_data import uniform_sampling, ssim_sampling, duplicate_frames  # Replace `some_module` with the actual module name
import json

LABEL_MAPPING = {
    0: "Harmful",
    1: "Adult",
    2: "Safe",
    3: "Suicide"
}

def print_usage():
    print("usage: ")
    print("python3 deployment.py --model [MODEL PATH] --videos [VIDEO DIR]")


# Function to classify videos and return results as JSON
def classify_and_display(model, data_tensors, video_names):
    results = []
    label_counter = Counter()  # Counter to track label occurrences
    
    for idx, video_tensor in enumerate(data_tensors):
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(video_tensor)
            _, predicted = torch.max(output, 1)
        
        label_number = predicted.item()
        label_string = LABEL_MAPPING.get(label_number, "Unknown")  # Convert to string label
        
        timestamp = datetime.now()
        result = {
            "video_name": video_names[idx],
            "label": label_string,
            "timestamp": timestamp.isoformat()
        }
        results.append(result)
        label_counter[label_string] += 1  # Increment label count
        print(f"Processed {video_names[idx]}: {label_string}")

    # Display results as JSON
    import json
    print(json.dumps(results, indent=4))
    
    # Display label counts
    print("\nLabel Counts:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

# Main function
def main(model_path, video_folder=all_config.VIDEO_DIR, sampling_method=all_config.SAMPLING_METHOD, sequence_length=all_config.SEQUENCE_LENGTH):
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path).to(all_config.CONF_DEVICE)
    model.eval()

    # Load dataset (videos to classify)
    data, video_names = load_dataset_inference(video_folder, sampling_method, sequence_length)

    # Transform data to PyTorch tensors and permute dimensions
    data_tensors = [torch.tensor(video).permute(0, 3, 1, 2).float().to(all_config.CONF_DEVICE) for video in data]

    # Classify videos and display results as JSON
    classify_and_display(model, data_tensors, video_names)

if __name__ == "__main__":
    # Set up CLI arguments
    parser = argparse.ArgumentParser(description="Classify videos using a trained model.")
    parser.add_argument("--model", required=True, help="Path to the trained model file (e.g., .pth).")
    parser.add_argument("--videos", help="Path to the folder containing videos to classify.")
    parser.add_argument("--sampling", default="uniform", choices=["uniform", "ssim"], help="Frame sampling method (default: uniform).")
    parser.add_argument("--sequence_length", default=40, type=int, help="Number of frames per video for classification.")
    args = parser.parse_args()

    model_path = os.path.join(all_config.BEST_MODEL_DIR, args.model)

    # Run the main function
    #main(args.model, args.videos, args.sampling, args.sequence_length)
    main(model_path, args.videos, args.sampling, args.sequence_length)



#python3 deployment.py 
# --model /home/arifadh/Desktop/Skripsi-Magang-Proyek/best_models_medsos/best_model_seq40_batch16_hidden32_cnnresnet34_rnn16_layer2_rnnTypemamba_methoduniform_outall_max700_epochs8_finetuneTrue_classifmodemulticlass_f10.7453.pth 
# --videos /home/arifadh/Desktop/Dataset/tikHarm/Dataset/test/Adult/