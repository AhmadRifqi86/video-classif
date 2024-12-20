import zmq
import torch
from collections import Counter
import datetime
import requests
import json
import os
import all_config
import loader_data
import time

# Load config or environment variables
APP_STAGE = os.getenv("APP_STAGE", "devel")

if APP_STAGE == "prod":
    import custom_pyktok.pyktok as pyk
    print("Using custom packaged version of pyktok")
else:
    import pyktok as pyk
    print("Using development version of pyktok")

pyk.specify_browser('firefox')
# Load parameters from environment variables or all_config
MODEL_PATH = os.getenv("MODEL_PATH", "/home/arifadh/Desktop/Skripsi-Magang-Proyek/new_best_model/seq60_max1000_mobilenetv2_mamba_batch32_hidden32_rnninp8_layer3_unidir_adp123normsiludrop0.3_dinner4dmodel_dtrankeqnstate_epoch8_acc0.8189.pth")
SAMPLING_METHOD = os.getenv("SAMPLING_METHOD", "uniform")  # Example: sampling method
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 60))  # Example: sequence length
VIDEO_DIR = os.getenv("VIDEO_DIR", "/home/arifadh/Downloads/tiktok_videos")
ZEROMQ_URI = "tcp://0.0.0.0:54000"

LABEL_MAPPING = {
    0: "Harmful",
    1: "Adult",
    2: "Safe",
    3: "Suicide"
}

# Classify and display results
def classify_and_display(model, data_tensors, video_names):
    results = []
    label_counter = Counter()  # Counter to track label occurrences

    for idx, video_tensor in enumerate(data_tensors):
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(video_tensor)
            probabilities = torch.softmax(output, dim=1)
            sorted_indices = torch.argsort(probabilities, dim=1, descending=True)  # Sort by probability

            sorted_labels = [
                LABEL_MAPPING.get(idx.item(), "Unknown") for idx in sorted_indices[0]
            ]
            sorted_scores = probabilities[0, sorted_indices[0]].tolist()

        timestamp = datetime.datetime.now()
        result = {
            "video_name": video_names[idx],
            "labels": sorted_labels,
            "scores": sorted_scores,
            "timestamp": timestamp.isoformat()
        }
        results.append(result)

        top_label = sorted_labels[0]
        label_counter[top_label] += 1
        print(f"Processed {video_names[idx]}: {sorted_labels[0]}")

    print(json.dumps(results, indent=4))
    print("\nLabel Counts:")
    for label, count in label_counter.items():
        print(f"{label}: {count}")

    return results

# Post results to backend
def post_results(results, latency):
    for result in results:
        video_name = result["video_name"]
        labels = result["labels"]
        scores = result["scores"]
        ts = result["timestamp"]

        video_url = loader_data.construct_url(video_name)
        if not video_url:
            print(f"Failed to construct URL for {video_name}")
            continue

        payload = {
            "url": video_url,
            "labels": labels,
            "scores": scores,
            "timestamp": ts,
            "latency": latency
        }

        try:
            response = requests.post(all_config.BACKEND_URL, json=payload)
            if response.status_code in [200, 201]:
                print(f"Successfully sent classification result to backend for {video_name}")
            else:
                print(f"Failed to send classification result for {video_name}. HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error sending result to backend for {video_name}: {e}")

# Callback function that is triggered when ZeroMQ sends a message
def callback(message, model):
    tmp = []
    url = message.decode()
    print(f"Processing URL: {url}")
    tmp.append(url)
    # Download the video using pyktok
    pyk.save_tiktok_multi_urls(tmp, True, '', 1, save_dir=VIDEO_DIR)  # problem nya disini
    print("finish downloading")
    
    start = time.time()

    # Load dataset (videos to classify)
    data, video_names = loader_data.load_dataset_inference(VIDEO_DIR, SAMPLING_METHOD, SEQUENCE_LENGTH)

    # Transform data to PyTorch tensors
    data_tensors = [torch.tensor(video).permute(0, 3, 1, 2).float().to(all_config.CONF_DEVICE) for video in data]

    # Classify videos and display results
    result = classify_and_display(model, data_tensors, video_names)
    latency = time.time() - start

    # Post the results to backend
    post_results(result, latency)

    print(f"Finished processing: {url}")

# Consume messages from ZeroMQ (using PULL socket)
def consume_messages():
    # Load model once during service startup
    print("Loading model once...")
    model = torch.load(MODEL_PATH).to(all_config.CONF_DEVICE)
    model.eval()

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    
    # Workers connect to the backend's PUSH socket
    socket.bind(ZEROMQ_URI)

    print("Worker connected to ZeroMQ queue. Waiting for messages...")

    while True:
        try:
            message = socket.recv()
            callback(message, model)  # Pass the model to the callback
        except zmq.ZMQError as e:
            print(f"ZeroMQ Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

if __name__ == "__main__":
    consume_messages()
