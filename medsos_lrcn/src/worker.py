import torch
from collections import Counter
import datetime
import requests
import json
import os
import pika
import all_config
import loader_data

# Load config or environment variables
APP_STAGE = os.getenv("APP_STAGE", "devel")

if APP_STAGE == "prod":
    import custom_pyktok.pyktok as pyk
    print("Using custom packaged version of pyktok")
else:
    import pyktok as pyk
    print("Using development version of pyktok")

# Load parameters from environment variables or all_config
MODEL_PATH = os.getenv("MODEL_PATH", "/home/arifadh/Desktop/Skripsi-Magang-Proyek/best_models_medsos2/seq60_batch32_hidden32_cnnresnet50_rnninput8_layer3_typemamba_acc0.7842_unidir.pth")  # Example: model file path
SAMPLING_METHOD = os.getenv("SAMPLING_METHOD", "uniform")  # Example: sampling method
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", 60))  # Example: sequence length
VIDEO_DIR = os.getenv("VIDEO_DIR","/home/arifadh/Downloads/tiktok_videos")

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
def post_results(results):
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
            "timestamp": ts
        }

        try:
            response = requests.post(all_config.BACKEND_URL, json=payload)
            if response.status_code in [200, 201]:
                print(f"Successfully sent classification result to backend for {video_name}")
            else:
                print(f"Failed to send classification result for {video_name}. HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Error sending result to backend for {video_name}: {e}")

# Callback function that is triggered when RabbitMQ sends a message
def callback(ch, method, properties, body):
    url = body.decode()
    print(f"Processing URL: {url}")

    # Download the video using pyktok
    pyk.save_tiktok_multi_urls(url, True, '', 1, save_dir=VIDEO_DIR)

    # Load the model
    print("Loading model ...")
    model = torch.load(MODEL_PATH).to(all_config.CONF_DEVICE)
    model.eval()

    # Load dataset (videos to classify)
    data, video_names = loader_data.load_dataset_inference(VIDEO_DIR, SAMPLING_METHOD, SEQUENCE_LENGTH)

    # Transform data to PyTorch tensors
    data_tensors = [torch.tensor(video).permute(0, 3, 1, 2).float().to(all_config.CONF_DEVICE) for video in data]

    # Classify videos and display results
    result = classify_and_display(model, data_tensors, video_names)

    # Post the results to backend
    post_results(result)

    # Acknowledge the message once processing is complete
    ch.basic_ack(delivery_tag=method.delivery_tag)
    print(f"Finished processing: {url}")

# Consume messages from RabbitMQ
def consume_messages():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=os.getenv('RABBITMQ_HOST', 'rabbitmq')))
    channel = connection.channel()

    queue = os.getenv('RABBITMQ_QUEUE', 'queue_name')
    
    # Declare the queue
    channel.queue_declare(queue=queue, durable=True)

    # Enable QoS to limit the number of messages each worker gets (1 or 2 at a time)
    channel.basic_qos(prefetch_count=2)  # This ensures each worker gets a maximum of 2 messages at a time

    # Start consuming messages from the queue
    channel.basic_consume(queue=queue, on_message_callback=callback)

    print(f"Waiting for messages in {queue}. To exit press CTRL+C")
    channel.start_consuming()

if __name__ == "__main__":
    consume_messages()



#note
# ENV nya harus di set pas bikin dockerfile