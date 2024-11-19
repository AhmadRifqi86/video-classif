import os
import numpy as np
import cv2
import torch
import pickle
import all_config
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
import json
import re

def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    return ssim(img1, img2, multichannel=True,win_size=3, channel_axis=-1)

def ssim_sampling(frames, sequence_length):
    """Sample frames based on structural similarity."""
    if len(frames) <= sequence_length:
        return frames

    ssim_diffs = []
    for i in range(1, len(frames)):
        diff = compute_ssim(frames[i-1], frames[i])
        ssim_diffs.append((diff, i))
    
    ssim_diffs.sort(key=lambda x: x[0])
    selected_indices = [0] + [idx for _, idx in ssim_diffs[:sequence_length-1]]
    selected_indices.sort()
    
    return [frames[i] for i in selected_indices[:sequence_length]]

def uniform_sampling(frames, sequence_length):
    """Sample frames uniformly."""
    if len(frames) <= sequence_length:
        return frames

    interval = len(frames) // sequence_length
    return [frames[i] for i in range(0, len(frames), interval)][:sequence_length]

def duplicate_frames(frames, sequence_length):
    """Duplicate frames to reach desired sequence length."""
    if len(frames) >= sequence_length:
        return frames[:sequence_length]
        
    duplicated = []
    while len(duplicated) < sequence_length:
        duplicated.extend(frames)
    return duplicated[:sequence_length]

class VideoDataset(Dataset):
    def __init__(self, data, labels, task_type="multiclass"):
        self.data = data
        self.labels = labels
        self.task_type = task_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video = self.data[index]
        label = self.labels[index]
        
        video = torch.tensor(video, dtype=torch.float32).permute(0,3,1,2)
        if self.task_type == "multiclass":
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)
            
        return video, label

def load_dataset(path, max_videos_per_class=100, task_type="multiclass", sampling_method="uniform"):
    data = []
    labels = []
    class_labels = []

    # First pass: collect all class names to determine total number of classes
    all_classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print(all_classes)
    num_classes = len(all_classes)

    for class_name in all_classes:
        class_dir = os.path.join(path, class_name)
        print(f"Loading class: {class_name}")
        class_labels.append(class_name)
        label = len(class_labels) - 1

        video_count = 0
        for video_name in os.listdir(class_dir):
            if video_count >= max_videos_per_class:
                break

            video_path = os.path.join(class_dir, video_name)
            if video_path.endswith('.mp4'):  # Adjust as per the video format
                print("processing: ",video_name)
                try:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Warning: Could not open video file {video_name}")
                        continue  # Skip this file if it can't be opened
                    
                    frames = []
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (all_config.IMG_HEIGHT, all_config.IMG_WIDTH))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)

                    cap.release()
                    
                    if len(frames) == 0:
                        print(f"Warning: No frames found in {video_name}")
                        continue  # Skip if no frames were read

                    # Apply frame sampling
                    if sampling_method == "ssim":
                        frames = ssim_sampling(frames, all_config.CONF_SEQUENCE_LENGTH)
                    else:
                        frames = uniform_sampling(frames, all_config.CONF_SEQUENCE_LENGTH)
                    
                    # Handle short videos
                    if len(frames) < all_config.CONF_SEQUENCE_LENGTH:
                        frames = duplicate_frames(frames, all_config.CONF_SEQUENCE_LENGTH)
                    
                    frames = np.array(frames) / 255.0  # Normalize pixel values
                    data.append(frames)

                    # Set labels based on task type
                    if task_type == "multiclass":
                        labels.append(label)
                    else:
                        # For binary/multilabel, create a one-hot label array
                        binary_label = np.zeros(num_classes, dtype=np.float32)
                        binary_label[label] = 1
                        labels.append(binary_label)

                    video_count += 1

                except Exception as e:
                    print(f"Error processing video {video_name}: {str(e)}")
                    continue  # Skip to the next video if there's an error

    # Convert lists to numpy arrays for consistency
    data_array = np.array(data, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int64 if task_type == "multiclass" else np.float32)
    
    print(f"Final data shape: {data_array.shape}")
    print(f"Final labels shape: {labels_array.shape}")
    
    return data_array, labels_array, class_labels

def save_processed_data(X, y, class_labels):
    """Save processed data to disk."""
    np.save(all_config.DATA_FILE, X)
    np.save(all_config.LABELS_FILE, y)
    with open(all_config.CLASSES_FILE, "wb") as f:
        pickle.dump(class_labels, f)
    print(f"Data saved to {all_config.PROCESSED_DATA_PATH}")

def load_processed_data():
    """Load processed data from disk."""
    X = np.load(all_config.DATA_FILE)
    y = np.load(all_config.LABELS_FILE)
    with open(all_config.CLASSES_FILE, "rb") as f:
        class_labels = pickle.load(f)
    print(f"Data loaded from {all_config.PROCESSED_DATA_PATH}")
    return X, y, class_labels


def load_dataset_inference(path, sampling_method="uniform", sequence_length=30):
    data = []
    video_names = []

    for video_name in os.listdir(path):
        video_path = os.path.join(path, video_name)
        if not video_path.endswith('.mp4'):  # Adjust for supported formats
            continue

        print(f"Processing video: {video_name}")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video file {video_name}")
                continue  # Skip this file if it can't be opened
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (all_config.IMG_HEIGHT, all_config.IMG_WIDTH))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                print(f"Warning: No frames found in {video_name}")
                continue  # Skip if no frames were read

            # Apply frame sampling
            if sampling_method == "ssim":
                frames = ssim_sampling(frames, sequence_length)
            else:
                frames = uniform_sampling(frames, sequence_length)

            # Handle short videos
            if len(frames) < sequence_length:
                frames = duplicate_frames(frames, sequence_length)

            frames = np.array(frames) / 255.0  # Normalize pixel values
            data.append(frames)
            video_names.append(video_name)

        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
            continue  # Skip to the next video if there's an error

    # Convert data to numpy array
    data_array = np.array(data, dtype=np.float32)
    print(f"Final data shape: {data_array.shape}")
    return data_array, video_names


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

def serialize_document(document):
    if document is None:
        return None
    document["_id"] = str(document["_id"])  # Convert ObjectId to string
    return document

def construct_url(video_name):
    pattern = r"(?P<username>@.+?)_video_(?P<video_id>\d+)"
    match = re.match(pattern, video_name)
    if match:
        username = match.group("username")
        video_id = match.group("video_id")
        return f"https://www.tiktok.com/{username}/video/{video_id}"
    return None
