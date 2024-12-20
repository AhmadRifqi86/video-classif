import os
import numpy as np
import cv2
import torch
import pickle
import h5py
import all_config
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
import json
import re
import requests
import random
#from tqdm import tqdm

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

# class VideoDataset(Dataset):
#     def __init__(self, data, labels, task_type="multiclass"):
#         self.data = data
#         self.labels = labels
#         self.task_type = task_type

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         video = self.data[index]
#         label = self.labels[index]
        
#         video = torch.tensor(video, dtype=torch.float32).permute(0,3,1,2)
#         if self.task_type == "multiclass":
#             label = torch.tensor(label, dtype=torch.long)
#         else:
#             label = torch.tensor(label, dtype=torch.float32)
            
#         return video, label

class VideoDataset(Dataset):
    def __init__(self, data, labels, task_type="multiclass"):
        """
        Initialize VideoDataset with either HDF5 file paths or numpy arrays
        
        Args:
            data: Either HDF5 file path (str) or numpy array of video data
            labels: Either HDF5 file path (str) or numpy array of labels
            task_type: "multiclass" or "multiple_binary" for classification type
        """
        self.task_type = task_type
        
        if isinstance(data, str):
            # If data is a file path, open HDF5 file
            self.h5_file = h5py.File(data, 'r')
            self.data = self.h5_file['videos']
            self.labels = self.h5_file['labels']
            self.using_h5 = True
        else:
            # If data is numpy array, use directly
            self.h5_file = None
            self.data = data
            self.labels = labels
            self.using_h5 = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Load video data
        if self.using_h5:
            video = self.data[index]
            label = self.labels[index]
        else:
            video = self.data[index]
            label = self.labels[index]
        
        # Convert to tensor and permute dimensions (sequence, channels, height, width)
        video = torch.tensor(video, dtype=torch.float32).permute(0, 3, 1, 2)
        
        # Handle label based on task type
        if self.task_type == "multiclass":
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.float32)
            
        return video, label
    
    def __del__(self):
        # Clean up HDF5 file handle if using one
        if self.h5_file is not None:
            self.h5_file.close()

def load_dataset_simple(path, max_videos_per_class=100, task_type="multiclass", sampling_method="uniform"):
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
                #print("processing: ",video_name)
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


def sample_frames_uniform(video_path, num_frames, img_height, img_width):
    """
    Uniformly sample frames from a video
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to sample
        img_height (int): Target image height
        img_width (int): Target image width
    
    Returns:
        np.ndarray: Sampled and processed frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Uniform frame selection
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and preprocess frame
            frame = cv2.resize(frame, (img_height, img_width))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If frame reading fails, use last successful frame
            if frames:
                frames.append(frames[-1])
            else:
                cap.release()
                return None
    
    cap.release()
    
    # Ensure exactly num_frames are returned
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    
    return np.array(frames[:num_frames]) / 255.0

def sample_frames_random(video_path, num_frames, img_height, img_width):
    """
    Randomly sample frames from a video
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to sample
        img_height (int): Target image height
        img_width (int): Target image width
    
    Returns:
        np.ndarray: Sampled and processed frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Randomly select frame indices
    frame_indices = np.sort(np.random.choice(total_frames, num_frames, replace=False))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize and preprocess frame
            frame = cv2.resize(frame, (img_height, img_width))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If frame reading fails, use last successful frame
            if frames:
                frames.append(frames[-1])
            else:
                cap.release()
                return None
    
    cap.release()
    
    # Ensure exactly num_frames are returned
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    
    return np.array(frames[:num_frames]) / 255.0

def sample_frames_ssim(video_path, num_frames, img_height, img_width):
    """
    Sample frames based on Structural Similarity Index (SSIM)
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to sample
        img_height (int): Target image height
        img_width (int): Target image width
    
    Returns:
        np.ndarray: Sampled and processed frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Read all frames and preprocess
    all_frames = []
    for i in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Resize and preprocess frame
            frame = cv2.resize(frame, (img_height, img_width))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)
    
    cap.release()
    
    if len(all_frames) == 0:
        return None
    
    # Convert to grayscale for SSIM
    all_frames_gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in all_frames]
    
    # Compute SSIM between consecutive frames
    ssim_values = []
    for i in range(1, len(all_frames_gray)):
        ssim_value = ssim(all_frames_gray[i-1], all_frames_gray[i])
        ssim_values.append(ssim_value)
    
    # Add initial frame
    ssim_values.insert(0, 0)
    
    # Sort frames by their SSIM values (lower SSIM indicates more distinct frames)
    sorted_frame_indices = sorted(range(len(ssim_values)), key=lambda k: ssim_values[k])
    
    # Select frames with lowest SSIM (most distinct)
    selected_indices = sorted(sorted_frame_indices[:num_frames])
    
    # Ensure exactly num_frames are returned
    selected_frames = [all_frames[idx] for idx in selected_indices]
    
    if len(selected_frames) < num_frames:
        selected_frames += [selected_frames[-1]] * (num_frames - len(selected_frames))
    
    return np.array(selected_frames[:num_frames]) / 255.0

def load_dataset(path, max_videos_per_class=all_config.MAX_VIDEOS, 
                 batch=all_config.LOAD_BATCH, 
                 task_type=all_config.CONF_CLASSIF_MODE,
                 sampling_method='uniform'):
    """
    Enhanced dataset loading with configurable frame sampling
    
    Args:
        path (str): Path to the dataset directory
        max_videos_per_class (int): Maximum videos to process per class
        batch (int): Batch size for processing
        task_type (str): Classification mode ('multiclass' or 'multilabel')
        sampling_method (str): Frame sampling method 
            - 'uniform': Evenly spaced frames
            - 'random': Randomly selected frames
            - 'ssim': Select frames with lowest structural similarity
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Select sampling method
    sampling_func = {
        'uniform': sample_frames_uniform,
        'random': sample_frames_random,
        'ssim': sample_frames_ssim
    }.get(sampling_method, sample_frames_uniform)
    
    class_labels = []
    total_videos = 0
    
    # First pass: collect all class names
    all_classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print("Found classes:", all_classes)
    num_classes = len(all_classes)
    
    # Create HDF5 file for storing processed videos
    with h5py.File(all_config.DATA_FILE, 'w') as hf_data:
        # Create datasets with max size
        max_total_videos = max_videos_per_class * num_classes
        hf_data.create_dataset(
            'videos', 
            shape=(0, all_config.CONF_SEQUENCE_LENGTH, all_config.IMG_HEIGHT, all_config.IMG_WIDTH, 3),
            maxshape=(max_total_videos, all_config.CONF_SEQUENCE_LENGTH, all_config.IMG_HEIGHT, all_config.IMG_WIDTH, 3),
            dtype=np.float32,
            chunks=True
        )
        
        # Create labels dataset based on task type
        if task_type == "multiclass":
            label_shape = (0,)
            label_maxshape = (max_total_videos,)
            label_dtype = np.int64
        else:
            label_shape = (0, num_classes)
            label_maxshape = (max_total_videos, num_classes)
            label_dtype = np.float32
            
        hf_data.create_dataset(
            'labels',
            shape=label_shape,
            maxshape=label_maxshape,
            dtype=label_dtype,
            chunks=True
        )
        
        # Process videos class by class
        for class_idx, class_name in enumerate(all_classes):
            class_dir = os.path.join(path, class_name)
            print(f"\nProcessing class: {class_name}")
            class_labels.append(class_name)
            
            video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            videos_to_process = min(len(video_files), max_videos_per_class)
            
            # Process videos in batches
            for batch_start in range(0, videos_to_process, batch):
                batch_end = min(batch_start + batch, videos_to_process)
                batch_videos = []
                batch_labels = []
                
                # Process batch of videos
                for video_idx in range(batch_start, batch_end):
                    video_name = video_files[video_idx]
                    video_path = os.path.join(class_dir, video_name)
                    
                    try:
                        # Sample frames using the specified method
                        frames = sampling_func(
                            video_path, 
                            all_config.CONF_SEQUENCE_LENGTH, 
                            all_config.IMG_HEIGHT, 
                            all_config.IMG_WIDTH
                        )
                        
                        if frames is None:
                            print(f"Skipping {video_name}: Unable to sample frames")
                            continue
                        
                        batch_videos.append(frames)
                        
                        # Create label
                        if task_type == "multiclass":
                            batch_labels.append(class_idx)
                        else:
                            binary_label = np.zeros(num_classes, dtype=np.float32)
                            binary_label[class_idx] = 1
                            batch_labels.append(binary_label)
                    
                    except Exception as e:
                        print(f"Error processing {video_name}: {str(e)}")
                        continue
                
                # Save batch to HDF5 file
                if batch_videos:
                    batch_videos = np.array(batch_videos)
                    batch_labels = np.array(batch_labels)
                    
                    # Resize HDF5 datasets and append new data
                    current_size = hf_data['videos'].shape[0]
                    new_size = current_size + len(batch_videos)
                    
                    hf_data['videos'].resize(new_size, axis=0)
                    hf_data['labels'].resize(new_size, axis=0)
                    
                    hf_data['videos'][current_size:new_size] = batch_videos
                    hf_data['labels'][current_size:new_size] = batch_labels
                    
                    total_videos += len(batch_videos)
                    print(f"Saved batch: {len(batch_videos)} videos, Total: {total_videos}")
        
        # Save class labels
        np.save(all_config.CLASSES_FILE, class_labels)
        print(f"Dataset processing complete. Total videos: {total_videos}")
        
        # Return loaded data
        return class_labels
# def load_dataset(path, max_videos_per_class=all_config.CONF_MAX_VIDEOS, frames_per_video=all_config.CONF_SEQUENCE_LENGTH, chunk_size=64):
#     """
#     Load videos in chunks to manage memory efficiently
    
#     Args:
#     - path: Dataset root directory
#     - max_videos_per_class: Max videos to process per class
#     - frames_per_video: Number of frames to sample per video
#     - chunk_size: Number of videos to process in each memory chunk
#     """
#     processed_videos = []
#     processed_labels = []
    
#     # Get all video classes
#     classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
#     for class_idx, class_name in enumerate(classes):
#         class_path = os.path.join(path, class_name)
#         video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4')]
        
#         # Limit videos per class
#         video_files = video_files[:max_videos_per_class]
        
#         # Process videos in chunks
#         for i in range(0, len(video_files), chunk_size):
#             chunk_videos = video_files[i:i+chunk_size]
#             chunk_data = []
#             chunk_labels = []
            
#             for video_file in chunk_videos:
#                 video_path = os.path.join(class_path, video_file)
                
#                 # Load and sample video frames
#                 frames = load_video_frames(video_path, frames_per_video)
                
#                 if frames is not None:
#                     chunk_data.append(frames)
#                     chunk_labels.append(class_idx)
            
#             # Optional: Further process or save chunk
#             processed_videos.extend(chunk_data)
#             processed_labels.extend(chunk_labels)
            
#             # Clear memory after processing chunk
#             del chunk_data
#             del chunk_labels
    
#     return np.array(processed_videos), np.array(processed_labels)

# def load_video_frames(video_path, frames_per_video=60):
#     try:
#         cap = cv2.VideoCapture(video_path)
#         frames = []
        
#         while len(frames) < frames_per_video:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Resize and normalize frame
#             frame = cv2.resize(frame, (224, 224))
#             frame = frame / 255.0  # Normalize
#             frames.append(frame)
        
#         cap.release()
        
#         # Handle videos shorter than desired frame count
#         if len(frames) < frames_per_video:
#             # Repeat last frame to match desired length
#             frames += [frames[-1]] * (frames_per_video - len(frames))
        
#         # Sample exactly desired number of frames
#         frames = frames[:frames_per_video]
        
#         return np.array(frames)
    
#     except Exception as e:
#         print(f"Error processing {video_path}: {e}")
#         return None

def save_processed_data(X, y, class_labels):
    """Save processed data to disk."""
    np.save(all_config.DATA_FILE, X)
    np.save(all_config.LABELS_FILE, y)
    with open(all_config.CLASSES_FILE, "wb") as f:
        pickle.dump(class_labels, f)
    print(f"Data saved to {all_config.PROCESSED_DATA_PATH}")

def load_processed_data():
    """Load processed data from disk."""
    X = np.load(all_config.DATA_FILE,allow_pickle=True)
    y = np.load(all_config.LABELS_FILE)
    with open(all_config.CLASSES_FILE, "rb") as f:
        class_labels = pickle.load(f)
    print(f"Data loaded from {all_config.PROCESSED_DATA_PATH}")
    return X, y, class_labels

def save_sampled_data(X, y, class_labels):
    """Append processed data to existing files."""
    # Check if data files already exist
    if os.path.exists(all_config.DATA_FILE) and os.path.exists(all_config.LABELS_FILE) and os.path.exists(all_config.CLASSES_FILE):
        # Load existing data
        try:
            existing_X = np.load(all_config.DATA_FILE)
            existing_y = np.load(all_config.LABELS_FILE)
            with open(all_config.CLASSES_FILE, "rb") as f:
                existing_class_labels = pickle.load(f)
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            existing_X, existing_y, existing_class_labels = np.empty((0,) + X.shape[1:]), np.empty((0,)), []

        # Append new data
        X = np.concatenate((existing_X, X), axis=0)
        y = np.concatenate((existing_y, y), axis=0)
        class_labels = sorted(list(set(existing_class_labels + class_labels)))
    else:
        print("No existing data found. Creating new files.")

    # Validate data shapes
    assert X.shape[0] == y.shape[0], f"Mismatch between data and labels: {X.shape}, {y.shape}"

    # Save the updated data
    np.save(all_config.DATA_FILE, X)
    np.save(all_config.LABELS_FILE, y)
    with open(all_config.CLASSES_FILE, "wb") as f:
        pickle.dump(class_labels, f)

    print(f"Data appended and saved to {all_config.PROCESSED_DATA_PATH}")

def load_dataset_inference(path, sampling_method="uniform", sequence_length=30):
    data = []
    video_names = []

    for video_name in os.listdir(path):
        video_path = os.path.join(path, video_name)
        if not video_path.endswith('.mp4'):  # Adjust for supported formats
            continue

        if is_url_classified(video_name):
            print("deleting video: ",video_name)  #deleting video_path
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"Deleted video: {video_path}")
                except Exception as e:
                    print(f"Error deleting video {video_path}: {e}")
            else:
                print(f"Video {video_path} not found.")
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

def is_url_classified(video_url):
    video_url = construct_url(video_url)
    # print("checker, video_url: ",video_url)
    try:
        response = requests.get(all_config.BACKEND_CHECKER, params={"url": video_url})
        if response.status_code == 200:
            data = response.json()
            if "url" in data and "labels" in data:
                print(f"URL {video_url} is already classified with label: {data['labels']}")
                return True
            else:
                print(f"URL {video_url} is not classified yet.")
                return False
        else:
            print(f"Failed to check classification status for {video_url}. HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"Error checking classification status for {video_url}: {e}")
        return False



def load_dataset_simple(path, 
                         max_videos_per_class=all_config.MAX_VIDEOS, 
                         batch=all_config.LOAD_BATCH, 
                         task_type=all_config.CONF_CLASSIF_MODE,
                         sampling_method='uniform'):
    """
    Enhanced dataset loading with configurable frame sampling and batch processing
    
    Args:
        path (str): Path to the dataset directory
        max_videos_per_class (int): Maximum videos to process per class
        batch (int): Batch size for processing
        task_type (str): Classification mode ('multiclass' or 'multilabel')
        sampling_method (str): Frame sampling method 
            - 'uniform': Evenly spaced frames
            - 'random': Randomly selected frames
            - 'ssim': Select frames with lowest structural similarity
    
    Returns:
        tuple: Processed data, labels, and class labels
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Select sampling method
    sampling_func = {
        'uniform': uniform_sampling,
        'random': sample_frames_random,  # Assuming this function exists
        'ssim': sample_frames_ssim       # Assuming this function exists
    }.get(sampling_method, uniform_sampling)
    
    class_labels = []
    total_videos = 0
    
    # First pass: collect all class names
    all_classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print("Found classes:", all_classes)
    num_classes = len(all_classes)
    
    # Create HDF5 file for storing processed videos
    with h5py.File(all_config.DATA_FILE, 'w') as hf_data:
        # Create datasets with max size
        max_total_videos = max_videos_per_class * num_classes
        hf_data.create_dataset(
            'videos', 
            shape=(0, all_config.CONF_SEQUENCE_LENGTH, all_config.IMG_HEIGHT, all_config.IMG_WIDTH, 3),
            maxshape=(max_total_videos, all_config.CONF_SEQUENCE_LENGTH, all_config.IMG_HEIGHT, all_config.IMG_WIDTH, 3),
            dtype=np.float32,
            chunks=True
        )
        
        # Create labels dataset based on task type
        if task_type == "multiclass":
            label_shape = (0,)
            label_maxshape = (max_total_videos,)
            label_dtype = np.int64
        else:
            label_shape = (0, num_classes)
            label_maxshape = (max_total_videos, num_classes)
            label_dtype = np.float32
            
        hf_data.create_dataset(
            'labels',
            shape=label_shape,
            maxshape=label_maxshape,
            dtype=label_dtype,
            chunks=True
        )
        
        # Process videos class by class
        for class_idx, class_name in enumerate(all_classes):
            class_dir = os.path.join(path, class_name)
            print(f"\nProcessing class: {class_name}")
            class_labels.append(class_name)
            
            video_files = [f for f in os.listdir(class_dir) if f.endswith('.mp4')]
            videos_to_process = min(len(video_files), max_videos_per_class)
            
            # Process videos in batches
            for batch_start in range(0, videos_to_process, batch):
                batch_end = min(batch_start + batch, videos_to_process)
                batch_videos = []
                batch_labels = []
                
                # Process batch of videos
                for video_idx in range(batch_start, batch_end):
                    video_name = video_files[video_idx]
                    video_path = os.path.join(class_dir, video_name)
                    
                    try:
                        # Open video capture
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            print(f"Warning: Could not open video file {video_name}")
                            continue
                        
                        # Read and process frames
                        frames = []
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame = cv2.resize(frame, (all_config.IMG_HEIGHT, all_config.IMG_WIDTH))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        cap.release()
                        
                        # Skip if no frames
                        if len(frames) == 0:
                            print(f"Skipping {video_name}: No frames found")
                            continue
                        
                        # Sample frames using the specified method
                        sampled_frames = sampling_func(
                            frames, 
                            all_config.CONF_SEQUENCE_LENGTH, 
                            all_config.IMG_HEIGHT, 
                            all_config.IMG_WIDTH
                        )
                        
                        if sampled_frames is None:
                            print(f"Skipping {video_name}: Unable to sample frames")
                            continue
                        
                        batch_videos.append(sampled_frames)
                        
                        # Create label
                        if task_type == "multiclass":
                            batch_labels.append(class_idx)
                        else:
                            binary_label = np.zeros(num_classes, dtype=np.float32)
                            binary_label[class_idx] = 1
                            batch_labels.append(binary_label)
                    
                    except Exception as e:
                        print(f"Error processing {video_name}: {str(e)}")
                        continue
                
                # Save batch to HDF5 file
                if batch_videos:
                    batch_videos = np.array(batch_videos)
                    batch_labels = np.array(batch_labels)
                    
                    # Resize HDF5 datasets and append new data
                    current_size = hf_data['videos'].shape[0]
                    new_size = current_size + len(batch_videos)
                    
                    hf_data['videos'].resize(new_size, axis=0)
                    hf_data['labels'].resize(new_size, axis=0)
                    
                    hf_data['videos'][current_size:new_size] = batch_videos
                    hf_data['labels'][current_size:new_size] = batch_labels
                    
                    total_videos += len(batch_videos)
                    print(f"Saved batch: {len(batch_videos)} videos, Total: {total_videos}")
        
        # Save class labels
        np.save(all_config.CLASSES_FILE, class_labels)
        print(f"Dataset processing complete. Total videos: {total_videos}")
        
        # Optionally, return data from HDF5 file
        return class_labels
