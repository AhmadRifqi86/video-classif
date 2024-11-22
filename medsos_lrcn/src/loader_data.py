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
from tqdm import tqdm

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


def load_dataset(path, max_videos_per_class=all_config.MAX_VIDEOS, batch=all_config.CONF_BATCH_SIZE, task_type=all_config.CONF_CLASSIF_MODE):
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
                        # Read video and sample frames
                        cap = cv2.VideoCapture(video_path)
                        frames = []
                        while len(frames) < all_config.CONF_SEQUENCE_LENGTH:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Resize and preprocess frame
                            frame = cv2.resize(frame, (all_config.IMG_HEIGHT, all_config.IMG_WIDTH))
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                        
                        cap.release()
                        
                        # Handle short videos by duplicating frames
                        if len(frames) < all_config.CONF_SEQUENCE_LENGTH:
                            frames += [frames[-1]] * (all_config.CONF_SEQUENCE_LENGTH - len(frames))
                        
                        # Truncate to exact sequence length and normalize
                        frames = np.array(frames[:all_config.CONF_SEQUENCE_LENGTH]) / 255.0
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
        #return np.array(hf_data['videos']), np.array(hf_data['labels']), class_labels

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
