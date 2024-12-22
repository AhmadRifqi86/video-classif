import os
import torch
import torch.nn as nn
import torch.optim as optim
#from models_back import LRCN
from model2 import LRCN
from loader_data import load_dataset, VideoDataset
from sklearn.model_selection import train_test_split
from train_eval import train_model, evaluate_model, count_parameters
import all_config
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import h5py
import numpy as np
from collections import Counter

def compute_dataset_class_weights(dataset, indices, num_classes, task_type="multiclass"):
    """
    Compute class weights for the dataset using the training indices
    """
    # Load all labels for the training set
    train_labels = []
    for idx in indices:
        _, label = dataset[idx]
        train_labels.append(label.numpy())
    train_labels = np.array(train_labels)
    
    if task_type == "multiclass":
        # Compute weights for multiclass classification
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),  # Use all possible classes
            y=train_labels
        )
        return nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float).to(all_config.CONF_DEVICE)
        )
    else:
        # Compute weights for multiple binary classification
        criterion_list = []
        for i in range(num_classes):
            binary_labels = train_labels[:, i]
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.array([0, 1]),
                y=binary_labels
            )
            pos_weight = torch.tensor([class_weights[1]/class_weights[0]]).to(all_config.CONF_DEVICE)
            criterion_list.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight))
        return criterion_list

# def display_class_distribution(indices, labels, class_labels):
#     """
#     Display the number of videos for each class in the dataset split.

#     Args:
#         indices (list): Indices of the videos in the split (train/test).
#         labels (np.ndarray): Array of labels corresponding to the videos.
#         class_labels (list): List of class label names.
#     """
#     class_counts = {class_name: 0 for class_name in class_labels}

#     for idx in indices:
#         label = labels[idx]
#         class_counts[class_labels[label]] += 1

#     print("Class distribution:")
#     for class_name, count in class_counts.items():
#         print(f"  {class_name}: {count} videos")

def main():
    
    print("Train Config: ")
    print(f"Seq_Length:      {all_config.CONF_SEQUENCE_LENGTH}")
    print(f"Batch_Size:      {all_config.CONF_BATCH_SIZE}")
    print(f"Hidden_Size:     {all_config.CONF_HIDDEN_SIZE}")
    print(f"CNN_Backbone:    {all_config.CONF_CNN_BACKBONE}")
    print(f"RNN_Input_Size:  {all_config.CONF_RNN_INPUT_SIZE}")
    print(f"RNN_Layer:       {all_config.CONF_RNN_LAYER}")
    print(f"RNN_type:        {all_config.CONF_RNN_TYPE}")
    print(f"Sampling_Method: {all_config.CONF_SAMPLING_METHOD}")
    print(f"RNN_Out:         {all_config.CONF_RNN_OUT}")
    print(f"Max_Videos:      {all_config.CONF_MAX_VIDEOS}") 
    print(f"Epoch:           {all_config.CONF_EPOCH}")
    print(f"Classif_Mode:    {all_config.CONF_CLASSIF_MODE}")
    print(f"Dropout:         {all_config.CONF_DROPOUT}")
    print(f"Bidirectional:   {all_config.CONF_BIDIR}")

    if os.path.exists(all_config.DATA_FILE) and os.path.exists(all_config.CLASSES_FILE):
        print("Processed data found. Loading class labels...")
        class_labels = np.load(all_config.CLASSES_FILE)
    else:
        print("No processed data found. Loading and processing raw dataset...")
        class_labels = load_dataset(
            all_config.DATASET_PATH, 
            max_videos_per_class=all_config.CONF_MAX_VIDEOS,
            batch=all_config.LOAD_BATCH,
            task_type=all_config.CONF_CLASSIF_MODE
        )
    
    # Get total dataset size and labels from HDF5 file
    with h5py.File(all_config.DATA_FILE, 'r') as hf:
        total_samples = len(hf['videos'])
        labels = hf['labels'][:]

    # Calculate split indices
    print("splitting")
    # train_size = int(0.9 * total_samples)
    # indices = np.random.permutation(total_samples)
    # train_indices = indices[:train_size]
    # test_indices = indices[train_size:]
    train_indices, test_indices = train_test_split(
        np.arange(total_samples),
        test_size=0.1,  # 10% for testing
        stratify=labels  # Ensure similar distribution
    )

    # Display class distribution
    def display_class_distribution(indices, labels, class_labels):
        class_counts = {class_name: 0 for class_name in class_labels}
        for idx in indices:
            label = labels[idx]
            class_counts[class_labels[label]] += 1
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} videos")

    print("\nTrain dataset class distribution:")
    display_class_distribution(train_indices, labels, class_labels)

    print("\nTest dataset class distribution:")
    display_class_distribution(test_indices, labels, class_labels)
    
    # Create train and test datasets using indices
    print("load train")
    train_dataset = VideoDataset(
        all_config.DATA_FILE,
        all_config.DATA_FILE,  # Same file contains both videos and labels
        task_type=all_config.CONF_CLASSIF_MODE
    )
    
    print("load test")
    test_dataset = VideoDataset(
        all_config.DATA_FILE,
        all_config.DATA_FILE,  # Same file contains both videos and labels
        task_type=all_config.CONF_CLASSIF_MODE
    )
    
    # Create data loaders with samplers for the splits
    print("randomize")
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    print("Dataloader train")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=all_config.CONF_BATCH_SIZE,
        sampler=train_sampler,
        #shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print("Dataloader test")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=all_config.CONF_BATCH_SIZE,
        sampler=test_sampler,
        #shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print("creating model")
    # Initialize model
    model = LRCN(
        num_classes=len(class_labels), 
        sequence_length=all_config.CONF_SEQUENCE_LENGTH, 
        hidden_size=all_config.CONF_HIDDEN_SIZE, 
        rnn_input_size=all_config.CONF_RNN_INPUT_SIZE
    ).to(all_config.CONF_DEVICE)

    print("creating criterion")
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    print("creating optimizer and display param count")
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_param = count_parameters(model)
    print("Param info: ", num_param)

    # Train the model
    print("start training")
    train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        num_epochs=all_config.CONF_EPOCH
    )
    
    # Evaluate the model
    print("evaluate")
    evaluate_model(model, test_loader, class_labels)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

