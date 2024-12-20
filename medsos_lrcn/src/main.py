import os
import torch
import torch.nn as nn
import torch.optim as optim
#from models_back import LRCN
from model2 import LRCN
from loader_data import load_dataset, VideoDataset, load_processed_data, save_processed_data, save_sampled_data
from train_eval import train_model, evaluate_model, count_parameters
import all_config
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import h5py
import numpy as np

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
    print(f"Bidirectional    {all_config.CONF_BIDIR}")

    if os.path.exists(all_config.DATA_FILE) and os.path.exists(all_config.CLASSES_FILE):
        print("Processed data found. Loading class labels...")
        class_labels = np.load(all_config.CLASSES_FILE)
    else:
        print("No processed data found. Loading and processing raw dataset...")
        load_dataset(
            all_config.DATASET_PATH, 
            max_videos_per_class=all_config.CONF_MAX_VIDEOS,
            batch = all_config.BATCH_SIZE,
            task_type=all_config.CONF_CLASSIF_MODE
        )
    
    # Get total dataset size from HDF5 file
    with h5py.File(all_config.DATA_FILE, 'r') as hf:
        total_samples = len(hf['videos'])
    
    # Calculate split indices
    print("splitting")
    train_size = int(0.8 * total_samples)
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
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
        num_workers=4,
        pin_memory=True
    )
    
    print("Dataloader test")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=all_config.CONF_BATCH_SIZE,
        sampler=test_sampler,
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

    # print("creating criterion")
    # criterion = compute_dataset_class_weights(
    #     dataset=train_dataset,
    #     indices=train_indices,
    #     num_classes=len(class_labels),
    #     task_type=all_config.CONF_CLASSIF_MODE
    # )
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
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    # print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")
    # Evaluate the model
    print("evaluate")
    evaluate_model(model, test_loader, class_labels)
    
    # print(f"Allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    # print(f"Cached: {torch.cuda.memory_reserved() / 1e9} GB")

if __name__ == "__main__":
    main()