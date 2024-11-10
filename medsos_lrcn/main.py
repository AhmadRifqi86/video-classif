import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import LRCN
from loader_data import load_dataset, VideoDataset, load_processed_data, save_processed_data
from train_eval import train_model, evaluate_model
import all_config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

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
    
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    print(f"{all_config.DATA_FILE},{all_config.LABELS_FILE},{all_config.CLASSES_FILE}")
    # Load and prepare data
    if os.path.exists(all_config.DATA_FILE) and os.path.exists(all_config.LABELS_FILE) and os.path.exists(all_config.CLASSES_FILE):
        print("Processed data found. Loading data...")
        X, y, class_labels = load_processed_data()
    else:
        print("No processed data found. Loading and processing raw dataset...")
        X, y, class_labels = load_dataset(
            all_config.DATASET_PATH, 
            max_videos_per_class=all_config.CONF_MAX_VIDEOS,
            task_type=all_config.CONF_CLASSIF_MODE,
            sampling_method=all_config.CONF_SAMPLING_METHOD
        )
        save_processed_data(X, y, class_labels)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Compute class weights for balanced learning
    if all_config.CONF_CLASSIF_MODE == "multiclass":
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(all_config.CONF_DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:  # multiple_binary
        class_weights_list = []
        for i in range(len(class_labels)):
            # Compute weights for each binary class
            class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train[:, i])
            pos_weight = torch.tensor([class_weights[1]/class_weights[0]]).to(all_config.CONF_DEVICE)
            class_weights_list.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight))
        criterion = class_weights_list
    
    # Prepare datasets and dataloaders
    train_dataset = VideoDataset(X_train, y_train)
    test_dataset = VideoDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=all_config.CONF_BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=all_config.CONF_BATCH_SIZE, 
        shuffle=False
    )
    
    # Initialize model
    model = LRCN(
        num_classes=len(class_labels), 
        sequence_length=all_config.CONF_SEQUENCE_LENGTH, 
        hidden_size=all_config.CONF_HIDDEN_SIZE, 
        rnn_input_size=all_config.CONF_RNN_INPUT_SIZE
    ).to(all_config.CONF_DEVICE)
    
    # Freeze CNN backbone if not fine-tuning
    # if not CONF_FINETUNE:
    #     for param in model.cnn_backbone.parameters():
    #         param.requires_grad = False
    
    # Select optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model
    train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        num_epochs=all_config.CONF_EPOCH,
        early_stop=all_config.CONF_EARLY_STOP
    )
    
    # Evaluate the model
    evaluate_model(model, test_loader, class_labels)

if __name__ == "__main__":
    main()