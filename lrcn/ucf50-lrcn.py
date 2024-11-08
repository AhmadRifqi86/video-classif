import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration constants
DATASET_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/train'  # Path to dataset
TEST_PATH = '/path/to/test'
IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
SEQUENCE_LENGTH = 40
BATCH_SIZE = 2
HIDDEN_SIZE = 48
CNN_BACKBONE = "densenet121"
RNN_INPUT_SIZE = 512
RNN_LAYER = 4
RNN_TYPE = "lstm"
SAMPLING_METHOD = "uniform"
RNN_OUT = "last"
MAX_VIDEOS = 300
EPOCH = 30
FINETUNE = True
CLASSIF_MODE = "multiclass"
MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/model.pth'  # Path to save model
EARLY_STOP = 0.0

# Transfer configuration to variables
CONF_SEQUENCE_LENGTH = SEQUENCE_LENGTH
CONF_BATCH_SIZE = BATCH_SIZE
CONF_HIDDEN_SIZE = HIDDEN_SIZE
CONF_CNN_BACKBONE = CNN_BACKBONE
CONF_RNN_INPUT_SIZE = RNN_INPUT_SIZE
CONF_RNN_LAYER = RNN_LAYER
CONF_RNN_TYPE = RNN_TYPE
CONF_SAMPLING_METHOD = SAMPLING_METHOD
CONF_RNN_OUT = RNN_OUT
CONF_MAX_VIDEOS = MAX_VIDEOS
CONF_EPOCH = EPOCH
CONF_FINETUNE = FINETUNE
CONF_MODEL_PATH = MODEL_PATH
CONF_CLASSIF_MODE = CLASSIF_MODE
CONF_EARLY_STOP = EARLY_STOP

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
    def __init__(self, data, labels, task_type=CONF_CLASSIF_MODE):
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

class LRCN(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size, rnn_input_size, 
                 cnn_backbone=CONF_CNN_BACKBONE, rnn_type=CONF_RNN_TYPE, 
                 rnn_out=CONF_RNN_OUT):
        super(LRCN, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.backbone = cnn_backbone
        self.rnn_type = rnn_type
        
        # Load CNN backbone
        self.cnn_backbone = getattr(models, cnn_backbone)(pretrained=True)
        if hasattr(self.cnn_backbone, 'fc'):
            cnn_out_size = self.cnn_backbone.fc.in_features
            self.cnn_backbone.fc = nn.Identity()
        elif hasattr(self.cnn_backbone, 'classifier'):
            cnn_out_size = self.cnn_backbone.classifier.in_features
            self.cnn_backbone.classifier = nn.Identity()
        
        for param in self.cnn_backbone.parameters():  #backbone param is freezed
            param.requires_grad = False

        # Adaptation layer
        self.adapt1 = nn.Linear(cnn_out_size, cnn_out_size//2)
        self.adapt2 = nn.Linear(cnn_out_size//2, cnn_out_size//4)
        self.adapt3 = nn.Linear(cnn_out_size//4, rnn_input_size)
        
        # RNN layer
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size,
                              num_layers=CONF_RNN_LAYER, bidirectional=True, 
                              batch_first=True)
        else:  # GRU
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size,
                             num_layers=CONF_RNN_LAYER, bidirectional=True, 
                             batch_first=True)
        
        # Output layer
        if CONF_CLASSIF_MODE == "multiclass":
            self.fc = nn.Linear(hidden_size * 2 * (sequence_length if rnn_out == "all" else 1), 
                              num_classes)
        else:
            self.fc = nn.ModuleList([
                nn.Linear(hidden_size * 2 * (sequence_length if rnn_out == "all" else 1), 1) 
                for _ in range(num_classes)
            ])

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process through CNN
        x = x.view(batch_size * seq_len, c, h, w)
        #print("x before cnn: ",x.size())
        x = self.cnn_backbone(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.adapt1(x)
        x = self.adapt2(x)
        x = self.adapt3(x)
        # Process through RNN
        rnn_out, _ = self.rnn(x)
        
        # Handle different output modes
        if CONF_RNN_OUT == "all":
            rnn_out = rnn_out.contiguous().view(batch_size, -1)
        else:
            rnn_out = rnn_out[:, -1, :]
        
        # Final classification
        if CONF_CLASSIF_MODE == "multiclass":
            out = self.fc(rnn_out)
        else:
            out = torch.cat([fc(rnn_out) for fc in self.fc], dim=1)
        
        return out

def load_dataset(path, max_videos_per_class=100, task_type="multiclass", sampling_method="uniform"):
    data = []
    labels = []
    class_labels = []

    # First pass: collect all class names to determine total number of classes
    all_classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
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
                        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)

                    cap.release()
                    
                    if len(frames) == 0:
                        print(f"Warning: No frames found in {video_name}")
                        continue  # Skip if no frames were read

                    # Apply frame sampling
                    if sampling_method == "ssim":
                        frames = ssim_sampling(frames, CONF_SEQUENCE_LENGTH)
                    else:
                        frames = uniform_sampling(frames, CONF_SEQUENCE_LENGTH)
                    
                    # Handle short videos
                    if len(frames) < CONF_SEQUENCE_LENGTH:
                        frames = duplicate_frames(frames, CONF_SEQUENCE_LENGTH)
                    
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
        
    
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, 
                save_model=True, early_stop=0.0):
    print(f"Training with {CONF_CLASSIF_MODE} classification mode")
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if CONF_CLASSIF_MODE == "multiclass":
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                batch_losses = []
                for i in range(outputs.size(1)):
                    output_i = outputs[:, i]
                    label_i = labels[:, i].float()
                    batch_losses.append(criterion[i](output_i, label_i))
                loss = sum(batch_losses)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.numel()
                correct += (predictions == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, "
              f"Accuracy: {epoch_acc:.4f}")
        
        if epoch_loss < early_stop:
            print("Early stopping triggered")
            break
    
    if save_model:
        torch.save(model.state_dict(), CONF_MODEL_PATH)
        print(f"Model saved to {CONF_MODEL_PATH}")

def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(2)
            outputs = model(inputs)

            if CONF_CLASSIF_MODE == "multiple_binary":
                predictions = torch.sigmoid(outputs) > 0.5
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            else:
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    if CONF_CLASSIF_MODE == "multiple_binary":
        all_labels = np.concatenate(all_labels, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # Calculate accuracy, precision, recall, and f1-score for each class
        accuracies = []
        for i, class_name in enumerate(class_names):
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels[:, i], all_predictions[:, i], average="binary")
            accuracy = np.mean(all_predictions[:, i] == all_labels[:, i])  # Calculate accuracy for this class
            accuracies.append(accuracy)
            print(f"Class {class_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, f1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        
        # Calculate overall precision, recall, and F1-score across all classes
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro")
        print(f"Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}, Overall F1-Score: {overall_f1:.4f}")

        # Calculate and print overall accuracy across all classes
        overall_accuracy = np.mean(np.all(all_predictions == all_labels, axis=1))
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

    elif CONF_CLASSIF_MODE == "multiclass":
        accuracy = correct / total  # This is the overall accuracy
        print(f"Overall Accuracy: {accuracy:.4f}")

        # Class-wise precision, recall, and F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)

        # Print per-class results
        for i, class_name in enumerate(class_names):
            print(f"Class: {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, f1-Score: {f1[i]:.4f}")

        # Calculate overall precision, recall, and F1-score across all classes
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average="macro")
        print(f"Overall Precision: {overall_precision:.4f}, Overall Recall: {overall_recall:.4f}, Overall F1-Score: {overall_f1:.4f}")
    else:
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")


def main():
    print("Train Config: ")
    print(f"Seq_Length:      {CONF_SEQUENCE_LENGTH}")
    print(f"Batch_Size:      {CONF_BATCH_SIZE}")
    print(f"Hidden_Size:     {CONF_HIDDEN_SIZE}")
    print(f"CNN_Backbone:    {CONF_CNN_BACKBONE}")
    print(f"RNN_Input_Size:  {CONF_RNN_INPUT_SIZE}")
    print(f"RNN_Layer:       {CONF_RNN_LAYER}")
    print(f"RNN_type:        {CONF_RNN_TYPE}")
    print(f"Sampling_Method: {CONF_SAMPLING_METHOD}")
    print(f"RNN_Out:         {CONF_RNN_OUT}")
    print(f"Max_Videos:      {CONF_MAX_VIDEOS}") 
    print(f"Epoch:           {CONF_EPOCH}")
    print(f"Classif_Mode:    {CONF_CLASSIF_MODE}")
    
    # Set device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    
    # Load and prepare data
    X, y, class_labels = load_dataset(
        DATASET_PATH, 
        max_videos_per_class=CONF_MAX_VIDEOS,
        task_type=CONF_CLASSIF_MODE,
        sampling_method=CONF_SAMPLING_METHOD
    )
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Compute class weights for balanced learning
    if CONF_CLASSIF_MODE == "multiclass":
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:  # multiple_binary
        class_weights_list = []
        for i in range(len(class_labels)):
            # Compute weights for each binary class
            class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train[:, i])
            pos_weight = torch.tensor([class_weights[1]/class_weights[0]]).to(device)
            class_weights_list.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight))
        criterion = class_weights_list
    
    # Prepare datasets and dataloaders
    train_dataset = VideoDataset(X_train, y_train)
    test_dataset = VideoDataset(X_test, y_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONF_BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONF_BATCH_SIZE, 
        shuffle=False
    )
    
    # Initialize model
    model = LRCN(
        num_classes=len(class_labels), 
        sequence_length=CONF_SEQUENCE_LENGTH, 
        hidden_size=CONF_HIDDEN_SIZE, 
        rnn_input_size=CONF_RNN_INPUT_SIZE
    ).to(device)
    
    # Freeze CNN backbone if not fine-tuning
    if not CONF_FINETUNE:
        for param in model.cnn_backbone.parameters():
            param.requires_grad = False
    
    # Select optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train the model
    train_model(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        num_epochs=CONF_EPOCH,
        early_stop=CONF_EARLY_STOP
    )
    
    # Evaluate the model
    evaluate_model(model, test_loader, class_labels)

if __name__ == "__main__":
    main()
