
#best epoch = 40, RNNdropout = 0.4, RNNhidden = 24, RNNlayer = 3, batch_size=4, seq_length = 60, conv_layer=[16,32,64], CNN 3 layer
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48/56, RNNlayer = 4, batch_size=8, seq_length = 60, conv = resnet18, RNNType=LSTM, num_vid = 250

import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from skimage.metrics import structural_similarity as ssim

# Configuration constants
# Set parameters
DATASET_PATH = '/home/arifadh/Desktop/Dataset/crime-video/Train'  # Path to dataset
TEST_PATH = '/home/arifadh/Desktop/Dataset/crime-video/Test'
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Image dimensions
EARLY_STOP = 0.0
SEQUENCE_LENGTH = 40
BATCH_SIZE = 8
HIDDEN_SIZE = 56
CNN_BACKBONE = "resnet50"
RNN_INPUT_SIZE = 768
RNN_LAYER = 6
RNN_TYPE = "gru"
SAMPLING_METHOD = "optiflow"
RNN_OUT = "all"
MAX_VIDEOS = 300
EPOCH = 10
FINETUNE = True
CLASSIF_MODE = "multiple_binary"
MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/model.pth'

CONF_EARLY_STOP = EARLY_STOP
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
# Define class labels
CLASS_LABELS = ['Theft','Normal','Vandalism']



def compute_ssim(img1, img2):
    """Compute SSIM between two images."""
    return ssim(img1, img2, multichannel=True)

def ssim_sampling_most_unique(frames, sequence_length, factor=0.5):
    """Sample frames based on SSIM to select the most unique frames with high differences."""
    if len(frames) <= sequence_length:
        return frames  # If there are fewer or equal frames than needed, return them all
    
    ssim_diffs = []
    
    # Compute SSIM for each pair of consecutive frames
    for i in range(1, len(frames) - 1):
        ssim_before = compute_ssim(frames[i], frames[i - 1])
        ssim_after = compute_ssim(frames[i], frames[i + 1])
        
        # Calculate the maximum difference between current frame and neighbors
        ssim_diff = max(abs(ssim_before - 1), abs(ssim_after - 1))
        ssim_diffs.append((ssim_diff, i))  # Store SSIM difference and frame index
    
    # Sort frames based on SSIM differences (from largest to smallest)
    ssim_diffs.sort(reverse=True, key=lambda x: x[0])
    
    # Select the top N frames with the largest SSIM differences
    selected_frames = [frames[0]]  # Always include the first frame
    selected_indices = set([0])  # Track selected indices
    
    for _, idx in ssim_diffs:
        if len(selected_frames) >= sequence_length:
            break
        if idx not in selected_indices:  # Ensure we don't select the same frame multiple times
            selected_frames.append(frames[idx])
            selected_indices.add(idx)

    # Sort the selected frames back into the original order (to maintain video continuity)
    selected_frames.sort(key=lambda x: frames.index(x))

    # Ensure the final length is sequence_length
    return selected_frames[:sequence_length]

def duplicate_frames(frames, sequence_length):
    """Duplicate frames to reach the desired sequence length while maintaining order."""
    num_frames = len(frames)
    if num_frames >= sequence_length:
        return frames[:sequence_length]  # No need to duplicate if we have enough frames

    # Calculate how many frames need to be duplicated
    num_duplicates = sequence_length - num_frames
    
    # Get the positions where the frames should be duplicated
    duplicate_positions = np.linspace(1, num_frames - 1, num_duplicates, dtype=int)
    
    # Insert duplicates at the calculated positions
    duplicated_frames = []
    for i in range(num_frames):
        duplicated_frames.append(frames[i])
        # If the current index is in the duplication positions, insert a duplicate
        if i in duplicate_positions:
            duplicated_frames.append(frames[i])  # Duplicate the current frame

    # Ensure the final length is exactly sequence_length
    while len(duplicated_frames) < sequence_length:
        duplicated_frames.append(frames[-1])  # Append the last frame until we reach the target length

    return duplicated_frames[:sequence_length]

def uniform_sampling(frames, sequence_length):
    """Sample frames uniformly based on the sequence length."""
    if len(frames) <= sequence_length:
        return frames  # If we have fewer frames than needed, return them all

    # Calculate the interval to uniformly sample frames
    interval = len(frames) // sequence_length
    sampled_frames = [frames[i] for i in range(0, len(frames), interval)[:sequence_length]]

    # If fewer than sequence_length frames are selected, pad with zero frames
    if len(sampled_frames) < sequence_length:
        padding = [np.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (sequence_length - len(sampled_frames))
        sampled_frames += padding

    return sampled_frames[:sequence_length]  # Return exactly 'sequence_length' frames

def compute_optical_flow(imageA, imageB):
    """Optical Flow calculation using Farneback method."""
    
    # Convert the images to grayscale, as optical flow is typically computed on single channel images
    prev_gray = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    
    # Calculate optical flow between two consecutive frames
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Compute the magnitude of the flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Sum up the magnitude as a measure of motion intensity
    return np.sum(magnitude)

def optical_flow_sampling_most_unique(frames, sequence_length, factor=0.5):
    """Sample frames based on optical flow to select the most unique frames with high motion differences."""
    if len(frames) <= sequence_length:
        return frames  # If there are fewer or equal frames than needed, return them all

    optical_flow_diffs = []

    # Compute optical flow between each pair of consecutive frames
    for i in range(1, len(frames)):
        flow_magnitude = compute_optical_flow(frames[i-1], frames[i])
        optical_flow_diffs.append((flow_magnitude, i))

    # Sort frames based on optical flow differences (from largest to smallest)
    optical_flow_diffs.sort(reverse=True, key=lambda x: x[0])

    # Select the top N frames with the largest optical flow differences
    selected_frames = [frames[0]]  # Always include the first frame
    selected_indices = set([0])  # Track selected indices

    for _, idx in optical_flow_diffs:
        if len(selected_frames) >= sequence_length:
            break
        if idx not in selected_indices:  # Ensure we don't select the same frame multiple times
            selected_frames.append(frames[idx])
            selected_indices.add(idx)

    # Sort the selected frames back into the original order (to maintain video continuity)
    selected_frames.sort(key=lambda x: frames.index(x))

    # Ensure the final length is sequence_length
    return selected_frames[:sequence_length]


# Custom Dataset class to load video sequences
class VideoDataset(Dataset):
    def __init__(self, data, labels, task_type=CONF_CLASSIF_MODE):
        self.data = data
        self.labels = labels
        self.task_type = task_type  # multiclass or multiple_binary

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video = self.data[index]
        label = self.labels[index]
        video = torch.tensor(video, dtype=torch.float32).unsqueeze(1)  # Add channel dimension

        if self.task_type == "multiclass":
            label = torch.tensor(label, dtype=torch.long)
        else:  # multiple_binary
            label = torch.tensor(label, dtype=torch.float32)

        return video, label


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_dataset(dataset_path, sequence_length, max_videos_per_class=400, task_type="multiclass", sampling_method="uniform", factor=0.5):
    video_sequences = []
    labels = []
    
    for class_label in CLASS_LABELS:
        print("class: ", class_label)
        class_path = os.path.join(dataset_path, class_label)
        video_dict = {}
        video_count = 0

        for img_name in sorted(os.listdir(class_path), key=natural_sort_key):
            if video_count >= max_videos_per_class:
                break

            if img_name.endswith('.png'):
                video_name = '_'.join(img_name.split('_')[:2])
                img_path = os.path.join(class_path, img_name)
                
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                img = img / 255.0

                if video_name not in video_dict:
                    video_dict[video_name] = []
                video_dict[video_name].append(img.transpose(2, 0, 1))

        for video_name, frames in video_dict.items():
            if video_count >= max_videos_per_class:
                break

            if sampling_method == "ssim":
                frames = ssim_sampling_most_unique(frames, sequence_length, factor=factor)
            elif sampling_method == "uniform":
                frames = uniform_sampling(frames, sequence_length)
            elif sampling_method == "optiflow":
                frames = optical_flow_sampling_most_unique(frames, sequence_length, factor=factor)
            # If fewer than sequence_length frames, apply frame duplication
            if len(frames) < sequence_length:
                frames = duplicate_frames(frames, sequence_length)

            video_sequences.append(np.stack(frames))

            # Assign labels based on the task type
            if task_type == "multiclass":
                # Single label for multiclass classification
                labels.append(CLASS_LABELS.index(class_label))
            else:
                # Binary classification (one-vs-rest), one-hot encode the class label
                binary_label = [1 if i == CLASS_LABELS.index(class_label) else 0 for i in range(len(CLASS_LABELS))]
                labels.append(binary_label)

            video_count += 1

    return np.array(video_sequences), np.array(labels)


# Model Configuration
class LRCN(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size, rnn_input_size, cnn_backbone=CONF_CNN_BACKBONE, rnn_type=CONF_RNN_TYPE,rnn_out=CONF_RNN_OUT, freeze_until_layer=None):
        super(LRCN, self).__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.backbone = cnn_backbone
        self.rnn_type = rnn_type
        # Load the CNN backbone (from torchvision models)
        self.cnn_backbone = getattr(models, cnn_backbone)(pretrained=True)

        # Handle different CNN backbones and their output size
        self.cnn_backbone = getattr(models, cnn_backbone)(pretrained=True)
        if hasattr(self.cnn_backbone, 'fc'):
            cnn_out_size = self.cnn_backbone.fc.in_features
            self.cnn_backbone.fc = nn.Identity()
        elif hasattr(self.cnn_backbone, 'classifier'):
            cnn_out_size = self.cnn_backbone.classifier.in_features
            self.cnn_backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")

        # Freeze layers up to the specified layer
        #self.freeze_cnn_layers(freeze_until_layer, unfreeze_dense_layer=CONF_FINETUNE)

        # Linear layer to adapt CNN output to RNN input
        self.adapt = nn.Linear(cnn_out_size, rnn_input_size)

        # LSTM layer for sequence modeling
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=self.hidden_size, num_layers=CONF_RNN_LAYER, bidirectional=True, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=self.hidden_size, num_layers=CONF_RNN_LAYER, bidirectional=True, batch_first=True)
        # Fully connected layers for classification
        if CONF_CLASSIF_MODE == "multiclass":  #112*seq_len to 3
            self.fc = nn.Linear(self.hidden_size * 2 * (self.sequence_length if rnn_out == "all" else 1), num_classes)
        else:  # multiple_binary
            print("Create several fc") #112*seq_len to 1 each
            self.fc = nn.ModuleList([nn.Linear(self.hidden_size * 2 * (self.sequence_length if rnn_out == "all" else 1), 1) for _ in range(num_classes)])

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn_backbone(x)  # Pass through CNN
        #print("after CNN: ",x.size())
        x = x.view(batch_size, seq_len, -1)
        x = self.adapt(x)  # Adapt CNN output to RNN input size
        #print("after adapt: ",x.size())
        lstm_out, _ = self.rnn(x)
        #print("after lstm: ",lstm_out.size())
        if CONF_RNN_OUT == "all":
            lstm_out = lstm_out.contiguous().view(batch_size, -1)
        else:
            lstm_out = lstm_out[:, -1, :]
        #print("lstm_out size all/last: ",lstm_out.size())
        if CONF_CLASSIF_MODE == "multiclass":
            out = self.fc(lstm_out)
        else:
            # For multiple binary classification, apply each FC layer separately
            out = torch.cat([fc_layer(lstm_out) for fc_layer in self.fc], dim=1)

        return out

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, save_model=True, early_stop=0.0):
    print("Classif mode:", CONF_CLASSIF_MODE)
    #early_stop = float(early_stop)
    #print(type(early_stop))
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(2)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if CONF_CLASSIF_MODE == "multiclass":
                loss = criterion(outputs, labels)
            else:  # multiple_binary
                batch_losses = []
                for i in range(len(CLASS_LABELS)):
                    # Ensure inputs and targets have the same batch size
                    output_i = outputs[:, i].view(-1)  # Flatten to [batch_size]
                    label_i = labels[:, i].float()     # Convert to float for BCE
                    # Apply the loss function for this class
                    class_loss = criterion[i](output_i, label_i)
                    batch_losses.append(class_loss)
                loss = sum(batch_losses)
            
            loss.backward()
            optimizer.step()
            
            if CONF_CLASSIF_MODE == "multiclass":
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else: #multiple_binary
                # For binary classification
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.numel()
                correct += (predictions == labels).sum().item()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch_loss < early_stop:  #break the loop if loss less than early stop
            break
    
    if save_model:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

# Evaluation function
from sklearn.metrics import precision_recall_fscore_support

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


#print("load dataset")
X, y = load_dataset(DATASET_PATH,sequence_length=CONF_SEQUENCE_LENGTH,max_videos_per_class=CONF_MAX_VIDEOS, task_type=CONF_CLASSIF_MODE)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch datasets
train_dataset = VideoDataset(X_train, y_train)
test_dataset = VideoDataset(X_test, y_test)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=CONF_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONF_BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

#criterion = nn.CrossEntropyLoss() if CONF_CLASSIF_MODE == "multiclass" else [nn.BCEWithLogitsLoss() for _ in range(len(CLASS_LABELS))]
if CONF_CLASSIF_MODE == "multiclass":
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
else:  # multiple_binary
    class_weights_list = []
    for i in range(len(CLASS_LABELS)):
        # Compute weights for each binary class
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train[:, i])
        pos_weight = torch.tensor([class_weights[1]/class_weights[0]]).to(device)
        class_weights_list.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight))
    criterion = class_weights_list
# Now proceed with the rest of the code
model = LRCN(num_classes=len(CLASS_LABELS), sequence_length=CONF_SEQUENCE_LENGTH, hidden_size=CONF_HIDDEN_SIZE, rnn_input_size=CONF_RNN_INPUT_SIZE).to(device)

if not CONF_FINETUNE:
    for param in model.cnn_backbone.parameters():
        param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_model(model, train_loader, criterion, optimizer, num_epochs=CONF_EPOCH, save_model=True, early_stop=CONF_EARLY_STOP)
evaluate_model(model, test_loader, CLASS_LABELS)



#plan: batch norm siang, result visualization malam
#Command buat ngecount number of video:
# ls Train/{class} | cut -d'_' -f1 | sort | uniq | wc -l

#Command buat ngecount number of frame in each video:
# find Train/{class}/ -name "*.png" | cut -d'_' -f1 | sort | uniq -c


