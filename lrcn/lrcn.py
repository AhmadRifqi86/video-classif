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
SEQUENCE_LENGTH = 40
BATCH_SIZE = 8
HIDDEN_SIZE = 56
CNN_BACKBONE = "densenet121"
RNN_INPUT_SIZE = 512
RNN_LAYER = 4
SAMPLING_METHOD = "uniform"
RNN_OUT = "all"
MAX_VIDEOS = 400
EPOCH = 15
FINETUNE = True
CLASSIF_MODE = "multiple_binary"
MODEL_PATH = '/home/arifadh/Desktop/Skripsi-Magang-Proyek/skripsi/lrcn/model.pth'



CONF_SEQUENCE_LENGTH = SEQUENCE_LENGTH
CONF_BATCH_SIZE = BATCH_SIZE
CONF_HIDDEN_SIZE = HIDDEN_SIZE
CONF_CNN_BACKBONE = CNN_BACKBONE
CONF_RNN_INPUT_SIZE = RNN_INPUT_SIZE
CONF_RNN_LAYER = RNN_LAYER
CONF_SAMPLING_METHOD = SAMPLING_METHOD
CONF_RNN_OUT = RNN_OUT
CONF_MAX_VIDEOS = MAX_VIDEOS
CONF_EPOCH = EPOCH
CONF_FINETUNE = FINETUNE
CONF_MODEL_PATH = MODEL_PATH
CONF_CLASSIF_MODE = CLASSIF_MODE
# Define class labels
CLASS_LABELS = ['Theft', 'Violence', 'Vandalism']

# Frame Difference Methods
def compute_sad(imageA, imageB):
    """Sum of Absolute Differences (SAD)"""
    return np.sum(np.abs(imageA - imageB))

def compute_ssim(imageA, imageB):
    """Structural Similarity Index (SSIM)"""
    return ssim(imageA, imageB, multichannel=True,win_size=3, channel_axis=-1, data_range=1.0)

def compute_optical_flow(imageA, imageB):
    """Optical Flow calculation using Farneback method."""
    prev_gray = cv2.cvtColor(imageA, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(imageB, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(magnitude)

# Sampling method selector
def sample_frames(frames):
    if CONF_SAMPLING_METHOD == "uniform":
        differences = [compute_sad(frames[i], frames[i-1]) for i in range(1, len(frames))]
    elif CONF_SAMPLING_METHOD == "ssim":
        differences = [1 - compute_ssim(frames[i], frames[i-1]) for i in range(1, len(frames))]  # 1 - SSIM to indicate dissimilarity
    elif CONF_SAMPLING_METHOD == "optical-flow":
        differences = [compute_optical_flow(frames[i], frames[i-1]) for i in range(1, len(frames))]
    else:
        raise ValueError("Invalid SAMPLING_METHOD")
    return np.array(differences)

# Custom Dataset class to load video sequences
class VideoDataset(Dataset):
    def __init__(self, data, labels, task_type=CONF_CLASSIF_MODE, transform=None):
        self.data = data
        self.labels = labels
        self.task_type = task_type  # Either "multiclass" or "multiple_binary"
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        video = self.data[index]  # Shape: (sequence_length, H, W)
        label = self.labels[index]

        # Apply transformation to each frame
        if self.transform:
            video = np.stack([self.transform(frame) for frame in video])

        # Convert to tensor
        video = torch.tensor(video, dtype=torch.float32).unsqueeze(1)  # Add channel dimension (1 for grayscale)

        # Handle labels based on task type
        if self.task_type == "multiclass":
            label = torch.tensor(label, dtype=torch.long)
        else:  # multiple_binary
            label = torch.tensor(label, dtype=torch.float32)  # Binary labels as floats

        return video, label


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_dataset(dataset_path, task_type=CONF_CLASSIF_MODE, sequence_length=CONF_SEQUENCE_LENGTH, max_videos_per_class=CONF_MAX_VIDEOS, sampling_method="uniform"):
    video_sequences = []
    labels = []
    
    for class_label in CLASS_LABELS:
        class_path = os.path.join(dataset_path, class_label)
        video_dict = {}
        video_count = 0  # Counter for videos in this class

        for img_name in sorted(os.listdir(class_path), key=natural_sort_key):  # Natural sorting by numbers
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

        # Process videos and apply the selected sampling method
        for video_name, frames in video_dict.items():
            if video_count >= max_videos_per_class:
                break

            if sampling_method == "uniform":
                # Uniform sampling
                if len(frames) >= sequence_length:
                    interval = len(frames) // sequence_length
                    frames = [frames[i] for i in range(0, len(frames), interval)[:sequence_length]]
                else:
                    frames += [np.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (sequence_length - len(frames))
            else:
                # Frame difference-based sampling
                if len(frames) >= sequence_length:
                    differences = sample_frames(frames)
                    frame_indices = np.argsort(differences)[-sequence_length:]  # Select top 'sequence_length' frames
                    frames = [frames[i] for i in sorted(frame_indices)]
                else:
                    frames += [np.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (sequence_length - len(frames))

            video_sequences.append(np.stack(frames))

            # Handle labels based on task type
            if task_type == "multiclass":
                labels.append(CLASS_LABELS.index(class_label))  # Single label for multiclass
            else:
                # Create binary labels for each class (1 for current class, 0 for others)
                binary_label = [1 if i == CLASS_LABELS.index(class_label) else 0 for i in range(len(CLASS_LABELS))]
                labels.append(binary_label)  # One-hot encoded label for binary classification

            video_count += 1

    return np.array(video_sequences), np.array(labels)


# Model Configuration
class LRCN(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size, rnn_input_size, cnn_backbone=CONF_CNN_BACKBONE, rnn_out=CONF_RNN_OUT, freeze_until_layer=None):
        super(LRCN, self).__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.backbone = cnn_backbone

        # Load the configurable backbone
        self.cnn_backbone = getattr(models, cnn_backbone)(pretrained=True)
        # Handling different CNN backbone architectures
        if "resnet" in cnn_backbone or "efficientnet" in cnn_backbone or "inception" in cnn_backbone:
            cnn_out_size = self.cnn_backbone.fc.in_features
            self.cnn_backbone.fc = nn.Identity()  # Replace the final FC layer with identity

        elif "densenet" in cnn_backbone:
            cnn_out_size = self.cnn_backbone.classifier.in_features
            self.cnn_backbone.classifier = nn.Identity()  # Replace the classifier with identity

        elif "vgg" in cnn_backbone or "alexnet" in cnn_backbone:
            #cnn_out_size = self.cnn_backbone.classifier[-1].in_features #25088 gasi?
            cnn_out_size = 25088
            self.cnn_backbone.classifier = nn.Identity()  # Replace the classifier with identity
            self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # Adjust output size to match

        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone}")
            # self.cnn_backbone = nn.Sequential(
            # nn.Conv2d(3, 16, kernel_size=3, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(0.5),
            
            # nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(0.5),

            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # nn.Dropout(0.5)
            # )
            # cnn_out_size = (64 // 4) * (64 // 4) * 64

        # Freeze all layers initially
        self.freeze_cnn_layers(freeze_until_layer, unfreeze_dense_layer=CONF_FINETUNE)

        # Linear layer to adapt CNN output size to RNN input size
        self.adapt = nn.Linear(cnn_out_size, rnn_input_size)

        # LSTM/GRU layer for temporal processing
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=self.hidden_size, num_layers=CONF_RNN_LAYER, bidirectional=True, batch_first=True)

        # Fully connected layer to classify the entire sequence output
        if CONF_CLASSIF_MODE == "multiclass":
            self.fc = nn.Linear(self.hidden_size * 2 * (self.sequence_length if rnn_out == "all" else 1), num_classes)
        elif CONF_CLASSIF_MODE == "multiple_binary":
            self.fc = nn.ModuleList([nn.Linear(self.hidden_size * 2 * (self.sequence_length if rnn_out == "all" else 1), 1) for _ in range(num_classes)])
        else:
            raise ValueError(f"Unsupported CLASSIF_MODE: {CONF_CLASSIF_MODE}")

    def freeze_cnn_layers(self, freeze_until_layer=None, unfreeze_dense_layer=False):
        if unfreeze_dense_layer:
            # Handle the unfreezing of the dense layer based on the architecture
            if "resnet" in self.cnn_backbone.__class__.__name__.lower() or \
               "efficientnet" in self.cnn_backbone.__class__.__name__.lower() or \
               "inception" in self.cnn_backbone.__class__.__name__.lower():
                # Unfreeze only the FC layer for these architectures
                for param in self.cnn_backbone.fc.parameters():
                    param.requires_grad = True

            elif "densenet" in self.cnn_backbone.__class__.__name__.lower():
                # Unfreeze only the classifier layer for DenseNet
                for param in self.cnn_backbone.classifier.parameters():
                    param.requires_grad = True

            elif "vgg" in self.cnn_backbone.__class__.__name__.lower() or \
                 "alexnet" in self.cnn_backbone.__class__.__name__.lower():
                # Unfreeze only the fully connected (dense) layers for VGG and AlexNet
                for param in self.cnn_backbone.classifier.parameters():
                    param.requires_grad = True
            
            else:
                raise ValueError(f"Unsupported CNN backbone: {self.cnn_backbone.__class__.__name__}")

        else:
            # Freeze layers up to freeze_until_layer
            if freeze_until_layer is None:
                # Freeze all layers if no specific layer is mentioned
                for param in self.cnn_backbone.parameters():
                    param.requires_grad = False
            else:
                layer_count = 0
                for name, param in self.cnn_backbone.named_parameters():
                    if layer_count <= freeze_until_layer:
                        param.requires_grad = False  # Freeze this layer
                    else:
                        param.requires_grad = True  # Unfreeze this layer
                    layer_count += 1

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        x = self.cnn_backbone(x) # pass through CNN
        x = x.view(batch_size, seq_len, -1)
        #print("x size after CNN: ",x.size())
        x = self.adapt(x)
        #print("x size after adapt: ",x.size())
        lstm_out, _ = self.lstm(x) # Pass through LSTM/GRU
        if CONF_RNN_OUT == "all":
            lstm_out = lstm_out.contiguous().view(batch_size, -1)  # Use all LSTM outputs
        else:
            lstm_out = lstm_out[:, -1, :]  # Use only the last LSTM output

        if CONF_CLASSIF_MODE == "multiclass":
            out = self.fc(lstm_out)
        else:
            out = torch.cat([fc_layer(lstm_out) for fc_layer in self.fc], dim=1)
        
        return out



def train_model(model, train_loader, criterion, optimizer, num_epochs=10,save_model=True):
    print("Classif mode:",CONF_CLASSIF_MODE)
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
            else:
                loss = sum([criterion[i](outputs[:, i], labels[:, i]) for i in range(model.num_classes)])

            loss.backward()
            optimizer.step()

            if CONF_CLASSIF_MODE == "multiclass":
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            else:
                predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
                correct += (predictions == labels).sum().item()  # Compare predictions with true labels
                total += labels.numel()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    
    if save_model:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

# Evaluation function
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
        
        # Calculate and print overall accuracy across all classes
        overall_accuracy = np.mean(np.all(all_predictions == all_labels, axis=1))
        print(f"Overall Accuracy: {overall_accuracy:.4f}")

    elif CONF_CLASSIF_MODE == "multiclass":
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
        for i, class_name in enumerate(class_names):
            print(f"Class: {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, f1-Score: {f1[i]:.4f}")
    else:
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")




#print("load dataset")
X, y = load_dataset(DATASET_PATH,max_videos_per_class=CONF_MAX_VIDEOS)

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
print(f"Sampling_Method: {CONF_SAMPLING_METHOD}")
print(f"RNN_Out:         {CONF_RNN_OUT}")
print(f"Max_Videos:      {CONF_MAX_VIDEOS}") 
print(f"Epoch:           {CONF_EPOCH}")
print(f"Classif_Mode:    {CONF_CLASSIF_MODE}")

model = LRCN(num_classes=len(CLASS_LABELS), sequence_length=CONF_SEQUENCE_LENGTH, hidden_size=CONF_HIDDEN_SIZE, rnn_input_size=CONF_RNN_INPUT_SIZE).to(device)
criterion = nn.CrossEntropyLoss() if CONF_CLASSIF_MODE == "multiclass" else [nn.BCEWithLogitsLoss() for _ in range(len(CLASS_LABELS))]
optimizer = optim.Adam(model.parameters(), lr=0.0001)
train_model(model, train_loader, criterion, optimizer, num_epochs=CONF_EPOCH, save_model=False)
evaluate_model(model, test_loader, CLASS_LABELS)