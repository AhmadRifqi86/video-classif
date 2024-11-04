
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
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
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
        return video, torch.tensor(label, dtype=torch.long)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_dataset(dataset_path, sequence_length=CONF_SEQUENCE_LENGTH, max_videos_per_class=CONF_MAX_VIDEOS, sampling_method="uniform"):
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
            labels.append(CLASS_LABELS.index(class_label))
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

        # Freeze all layers initially
        self.freeze_cnn_layers(freeze_until_layer, unfreeze_dense_layer=CONF_FINETUNE)

        # Linear layer to adapt CNN output size to RNN input size
        self.adapt = nn.Linear(cnn_out_size, rnn_input_size)

        # LSTM/GRU layer for temporal processing
        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=self.hidden_size, num_layers=CONF_RNN_LAYER, bidirectional=True, batch_first=True)

        # Fully connected layer to classify the entire sequence output
        self.fc = nn.Linear(self.hidden_size * 2 * (self.sequence_length if rnn_out == "all" else 1), num_classes)

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

        out = self.fc(lstm_out)
        return out



def train_model(model, train_loader, criterion, optimizer, num_epochs=10, scheduler=None, save_model=False, save_path=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(2)
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

    # Save the model if save_model is True
    if save_model:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")


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
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    #cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate precision and recall
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
    
    # Print precision and recall for each class
    for i, class_name in enumerate(class_names):
        print(f"Class: {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, f1-Score: {f1[i]:.4f}")
    
    # Plot confusion matrix
    # plt.figure(figsize=(10, 7))
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix')
    # plt.show()



#print("load dataset")
X, y = load_dataset(DATASET_PATH,max_videos_per_class=CONF_MAX_VIDEOS,sampling_method=CONF_SAMPLING_METHOD)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch datasets

train_dataset = VideoDataset(X_train, y_train)
test_dataset = VideoDataset(X_test, y_test)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=CONF_BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=CONF_BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)


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

model = LRCN(num_classes=len(CLASS_LABELS), sequence_length=CONF_SEQUENCE_LENGTH,hidden_size=CONF_HIDDEN_SIZE, rnn_input_size=CONF_RNN_INPUT_SIZE).to(device) #best hidden size = 24
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=1e-5)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=CONF_EPOCH, scheduler=None, save_model=True, save_path=CONF_MODEL_PATH)
# Evaluate the model
evaluate_model(model, test_loader,CLASS_LABELS)



#plan: batch norm siang, result visualization malam
#Command buat ngecount number of video:
# ls Train/{class} | cut -d'_' -f1 | sort | uniq | wc -l

#Command buat ngecount number of frame in each video:
# find Train/{class}/ -name "*.png" | cut -d'_' -f1 | sort | uniq -c


##Result Note
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=8, seq_length = 60, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [56.84, 52.63, 50.53, 56.32, 51.90 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48, RNNlayer = 4, batch_size=8, seq_length = 60, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [52.11, 52.11, 49.47, 51.05, 55.26 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=8, seq_length = 80, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [52.63, 57.89, 49.47, 52.11, 55.26 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48, RNNlayer = 4, batch_size=8, seq_length = 80, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [51.05, 53.16, 57.89, 53.68, 50.53 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=8, seq_length = 40, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [55.36, 61.58, 50.00, 56.32, 54.21 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 58, RNNlayer = 4, batch_size=8, seq_length = 40, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [55.26, 54.74, 51.58, 53.16, 53.16 ]

#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=16, seq_length = 60, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [55.79, 50.00, 51.58, 55.26, 58.42]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48, RNNlayer = 4, batch_size=16, seq_length = 60, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=16, seq_length = 80, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48, RNNlayer = 4, batch_size=16, seq_length = 80, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=16, seq_length = 40, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [54.74, 54.74, 55.79, 53.68, 54.74 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 58, RNNlayer = 4, batch_size=16, seq_length = 40, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 

#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=4, seq_length = 60, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48, RNNlayer = 4, batch_size=4, seq_length = 60, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=4, seq_length = 80, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 48, RNNlayer = 4, batch_size=4, seq_length = 80, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 56, RNNlayer = 4, batch_size=4, seq_length = 40, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = [52.63, 51.58, 51.58, 44.74, 51.05 ]
#best epoch = 10, RNNDrop = 0.0, RNNHidden = 58, RNNlayer = 4, batch_size=4, seq_length = 40, conv = resnet50, RNNType=LSTM, num_vid = 400, weighted class, result = 



#
# class LRCN(nn.Module):
#     def __init__(self, num_classes, sequence_length, hidden_size, input_shape=(3, 64, 64)):
#         super(LRCN, self).__init__()
#         self.sequence_length = sequence_length
#         self.num_classes = num_classes
#         self.hidden_size = hidden_size  # Store hidden size
        
#         # CNN feature extractor updated for RGB (3 input channels)
#         # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         # self.conv3 = nn.Conv2d(32,64,kernel_size=3, padding=1)
#         # self.bn1 = nn.BatchNorm2d(16)
#         # self.bn2 = nn.BatchNorm2d(32)
#         # self.bn3 = nn.BatchNorm2d(64)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.dropout = nn.Dropout(0.5)

#         self.resnet = models.resnet50(pretrained=True)
#         # Get the output size of the pre-trained CNN (ResNet18 in this case)
#         cnn_out_size = self.resnet.fc.in_features 
#         # Remove the last fully connected layer
#         self.resnet.fc = nn.Identity()  # We replace the final FC layer with an identity layer
#         for param in self.resnet.parameters():
#             param.requires_grad = False
#         for param in self.resnet.layer4.parameters():  # Unfreeze last ResNet block
#             param.requires_grad = True
        
#         # Calculate CNN output size
#         #cnn_out_size = (input_shape[1] // 4) * (input_shape[2] // 4) * 64  # Adjust based on pooling /128 kalo 2 layer conv  /64 kalo 3 layer conv
        
#         # LSTM to process CNN features over time with dynamic hidden size
#         self.gru = nn.LSTM(input_size=cnn_out_size, hidden_size=self.hidden_size, num_layers=4,bidirectional=True, batch_first=True)  #best drop-out=0.5, num_layer=4
#         self.fc = nn.Linear(self.hidden_size * sequence_length * 2, num_classes)
#         #If unidirectional
#         #self.fc = nn.Linear(self.hidden_size * sequence_length , num_classes)

#     def forward(self, x):
#         batch_size, seq_len, c, h, w = x.size()

#         # Reshape to (batch_size * seq_len, c, h, w) to process each frame with CNN
#         x = x.view(batch_size * seq_len, c, h, w)

#         # x = F.relu(self.bn1(self.conv1(x)))
#         # x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))

#         # x = F.relu(self.bn1(self.conv1(x)))
#         # x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
#         # x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))

#         x = self.resnet(x)

#         # Flatten CNN output for each frame
#         x = x.view(batch_size, seq_len, -1)
#         gru_out, _ = self.gru(x)

#         # Flatten the last hidden states for the fully connected layer
#         gru_out = gru_out.contiguous().view(batch_size, -1)

#         # Pass the flattened LSTM output to the fully connected layer
#         out = self.fc(gru_out)

#         return out