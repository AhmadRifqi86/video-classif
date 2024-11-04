import os
import cv2
import torch
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

# Set parameters
DATASET_PATH = '/home/arifadh/Desktop/Dataset/crime-video/Train'  # Path to dataset
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Image dimensions
SEQUENCE_LENGTH = 60  # Number of frames per video
BATCH_SIZE = 8  # Batch size for training
CLASS_LABELS = ['Theft', 'Violence', 'Vandalism']

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

def load_dataset(dataset_path, sequence_length=SEQUENCE_LENGTH, max_videos_per_class=400):
    video_sequences = []
    labels = []

    for class_label in CLASS_LABELS:
        class_path = os.path.join(dataset_path, class_label)
        video_dict = {}
        video_count = 0  # Counter for videos in this class
        print("loaded class: ", class_label)
        # Collect images and group them as videos based on their naming convention
        for img_name in sorted(os.listdir(class_path), key=natural_sort_key):  # Natural sorting by numbers
            if video_count >= max_videos_per_class:  # Stop loading if we have enough videos for this class
                break

            if img_name.endswith('.png'):
                video_name = '_'.join(img_name.split('_')[:2])  # e.g., 'Normal_Videos_944_x264'
                img_path = os.path.join(class_path, img_name)

                # Load and preprocess image (for RGB, use IMREAD_COLOR)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load RGB image
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # Resize image to 64x64
                img = img / 255.0  # Normalize image (0-1)

                if video_name not in video_dict:
                    video_dict[video_name] = []
                # Ensure that frames are of shape (C, H, W) -> (3, 64, 64)
                video_dict[video_name].append(img.transpose(2, 0, 1))  # Shape: (3, 64, 64)

        # Process videos and ensure they have a consistent length (padding if necessary)
        for video_name, frames in video_dict.items():
            if video_count >= max_videos_per_class:  # Stop loading if we have enough videos for this class
                break

            if len(frames) >= sequence_length:
                frames = frames[:sequence_length]  # Use only the first 'sequence_length' frames
            else:
                # Pad with zeros if less than 'sequence_length'
                frames += [np.zeros((3, IMG_HEIGHT, IMG_WIDTH))] * (sequence_length - len(frames))

            # Stack frames into an array with shape (sequence_length, 3, IMG_HEIGHT, IMG_WIDTH)
            video_sequences.append(np.stack(frames))  # Stack to ensure the correct shape
            labels.append(CLASS_LABELS.index(class_label))

            video_count += 1  # Increment video count for this class

        # Print out the number of videos in the current class
        print(f"Class '{class_label}' has {video_count} videos.")

    # Ensure the output is of shape (batch_size, sequence_length, 3, IMG_HEIGHT, IMG_WIDTH)
    return np.array(video_sequences), np.array(labels)


# Define TimeDistributed CNN-LSTM Model
class TimeDistributedCNNLSTM(nn.Module):
    def __init__(self, num_classes=5, input_shape=(80, 64, 64, 1)):  # Assuming (time_steps, height, width, channels)
        super(TimeDistributedCNNLSTM, self).__init__()

        # TimeDistributed Conv2D and MaxPooling2D
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)

        # Flatten and LSTM
        self.flatten = nn.Flatten()
        self.lstm = nn.LSTM(64, 32, batch_first=True)  # LSTM with hidden size 32

        # Fully connected layer
        self.fc1 = nn.Linear(32, num_classes)
        #If using entire sequence
        self.fc2 = nn.Linear(32 * 80, num_classes)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()

        # Reshape to combine batch and time steps for Conv2D
        x = x.view(batch_size * time_steps, C, H, W)

        # TimeDistributed Conv2D + MaxPooling
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Reshape back to (batch_size, time_steps, flattened_size)
        x = x.view(batch_size, time_steps, -1)

        # LSTM layer
        x, (hn, cn) = self.lstm(x)

        # Final Dense (fully connected) layer
        #x = x.contiguous().view(batch_size, -1)
        x = self.fc1(x[:, -1, :])  # Take the last output of the LSTM for classification
       
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, scheduler=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(2)
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

        if scheduler is not None:
            scheduler.step()

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

    cm = confusion_matrix(all_labels, all_predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
    for i, class_name in enumerate(class_names):
        print(f"Class: {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}")

    plt.figure(figsize=(10, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(10, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Load dataset and split
print("load dataset")
X, y = load_dataset(DATASET_PATH)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create PyTorch datasets
train_dataset = VideoDataset(X_train, y_train)
test_dataset = VideoDataset(X_test, y_test)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

# Instantiate the model, loss function, and optimizer
model = TimeDistributedCNNLSTM(num_classes=len(CLASS_LABELS), input_shape=(SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)).to(device)

# Use weighted cross entropy to account for class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10, scheduler=None)

# Evaluate the model
evaluate_model(model, test_loader, CLASS_LABELS)

