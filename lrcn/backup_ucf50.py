#result: Test Accuracy: 0.7230
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

# Set parameters
DATASET_PATH = '/home/arifadh/Desktop/Dataset/UCF50'  # Path to dataset
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Image dimensions
SEQUENCE_LENGTH = 60  # Number of frames per video
BATCH_SIZE = 8  # Batch size for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the dataset

def load_dataset(path, max_videos_per_class=100):
    data = []
    labels = []
    class_labels = []

    # Iterate over each subdirectory in the dataset path
    for class_name in os.listdir(path):
        class_dir = os.path.join(path, class_name)
        if os.path.isdir(class_dir):
            print("class loaded: ",class_name)
            class_labels.append(class_name)
            label = len(class_labels) - 1  # Assign an integer label to each class

            video_count = 0  # Initialize a counter for the number of videos loaded per class

            # Iterate over each video file in the subdirectory
            for video_name in os.listdir(class_dir):
                if video_count >= max_videos_per_class:
                    break  # Stop loading videos once the limit is reached

                video_path = os.path.join(class_dir, video_name)
                #print("loading: ",video_path)
                if video_path.endswith('.avi'):  # Adjust based on your dataset file format
                    # Read video frames using OpenCV
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames

                    # If video has fewer frames than required, skip it
                    if total_frames < SEQUENCE_LENGTH:
                        cap.release()
                        continue

                    # Calculate the sampling interval
                    interval = total_frames // SEQUENCE_LENGTH

                    frames = []
                    for i in range(SEQUENCE_LENGTH):
                        # Seek to the required frame
                        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))  # Resize frames
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                        frames.append(frame)

                    cap.release()
                    
                    # Ensure we have exactly SEQUENCE_LENGTH frames
                    if len(frames) == SEQUENCE_LENGTH:
                        data.append(frames)
                        labels.append(label)
                        video_count += 1  # Increment the counter for videos

    return np.array(data), np.array(labels), class_labels



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
        video = torch.tensor(video, dtype=torch.float32).permute(0, 3, 1, 2)  # (sequence_length, channels, H, W)
        return video, torch.tensor(label, dtype=torch.long)

# LRCN (Long-term Recurrent Convolutional Network) model
class LRCN2(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size, input_shape=(3, 64, 64)):
        super(LRCN2, self).__init__()
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3) #Dropout jadi 0.25
        
        # Calculate CNN output size
        cnn_out_size = (input_shape[1] // 4) * (input_shape[2] // 4) * 64
        
        # LSTM
        self.lstm = nn.GRU(input_size=cnn_out_size, hidden_size=self.hidden_size, num_layers=1, bidirectional = True, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size * sequence_length * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Reshape to (batch_size * seq_len, c, h, w) for CNN
        x = x.view(batch_size * seq_len, c, h, w)

        # Pass through CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))

        # Flatten CNN output
        x = x.reshape(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)

        # Flatten the LSTM output for fully connected layer
        lstm_out = lstm_out.contiguous().view(batch_size, -1)

        # Pass to fully connected layer
        out = self.fc(lstm_out)
        return out

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, num_epochs=10, scheduler=None):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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

        if scheduler is not None:
            scheduler.step()

# Function to evaluate the model, add per class accuracy
def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Calculate precision and recall
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None, zero_division=0)
    
    # Print precision and recall for each class
    for i, class_name in enumerate(class_names):
        print(f"Class: {class_name} - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(40, 30),dpi=100)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

# Main function
def main():
    # Load dataset
    X, y, CLASS_LABELS = load_dataset(DATASET_PATH)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create PyTorch datasets
    train_dataset = VideoDataset(X_train, y_train)
    test_dataset = VideoDataset(X_test, y_test)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model, loss function, and optimizer
    model = LRCN2(num_classes=len(CLASS_LABELS), sequence_length=SEQUENCE_LENGTH, hidden_size=32).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=30)

    # Evaluate the model
    evaluate_model(model, test_loader, CLASS_LABELS)

# Run the main function
if __name__ == "__main__":
    main()



