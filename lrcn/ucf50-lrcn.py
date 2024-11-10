import os
import cv2
import torch
import time
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from einops import rearrange
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration constants
DATASET_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/train'  # Path to dataset
VAL_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/val'
TEST_PATH = '/home/arifadh/Desktop/Dataset/tikHarm/Dataset/test'
PROCESSED_DATA_PATH = "/home/arifadh/Desktop/Skripsi-Magang-Proyek/temporary"
DATA_FILE = os.path.join(PROCESSED_DATA_PATH, "X_data_700.npy")
LABELS_FILE = os.path.join(PROCESSED_DATA_PATH, "y_labels_700.npy")
CLASSES_FILE = os.path.join(PROCESSED_DATA_PATH, "class_labels_700.pkl")
TEST_PATH = '/path/to/test'
IMG_HEIGHT, IMG_WIDTH = 80, 80 # Image dimensions
SEQUENCE_LENGTH = 40
BATCH_SIZE = 2
HIDDEN_SIZE = 56
CNN_BACKBONE = "resnet50"
RNN_INPUT_SIZE = 512
RNN_LAYER = 4
RNN_TYPE = "lstm"
SAMPLING_METHOD = "uniform"
RNN_OUT = "all"
MAX_VIDEOS = 700
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
    def __init__(self, data, labels, task_type="multiclass"):
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

class RMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))


    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class ParallelMamba(nn.Module):
    def __init__(self, d_model, d_inner, n_state, dt_rank, bias=True, conv_bias=True, kernel_size=3):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_state = n_state
        self.dt_rank = dt_rank

        # Parameters for the state-space model
        self.A_log = nn.Parameter(torch.randn(d_inner, n_state))
        self.D = nn.Parameter(torch.randn(d_inner))

        # Projections
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=kernel_size,
            groups=d_inner,
            padding=kernel_size - 1
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + n_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)

    def parallel_scan(self, u, delta, A, B, C):
        """
        Parallel selective scan implementation
        Args:
            u: input tensor (B, L, D)
            delta: time delta (B, L, D)
            A: state matrix (D, N)
            B: input matrix (B, L, N)
            C: output matrix (B, L, N)
        """
        batch_size, seq_len, d_inner = u.shape
        n_state = A.shape[1]
        
        # Discretize A and B (parallel across batch and length)
        deltaA = torch.exp(torch.einsum('b l d, d n -> b l d n', delta, A))
        deltaB_u = torch.einsum('b l d, b l n, b l d -> b l d n', delta, B, u)
        
        # Parallel scan implementation in chunks
        chunk_size = 256
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        def chunk_scan(chunk_idx):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            chunk_len = end_idx - start_idx
            
            # Initialize state for the chunk
            x = torch.zeros((batch_size, d_inner, n_state), device=deltaA.device)
            chunk_states = []
            
            for i in range(chunk_len):
                x = deltaA[:, start_idx + i] * x + deltaB_u[:, start_idx + i]
                y = torch.einsum('b d n, b n -> b d', x, C[:, start_idx + i])
                chunk_states.append(y)
            
            chunk_states = torch.stack(chunk_states, dim=1)
            return chunk_states

        # Process each chunk and combine results
        chunk_results = [chunk_scan(i) for i in range(num_chunks)]
        states = torch.cat(chunk_results, dim=1)  # shape (B, L, D)

        return states

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        n_state = self.A_log.shape[1]
        
        # Input projection with residual separation
        x_and_res = self.in_proj(x)  # (B, L, 2 * d_inner)
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Apply convolution
        x = rearrange(x, 'b l d -> b d l')  # for conv1d (B, D, L)
        x = self.conv1d(x)
        x = x[:, :, :seq_len]  # remove extra padding if any
        x = rearrange(x, 'b d l -> b l d')  # back to (B, L, D)
        
        # Apply activation
        x = F.silu(x)

        # Project to obtain delta, B, C, this could be substitute by ssm(x)
        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([self.dt_rank, n_state, n_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # shape (B, L, d_inner)
        
        # Compute A from log
        A = -torch.exp(self.A_log)

        # Run the parallel scan SSM
        y = self.parallel_scan(x, delta, A, B, C)
        
        # Apply residual connection and activation, end of ssm(x)
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        return output

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_state, dt_rank, bias=True, conv_bias=True, kernel_size=3):
        super().__init__()
        self.mixer = ParallelMamba(d_model, d_inner, n_state, dt_rank, bias=bias, conv_bias=conv_bias, kernel_size=kernel_size)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

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
        elif rnn_type == "mamba":
            self.rnn = nn.ModuleList([
            ResidualBlock(rnn_input_size, rnn_input_size*2, hidden_size, hidden_size, bias=True, conv_bias=True, kernel_size=3)
            for _ in range(CONF_RNN_LAYER)
        ])
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
        if self.rnn_type == "mamba":
            for layer in self.rnn:
                x = layer(x)  # Apply each ResidualBlock layer
            rnn_out = x
        else:
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
    
    start = time.time()
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
    
    duration = time.time() - start
    print(f"training_duration: {duration:.4f}")
    if save_model:
        torch.save(model, CONF_MODEL_PATH)
        print(f"Model saved to {CONF_MODEL_PATH}")

def evaluate_model(model, test_loader, class_names):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    start = time.time()
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
    duration = time.time()-start
    print(f"inference_duration: {duration:.4f}")

def save_processed_data(X, y, class_labels):
    """Save processed data to disk."""
    np.save(DATA_FILE, X)
    np.save(LABELS_FILE, y)
    with open(CLASSES_FILE, "wb") as f:
        pickle.dump(class_labels, f)
    print(f"Data saved to {PROCESSED_DATA_PATH}")

def load_processed_data():
    """Load processed data from disk."""
    X = np.load(DATA_FILE)
    y = np.load(LABELS_FILE)
    with open(CLASSES_FILE, "rb") as f:
        class_labels = pickle.load(f)
    print(f"Data loaded from {PROCESSED_DATA_PATH}")
    return X, y, class_labels

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
    print(f"{DATA_FILE},{LABELS_FILE},{CLASSES_FILE}")
    # Load and prepare data
    if os.path.exists(DATA_FILE) and os.path.exists(LABELS_FILE) and os.path.exists(CLASSES_FILE):
        print("Processed data found. Loading data...")
        X, y, class_labels = load_processed_data()
    else:
        print("No processed data found. Loading and processing raw dataset...")
        X, y, class_labels = load_dataset(
            DATASET_PATH, 
            max_videos_per_class=CONF_MAX_VIDEOS,
            task_type=CONF_CLASSIF_MODE,
            sampling_method=CONF_SAMPLING_METHOD
        )
        save_processed_data(X, y, class_labels)
    
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
        num_epochs=CONF_EPOCH,
        early_stop=CONF_EARLY_STOP
    )
    
    # Evaluate the model
    evaluate_model(model, test_loader, class_labels)

if __name__ == "__main__":
    main()
