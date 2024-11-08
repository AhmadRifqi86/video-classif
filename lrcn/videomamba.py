
import os
import cv2
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from skimage.metrics import structural_similarity as ssim
import logging
import random
from einops import rearrange

DATASET_PATH = '/home/arifadh/Desktop/Dataset/UCF50'  # Path to dataset
TEST_PATH = '/path/to/test'
IMG_HEIGHT, IMG_WIDTH = 64, 64 # Image dimensions
SEQUENCE_LENGTH = 30  # Number of frames per video
BATCH_SIZE = 2  # Batch size for training
HIDDEN_SIZE = 32  # RNN hidden size
CNN_BACKBONE = "densenet121"  # CNN backbone model
RNN_INPUT_SIZE = 512  # Size of RNN input
RNN_LAYER = 4  # Number of RNN layers
RNN_TYPE = "gru"  # Type of RNN (lstm or gru)
SAMPLING_METHOD = "uniform"  # Frame sampling method
RNN_OUT = "all"  # RNN output type
MAX_VIDEOS = 50  # Maximum videos per class
EPOCH = 20  # Number of training epochs
FINETUNE = True  # Whether to fine-tune CNN
CLASSIF_MODE = "multiclass"  # Classification mode
MODEL_PATH = 'model.pth'  # Path to save model
EARLY_STOP = 0.0  # Early stopping threshold

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
    return ssim(img1, img2, multichannel=True)

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


def load_dataset(path, max_videos_per_class=100, task_type="multiclass", 
                sampling_method="uniform"):
    data = []
    labels = []
    class_labels = []
    
    # Pre-define the expected shape of each video
    expected_shape = (CONF_SEQUENCE_LENGTH, IMG_HEIGHT, IMG_WIDTH, 3)
    
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
            if video_path.endswith('.avi'):
                try:
                    cap = cv2.VideoCapture(video_path)
                    frames = []
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    
                    cap.release()
                    
                    if len(frames) > 0:
                        # Apply frame sampling
                        if sampling_method == "ssim":
                            frames = ssim_sampling(frames, CONF_SEQUENCE_LENGTH)
                        else:
                            frames = uniform_sampling(frames, CONF_SEQUENCE_LENGTH)
                            
                        # Handle short videos
                        if len(frames) < CONF_SEQUENCE_LENGTH:
                            frames = duplicate_frames(frames, CONF_SEQUENCE_LENGTH)
                        
                        # Ensure frames array has correct shape
                        frames = np.array(frames)
                        if frames.shape != expected_shape:
                            # Skip videos with incorrect shape
                            print(f"Skipping video {video_name} due to incorrect shape: {frames.shape}")
                            continue
                        
                        frames = frames / 255.0
                        data.append(frames)
                        
                        if task_type == "multiclass":
                            labels.append(label)
                        else:  # binary or multilabel classification
                            # Create a zero array of proper size first
                            binary_label = np.zeros(num_classes, dtype=np.float32)
                            binary_label[label] = 1
                            labels.append(binary_label)
                            
                        video_count += 1
                except Exception as e:
                    print(f"Error processing video {video_name}: {str(e)}")
                    continue
    
    # Convert to numpy arrays with explicit dtype
    data_array = np.array(data, dtype=np.float32)
    
    # Handle labels conversion based on task type
    if task_type == "multiclass":
        labels_array = np.array(labels, dtype=np.int64)
    else:
        # For binary/multilabel, we already have consistent shaped arrays
        labels_array = np.array(labels, dtype=np.float32)
    
    print(f"Final data shape: {data_array.shape}")
    print(f"Final labels shape: {labels_array.shape}")
    
    return data_array, labels_array, class_labels


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

class VideoMamba(nn.Module):
    def __init__(
        self, 
        num_classes,
        cnn_backbone="resnet50",
        n_layer=4,
        d_model=512,
        d_inner=2048,
        n_state=16,
        dt_rank=16,
        num_frames=16,
        temporal_mode='mean',  # 'mean', 'max', 'last', 'all'
        classification_mode='multiclass'  # 'multiclass' or 'multiple_binary'
    ):
        super().__init__()
        
        self.temporal_mode = temporal_mode
        self.classification_mode = classification_mode
        self.num_frames = num_frames
        self.d_model = d_model
        
        # CNN Backbone for spatial features, apa tambah vgg ya?
        self.cnn_backbone = getattr(models, cnn_backbone)(pretrained=True)
        if hasattr(self.cnn_backbone, 'fc'):
            cnn_out_size = self.cnn_backbone.fc.in_features
            self.cnn_backbone.fc = nn.Identity()
        elif hasattr(self.cnn_backbone, 'classifier'):
            cnn_out_size = self.cnn_backbone.classifier.in_features
            self.cnn_backbone.classifier = nn.Identity()
        
        for param in self.cnn_backbone.parameters():  #backbone param is freezed
            param.requires_grad = False
 
        # Project CNN features to Mamba dimension
        self.adapt = nn.Linear(cnn_out_size, d_model)
        
        # Mamba layers for temporal modeling
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_inner, n_state, dt_rank, bias=True, conv_bias=True, kernel_size=3)
            for _ in range(n_layer)
        ])
        
        self.norm_f = RMSNorm(d_model)
        
        # Calculate classifier input size based on temporal mode
        classifier_input_size = d_model * num_frames if temporal_mode == 'all' else d_model
        
        # Create classifier based on classification mode
        if classification_mode == 'multiclass':
            self.classifier = nn.Linear(classifier_input_size, num_classes)
        else:  # multiple_binary
            self.classifier = nn.ModuleList([
                nn.Linear(classifier_input_size, 1) 
                for _ in range(num_classes)
            ])

    def temporal_pool(self, x):
        """Handle different temporal pooling modes"""
        if self.temporal_mode == 'mean':
            return x.mean(dim=1)
        elif self.temporal_mode == 'max':
            return x.max(dim=1)[0]
        elif self.temporal_mode == 'last':
            return x[:, -1]
        elif self.temporal_mode == 'all':
            # Flatten all timesteps together
            return x.reshape(x.size(0), -1)
        else:
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode}")

    def forward(self, x):
        """
        Input: x of shape (batch_size, num_frames, channels, height, width)
        Output: 
            - multiclass: tensor of shape (batch_size, num_classes)
            - multiple_binary: list of tensors, each of shape (batch_size, 1)
        """
        batch_size, num_frames, c, h, w = x.shape
        
        # Process each frame through CNN
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.cnn_backbone(x)
        
        # Project to Mamba dimension
        x = self.adapt(x)
        
        # Reshape to sequence form
        x = x.view(batch_size, num_frames, -1)
        
        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm_f(x)
        
        # Apply temporal pooling
        x = self.temporal_pool(x)
        
        # Apply classification head(s)
        if self.classification_mode == 'multiclass':
            return self.classifier(x)
        else:  # multiple_binary
            return [classifier(x) for classifier in self.classifier]
        

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(videos)
            
            if isinstance(outputs, list):  # multiple binary classification
                # Initialize separate loss for each classification head
                loss = torch.zeros(1, device=device)
                batch_preds = []
                
                # Calculate loss for each classification head separately
                for i, output in enumerate(outputs):
                    head_criterion = nn.BCEWithLogitsLoss()  # Create separate criterion for each head
                    head_loss = head_criterion(output.squeeze(), labels[:, i])
                    loss += head_loss
                    batch_preds.append(torch.sigmoid(output).squeeze().detach())
                
                # Average the losses across all heads
                loss = loss / len(outputs)
                batch_preds = torch.stack(batch_preds, dim=1)
            else:  # multiclass classification
                loss = criterion(outputs, labels)
                batch_preds = torch.softmax(outputs, dim=1).detach()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            all_preds.append(batch_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels, model.classification_mode)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics

def evaluate(model, val_loader, criterion, device):
    """Evaluate the model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Evaluating") as pbar:
            for videos, labels in pbar:
                videos = videos.to(device)
                labels = labels.to(device)
                
                outputs = model(videos)
                
                if isinstance(outputs, list):  # multiple binary classification
                    # Initialize separate loss for each classification head
                    loss = torch.zeros(1, device=device)
                    batch_preds = []
                    
                    # Calculate loss for each classification head separately
                    for i, output in enumerate(outputs):
                        head_criterion = nn.BCEWithLogitsLoss()  # Create separate criterion for each head
                        head_loss = head_criterion(output.squeeze(), labels[:, i])
                        loss += head_loss
                        batch_preds.append(torch.sigmoid(output).squeeze())
                    
                    # Average the losses across all heads
                    loss = loss / len(outputs)
                    batch_preds = torch.stack(batch_preds, dim=1)
                else:  # multiclass classification
                    loss = criterion(outputs, labels)
                    batch_preds = torch.softmax(outputs, dim=1)
                
                total_loss += loss.item()
                
                # Store predictions and labels for metrics
                all_preds.append(batch_preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels, model.classification_mode)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics

def calculate_metrics(predictions, labels, classification_mode):
    """Calculate relevant metrics based on classification mode."""
    metrics = {}
    
    if classification_mode == 'multiclass':
        pred_classes = np.argmax(predictions, axis=1)
        metrics['accuracy'] = accuracy_score(labels, pred_classes)
        metrics['f1'] = f1_score(labels, pred_classes, average='macro')
        try:
            metrics['auc'] = roc_auc_score(labels, predictions, multi_class='ovr')
        except ValueError:
            metrics['auc'] = 0.0
    else:  # multiple_binary
        # Convert predictions to binary (0/1) using 0.5 threshold
        pred_classes = (predictions > 0.5).astype(int)
        metrics['accuracy'] = accuracy_score(labels, pred_classes)
        metrics['f1'] = f1_score(labels, pred_classes, average='macro')
        try:
            metrics['auc'] = roc_auc_score(labels, predictions, average='macro')
        except ValueError:
            metrics['auc'] = 0.0
    
    return metrics

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load dataset
    logger.info("Loading dataset...")
    data, labels, class_labels = load_dataset(
        DATASET_PATH,
        max_videos_per_class=CONF_MAX_VIDEOS,
        task_type=CONF_CLASSIF_MODE,
        sampling_method=CONF_SAMPLING_METHOD
    )
    
    # Split dataset
    num_samples = len(data)
    indices = np.random.permutation(num_samples)
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = VideoDataset(data[train_indices], labels[train_indices], CONF_CLASSIF_MODE)
    val_dataset = VideoDataset(data[val_indices], labels[val_indices], CONF_CLASSIF_MODE)
    test_dataset = VideoDataset(data[test_indices], labels[test_indices], CONF_CLASSIF_MODE)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONF_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONF_BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=CONF_BATCH_SIZE)
    
    # Initialize model
    num_classes = len(class_labels)
    model = VideoMamba(
        num_classes=num_classes,
        cnn_backbone=CONF_CNN_BACKBONE,
        n_layer=CONF_RNN_LAYER,
        d_model=CONF_RNN_INPUT_SIZE,
        num_frames=CONF_SEQUENCE_LENGTH,
        temporal_mode=CONF_RNN_OUT,
        classification_mode=CONF_CLASSIF_MODE
    ).to(device)
    
    # Define loss function based on classification mode
    if CONF_CLASSIF_MODE == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        #criterion = nn.BCEWithLogitsLoss()
        criterion = None
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training loop
    best_val_metric = 0.0
    early_stop_counter = 0
    
    for epoch in range(CONF_EPOCH):
        logger.info(f"\nEpoch {epoch+1}/{CONF_EPOCH}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Training metrics: {train_metrics}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"Validation metrics: {val_metrics}")
        
        # Save best model
        current_metric = val_metrics['f1']  # Can be changed to other metrics
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), CONF_MODEL_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        # Early stopping
        if CONF_EARLY_STOP > 0 and early_stop_counter >= CONF_EARLY_STOP:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(CONF_MODEL_PATH))
    test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(f"\nTest metrics: {test_metrics}")

if __name__ == "__main__":
    main()