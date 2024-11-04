import math
import json
import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
import random
from einops import rearrange



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


class VideoMambaTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        classification_mode='multiclass',
        scheduler=None,
        early_stopping_patience=10,
        clip_grad_norm=1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.classification_mode = classification_mode
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.clip_grad_norm = clip_grad_norm
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (videos, labels) in enumerate(pbar):
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(videos)
            
            if self.classification_mode == 'multiclass':
                loss = self.criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:  # multiple_binary
                loss = 0
                batch_preds = []
                for i, output in enumerate(outputs):
                    loss += self.criterion(output, labels[:, i].unsqueeze(1).float())
                    batch_preds.append((output > 0).float())
                loss = loss / len(outputs)  # Average loss across all binary classifiers
                all_preds.extend(torch.cat(batch_preds, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            loss.backward()
            
            # Gradient clipping
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            
            self.optimizer.step()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = total_loss / len(train_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        return epoch_loss, metrics
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc='Validation'):
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(videos)
                
                if self.classification_mode == 'multiclass':
                    loss = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                else:  # multiple_binary
                    loss = 0
                    batch_preds = []
                    for i, output in enumerate(outputs):
                        loss += self.criterion(output, labels[:, i].unsqueeze(1).float())
                        batch_preds.append((output > 0).float())
                    loss = loss / len(outputs)
                    all_preds.extend(torch.cat(batch_preds, dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                total_loss += loss.item()
        
        val_loss = total_loss / len(val_loader)
        metrics = self.calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        return val_loss, metrics
    
    def calculate_metrics(self, labels, predictions):
        if self.classification_mode == 'multiclass':
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='weighted')
            }
        else:  # multiple_binary
            return {
                'accuracy': accuracy_score(labels.flatten(), predictions.flatten()),
                'f1': f1_score(labels, predictions, average='macro'),
                'auc': roc_auc_score(labels, predictions, average='macro')
            }
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = []
        
        for epoch in range(num_epochs):
            self.logger.info(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Log metrics
            self.logger.info(
                f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n'
                f'Train Metrics: {train_metrics}\n'
                f'Val Metrics: {val_metrics}'
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info('Early stopping triggered')
                    break
            
            # Store history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })
        
        return training_history

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'num_classes': 5,
        'num_frames': 16,
        'batch_size': 8,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'classification_mode': 'multiple_binary',  # or 'multiclass'
        'temporal_mode': 'mean',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    # Generate dummy video data for demonstration
    def generate_dummy_data(num_samples):
        videos = torch.randn(num_samples, config['num_frames'], 3, 64, 64)
        if config['classification_mode'] == 'multiclass':
            labels = torch.randint(0, config['num_classes'], (num_samples,))
        else:
            labels = torch.randint(0, 2, (num_samples, config['num_classes']))
        return videos, labels
    
    # Create datasets
    train_videos, train_labels = generate_dummy_data(100)
    val_videos, val_labels = generate_dummy_data(20)
    
    train_dataset = TensorDataset(train_videos, train_labels)
    val_dataset = TensorDataset(val_videos, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = VideoMamba(
        num_classes=config['num_classes'],
        classification_mode=config['classification_mode'],
        temporal_mode=config['temporal_mode']
    ).to(config['device'])
    
    # Define optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    if config['classification_mode'] == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Initialize trainer
    trainer = VideoMambaTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config['device'],
        classification_mode=config['classification_mode'],
        scheduler=scheduler,
        early_stopping_patience=10,
        clip_grad_norm=1.0
    )
    
    # Train the model
    training_history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs']
    )
    
    # Save training history
    import json
    with open('training_history.json', 'w') as f:
        json.dump(training_history, f)
    
    print("Training completed!")

if __name__ == "__main__":
    main()