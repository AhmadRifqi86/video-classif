import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
import all_config


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
                 cnn_backbone=all_config.CONF_CNN_BACKBONE, rnn_type=all_config.CONF_RNN_TYPE, 
                 rnn_out=all_config.CONF_RNN_OUT):
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
                              num_layers=all_config.CONF_RNN_LAYER, bidirectional=True, 
                              batch_first=True)
        elif rnn_type == "mamba":
            self.rnn = nn.ModuleList([
            ResidualBlock(rnn_input_size, rnn_input_size*2, hidden_size, hidden_size, bias=True, conv_bias=True, kernel_size=3)
            for _ in range(all_config.CONF_RNN_LAYER)
        ])
        else:  # GRU
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size,
                             num_layers=all_config.CONF_RNN_LAYER, bidirectional=True, 
                             batch_first=True)
        
        # Output layer
        if all_config.CONF_CLASSIF_MODE == "multiclass":
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
        if all_config.CONF_RNN_OUT == "all":
            rnn_out = rnn_out.contiguous().view(batch_size, -1)
        else:
            rnn_out = rnn_out[:, -1, :]
        
        # Final classification
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            out = self.fc(rnn_out)
        else:
            out = torch.cat([fc(rnn_out) for fc in self.fc], dim=1)
        
        return out