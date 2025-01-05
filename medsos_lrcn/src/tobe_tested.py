import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
import all_config

class RMSNorm(nn.Module):
    def __init__(self,d_model: int,eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):  #kenapa bidir nya jadi false ya masuk sini?
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class ParallelMamba(nn.Module):
    def __init__(self, d_model, d_inner, n_state, dt_rank, bias=True, conv_bias=True, kernel_size=3, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_state = n_state
        self.dt_rank = dt_rank
        self.bidirectional = bidirectional

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
        self.out_proj = nn.Linear(d_inner * (2 if bidirectional else 1), d_model, bias=bias)

    def parallel_scan(self, u, delta, A, B, C, direction="forward"):
        batch_size, seq_len, d_inner = u.shape
        n_state = A.shape[1]

        if direction == "backward":
            u = torch.flip(u, dims=[1])
            delta = torch.flip(delta, dims=[1])

        deltaA = torch.exp(torch.einsum('b l d, d n -> b l d n', delta, A))
        deltaB_u = torch.einsum('b l d, b l n, b l d -> b l d n', delta, B, u)

        x = torch.zeros((batch_size, d_inner, n_state), device=deltaA.device)
        states = []

        for t in range(seq_len):
            x = deltaA[:, t] * x + deltaB_u[:, t]
            y = torch.einsum('b d n, b n -> b d', x, C[:, t])
            states.append(y)

        states = torch.stack(states, dim=1)

        if direction == "backward":
            states = torch.flip(states, dims=[1])

        return states

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        n_state = self.A_log.shape[1]

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(self.d_inner, dim=-1)

        if self.bidirectional:
            res = torch.cat([res, res], dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)
        x = x[:, :, :seq_len]
        x = rearrange(x, 'b d l -> b l d')

        x = F.silu(x)

        x_proj = self.x_proj(x)
        delta, B, C = x_proj.split([self.dt_rank, n_state, n_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log)

        y_forward = self.parallel_scan(x, delta, A, B, C, direction="forward")

        if self.bidirectional:
            y_backward = self.parallel_scan(x, delta, A, B, C, direction="backward")
            y = torch.cat([y_forward, y_backward], dim=-1)
        else:
            y = y_forward
        y = y * F.silu(res)
        output = self.out_proj(y)

        return output

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_state, dt_rank, bias=True, conv_bias=True, kernel_size=3, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.mixer = ParallelMamba(d_model, 
                                   d_inner, n_state, 
                                   dt_rank, bias=bias, conv_bias=conv_bias, 
                                   kernel_size=kernel_size, bidirectional=bidirectional)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        context = torch.matmul(attention_probs, value)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(context)
        
        return output


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, seq_len, c = x.size()
        # Perform pooling along sequence length
        y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
        # Generate scale
        y = self.fc(y)
        # Reshape to (batch, channel, 1) and expand to (batch, channel, seq_len)
        y = y.view(b, c, 1).expand(b, c, seq_len)
        # Transpose back to original shape and apply scaling
        return x * y.transpose(1, 2)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
import math

class TemporalPyramidPooling(nn.Module):
    """
    Temporal Pyramid Pooling module that captures temporal features at different scales.
    levels: List of pooling levels (e.g., [1, 2, 4] for full, half, and quarter sequence lengths)
    """
    def __init__(self, levels=[1, 2, 4]):
        super().__init__()
        self.levels = levels
        
    def forward(self, x):
        batch, seq_len, channels = x.size()
        pooled = []
        for level in self.levels:
            kernel_size = max(seq_len // level, 1)
            pool = F.adaptive_max_pool1d(
                x.transpose(1, 2), kernel_size
            ).transpose(1, 2)
            pooled.append(pool)
        return torch.cat([x] + pooled, dim=2)

class TemporalFPN(nn.Module):
    """
    Feature Pyramid Network for temporal feature hierarchies.
    Creates multi-scale temporal features through down/up sampling.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = nn.Conv1d(in_channels, out_channels, 3, stride=2, padding=1)
        self.down2 = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose1d(out_channels, out_channels, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(out_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x_orig = x
        x1 = self.down1(x)
        x2 = self.down2(x1)
        up1 = self.up1(x2)
        # Handle potential size mismatch due to padding
        if up1.size() != x1.size():
            up1 = F.interpolate(up1, size=x1.size(-1))
        up1 = up1 + x1
        up2 = self.up2(up1)
        if up2.size() != x_orig.size():
            up2 = F.interpolate(up2, size=x_orig.size(-1))
        return torch.cat([x_orig, up2], dim=1)

class DynamicRouting(nn.Module):
    """
    Dynamic routing between spatial and temporal features using learned gates.
    Allows the model to adaptively combine information from both streams.
    """
    def __init__(self, channels):
        super().__init__()
        self.routing = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.Sigmoid()
        )
        
    def forward(self, spatial, temporal):
        gates = self.routing(spatial)
        return temporal * gates + spatial * (1 - gates)

class SinusoidalPositionEncoding(nn.Module):
    """
    Adds positional information to temporal features using sinusoidal encoding.
    Helps the model understand temporal ordering of frames.
    """
    def __init__(self, channels, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class ImprovedFeatureAdapter(nn.Module):
    """
    Enhanced feature adaptation module with residual connections and configurable components.
    Transforms CNN features for temporal processing while preserving spatial information.
    """
    def __init__(self, input_size, output_size, dropout_rate=0.1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        self.main_path = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LayerNorm(input_size),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )
        
        if input_size != output_size:
            self.proj = nn.Linear(input_size, output_size)
        else:
            self.proj = nn.Identity()
            
        if use_residual:
            self.residual = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
    
    def forward(self, x):
        main = self.proj(self.main_path(x))
        if self.use_residual:
            return main + self.residual(x)
        return main

class LRCN(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size, rnn_input_size,
                 cnn_backbone=all_config.CONF_CNN_BACKBONE,
                 rnn_type=all_config.CONF_RNN_TYPE,
                 rnn_out=all_config.CONF_RNN_OUT,
                 bidirectional=all_config.CONF_BIDIR,
                 use_tpp=all_config.CONF_USE_TPP,
                 use_fpn=all_config.CONF_USE_FPN,
                 use_routing=all_config.CONF_USE_ROUTING,
                 use_pos_encoding=all_config.CONF_USE_POS_ENCODING,
                 mamba_configs=None):
        super(LRCN, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.backbone = cnn_backbone
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Initialize CNN backbone
        self.cnn_backbone = getattr(models, cnn_backbone)(pretrained=True)
        if hasattr(self.cnn_backbone, 'fc'):
            cnn_out_size = self.cnn_backbone.fc.in_features
            self.cnn_backbone.fc = nn.Identity()
        elif hasattr(self.cnn_backbone, 'classifier'):
            if isinstance(self.cnn_backbone.classifier, nn.Sequential):
                cnn_out_size = self.cnn_backbone.classifier[-1].in_features
            else:
                cnn_out_size = self.cnn_backbone.classifier.in_features
            self.cnn_backbone.classifier = nn.Identity()

        # Improved feature adaptation
        adaptation_sizes = [cnn_out_size, cnn_out_size//2, cnn_out_size//4, rnn_input_size]
        self.adaptation_layers = nn.ModuleList([
            ImprovedFeatureAdapter(adaptation_sizes[i], adaptation_sizes[i+1])
            for i in range(len(adaptation_sizes)-1)
        ])

        # Optional components based on config
        if use_tpp:
            self.tpp = TemporalPyramidPooling([1, 2, 4])
            rnn_input_size *= 4  # Adjust for concatenated pooling levels
        
        if use_fpn:
            self.fpn = TemporalFPN(rnn_input_size, rnn_input_size)
            rnn_input_size *= 2  # Adjust for concatenated FPN features
            
        if use_routing:
            self.routing = DynamicRouting(rnn_input_size)
            
        if use_pos_encoding:
            self.pos_encoding = SinusoidalPositionEncoding(rnn_input_size)

        # RNN/Mamba configuration
        if rnn_type == "mamba":
            if mamba_configs is None:
                # Default Mamba configurations - these are carefully chosen defaults
                mamba_configs = {
                    'd_model': rnn_input_size,
                    'd_inner': rnn_input_size * 4,  # Typically 2-4x d_model
                    'n_state': hidden_size,  # Usually similar to d_model
                    'dt_rank': hidden_size,  # Usually similar to n_state
                    'n_layers': all_config.CONF_RNN_LAYER
                }
            
            self.rnn = nn.ModuleList([
                ResidualBlock(
                    mamba_configs['d_model'],
                    mamba_configs['d_inner'],
                    mamba_configs['n_state'],
                    mamba_configs['dt_rank'],
                    bias=True,
                    conv_bias=True,
                    kernel_size=3,
                    bidirectional=bidirectional
                ) for _ in range(mamba_configs['n_layers'])
            ])
            self.rnn_output_size = mamba_configs['d_model']
            self.norm_f = RMSNorm(mamba_configs['d_model'])
        else:  # LSTM/GRU
            rnn_class = nn.LSTM if rnn_type == "lstm" else nn.GRU
            self.rnn = rnn_class(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=all_config.CONF_RNN_LAYER,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=0.3 if all_config.CONF_RNN_LAYER > 1 else 0
            )
            self.rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # Classification head remains largely unchanged
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.Sequential(
                nn.Linear(fc_input_size, fc_input_size//2),
                nn.LayerNorm(fc_input_size//2),
                nn.GELU(),
                nn.Dropout(0.5),
                nn.Linear(fc_input_size//2, fc_input_size//4),
                nn.LayerNorm(fc_input_size//4),
                nn.Linear(fc_input_size//4, num_classes)
            )
        else:
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(fc_input_size),
                    nn.Linear(fc_input_size, 1)
                ) for _ in range(num_classes)
            ])

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN feature extraction
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn_backbone(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Feature adaptation with improved layers
        for adapter in self.adaptation_layers:
            x = adapter(x)
            
        # Apply optional components
        if hasattr(self, 'pos_encoding'):
            x = self.pos_encoding(x)
            
        if hasattr(self, 'tpp'):
            x_tpp = x.transpose(1, 2)
            x = self.tpp(x_tpp).transpose(1, 2)
            
        if hasattr(self, 'fpn'):
            x_fpn = x.transpose(1, 2)
            x = self.fpn(x_fpn).transpose(1, 2)
        
        # Temporal modeling
        if self.rnn_type == "mamba":
            for layer in self.rnn:
                x = layer(x)
            x = self.norm_f(x)
        else:
            x, _ = self.rnn(x)
            
        if hasattr(self, 'routing'):
            x = self.routing(x, x)  # Self-routing between temporal features
            
        # Output handling
        if all_config.CONF_RNN_OUT == "all":
            x = x.contiguous().view(batch_size, -1)
        else:
            x = x[:, -1, :]
            
        # Classification
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            out = self.fc(x)
        else:
            out = torch.cat([fc(x) for fc in self.fc], dim=1)
            
        return out

# Add to your configuration file:
# """
# CONF_USE_TPP = False  # Use Temporal Pyramid Pooling
# CONF_USE_FPN = False  # Use Feature Pyramid Network
# CONF_USE_ROUTING = False  # Use Dynamic Routing
# CONF_USE_POS_ENCODING = True  # Use Positional Encoding

# # Mamba configurations
# CONF_MAMBA_CONFIGS = {
#     'd_model': None,  # Will be set based on rnn_input_size
#     'd_inner': None,  # Will be set to 4 * d_model
#     'n_state': None,  # Will be set based on hidden_size
#     'dt_rank': None,  # Will be set based on hidden_size
#     'n_layers': CONF_RNN_LAYER
# }
# """

