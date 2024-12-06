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
        self.mixer = ParallelMamba(d_model, d_inner, n_state, dt_rank, bias=bias, conv_bias=conv_bias, kernel_size=kernel_size, bidirectional=bidirectional)
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
            if isinstance(self.cnn_backbone.classifier, nn.Sequential):
                cnn_out_size = self.cnn_backbone.classifier[-1].in_features
            else:
                cnn_out_size = self.cnn_backbone.classifier.in_features
            self.cnn_backbone.classifier = nn.Identity()
        
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

        # Adaptation layer
        # RNN layer
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=cnn_out_size, hidden_size=hidden_size,
                              num_layers=all_config.CONF_RNN_LAYER, bidirectional=True, 
                              batch_first=True)
            self.rnn_output_size = hidden_size * 2  # bidirectional makes it *2
        elif rnn_type == "mamba":
            self.rnn = nn.ModuleList([
                ResidualBlock(cnn_out_size, rnn_input_size*2, hidden_size, hidden_size, 
                            bias=True, conv_bias=True, kernel_size=3)
                for _ in range(all_config.CONF_RNN_LAYER)
            ])
            self.rnn_output_size = rnn_input_size  # Mamba maintains input dimension
            self.norm_f = RMSNorm(rnn_input_size)
        else:  # GRU
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size,
                             num_layers=all_config.CONF_RNN_LAYER, bidirectional=True, 
                             batch_first=True)
            self.rnn_output_size = hidden_size * 2  # bidirectional makes it *2

        # Output layer
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.Linear(fc_input_size,num_classes)
            self.fca = nn.Linear(fc_input_size, fc_input_size//2)
            self.fcb = nn.Linear(fc_input_size//2,fc_input_size//4)
            self.final= nn.Linear(fc_input_size//4,num_classes)
            self.bna = nn.LayerNorm(fc_input_size//2)
            self.bnb = nn.LayerNorm(fc_input_size//4)
        else:
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.ModuleList([
                nn.Linear(fc_input_size, 1) for _ in range(num_classes)
            ])

    def forward(self, x):
        print("running models_back")
        batch_size, seq_len, c, h, w = x.size()
        
        # Process through CNN
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn_backbone(x)
        x = x.view(batch_size, seq_len, -1)
        # Process through RNN
        if self.rnn_type == "mamba":
            for layer in self.rnn:
                x = layer(x)
            rnn_out = self.norm_f(x)
        else:
            rnn_out, _ = self.rnn(x)
        
        # Handle different output modes
        if all_config.CONF_RNN_OUT == "all":
            rnn_out = rnn_out.contiguous().view(batch_size, -1)
        else:  # last
            rnn_out = rnn_out[:, -1, :]
        
        # Final classification
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            #out = self.fc(rnn_out)
            out = self.bna(self.fca(rnn_out))
            out = self.bnb(self.fcb(out))
            out = self.final(out)
        else:
            out = torch.cat([fc(rnn_out) for fc in self.fc], dim=1)
        
        return out