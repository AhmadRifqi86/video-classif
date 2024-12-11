import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
import all_config


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
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


class Adapt(nn.Module):
    def __init__(self, in_size, out_size, mode, drop=all_config.CONF_DROPOUT, depth=3):
        super().__init__()
        self.input_size = in_size
        self.output_size = out_size
        self.depth = depth
        
        # Generate size_list based on input_size, depth, and output_size
        size_list = [in_size]
        for i in range(1, depth):
            size_list.append(size_list[-1] // 2)
        size_list.append(out_size)
        
        layer_list = []
        i = 0  # Index for `size_list`

        for char in mode:
            if char == 'l':
                layer_list.append(nn.Linear(size_list[i], size_list[i+1]))
                i += 1  # Move to the next pair in `size_list`
            elif char == 'n':
                layer_list.append(nn.LayerNorm(size_list[i]))  # Use current size
            elif char == 's':
                layer_list.append(nn.SiLU())
            elif char == 'g':
                layer_list.append(nn.GELU())
            elif char == 'r':
                layer_list.append(nn.ReLU())
            elif char == 'd' and i == depth:
                layer_list.append(nn.Dropout(drop))
            else:
                raise ValueError(f"Undefined layer type: {char}")
        
        self.adapt = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.adapt(x)


class LRCN(nn.Module):
    def __init__(self, num_classes, sequence_length, hidden_size, rnn_input_size, 
                 cnn_backbone=all_config.CONF_CNN_BACKBONE, 
                 rnn_type=all_config.CONF_RNN_TYPE, rnn_out=all_config.CONF_RNN_OUT, 
                 bidirectional=all_config.CONF_BIDIR):
        super(LRCN, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.backbone = cnn_backbone
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

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
        
        self.adapt = Adapt(cnn_out_size,rnn_input_size,mode=all_config.CONF_ADAPT,drop=all_config.CONF_DROPOUT)
 

        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size,
                               num_layers=all_config.CONF_RNN_LAYER, bidirectional=bidirectional, 
                               batch_first=True)
            self.rnn_output_size = hidden_size * (2 if bidirectional else 1)
        elif rnn_type == "mamba":
            self.rnn = nn.ModuleList([
                ResidualBlock(rnn_input_size, rnn_input_size * 2, hidden_size, hidden_size, bidirectional=bidirectional)
                for _ in range(all_config.CONF_RNN_LAYER)
            ])
            self.rnn_output_size = rnn_input_size #* (2 if bidirectional else 1)
        else:
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size,
                              num_layers=all_config.CONF_RNN_LAYER, bidirectional=bidirectional, 
                              batch_first=True)
            self.rnn_output_size = hidden_size * (2 if bidirectional else 1)

        if all_config.CONF_CLASSIF_MODE == "multiclass":
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.Linear(fc_input_size, fc_input_size // 2)
            self.fca = nn.Linear(fc_input_size // 2, fc_input_size // 4)
            self.fcb = nn.Linear(fc_input_size // 4, num_classes)
            self.bn0 = nn.LayerNorm(fc_input_size)
            self.bna = nn.LayerNorm(fc_input_size // 2)
            self.bnb = nn.LayerNorm(fc_input_size // 4)
            self.drop2 = nn.Dropout(all_config.CONF_DROPOUT)
            # print("fc in size: ",fc_input_size)
        else:
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.ModuleList([nn.Linear(fc_input_size, 1) for _ in range(num_classes)])

    def forward(self, x):
        #print("running models bidir")
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn_backbone(x)
        x = x.view(batch_size, seq_len, -1)
        x = self.adapt(x)

        if self.rnn_type == "mamba":
            for layer in self.rnn:
                x = layer(x)
            rnn_out = x
        else:
            rnn_out, _ = self.rnn(x)
        
        if all_config.CONF_RNN_OUT == "all":
            rnn_out = rnn_out.contiguous().view(batch_size, -1)
        else:
            rnn_out = rnn_out[:, -1, :]

        #print("rnn out size: ",rnn_out.size())
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            out = self.bn0(rnn_out)  #kalo masih bau coba pake batchNorm
            out = F.silu(self.bna(self.fc(out)))
            out = F.silu(self.bnb(self.fca(out)))
            # out = self.drop2(out)
            out = self.fcb(out)
        else:
            out = torch.cat([fc(rnn_out) for fc in self.fc], dim=1)

        return out
