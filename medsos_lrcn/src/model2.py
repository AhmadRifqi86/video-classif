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



# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, num_heads):
#         super().__init__()
#         assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
        
#         self.query = nn.Linear(d_model, d_model)
#         self.key = nn.Linear(d_model, d_model)
#         self.value = nn.Linear(d_model, d_model)
        
#         self.out_proj = nn.Linear(d_model, d_model)
        
#     def forward(self, x):
#         batch_size, seq_len, d_model = x.size()
        
#         # Linear projections
#         query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
#         key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
#         value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
#         # Transpose for attention computation
#         query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
#         key = key.transpose(1, 2)
#         value = value.transpose(1, 2)
        
#         # Compute attention scores
#         attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attention_probs = F.softmax(attention_scores, dim=-1)
        
#         # Apply attention
#         context = torch.matmul(attention_probs, value)
        
#         # Reshape and project
#         context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
#         output = self.out_proj(context)
        
#         return output


# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=8):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, seq_len, c = x.size()
#         # Perform pooling along sequence length
#         y = self.avg_pool(x.transpose(1, 2)).squeeze(-1)
#         # Generate scale
#         y = self.fc(y)
#         # Reshape to (batch, channel, 1) and expand to (batch, channel, seq_len)
#         y = y.view(b, c, 1).expand(b, c, seq_len)
#         # Transpose back to original shape and apply scaling
#         return x * y.transpose(1, 2)

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
        print("running models bidir")
        print("LRCN bidir: ",self.bidirectional)
        print("LRCN adapt dropout: ",all_config.CONF_DROPOUT)
        print("LRCN RNN Hidden size: ", hidden_size)

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

        # Improved feature adaptation with normalization, activation, and SE Block, original Linear->norm->silu
        self.adapt1 = nn.Sequential(
            # nn.Linear(cnn_out_size,rnn_input_size),
            # nn.LayerNorm(rnn_input_size),
            # nn.SiLU(),
            # nn.Dropout(p=all_config.CONF_DROPOUT)
            nn.Linear(cnn_out_size, cnn_out_size//2),
            nn.LayerNorm(cnn_out_size//2),
            nn.SiLU(),
            nn.Dropout(p=all_config.CONF_DROPOUT)
            #SEBlock(cnn_out_size//2)
        )
        self.adapt2 = nn.Sequential( # original Linear->norm->silu
            nn.Linear(cnn_out_size//2, cnn_out_size//4),
            nn.LayerNorm(cnn_out_size//4),
            nn.SiLU(),
            nn.Dropout(p=all_config.CONF_DROPOUT)
            #SEBlock(cnn_out_size//4)
        )
        self.adapt3 = nn.Sequential( #original Linear->norm->silu->dropout
            nn.Linear(cnn_out_size//4, rnn_input_size),  #nn.Linear(cnn_out_size//4, rnn_input_size),
            nn.LayerNorm(rnn_input_size), #nn.LayerNorm(rnn_input_size), 
            nn.SiLU(),
            nn.Dropout(p=all_config.CONF_DROPOUT),
            #SEBlock(rnn_input_size)
        )
        # self.adapt4 = nn.Sequential( #original Linear->norm->silu->dropout
        #     nn.Linear(cnn_out_size//8, rnn_input_size),
        #     nn.LayerNorm(rnn_input_size),
        #     nn.SiLU(),
        #     nn.Dropout(p=all_config.CONF_DROPOUT),
        # )
        #self.res_proj = nn.Linear(cnn_out_size,rnn_input_size)
        #self.adapt = nn.Linear(cnn_out_size,rnn_input_size)
        # # Gradual unfreezing of CNN backbone
        # for name, param in self.cnn_backbone.named_parameters():
        #     if 'layer4' in name or 'fc' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        for param in self.cnn_backbone.parameters():
            param.requires_grad = False

        # RNN Layer with enhanced configuration
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size,
                              num_layers=all_config.CONF_RNN_LAYER, bidirectional=bidirectional, 
                              batch_first=True)
            self.rnn_output_size = hidden_size * (2 if bidirectional else 1)
            
            # Add SE Block after RNN
            #self.rnn_se = SEBlock(self.rnn_output_size)
        elif rnn_type == "mamba":
            # Kept original Mamba implementation
            self.rnn = nn.ModuleList([ #to be tested: rnn_input_size, rnn_input_size*4, hidden_size, hidden_size,
                ResidualBlock(rnn_input_size, rnn_input_size*4, hidden_size, hidden_size,  # original: rnn_input_size, rnn_input_size*2, hidden_size, hidden_size//2,
                            bias=True, conv_bias=True, kernel_size=3, bidirectional=bidirectional)
                for _ in range(all_config.CONF_RNN_LAYER)
            ])
            self.rnn_output_size = rnn_input_size  #* (2 if bidirectional else 1)
            self.norm_f = RMSNorm(rnn_input_size)
            
            # Add SE Block after Mamba layers
            #self.rnn_se = SEBlock(rnn_input_size)
        elif rnn_type == "transformer":
            print("hidden_size: ",hidden_size)
            print("num_head: ",all_config.CONF_NUM_HEAD)
            self.rnn = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=rnn_input_size,
                    nhead=all_config.CONF_NUM_HEAD,
                    dim_feedforward=hidden_size,
                    dropout=all_config.CONF_DROPOUT,
                    activation="gelu",
                    batch_first=True,
                ) for _ in range(all_config.CONF_RNN_LAYER)  # Creates unique layers
            ])
            # transformer_layer = nn.TransformerEncoderLayer(
            #     d_model=rnn_input_size,  # Match input size to transformer model
            #     nhead=all_config.CONF_NUM_HEAD,  # Number of attention heads (adjust as needed)
            #     dim_feedforward=hidden_size,  # Feedforward expansion
            #     dropout=all_config.CONF_DROPOUT,
            #     activation="gelu",
            #     batch_first=True,  # Ensure batch is first dimension
            # )
            # self.rnn = nn.TransformerEncoder(transformer_layer, num_layers=all_config.CONF_RNN_LAYER)
            self.rnn_output_size = rnn_input_size  # Transformer output matches input size

        else:  # GRU
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size,
                             num_layers=all_config.CONF_RNN_LAYER, bidirectional=bidirectional, 
                             batch_first=True, dropout=0.3)
            self.rnn_output_size = hidden_size * (2 if bidirectional else 1)
            
            # Add SE Block after RNN
            #self.rnn_se = SEBlock(self.rnn_output_size)
        

        # Improved output layer with normalizationkingdom of predators
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            #print("fc_input_size: ",fc_input_size)  #480
            self.fc = nn.Sequential(
                #nn.LayerNorm(fc_input_size),
                nn.Linear(fc_input_size, fc_input_size//2),
                nn.LayerNorm(fc_input_size//2),
                nn.SiLU(),
                nn.Dropout(0.5),
                nn.Linear(fc_input_size//2, fc_input_size//4),
                nn.LayerNorm(fc_input_size//4),
                #nn.SiLU(),  #nambah silu disini jelek hasilnya
                #nn.Dropout(0.5),
                nn.Linear(fc_input_size//4,num_classes)
            )
        else:
            fc_input_size = self.rnn_output_size * (sequence_length if rnn_out == "all" else 1)
            self.fc = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(fc_input_size),
                    #nn.Dropout(0.5),
                    nn.Linear(fc_input_size, 1)
                ) for _ in range(num_classes)
            ])

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        
        # Process through CNN
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn_backbone(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Enhanced feature adaptation with normalization, dropout, and SE Block, 
        x = self.adapt3(self.adapt2(self.adapt1(x)))  #+ self.res_proj(x)
        #x = self.adapt(x)
        #x = self.adapt3(self.adapt2(self.adapt1(x))) + self.res_proj(x)
        #print("adapt out size: ",x.size())
        # Process through RNN
        if self.rnn_type == "mamba":
            for layer in self.rnn:
                x = layer(x)
                #x = x + x * self.self_attention(x)   #nambah attention disini
            rnn_out = self.norm_f(x)
        elif self.rnn_type == "transformer":
            rnn_out = self.rnn(x)
        else:
            rnn_out, _ = self.rnn(x)
        # Add self-attention to enrich RNN output
        # attention_out = self.self_attention(rnn_out)
        
        # # # # Combine RNN and attention outputs (you can modify this combination strategy)
        # rnn_out = rnn_out + rnn_out*attention_out
        #print("before reshaping: ",rnn_out.size())
        # Handle different output modes
        if all_config.CONF_RNN_OUT == "all":
            rnn_out = rnn_out.contiguous().view(batch_size, -1)
        else:  # last
            rnn_out = rnn_out[:, -1, :]
        
        # Final classification
        if all_config.CONF_CLASSIF_MODE == "multiclass":
            #print("rnn_out size:",rnn_out.size()) #(batch_size x 480)
            out = self.fc(rnn_out)
        else:
            out = torch.cat([fc(rnn_out) for fc in self.fc], dim=1)
        
        return out
    


