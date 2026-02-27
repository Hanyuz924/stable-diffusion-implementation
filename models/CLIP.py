import torch
from torch import nn
from torch.nn import functional as F
from VAE import Attention

class CLIPEmbedding(nn.Module):
    def __init__(self, voca_size:int, hidden_dim:int, max_token:int):
        super().__init__()
        self.voca_size = voca_size
        self.hidden_dim = hidden_dim
        self.max_token = max_token
        self.token_embedding = nn.Embedding(self.voca_size, self.hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros((self.max_token, self.hidden_dim)))
    def forward(self, tokens:torch.Tensor):
        #[B, L] -> [B, L, hidden_dim]
        tokens = self.token_embedding(tokens)

        tokens += self.position_embedding
        return tokens
    
class CLIPLayer(nn.Module):
    def __init__(self, head_num:int, hidden_dim:int):
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.layernorm_1 = nn.LayerNorm(self.hidden_dim)
        self.attention = Attention(heads_num= self.head_num, dim_head=  hidden_dim // head_num,
                                   embedding_dim=hidden_dim, casual_mask=True)
        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(self.hidden_dim)
        # Feedforward layer
        self.linear_1 = nn.Linear(self.hidden_dim, 4 * self.hidden_dim)
        self.linear_2 = nn.Linear(4 * self.hidden_dim, self.hidden_dim)

    def forward(self, hidden_state:torch.Tensor):
        residual = hidden_state

        hidden_state = self.layernorm_1(hidden_state)
        #add casual_mask 
        hidden_state = self.attention(hidden_state)
         
        hidden_state = hidden_state + residual

        residual = hidden_state
        hidden_state = self.layernorm_2(hidden_state)
        hidden_state = self.linear_1(hidden_state)

        #using QuickGELU
        hidden_state = hidden_state * torch.sigmoid(1.702 * hidden_state)
        hidden_state = self.linear_2(hidden_state)
        hidden_state = hidden_state + residual
        return hidden_state
    
class CLIP(nn.Module):
    def __init__(self, voca_size: int = 49408, hidden_dim:int = 768, max_token_len:int = 77, attention_head_num:int = 12, num_clip_layers: int = 12):
        self.num_clip_layers = num_clip_layers
        self.voca_size = voca_size
        self.hidden_dim = hidden_dim
        self.max_token_len = max_token_len
        self.attention_head_num =attention_head_num
        self.embedding = CLIPEmbedding(self.voca_size, self.hidden_dim, self.max_token_len)
        self.layers = nn.ModuleList([CLIPLayer(self.attention_head_num, self.hidden_dim) for _ in range(self.num_clip_layers)])
        self.layernorm = nn.LayerNorm(self.hidden_dim)

    def forward(self, hidden_state:torch.Tensor):
        hidden_state = self.embedding(hidden_state)
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        hidden_state = self.layernorm(hidden_state)
        return hidden_state