import torch
from torch import nn
from torch.nn import functional as F
import math
from VAE import Attention
from VAE import DownBlock
from VAE import UpBlock



"""

this file is for implementation of non hugging face style code ......

In order to use the pretrained weights from hugging face , I have to write code in hugging face style 
which  I didn't originally, so I copy my original implementation to this file 

"""

class TimestepEmbedSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, time_embedding_channels:int =1280,
                 groups: int = 32, norm_eps : float = 1e-5, dropout:float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_channels = time_embedding_channels
        self.groups = groups
        self.norm_eps = norm_eps
        self.drop_out = dropout

        #pre norm1
        self.norm1 = nn.GroupNorm(self.groups, self.in_channels, eps=self.norm_eps)
        #channel covn
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)

        self.time_emb_proj = nn.Linear(self.time_embedding_channels, self.out_channels)

        self.norm2 = nn.GroupNorm(self.groups, self.out_channels, eps= self.norm_eps)

        self.dropout = nn.Dropout(self.drop_out)

        #conv2
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()
        #conv used for residual connection OR IDENTITY !!
        self.use_in_shortcut = (self.in_channels != self.out_channels)
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    
    def forward(self, hidden_state:torch.Tensor, time_embedding:torch.Tensor):
        residual = hidden_state
        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.conv1(hidden_state)

        time_embedding = self.act(time_embedding)
        time_embedding = self.time_emb_proj(time_embedding)

        hidden_state = hidden_state + time_embedding
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.conv2(hidden_state)

        if self.conv_shortcut is not None:
            hidden_state = hidden_state + self.conv_shortcut(residual)
        return hidden_state


class AttentionBlock(nn.Module):
    def __init__(self, head_num:int, dim_head:int, embedding_dim:int, text_dim:int, linear_out_channels:int):
        super().__init__()
        #        channels = head_num * dim_head
        self.head_num = head_num
        self.dim_head = dim_head
        self.embedding_dim = embedding_dim
        self.linear_out_channels = linear_out_channels
        self.groupnorm = nn.GroupNorm(32, embedding_dim, eps=1e-6)
        self.conv_in = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, padding=0)
        self.norm_1 = nn.LayerNorm(embedding_dim)
        self.attention_1 = Attention(self.head_num, self.dim_head, self.embedding_dim, in_proj_bias=False)
        self.norm_2 = nn.LayerNorm(self.embedding_dim)
        self.attention_2 = CrossAttention(self.head_num, self.dim_head, self.embedding_dim,text_dim,in_proj_bias=False)
        self.norm_3 = nn.LayerNorm(self.embedding_dim)
        #using GEGLU so half of the projection is for gating 
        self.linear_1 = nn.Linear(embedding_dim, self.linear_out_channels * 2)
        self.linear_2 = nn.Linear(self.linear_out_channels, embedding_dim)
        self.conv_out = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1, padding=0)
    
    def forward(self, hidden_image:torch.Tensor, hidden_text:torch.Tensor):
        B, C, H, W =hidden_image.shape
        residual_1 = hidden_image
        hidden_image = self.groupnorm(hidden_image)
        hidden_image = self.conv_in(hidden_image)
        hidden_image = self.norm_1(hidden_image)
        hidden_image = hidden_image.view(B, H, W, C).reshape(B, -1, C)
        hidden_image = self.attention_1(hidden_image)
        residual_2 = hidden_image
        hidden_image = self.norm_2(hidden_image)
        hidden_image = self.attention_2(hidden_image, hidden_text)
        hidden_image = hidden_image + residual_2
        residual_2 = hidden_image
        hidden_image = self.norm_3(hidden_image)
        hidden_image, gate = self.linear_1(hidden_image).chunk(2, dim = -1)
        hidden_image = hidden_image * F.gelu(gate)
        hidden_image = self.linear_2(hidden_image)
        hidden_image = hidden_image + residual_2
        hidden_image = hidden_image.transpose(-1,-2).view(B, C, H, W)
        hidden_image = self.conv_out(hidden_image)
        return hidden_image + residual_1
    
class CrossAttention(nn.Module):
    def __init__(self, head_nums:int, dim_head:int, embedding_dim:int, cross_dim: int,
                 in_proj_bias = True, out_proj_bias = True):
        super().__init__()
        self.head_num = head_nums
        self.dim_head = dim_head
        self.embedding_dim = embedding_dim
        self.to_q   = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_k   = nn.Linear(cross_dim, embedding_dim, bias=in_proj_bias)
        self.to_v   = nn.Linear(cross_dim, embedding_dim, bias=in_proj_bias)
        self.to_out = nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim, bias=out_proj_bias),
                    nn.Dropout(0.0)
        )
    
    def forward(self, hidden_image:torch.Tensor, hidden_text:torch.Tensor):
        B, T, D = hidden_image.shape
        T_text = hidden_text.shape[1]
        assert D == self.embedding_dim, print("input embedding dim does not equal module embedding dim")
        q = self.to_q(hidden_image)
        k = self.to_k(hidden_text)
        v = self.to_v(hidden_text)
        q = q.view(B, T, self.head_num, self.dim_head).transpose(1, 2)
        k = k.view(B, T_text, self.head_num, self.dim_head).transpose(1, 2)
        v = v.view(B, T_text, self.head_num, self.dim_head).transpose(1, 2)
        scaling_factor =  1 / math.sqrt(self.dim_head)
        qk_weights = q @ k.transpose(-1, -2)
        qk_weights = qk_weights * scaling_factor
        qk_weights = F.softmax(qk_weights, dim=-1)
        attention = qk_weights @ v
        attention = attention.transpose(1,2).reshape(B, T, -1).contiguous()
        out_put = self.to_out(attention)
        return out_put
    

    # self.mid_block = TimestepEmbedSequential(
    #     ResidualBlock(in_channels=output_channels, out_channels= output_channels, time_embedding_channels=time_embedding_dim),
    #     AttentionBlock(head_num=attention_head_num, dim_head= output_channels // attention_head_num, embedding_dim=output_channels,
    #                    text_dim=text_embedding_dim, linear_out_channels= 4 * output_channels ),
    #     ResidualBlock(in_channels=output_channels, out_channels= output_channels, time_embedding_channels=time_embedding_dim),
    # )

    # self.up_blocks = []
    # reversed_block_out_channels = block_out_channels[::-1]
    # output_channels = reversed_block_out_channels[0]
    # for index ,current_out_channels in enumerate(reversed_block_out_channels):
    #     for _ in range(layer_per_block):
    #         #skip connection from encoder side
    #         skip_channel = skip_connection_channels.pop()
    #         if index != 0:
    #             self.up_blocks.append(TimestepEmbedSequential(
    #                                         ResidualBlock(in_channels=skip_channel + output_channels, out_channels=current_out_channels,
    #                                                         time_embedding_channels=time_embedding_dim),
    #                                         AttentionBlock(attention_head_num, dim_head= current_out_channels // attention_head_num,
    #                                                         embedding_dim= current_out_channels, text_dim=text_embedding_dim,linear_out_channels= 4* current_out_channels)
    #                                     )
    #                                 )
    #         else:
    #             self.up_blocks.append(TimestepEmbedSequential(
    #                                         ResidualBlock(in_channels=skip_channel + output_channels, out_channels=current_out_channels,
    #                                                             time_embedding_channels=time_embedding_dim),
    #                                     )
    #                                 )
    #         #update the in channels for next layer !!
    #         if output_channels != current_out_channels: 
    #             output_channels = current_out_channels    
    #     if index != len(reversed_block_out_channels) -1 :
    #         #skip connection from encoder side
    #         skip_channel = skip_connection_channels.pop()
    #         if index != 0:
    #             self.up_blocks.append(TimestepEmbedSequential(
    #                                         ResidualBlock(in_channels=skip_channel + output_channels, out_channels=current_out_channels,
    #                                                         time_embedding_channels=time_embedding_dim),
    #                                         AttentionBlock(attention_head_num, dim_head= current_out_channels // attention_head_num,
    #                                                         embedding_dim= current_out_channels, text_dim=text_embedding_dim,linear_out_channels= 4* current_out_channels),
    #                                         UpBlock(output_channels, output_channels)
    #                                     )
    #                                 )
    #         else:
    #             self.up_blocks.append(TimestepEmbedSequential(
    #                                         ResidualBlock(in_channels=skip_channel + output_channels, out_channels=current_out_channels,
    #                                                             time_embedding_channels=time_embedding_dim),
    #                                         UpBlock(output_channels, output_channels)
    #                                     )
    #                                 )
                
    # #skip connection for conv in 
    # skip_channel = skip_connection_channels.pop()
    # self.up_blocks.append(TimestepEmbedSequential(
    #                             ResidualBlock(in_channels=skip_channel + output_channels, out_channels=current_out_channels,
    #                                           time_embedding_channels=time_embedding_dim),
    #                             AttentionBlock(attention_head_num, dim_head= current_out_channels // attention_head_num,
    #                                            embedding_dim= current_out_channels, text_dim=text_embedding_dim,linear_out_channels= 4* current_out_channels)
    #                                     )
    #                     )
    # self.up_blocks = nn.ModuleList(self.up_blocks)
    # assert len(skip_connection_channels) == 0, "All skip connection channels should be used"