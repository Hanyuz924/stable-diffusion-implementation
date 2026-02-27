import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, heads_num, dim_head, embedding_dim, casual_mask = False, norm_eps = 1e5,
                 in_proj_bias = True, out_proj_bias = True, add_pre_norm = True):
        super().__init__()
        if add_pre_norm:
            self.group_norm = nn.GroupNorm(num_groups=32, num_channels=embedding_dim, eps=norm_eps)
        self.add_pre_norm = add_pre_norm
        self.head_num = heads_num
        self.dim_head = dim_head
        self.embedding_dim = embedding_dim
        self.casual_mask = casual_mask
        self.to_q = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_k = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_v = nn.Linear(embedding_dim, embedding_dim, bias=in_proj_bias)
        self.to_out = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=out_proj_bias)
        )
    def forward(self, input, add_residual = True):
        # print(input.shape)
        # print(self.embedding_dim)
        residual = input
        if len(input.shape) == 4:
            B, D, H, W = input.shape
            seq_len = H * W
        else:
            B, seq_len, D = input.shape
        #TODO stupid here, need a new attention block , so leave the qkv logic here only in this attention claas (move shape process output this class)
        if self.add_pre_norm:
            input = self.group_norm(input)
        if len(input.shape) == 4:
            input = input.view(B, D, seq_len).transpose(1,2)        #[B,seq_len, D]
        assert  D == self.embedding_dim, print("input embedding dim does not equal module embedding dim")
        q = self.to_q(input) # [B, seq_len, D]
        k = self.to_k(input) # [B, seq_len, D]
        v = self.to_v(input) # [B, seq_len, D]
        # [B, head_num, seq_len, dim_head]
        q = q.view(B, seq_len, self.head_num, self.dim_head).transpose(1, 2)
        k = k.view(B, seq_len, self.head_num, self.dim_head).transpose(1, 2)
        v = v.view(B, seq_len, self.head_num, self.dim_head).transpose(1, 2)
        scaling_factor =  1 / math.sqrt(self.dim_head)
        q = q * scaling_factor
        qk_weight = q @ k.transpose(-1,-2)          #[B, head_num, seq_len, seq_len]
        if self.casual_mask:
            mask = torch.triu(torch.ones_like(qk_weight, dtype=torch.bool),diagonal=1)
            qk_weight.masked_fill_(mask, -torch.inf)
        # qk_weight = qk_weight * scaling_factor
        attention = F.softmax(qk_weight, dim = -1)
        
        attention = attention @ v  #[B, head_num, seq_len, dim_head]
        attention = attention.transpose(1,2).reshape(B,seq_len,-1)
        if len(residual.shape) == 4:
            out = self.to_out(attention).transpose(1,2).view(B,D,H,W)
        else:
            out = self.to_out(attention)
        if add_residual:
            return out + residual
        else:
            return out

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int, padding:int = 0, kernel_size = 3, bias = True, vae_padding = True):
        super().__init__()
        #in most case in_channels == out_channels, but those 2 values can be different
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = 2
        self.vae_padding = vae_padding
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=self.stride, padding=padding, bias = bias)

    def forward(self, hidden_state:torch.Tensor):
        #mannually add padding at down and right 
        if self.vae_padding:
            hidden_state = F.pad(hidden_state, (0, 1, 0, 1), mode="constant", value=0)
        hidden_state = self.conv(hidden_state)
        return hidden_state

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
    def forward(self, hidden_state:torch.Tensor):
        hidden_state = F.interpolate(hidden_state, scale_factor=2, mode="nearest")
        hidden_state = self.conv(hidden_state)
        return hidden_state

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups: int = 32, norm_eps : float = 1e-5, dropout:float = 0.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.norm_eps = norm_eps
        self.drop_out = dropout

        #pre norm1
        self.norm1 = nn.GroupNorm(self.groups, self.in_channels, eps=self.norm_eps)
        #channel covn
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)

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


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_state = input_tensor

        hidden_state = self.norm1(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.conv1(hidden_state)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.conv2(hidden_state)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        out_tensor = hidden_state + input_tensor
        return out_tensor 
    
class Encoder_Downblock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_res_blocks:int =1, dropout: float = 0.0, add_downblocks: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        resblocks = []

        for i in range(self.num_res_blocks):
            in_chans = self.in_channels if i == 0 else self.out_channels
            resblocks.append(ResBlock(in_chans, self.out_channels, dropout=dropout,  norm_eps=1e-6))
        self.resnets = nn.ModuleList(resblocks)
        if add_downblocks:
            self.downsamplers = nn.ModuleList([
                DownBlock(in_channels=self.out_channels, out_channels=self.out_channels)
            ])
        else:
            self.downsamplers = None
    def forward(self, hidden_states:torch.Tensor):
        for resblock in self.resnets:
            hidden_states = resblock(hidden_states)
        if self.downsamplers is not None:
            for downsample in self.downsamplers:
                hidden_states = downsample(hidden_states)
        return hidden_states
    
class Encoder_Midblock(nn.Module):
    def __init__(self, in_channels: int = 512, attention_head_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResBlock(in_channels, in_channels, dropout=dropout, norm_eps=1e-6)
        ])
        self.attentions = nn.ModuleList([
            Attention(heads_num=1, dim_head=in_channels, embedding_dim=attention_head_dim, norm_eps=1e-6)
        ])
        self.resnets.append(
            ResBlock(in_channels, in_channels, dropout=dropout, norm_eps=1e-6)
        )
    def forward(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x, add_residual = True)
        x = self.resnets[1](x)
        return x

class Decoder_UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_layers:int = 2, add_upsample:bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layer = num_layers
        self.add_upsample = add_upsample
        
        resnet = []
        for i in range(self.num_layer):
            resnet.append(ResBlock(in_channels=self.in_channels, out_channels=self.out_channels, norm_eps=1e-6))
            if in_channels != out_channels:
                self.in_channels = self.out_channels
        self.resnets = nn.ModuleList(resnet)
        if self.add_upsample:
            self.upsamplers = nn.ModuleList([UpBlock(in_channels=self.out_channels, out_channels=self.out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states:torch.Tensor):

        for block in self.resnets:
            hidden_states = block(hidden_states)
        if self.upsamplers is not None:
            for upsample in self.upsamplers:
                hidden_states = upsample(hidden_states)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, in_channels:int =3, out_channels:int=4, down_block_channels:tuple[int, ...] = (64,), layer_per_block:int = 2, dropout:float = 0.0):
        super().__init__()
        self.layer_per_block = layer_per_block
        self.in_channels = in_channels
        #input projection conv project the input channel from RGB 3 to resblock base channel for later group norm use 
        self.conv_in = nn.Conv2d(self.in_channels, down_block_channels[0], kernel_size=3, padding=1, stride=1)
        self.down_block_channels = down_block_channels

        #down sampling blocks
        self.down_blocks = nn.ModuleList([])
        down_block_in_channel = down_block_channels[0]
        for i, channels in enumerate(self.down_block_channels):
            down_block_out_channels = self.down_block_channels[i]
            down_blocks = Encoder_Downblock(in_channels=down_block_in_channel,out_channels=down_block_out_channels,
                                            num_res_blocks=self.layer_per_block, dropout=dropout,
                                            add_downblocks= (i < len(self.down_block_channels) -1 )
                                            )
            self.down_blocks.append(down_blocks)
            down_block_in_channel = down_block_out_channels
        self.mid_block = Encoder_Midblock(in_channels=self.down_block_channels[-1],attention_head_dim=self.down_block_channels[-1])
        self.conv_norm_out = nn.GroupNorm(32,num_channels=self.down_block_channels[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.down_block_channels[-1], 2 * out_channels, kernel_size=3, padding=1)

    def forward(self, x, noise = None):
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x) #[B, C * 2, H, W]

        mean, log_variance = torch.chunk(x, 2, dim=1)
        if not noise:
            noise = torch.randn_like(mean)
        log_variance = torch.clamp(log_variance, -30, 20)
        std = torch.exp(0.5 * log_variance)
        x = mean + std * noise
        x = x * 0.18215

        return x

class Decoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, down_block_channels:tuple[int, ...] = (512,), layer_per_block:int = 2, dropout:float = 0.0):
        super().__init__()
        self.layer_per_block = layer_per_block +1 #TODO official implementation has plus 1 need to figure out the reason ......
        self.in_channels = in_channels
        #input projection conv project the input channel from RGB 3 to resblock base channel for later group norm use 
        self.conv_in = nn.Conv2d(self.in_channels, down_block_channels[0], kernel_size=3, padding=1, stride=1)
        self.down_block_channels = down_block_channels
        
        self.mid_block = Encoder_Midblock(in_channels=self.down_block_channels[0],attention_head_dim=self.down_block_channels[0])
        self.up_blocks = nn.ModuleList([])

        up_pre_channels = down_block_channels[0]
        for i, up_channel in enumerate(down_block_channels):
            up_block = Decoder_UpBlock(up_pre_channels, up_channel, self.layer_per_block, add_upsample= (i !=len(down_block_channels) -1))
            up_pre_channels = up_channel
            self.up_blocks.append(up_block)
        self.conv_norm_out = nn.GroupNorm(32,num_channels=self.down_block_channels[-1],eps = 1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(self.down_block_channels[-1], out_channels, kernel_size=3, padding=1)
        self.register_module("final_act", nn.Tanh())

    def forward(self, x: torch.Tensor):

        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)
        # x = torch.tanh(x)
        
        return x 

















