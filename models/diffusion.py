import torch
from torch import nn
from torch.nn import functional as F
import math
from models.VAE import Attention
from models.VAE  import DownBlock
from models.VAE  import UpBlock


class TimeEmbedding(nn.Module):
    def __init__(self, time_step_dim : int, time_embedding_dim : int,  max_period : int = 10000) -> None:
        super().__init__()
        self.time_step_dim = time_step_dim
        self.max_period  = max_period
        self.linear_1 = nn.Linear(time_step_dim, time_embedding_dim)
        self.linear_2 = nn.Linear(time_embedding_dim, time_embedding_dim)
    
    def forward(self, time_step:torch.Tensor):
        time_embedding = self.get_time_embedding(time_step)
        time_embedding = self.linear_1(time_embedding)
        time_embedding = F.silu(time_embedding)
        time_embedding = self.linear_2(time_embedding)
        return time_embedding

    def get_time_embedding(self, time_step:torch.Tensor):
        assert len(time_step.shape) == 1, "timestep should be 1 D tensor"
        time_step = time_step.float()
        half_dim = self.time_step_dim // 2 
        args = -math.log(self.max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=time_step.device) 
        args = args / half_dim
        freqs = torch.exp(args)
        # [B, half_dim]
        time_embedding = time_step[:, None] * freqs[None, :] 
        time_embedding = torch.cat([torch.cos(time_embedding), torch.sin(time_embedding)], dim=-1)
        return time_embedding

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
        q = q * scaling_factor
        qk_weights = q @ k.transpose(-1, -2)
        qk_weights = F.softmax(qk_weights, dim=-1)
        attention = qk_weights @ v
        attention = attention.transpose(1,2).reshape(B, T, -1).contiguous()
        out_put = self.to_out(attention)
        return out_put

class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        prev_output_channels: int, 
        res_skip_channels_list,
        out_channels: int, 
        temb_channels: int, 
        num_layers: int = 3,
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        add_upsample: bool = True,
        add_attention: bool =True
    ):
        super().__init__()
        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        self.add_attention = add_attention
        self.add_upsample = add_upsample
        
        for i in range(num_layers):
            res_skip_channels = res_skip_channels_list[i]
            res_in = prev_output_channels if i == 0 else out_channels
            self.resnets.append(
                ResidualBlock(
                    in_channels=res_in + res_skip_channels, 
                    out_channels=out_channels, 
                    time_embedding_channels=temb_channels
                )
            )
            if add_attention:
                self.attentions.append(
                    SpatialTransformer(
                        out_channels, 
                        num_attention_heads, 
                        out_channels // num_attention_heads, 
                        cross_attention_dim
                    )
                )
        self.upsamplers = nn.ModuleList([])
        if add_upsample:
            self.upsamplers.append(UpBlock(out_channels,out_channels))

    def forward(self, hidden_states, text_hidden_state, temb, res_hidden_states_skip):
        for i in range(len(self.resnets)):
            #can not use pop here
            # res_hidden_states = res_hidden_states_tuple.pop()
            res_hidden_states = res_hidden_states_skip[i]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = self.resnets[i](hidden_states, temb)
            if self.add_attention:
                hidden_states = self.attentions[i](hidden_states, text_hidden_state)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
            
        return hidden_states

class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        temb_channels: int, 
        num_layers: int = 2, 
        num_attention_heads: int = 8,
        cross_attention_dim: int = 768,
        add_downsample: bool = True,
        add_attention : bool = True
    ):
        super().__init__()
        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])
        self.add_attention = add_attention
        self.add_downsample = add_attention
        for i in range(num_layers):
            res_in = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResidualBlock(
                    in_channels=res_in, 
                    out_channels=out_channels, 
                    time_embedding_channels=temb_channels
                )
            )
            if add_attention:
                self.attentions.append(
                    SpatialTransformer(
                        channels=out_channels,
                        head_num=num_attention_heads,
                        dim_head=out_channels // num_attention_heads,
                        context_dim=cross_attention_dim,
                    )
                )
        self.downsamplers = nn.ModuleList([])
        if add_downsample:
            self.downsamplers.append(
                DownBlock(out_channels, out_channels, kernel_size=3, padding=1, vae_padding=False)
            )

    def forward(self, hidden_states, text_hidden_states, temb):
        output_states = []
        for i in range(len(self.resnets)):
            hidden_states = self.resnets[i](hidden_states, temb)
            if self.add_attention:
                hidden_states = self.attentions[i](hidden_states, text_hidden_states)
            output_states.append(hidden_states)
            
        for downsampler in self.downsamplers:
            hidden_states = downsampler(hidden_states)
            output_states.append(hidden_states)
            
        return hidden_states, output_states


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
        # print(hidden_state.shape)
        # print(time_embedding.shape)
        hidden_state = hidden_state + time_embedding.unsqueeze(-1).unsqueeze(-1)
        hidden_state = self.norm2(hidden_state)
        hidden_state = self.act(hidden_state)
        hidden_state = self.dropout(hidden_state)
        hidden_state = self.conv2(hidden_state)

        if self.conv_shortcut is not None:
            hidden_state = hidden_state + self.conv_shortcut(residual)
        else:
            hidden_state = hidden_state + residual
        return hidden_state
    
class SpatialTransformer(nn.Module):
    def __init__(self, channels, head_num, dim_head, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-5)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(channels, head_num, dim_head, context_dim)
        ])
        
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, context):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        for block in self.transformer_blocks:
            x = block(x, context)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return self.proj_out(x) + residual

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, head_num, dim_head, context_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(head_num, dim_head, dim,in_proj_bias =False, add_pre_norm=False) 
        
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(head_num, dim_head, dim, context_dim, in_proj_bias=False)
        
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim) 

    def forward(self, x, context):
        x = self.attn1(self.norm1(x), add_residual = False) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate, approximate="tanh")

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.ModuleList([
            GEGLU(dim, dim * 4),     
            nn.Dropout(dropout),      
            nn.Linear(dim * 4, dim)    
        ])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class UNetMidBlock2D(nn.Module):
    def __init__(self, in_channels, temb_channels, num_layers=1, context_dim=768):
        super().__init__()
        self.resnets = nn.ModuleList([
            ResidualBlock(in_channels, in_channels, temb_channels)
        ])
        
        self.attentions = nn.ModuleList([
            SpatialTransformer(in_channels, head_num=8, dim_head=in_channels//8, context_dim=context_dim)
        ])
        for _ in range(num_layers):
            self.resnets.append(
                ResidualBlock(in_channels, in_channels, temb_channels)
            )

    def forward(self, x, context,temb):
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        x = self.resnets[1](x, temb)
        return x

class UNet(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 time_embedding_dim:int,
                 attention_head_num:int,
                 layer_per_block:int,
                 text_embedding_dim :int, 
                 block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
                 norm_num_groups: int = 32,
                ):
        super().__init__()
        #in conv project from in channels to model base channel
        self.conv_in = nn.Conv2d(in_channels=in_channels,out_channels=block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = []
        output_channels = block_out_channels[0]
        self.in_channels = in_channels
        #Better to Expand the loop :(  
        # need to make res block and attention block in one layer 
        skip_connection_channels = [output_channels]

        for index, current_out_channels in enumerate(block_out_channels):
            self.down_blocks.append(CrossAttnDownBlock2D(output_channels, current_out_channels, time_embedding_dim, 
                                                         num_layers=layer_per_block,
                                                         add_downsample=index != len(block_out_channels) -1, 
                                                         add_attention=index != len(block_out_channels) -1)
                                    )
            skip_connection_channels.append(current_out_channels)
            skip_connection_channels.append(current_out_channels)
            if index != len(block_out_channels) -1:
                skip_connection_channels.append(current_out_channels)
            output_channels = current_out_channels    
        self.down_blocks = nn.ModuleList(self.down_blocks)

        self.mid_block = UNetMidBlock2D(
            in_channels=output_channels,
            temb_channels=time_embedding_dim,
            context_dim=text_embedding_dim
        )

        reversed_block_out_channels = list(reversed(block_out_channels)) 
        self.up_blocks = nn.ModuleList([])
        for i, out_channels in enumerate(reversed_block_out_channels):
            is_final_block = (i == len(reversed_block_out_channels) - 1)
            current_layer_skips = [skip_connection_channels.pop() for _ in range(3)]
            add_attention = True
            if i == 0:
                prev_output_channels = reversed_block_out_channels[0]
                add_attention = False
            else:
                prev_output_channels = reversed_block_out_channels[i-1]
            in_channels = out_channels 

            self.up_blocks.append(
                CrossAttnUpBlock2D(
                    in_channels=in_channels,
                    prev_output_channels=prev_output_channels,
                    res_skip_channels_list = current_layer_skips,
                    out_channels=out_channels,
                    temb_channels=time_embedding_dim,
                    num_layers=3, 
                    add_upsample=not is_final_block,
                    cross_attention_dim=text_embedding_dim,
                    add_attention=add_attention
                )
            )
        self.conv_norm_out = nn.GroupNorm(norm_num_groups, block_out_channels[0])
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], self.in_channels, kernel_size=3, padding=1)

    def forward(self, hidden_image:torch.Tensor, hidde_text:torch.Tensor, time_embedding:torch.Tensor):
        skip_connection = []
        hidden_image = self.conv_in(hidden_image)
        skip_connection.append(hidden_image)
        for downblock in self.down_blocks:
            hidden_image, skip_connection_output = downblock(hidden_image, hidde_text, time_embedding)
            skip_connection.extend(skip_connection_output)
        hidden_image = self.mid_block(hidden_image, hidde_text, time_embedding)
        for upblock in self.up_blocks:
            current_layer_skips = [skip_connection.pop() for _ in range(3)]
            hidden_image = upblock(hidden_image, hidde_text, time_embedding, current_layer_skips)
        hidden_image = self.conv_norm_out(hidden_image)
        hidden_image = self.conv_act(hidden_image)
        hidden_image = self.conv_out(hidden_image)
        return hidden_image

class Diffusion(nn.Module):
    def __init__(self,
                 in_channels:int = 4, 
                 attention_head_num:int = 8,
                 layer_per_block:int = 2,
                 text_embedding_dim :int = 768, 
                 time_step_dim:int =320
                 ) -> None:
        super().__init__()
        self.time_embedding_dim = in_channels * time_step_dim
        self.time_embedding =TimeEmbedding(time_step_dim,self.time_embedding_dim)
        self.unet = UNet(in_channels,self.time_embedding_dim,attention_head_num,layer_per_block,
                         text_embedding_dim)
    
    def forward(self, latent_imgae:torch.Tensor, hidden_contex:torch.Tensor, time_step:torch.Tensor):
        time_embedding = self.time_embedding(time_step)
        output = self.unet(latent_imgae, hidden_contex, time_embedding)
        return output




                




            
            











