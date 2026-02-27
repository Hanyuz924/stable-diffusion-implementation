import torch
from torch import nn
import math


class LoraLayer(nn.Module):
    def __init__(self, base_layer, rank=8, lora_alpha=16, init_method="gaussian"):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False) 
        self.r = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r 
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros((self.r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, self.r)))
        
        if init_method == "gaussian":
            nn.init.normal_(self.lora_A, std=1 / self.r)
        else:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.merged = False

    def forward(self, hidden_state: torch.Tensor):
        if self.merged:
            return self.base_layer(hidden_state)

        base_layer_out = self.base_layer(hidden_state)
        # hidden_state: [batch, in]
        # lora_A.t(): [in, r]
        # lora_B.t(): [r, out]
        lora_out = (hidden_state @ self.lora_A.t()) @ self.lora_B.t()
        
        return base_layer_out + lora_out * self.scaling
    
    def merge(self):
        if self.merged :
            return 
        if isinstance(self.base_layer, nn.Linear):
            with torch.no_grad():
                delta_weights = (self.lora_B @ self.lora_A) * self.scaling
                self.base_layer.weight.data  += delta_weights
                self.merged = True
        else:
            raise ValueError("Lora only support for linear layer, conv not implemented yet")
    def unmerge(self):
        if not self.merged:
            return
        if isinstance(self.base_layer, nn.Linear):
            with torch.no_grad():
                delta_weights = (self.lora_B @ self.lora_A) * self.scaling
                self.base_layer.weight.data -= delta_weights
                self.merged = False





    

        
