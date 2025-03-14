import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LoRALinear(nn.Linear):

    def __init__(self,
                 # nn.Linear parameters
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 # LoRA parameters
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:
        
        #TODO: Initialize the inherited class, nn.linear 
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)

        self.has_weights_merged = False
        if lora_rank > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)  # TODO: Initialize dropout

            self.lora_scaling = lora_alpha / lora_rank  # TODO: Compute scaling factor

            #TODO: Fill in the "..."
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features, device=device, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank, device=device, dtype=dtype))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            #TODO: Initialize both lora_A and lora_B with torch.nn.init. Refer to the paper to see how each is initialize
            #Hint: lora_A is initialized using kaiming_uniform_ using negative slope (a) as math.sqrt(5)
            nn.init.kaiming_uniform_(self.lora_A) #, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)  # Initialize lora_B with zeros

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        #TODO: return input after the forward pass
        #TODO: Remember to use dropout on the input before multiplying with lora_B and lora_A if the weights are not merged
        if self.has_weights_merged:
            return F.linear(input, self.weight, self.bias)
        else:
            result = F.linear(input, self.weight, self.bias)
            if self.is_lora():
                # TODO: Remember to use dropout on the input before multiplying with lora_B and lora_A if the weights are not merged
                result += self.lora_scaling * F.linear(self.lora_dropout(input), self.lora_A @ self.lora_B)
            return result

    def train(self, mode: bool = True) -> "LoRALinear":
        #TODO: Set the linear layer into train mode
        #Hint: Make sure to demerge LORA matrices if already merged
        super().train(mode)
        if mode and self.has_weights_merged:
            self.weight.data -= self.lora_scaling * (self.lora_B @ self.lora_A)
            self.has_weights_merged = False
        return self

    def eval(self) -> "LoRALinear":
        #TODO: Set the linear layer into eval mode
        #Hint: Make sure to merge LORA matrices if already demerged
        super().eval()
        if not self.has_weights_merged:
            self.weight.data += self.lora_scaling * (self.lora_B @ self.lora_A)
            self.has_weights_merged = True
        return self
    
    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    #TODO: Loop through parameters and mark some as trainable. Which ones should these be?
    #Hint: How do you mark a parameter as trainable (or not trainable)?
    for param in model.parameters():
        param.requires_grad = False  # Freeze all parameters
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    return model

