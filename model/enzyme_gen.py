import torch
from torch import nn
from torch.nn import functional as F

from esm.pretrained import ESM3_structure_encoder_v0, ESM3_structure_decoder_v0, ESM3_function_decoder_v0, ESM3_sm_open_v0


from adapter import Adapter_Layer
from conditional_encoder import ConditioningEncoder

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def load_pretrained_model(device):
    structure_enc = ESM3_structure_encoder_v0(device)
    structure_dec = ESM3_structure_decoder_v0(device)
    function_dec = ESM3_function_decoder_v0(device)
    esm3 = ESM3_sm_open_v0(device)

    return structure_enc, structure_dec, function_dec, esm3

class Enzyme_Gen(nn.Module):
    def __init__(self):
        super(Enzyme_Gen, self)
        
        self.structure_enc, self.structure_dec, self.function_dec, self.esm3 = load_pretrained_model(device)
        
        for param in self.structure_enc.parameters():
            param.requires_grad = False
        for param in self.structure_dec.parameters():
            param.requires_grad = False
        for param in self.function_dec.parameters():
            param.requires_grad = False
        for param in self.esm3.parameters():
            param.requires_grad = False
        
        self.conditioning_enc = ConditioningEncoder(
            ...
        )

        self.adapters = nn.ModuleList()
        for layer in self.esm3.layers:
            hidden_size = layer.self_attn.embed_dim  # Adjust attribute based on your model
            adapter = Adapter_Layer(hidden_size, condition_size, adapter_size)
            self.adapters.append(adapter)