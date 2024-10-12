import torch
from torch import nn
from torch.nn import functional as F

from esm.pretrained import ESM3_sm_open_v0


from adapter import Adapter_Layer
from conditional_encoder import ConditioningEncoder

class Enzyme_Gen(nn.Module):
    def __init__(self):
        super(Enzyme_Gen, self)
        
        
        