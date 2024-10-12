import torch
from torch import nn
from torch.nn import functional as F

class Adapter_Layer(nn.Module):
    def __init__(self, hidden_dim, conditional_enc_dim):
        super(Adapter_Layer, self).__init__()
        self.fc1 = nn.Linear(hidden_dim + conditional_enc_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, hidden_rep, conditional_enc_rep):
        
        x = torch.cat(hidden_rep, conditional_enc_rep)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
        