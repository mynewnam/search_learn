import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_dim, epsilon=1e-6):
        super(RMSNorm, self).__init__()

        self.hidden_dim = hidden_dim
        self.epsilon = epsilon

        self.weight = nn.Parameter(torch.ones(hidden_dim))
    
    def forward(self, hid):
        mean_of_sqrt = torch.sqrt(hid.pow(2).mean(-1, keepdim=True) + self.epsilon)
        hid = hid / mean_of_sqrt
        hid = hid * self.weight

        return hid
