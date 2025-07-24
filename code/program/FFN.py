import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFN, self).__init__()
        # 一般来说 hidden_dim = 4 * input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # [B, T, H]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        
        self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, input_dim)

        self.silu = F.silu

    def forward(self, x):
        # [B, T, H]
        up_x = self.up_proj(x)
        gate_x = self.silu(self.gate_proj(x))
        gated_x = up_x * gate_x

        x = self.down_proj(gated_x)

        return x