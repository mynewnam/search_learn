import torch
import torch.nn as nn
import math


# 广播过程按如下规则逐维处理：

# + 从尾部维度开始对齐

# + 对每一维执行：

#   + 若两个维度相同，直接对齐；
#   + 若其中一个为 1，则扩展为另一个的维度；
#   + 若两者不相同且都不为 1，则报错（无法广播）。

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim, max_length=2048):
        super(AbsolutePositionalEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length

        self.register_buffer('posi_emb', torch.zeros(max_length, hidden_dim))
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, hidden_dim, 2).float()  / hidden_dim * math.log(10000.0))

        self.posi_emb[:, 0::2] = torch.sin(position * div_term)
        self.posi_emb[:, 1::2] = torch.cos(position * div_term)

    def forward(self, hid):
        hid = hid + self.posi_emb[:hid.size(1), :].unsqueeze(0)

        return hid


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super(RotaryPositionalEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.div_term = 1.0 / (10000 ** (torch.arange(0, self.hidden_dim, 2) / self.hidden_dim))
    
    def forward(self, hid):
        seq_length = hid.size(1)
        position = torch.arange(seq_length, dtype=torch.float).unsqueeze(1)
        angle = position * self.div_term.unsqueeze(0)

        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # 将 sin 和 cos 分别应用到 hid 的偶数和奇数维度
        hid[:, :, 0::2] = hid[:, :, 0::2] * cos - hid[:, :, 1::2] * sin
        hid[:, :, 1::2] = hid[:, :, 0::2] * sin + hid[:, :, 1::2] * cos

        return hid
    