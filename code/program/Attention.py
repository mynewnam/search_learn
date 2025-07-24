import torch
import torch.nn as nn


class GQATrainingAttention(nn.Module):
    def __init__(self, hidden_dim, nums_attention_heads, nums_kv_heads):
        super(GQATrainingAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.nums_attention_heads = nums_attention_heads
        self.nums_kv_heads = nums_kv_heads
        self.head_dim = hidden_dim // nums_attention_heads
        self.pad_id_token = 0  # 假设 pad_id_token 是 0

        # num of head 是 kv head 的整数倍
        # atten_dim 是 head_dim 的整数倍

        self.q_proj = nn.Linear(hidden_dim, self.nums_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.nums_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.nums_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.nums_attention_heads * self.head_dim, hidden_dim, bias=False)

    def forward(self, hidden_state, inputs_id):
        B, T = inputs_id.shape

        # 构造 mask，右侧 padding
        padding_mask = (inputs_id != self.pad_id_token)  # B x T
        causal_mask = torch.tril(torch.ones((T, T), device=inputs_id.device)).bool()  # T x T
        ## 维度扩展
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) # B x 1 x 1 x T
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # 1 x 1 x T x T
        mask = padding_mask & causal_mask  # B x 1 x T x T

        # 计算 Q, K, V
        q = self.q_proj(hidden_state).view(B, T, self.nums_attention_heads, self.head_dim).transpose(1, 2) # B x nums_attention_heads x T x head_dim
        k = self.k_proj(hidden_state).view(B, T, self.nums_kv_heads, self.head_dim) # B x T x nums_kv_heads x head_dim
        v = self.v_proj(hidden_state).view(B, T, self.nums_kv_heads, self.head_dim) # B x T x nums_kv_heads x head_dim
        ## GQA 扩展
        k = k.unsqueeze(2).expand(B, T, self.nums_attention_heads // self.nums_kv_heads, self.nums_kv_heads, self.head_dim).reshape(B, T, self.nums_attention_heads, self.head_dim).transpose(1, 2) # B x nums_attention_heads x T x head_dim
        v = v.unsqueeze(2).expand(B, T, self.nums_attention_heads // self.nums_kv_heads, self.nums_kv_heads, self.head_dim).reshape(B, T, self.nums_attention_heads, self.head_dim).transpose(1, 2) # B x nums_attention_heads x T x head_dim

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # B x nums_attention_heads x T x T
        ## 应用 mask, True 表示填充
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)  # B x nums_attention_heads x T x head_dim

        # 合并
        output = output.transpose(1, 2).view(B, T, self.hidden_dim)
        output = self.o_proj(output)

        return output


class GQAInferenceAttention(nn.Module):
    def __init__(self, hidden_dim, nums_attention_heads, nums_kv_heads, max_seq_len, batch_size):
        super(GQAInferenceAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.nums_attention_heads = nums_attention_heads
        self.nums_kv_heads = nums_kv_heads
        self.head_dim = hidden_dim // nums_attention_heads
        self.pad_token_id = 0  # 假设 pad_id_token 是 0
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(hidden_dim, self.nums_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, self.nums_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, self.nums_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.nums_attention_heads * self.head_dim, hidden_dim, bias=False)

        # 推理过程中，batch 通常设置为 1，避免了 padding 和生成长度不一致
        self.register_buffer('k_cache', torch.zeros((batch_size, self.nums_attention_heads, max_seq_len, self.head_dim), dtype=torch.float32))
        self.register_buffer('v_cache', torch.zeros((batch_size, self.nums_attention_heads, max_seq_len, self.head_dim), dtype=torch.float32))

    def forward(self, input_ids, hidden_state, start_pos):
        B, T = input_ids.shape

        # 构造 mask，左侧 padding
        padding_mask = (input_ids != self.pad_token_id).unsqueeze(1).unsqueeze(2)  # B x 1 x 1 x T
        causal_mask = torch.tril(torch.ones((T, T)), diagonal=1).bool()  # T x T
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # 1 x 1 x T x T
        mask = padding_mask & causal_mask  # B x 1 x T x T

        # 计算 Q, K, V
        q = self.q_proj(hidden_state).view(B, T, self.nums_attention_heads, self.head_dim).transpose(1, 2) # B x nums_attention_heads x T x head_dim
        k = self.k_proj(hidden_state).view(B, T, self.nums_kv_heads, self.head_dim) # B x T x nums_kv_heads x head_dim
        v = self.v_proj(hidden_state).view(B, T, self.nums_kv_heads, self.head_dim) # B x T x nums_kv_heads x head_dim

        k = k.unsqueeze(2).expand(B, T, self.nums_attention_heads // self.nums_kv_heads, self.nums_kv_heads, self.head_dim).reshape(B, T, self.nums_attention_heads, self.head_dim).transpose(1, 2) # B x nums_attention_heads x T x head_dim
        v = v.unsqueeze(2).expand(B, T, self.nums_attention_heads // self.nums_kv_heads, self.nums_kv_heads, self.head_dim).reshape(B, T, self.nums_attention_heads, self.head_dim).transpose(1, 2) # B x nums_attention_heads x T x head_dim
        ## 更新缓存
        self.k_cache[:, :, start_pos:start_pos + T, :] = k
        self.v_cache[:, :, start_pos:start_pos + T, :] = v
        ## 取出缓存
        k = self.k_cache[:, :, :start_pos + T, :]
        v = self.v_cache[:, :, :start_pos + T, :]

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        ## 应用 mask, True 表示填充
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)  # B x nums_attention_heads x T x head_dim

        # 合并
        output = output.transpose(1, 2).view(B, T, self.hidden_dim)
        output = self.o_proj(output)

        return output, start_pos + T