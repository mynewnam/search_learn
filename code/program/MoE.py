import torch
import torch.nn as nn
import torch.nn.functional as F


class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoE(nn.Module):
    def __init__(self, num_of_expert, selected_num_of_expert, hidden_size, intermediate_size, norm_expert=False):
        super().__init__()
        self.num_of_expert = num_of_expert # 专家总数
        self.selected_num_of_expert = selected_num_of_expert # 激活专家数
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.norm_expert = norm_expert # 是否对专家权重进行归一化
        self.experts = nn.ModuleList([MoeMLP(hidden_size, intermediate_size) for _ in range(num_of_expert)]) # 专家层

        self.gate = nn.Linear(hidden_size, num_of_expert, bias=False) # 门控层

    def forward(self, hidden_state):
        batch_size, sequence_length, hidden_dim = hidden_state.size()
        hidden_state = hidden_state.view(-1, hidden_dim)  # [batch_size * sequence_length, hidden_dim]

        # 路由层，确定每个 token 去哪个专家层 (注意维度变化)
        router_logits = self.gate(hidden_state)  # [batch_size * sequence_length, num_of_expert]
        router_probs = F.softmax(router_logits, dim=-1)  # [batch_size * sequence_length, num_of_expert]
        router_probs, selected_experts = torch.topk(router_probs, self.selected_num_of_expert, dim=-1)  # [batch_size * sequence_length, selected_num_of_expert]
        if self.norm_expert:
            router_probs = router_probs / router_probs.sum(dim=-1, keepdim=True)
        
        # 初始化最终结果
        final_hidden_state = torch.zeros((batch_size * sequence_length, self.hidden_size), device=hidden_state.device, dtype=hidden_state.dtype)

        # one-hot 编码选中的专家
        ## [batch_size * sequence_length, selected_num_of_expert, num_of_expert] -> [num_of_expert, selected_num_of_expert, batch_size * sequence_length]
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_of_expert).permute(2, 1, 0)

        # 寻找被激活的专家
        ## [num_of_hitted_expert, 1]，存储了被激活专家的编号
        expert_hitted = torch.greater(torch.sum(expert_mask, dim=(-1, -2)), 0).nonzero()

        for expert_id in expert_hitted:
            expert_layer = self.experts[expert_id] # [selected_num_of_expert, hidden_size]
            ## id: 选择该专家的 token id；top_x: 该 token 是第 x 顺位选择的该专家
            top_x, id = torch.where(expert_mask[expert_id].squeeze(0) > 0) # [batch_size * sequence_length]

            # 将这批 token 进行计算
            current_state = hidden_state[None, id].reshape(-1, hidden_dim) # [selected_token, hidden_size]
            current_hidden_state = expert_layer(current_state) * router_probs[id, top_x, None]  # [selected_token, hidden_size]

            # 将结果累加到最终结果中
            final_hidden_state.index_add_(0, id, current_hidden_state)
        final_hidden_state = final_hidden_state.view(batch_size, sequence_length, hidden_dim)  # 恢复维度

        return final_hidden_state, router_logits
    

def load_balance_loss(gate_logits, num_of_expert, selected_num_of_expert, attention_mask=None):
    """
    计算负载均衡损失
    gate_logits: [batch_size * sequence_length, num_of_expert]
    num_of_expert: 专家总数
    selected_num_of_expert: 激活专家数
    """
    
    concat_gate_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)  # [batch_size * sequence_length * num_of_layer, num_of_expert]

    gate_probs = F.softmax(concat_gate_logits, dim=-1)  # [batch_size * sequence_length * num_of_layer, num_of_expert]
    _, selected_experts = torch.topk(gate_probs, selected_num_of_expert, dim=-1)  # [batch_size * sequence_length * num_of_layer, selected_num_of_expert]
    expert_mask = F.one_hot(selected_experts, num_classes=num_of_expert).float()  # [batch_size * sequence_length * num_of_layer, selected_num_of_expert, num_of_expert]

    if attention_mask is None:
        # 每个专家被选中的 token 频率，即每个专家在每个顺位的频率
        tokens_per_expert = torch.mean(expert_mask, dim=0)  # [selected_num_of_expert, num_of_expert]
        # 每个专家 softmax 的得分
        gate_probs_per_expert = torch.mean(gate_probs, dim=0)  # [num_of_expert]
    else:
        batch_size, sequence_length = attention_mask.size()
        num_hidden_layers = concat_gate_logits.size(0) // (batch_size * sequence_length)

        # 将 attention_mask 扩展到 expert_mask 的形状
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(num_hidden_layers, batch_size, sequence_length, selected_num_of_expert, num_of_expert)
            .reshape(-1, selected_num_of_expert, num_of_expert)
        )
        router_attention_mask = (
            attention_mask[None, :, :, None]
            .expand(num_hidden_layers, batch_size, sequence_length, num_of_expert)
            .reshape(-1, num_of_expert)
        )

        # 计算得分
        tokens_per_expert = torch.sum(expert_attention_mask * expert_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)  # [selected_num_of_expert, num_of_expert]
        gate_probs_per_expert = torch.sum(gate_probs * router_attention_mask, dim=0) / torch.sum(router_attention_mask, dim=0)

    overall_loss = torch.sum(tokens_per_expert * gate_probs_per_expert.unsqueeze(0))

    return overall_loss * num_of_expert
