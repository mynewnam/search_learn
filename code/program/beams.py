import torch
import torch.nn.functional as F


# 改进的实现包括 Group Beam Search, 将 beam 内部分成多个 group, 施加 Diversity Penalty
def beam_search_sampling(model, input_ids, beam_scores, beam_width=4, sample_per_beam=3, temperature=1.0, top_k=50):
    """
    input_ids: [batch_size * num_beams, sequence_length]
    beam_scores: [batch_size * num_beams]
    """

    batch_beam, seq_len = input_ids.size()
    batch_size = batch_beam // beam_width
    device = input_ids.device

    # 获取下一个 token 的 logits
    outputs = model(input_ids)  # [batch_size * num_beams, sequence_length, vocab_size]

    # 进行采样
    next_token_logits = outputs[:, -1, :] / temperature  # [batch_size * num_beams, vocab_size]
    if top_k > 0:
        top_k = min(top_k, next_token_logits.size(-1))
        ## topk 的返回值是 (值， 索引)
        top_k_logits, _ = torch.topk(next_token_logits, top_k, dim=-1)  # [batch_size * num_beams, top_k]
        min_top_k = top_k_logits[:, -1].unsqueeze(-1)  # [batch_size * num_beams, 1]
        ## torch.where 传入一个参数 -> 返回索引；传入三个参数 (condition, value_if_true, value_if_false) -> 返回值
        next_token_logits = torch.where(next_token_logits < min_top_k, torch.full_like(next_token_logits, float('-inf')), next_token_logits) # [batch_size * num_beams, vocab_size]
    
    probs = F.softmax(next_token_logits, dim=-1)  # [batch_size * num_beams, vocab_size]

    ## 采样数量为 sample_per_beam
    sampled_next_tokens = torch.multinomial(probs, num_samples=sample_per_beam)  # [batch_size * num_beams, sample_per_beam]
    sampled_probs = torch.gather(probs, dim=-1, index=sampled_next_tokens)  # [batch_size * num_beams, sample_per_beam]
    sampled_log_probs = torch.log(sampled_probs)  # [batch_size * num_beams, sample_per_beam]

    # 扩展 beams
    beam_scores = beam_scores.unsqueeze(-1) + sampled_log_probs  # [batch_size * num_beams, sample_per_beam]
    beam_scores = beam_scores.view(batch_size, -1)  # [batch_size, num_beams * sample_per_beam]

    # 拼接新的 beams
    input_ids = input_ids.unsqueeze(1).expand(-1, sample_per_beam, -1).contiguous()  # [batch_size * num_beams, sample_per_beam, sequence_length]
    input_ids = input_ids.view(-1, seq_len)  # [batch_size * num_beams * sample_per_beam, sequence_length]
    new_tokens = sampled_next_tokens.view(-1, 1)  # [batch_size * num_beams * sample_per_beam, 1]
    input_ids = torch.cat([input_ids, new_tokens], dim=-1).view(batch_size, -1, seq_len + 1)  # [batch_size, num_beams * sample_per_beam, sequence_length + 1]

    # 选择 top-k beams
    top_k_scores, top_k_indices = torch.topk(beam_scores, k=beam_width, dim=1)  # [batch_size, beam_width]
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [batch_size, 1]
    selected_input_ids = input_ids[batch_indices, top_k_indices]  # [batch_size, beam_width, sequence_length + 1]
    selected_beam_scores = top_k_scores  # [batch_size, beam_width]

    # 将结果展平
    selected_input_ids = selected_input_ids.view(batch_size * beam_width, seq_len + 1)  # [batch_size * beam_width, sequence_length + 1]
    selected_beam_scores = selected_beam_scores.view(batch_size * beam_width)  # [batch_size * beam_width]

    return selected_input_ids, selected_beam_scores


if __name__ == "__main__":
    model = None  # 假设有一个预训练的模型
    input_ids = torch.randint(0, 10000, (8, 10))  # 假设有 8 个样本，每个样本长度为 10
    beam_scores = torch.zeros(8 * 4)  # 假设有 4 个 beams
    beam_width = 4

    for _ in range(5):
        input_ids, beam_scores = beam_search_sampling(model, input_ids, beam_scores, beam_width=beam_width)