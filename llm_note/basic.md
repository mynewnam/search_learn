# LLM 基础知识

## 生成阶段
+ 关键参数
  + max_length / max_new_tokens: 最大生成 token 数量
  + temperature: 控制生成 token 的随机性，根据公式 $p_i=\frac{\exp(z_i/T)}{\sum\exp(z_j/T)}$，高温度使分布趋于平坦，低温度使分布趋于尖锐
  + top_k: 在每一步中仅从概率最高的 k 个 token 中采样
  + top_p: 累计概率达到 top_p 的最小 token 合集
  + do_sample: 是否使用采样策略，否则启用 Greedy-Search (完全确定化)
  + num_beams: Beam Search 中束的数量；如果 do_sample=False，那么执行非采样的 Beam Search，否则执行 Beam Search + Sampling
  + length_penalty / repetition_penalty: 额外惩罚

## 复杂度估算
+ 参数量
  + 词嵌入：$V * d$
  + Attention: Q/K/V/O 矩阵 $4d^2$
  + FFN: 两层 MLP $2 * d * 4d = 8d^2$
  + RMSNorm: 两个 $2d$
  + 输出层的 RMSNorm：$d$
  + LM Head: $d * V$
  + $L$ 层，共计: $2Vd + 12d^2L + (2L + 1)d$
+ 训练时间复杂度
  + 矩阵乘法复杂度：元素数量 * 单一元素计算复杂度