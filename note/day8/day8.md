# 搜广推 day8

## 经典排序模型 特征交叉

+ Factor Machine (FM)
  + 传统 LR 模型：$y = w_0 + \sum w_i x_i + \sum w_{ij} x_i x_j$
    + 如果二阶交叉项没有出现过，那么对应的系数就无法学习
    + $x_i$ 都是 0/1 变量
  + FM 模型：$y = w_0 + \sum w_i x_i + \sum <v_i, v_j> x_i x_j$
    + 利用了 Embedding 的思想，只要 $x_i$ 出现过，其对应的二阶交互项系数就可以学习得到
    + 可以将计算复杂度由 $O(kn^2)$ 简化为 $O(kn)$
  + 样本选择 (以 CTR 预估为例)
    + 正样本：曝光后用户点击的样本；
    + 负样本：曝光后未点击的样本，有时还选择 above click 的形式；
  + 特征选择 (都需要转化为 0/1 变量)
    + user 类别特征、item 类别特征、交叉特征 (只能在精排阶段使用)
    + 引入交叉特征带来的复杂度是 $O(log N)$ -> $O(N)$
    + 离散特征线上计算速度更快，离散化成高维向量
  + 线上的速度优化
    + 对于一次推荐任务，user 的特征值需要计算一次，即可复用给所有的 item 使用

+ Product-based Neural Networks for User Response Prediction