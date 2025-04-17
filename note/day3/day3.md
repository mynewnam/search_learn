# 搜广推 day3

## 向量召回 (Vector Filtering)

向量召回将物品和用户转化为固定维度的 Embedding 表示，通过余弦相似度、内积等方式刻画相似度；属于**语义**维度；协同过滤召回是基于用户和物品的**历史交互**，二者角度不同。

+ FM 召回 (Factorization Machine)
  + 基本模型结构：包含特征（用户特征及物品特征）的一阶、二阶交互项；
  + 计算公式：如图所示，注意二阶交互项的计算复杂度可以优化到 **O(kn)**； 
    ![fm](pic/fm_arch.png) 
  + 用于召回时的简化：
    + 由于召回时用户是给定的，因此只考虑**物品特征**和**用户和物品的交互特征**，具体如下。
    ![fm](pic/fm_callback.png)
  + 样本选择 (不同于**排序**的方式)
    + 正样本：曝光且点击
    + 负样本：通过在 item 库中随机采样得到；**不可以拿曝光未点击**做负样本；有论文背书，且数据分布与真实环境不同 (**曝光未点击**既不算用户不喜欢也不算用户喜欢)；
    + 由于候选集是整个 item 库，这种采样方式让模型能够对大错特错的样本有很好的认知；
    + 在正样本采样时，要打压热门 item；这是因为正样本中大部分是热门 item，直接采样会导致在用户之间没有区分度；一种可用的降采样方式如下：
      ![down_sample](pic/down_sampling.png)
    + 在负样本采样时，要提升热门 item 但同时也要覆盖到整个样本库；一种常见的采样方式如下：
      ![up_sampling](pic/up_sampling.png)
    + 但是，上述方法得到的通常是 easy-negetive 样本，需要找到一些 hard-negetive，比如说根据业务逻辑；比如说根据上次召回结果的 101-500 名定性为 hard-negetive；
      + 在线筛选：对同一个 batch 内的 <user, item+>，从中选取 **1-2** 个与 user 最接近的 item+ 作为 hard-negetive；但可能面临着数据量不够大的问题；
      + 离线筛选：过一遍 item 集合，根据上次召回结果的 101-500 名定性为 hard-negetive；
      + 比例维持在 100：1，全是 hard-negetive 并不有效；
    + 在 hard-negetive 的使用上，既可以混合使用 (迁移使用)，也可以并行使用 (加权)，也可以串行使用 (多次过滤)。
  + 特征选择上，在预测时，不能够使用任何 user 与 item 的交叉特征，否则检索速度不够快。
  + 优化目标上，使用二分类并不好 (因为 hard-negetive 是随机抽取的)；希望**正样本的得分远远大于负样本**即可。因此输入的一个样本是 <user, item+, item->，采用的损失函数可以是 PairWise LTR，也可以是 BPR (鼓励二者的差距越大越好)。
    ![ltr](pic/LTR.png)
    ![bpr](pic/BPR.png)
    ![score](pic/fm_score.png)
  + 线上使用时，对于同一个 user 而言，遵循如下的召回计算方式即可；
    + FM 召回对于新样本适应较好，因为至少有一个特征 new_user，样本分类也一定具有特征。
    + 不能只考虑交互项，热门 item 自身的特征对于推荐也很有帮助。
      ![fm_online](pic/fm_online.png)
  + FM 做 embedding 的方式就是 one-hot 编码一段隐向量。

+ item2vec 召回
  + word2vec 原理 - Skip-gram
    + 给定中心词 c 的情况下，最大化周围单词 o 的出现概率；其中，概率分布使用 softmax 损失函数进行估计；
    + 注意到，分母考虑了词表的所有单词，计算量很大；且中心词与周围词使用的是两个不同的矩阵。
      ![softmax](pic/softmax.png)
      ![skip_gram](pic/skip_gram.png)
    + 考虑到词表数量过大，采用了依照频率的四分之三次方的采样的方式选择 5-20 / 2-5 个负样本；使用 NCE 损失函数 $L = log(σ(v_c · v_o)) + Σ_{i=1}^k E[log(σ(-v_c · v_{neg_i}))]$
    + 考虑到训练集中有大量的高频词 (a, the, ...)，对频繁词子采样 $P(wi) = 1 - \sqrt{(t/f(wi))}$
  + word2vec 原理 - CBOW
    + 给定周围词 ${o_i}$ 的情况下，最大化中心单词 c 的出现概率；对于多个出现的周围词，对其特征向量进行了平均；
    + 其他均与 Skip-gram 相同；
  + item2vec: 直接类比了 word2vec，将一类物品集合 (set) 替代了原本的句子 (sentence) 的概念；$L = \sum_{(i,j) \in D^+}( \log \sigma(v_i \cdot v_j) + \sum_{k=1}^{K} \mathbb{E}_{item_k \sim P_n(i)} [\log \sigma(-v_i \cdot v_k)])$

+ Airbnb 召回