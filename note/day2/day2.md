# 搜广推 day2

## 协同过滤召回 (Collaborative Filtering)

基于用户历史行为数据挖掘用户的喜爱偏好，并预测用户可能喜爱的产品。

+ UserCF (用户协同过滤) (判断用户的相似度)
  + 算法流程
    + 根据已有打分情况 (用户 Embedding)，找到与 user_0 最相关的 K 个 user；
    + user_0 对于 new item 的打分使用 K 个 user 的加权平均获得 (优化：对所有 user 的打分减去均值)；
  + 缺点
    + 用户购买的重叠度低，难以寻找相似用户；
    + 用户相似度的计算开销很大；
  + 评价指标：召回率、精确率、覆盖率 (所有推荐的商品占商品总数的比例)、新颖度
![usercf](pic/usercf.png)

+ ItemCF (物品协同过滤) (基于历史用户的行为判断物品的相似度)
  + 算法流程
    + 根据已有打分情况 (物品 Embedding)，找到与 new item 最相关的 K 个 item；
    + 利用 user_0 对于最相关的 K 个 item 的打分，加权计算得到对 new item 的打分 (优化：对每个 item 的打分减去均值)；
  + 算法问题
    + 热门物品与其他物品总是相似的，因此热门物品大量推荐；而冷门物品特征向量稀疏，很少推荐；
    + 无法将两个物品的相似推广到其他物品上；
    + 改进：矩阵分解技术 (稀疏矩阵 -> 稠密矩阵)
![itemcf](pic/itemcf.png)

+ MF (矩阵分解) (相当于为 user 和 item 构造一个 Embedding)
  + 算法流程：如图所示，转换为高维矩阵重建问题即可；
  + 算法问题
    + 只考虑了评分矩阵，没有考虑 user、item、context 等特征；
    + 对于没有历史行为的用户，无法冷启动；
![mf](pic/mf.png)

+ Swing (构建 user-item 二部图捕获产品间的相似关系)
  + Swing 算法：利用用户的购买行为计算商品之间的相似性；
![swing1](pic/example_of_swing.jpeg)
![swing2](pic/swing_algo.png)
  + Surprise 算法：