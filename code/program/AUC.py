def calculate_auc(predictions, labels):
    """
    计算AUC值
    Args:
        predictions: 预测值列表
        labels: 真实标签列表(0或1)
    Returns:
        auc: AUC值
    """
    n = len(predictions)
    pos_pairs = 0  # 正确排序的样本对数
    total_pairs = 0  # 总样本对数
    
    # 遍历所有样本对
    for i in range(n):
        for j in range(n):
            if labels[i] > labels[j]:  # 当i为正样本,j为负样本时
                total_pairs += 1
                if predictions[i] > predictions[j]:
                    pos_pairs += 1
    
    return pos_pairs / total_pairs if total_pairs > 0 else 0

def main():
    # 测试用例1: 完美排序
    predictions1 = [0.9, 0.8, 0.3, 0.1]
    labels1 = [1, 1, 0, 0]
    
    # 测试用例2: 部分排序错误
    predictions2 = [0.7, 0.3, 0.5]
    labels2 = [1, 0, 0]
    
    # 测试用例3: 极端情况
    predictions3 = [0.1, 0.4, 0.35, 0.8]
    labels3 = [0, 1, 0, 1]
    
    # 计算并输出结果
    test_cases = [
        (predictions1, labels1),
        (predictions2, labels2),
        (predictions3, labels3)
    ]
    
    for i, (pred, lab) in enumerate(test_cases, 1):
        auc = calculate_auc(pred, lab)
        print(f"测试用例 {i}:")
        print(f"预测值: {pred}")
        print(f"真实值: {lab}")
        print(f"AUC: {auc:.6f}")
        print("-" * 50)

if __name__ == "__main__":
    main()