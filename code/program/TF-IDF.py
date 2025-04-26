import math
from collections import Counter


def calculate_tf(document):
    """
    计算文档中每个词的词频(TF)
    Args:
        document: 分词后的文档列表
    Returns:
        tf_dict: 词频字典
    """
    # 计算词频
    word_count = Counter(document)
    doc_length = len(document)
    
    # 计算每个词的TF值
    tf_dict = {word: count/doc_length for word, count in word_count.items()}
    return tf_dict


def calculate_idf(documents):
    """
    计算语料库中每个词的逆文档频率(IDF)
    Args:
        documents: 分词后的文档列表的列表
    Returns:
        idf_dict: 逆文档频率字典
    """
    # 获取文档总数
    n_docs = len(documents)
    
    # 统计每个词出现在多少个文档中
    word_in_docs = {}
    for doc in documents:
        # 使用集合去重，每个词在一个文档中只计算一次
        unique_words = set(doc)
        for word in unique_words:
            word_in_docs[word] = word_in_docs.get(word, 0) + 1
    
    # 计算IDF
    idf_dict = {word: math.log(n_docs / (doc_freq + 1)) + 1 
               for word, doc_freq in word_in_docs.items()}
    return idf_dict


def calculate_tfidf(documents):
    """
    计算TF-IDF值
    Args:
        documents: 文档列表
    Returns:
        tfidf_docs: 文档的TF-IDF表示列表
    """
    # 针对**一个词**和**一个文档**的TF-IDF计算公式如下：
    # TF(t,d) = (词t在文档d中出现的次数) / (文档d中的总词数)
    # IDF(t) = log(文档总数 / (包含词t的文档数 + 1)) + 1
    # TF-IDF(t,d) = TF(t,d) * IDF(t)

    # 预处理：分词
    processed_docs = []
    for doc in documents:
        words = doc.lower().split()
        processed_docs.append(words)
    
    # 计算IDF
    idf = calculate_idf(processed_docs)
    
    # 计算每个文档的TF-IDF
    tfidf_docs = []
    for doc in processed_docs:
        tf = calculate_tf(doc)
        
        # 计算TF-IDF
        tfidf = {word: tf[word] * idf.get(word, 0) for word in tf}
        tfidf_docs.append(tfidf)
    
    return tfidf_docs


def main():
    # 测试文档
    documents = [
        "hello world",
        "hello apple",
        "hello fruit"
    ]
    
    # 计算TF-IDF
    tfidf_results = calculate_tfidf(documents)
    
    # 获取所有不重复的词汇
    all_words = set()
    for result in tfidf_results:
        all_words.update(result.keys())
    all_words = sorted(list(all_words))
    
    # 输出结果
    for i, doc in enumerate(documents):
        print(f"文档 {i+1}: {doc}")
        for word in all_words:
            if word in tfidf_results[i]:
                print(f"  {word}: {tfidf_results[i][word]:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()