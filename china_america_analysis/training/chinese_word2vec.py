import gensim
from gensim.models import Word2Vec
import jieba
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 设置数据和模型保存路径
data_path = './data_y'
model_path = './embeddings'

# 创建文件夹用于保存模型
os.makedirs(model_path, exist_ok=True)

def plot_list(y_data, x_data, title='List Plot', xlabel='Year', ylabel='Similarity'):
    """
    画出一个 y_data 相对于 x_data 的折线图。

    参数:
    - y_data: 需要画的 y 轴数据（例如：wv_sim）
    - x_data: 需要画的 x 轴数据（例如：x_axis）
    - title: 图的标题 (可选)
    - xlabel: x轴标签 (可选)
    - ylabel: y轴标签 (可选)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, marker='o')  # 使用折线图，带有点标记
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig('./temp.png')
    plt.show()

def word2vec(sy, ey):
    """
    训练并保存 Word2Vec 模型。
    
    参数:
    - sy: 起始年份
    - ey: 结束年份
    """
    # 设置使用的 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    file_name = f'_{sy}_{ey}_'
    sentences = []
    
    # 读取数据并分词
    with open(f'{data_path}/{file_name}.txt', 'r', encoding='utf-8') as f:
        for line in f:
            words = jieba.lcut(line.strip())
            sentences.append(words)
    
    # 训练 Word2Vec 模型
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=10, workers=8)
    
    # 保存模型
    model.save(f"{model_path}/{file_name}.model")

def load_model(sy, ey):
    """
    加载已保存的 Word2Vec 模型。
    
    参数:
    - sy: 起始年份
    - ey: 结束年份
    
    返回:
    - model: 加载的 Word2Vec 模型
    """
    file_name = f'_{sy}_{ey}_'
    model = Word2Vec.load(f"{model_path}/{file_name}.model")
    return model

def train_model():
    """
    训练多个年份区间的 Word2Vec 模型并保存。
    """
    for sy in tqdm(range(1946, 2024, 3)):
        ey = sy + 2
        word2vec(sy, ey)

def analyze_similarity(word1, word2):
    """
    分析两个词在不同年份区间内的相似度变化。
    
    参数:
    - word1: 第一个词
    - word2: 第二个词
    """
    wv_sim = []
    x_axis = []
    
    # 遍历所有年份区间
    for sy in tqdm(range(1946, 2024, 3)):
        ey = sy + 2
        try:
            model = load_model(sy, ey)
            # 计算两个词之间的相似度
            similarity = model.wv.similarity(word1, word2)
            wv_sim.append(similarity)
            x_axis.append(sy)
            print(f'{sy} {word1}-{word2} similarity: {similarity}')
        except KeyError:
            # 如果词不存在于词汇表中，设置相似度为0
            print(f'{sy}: {word1} or {word2} not present in model vocabulary.')
            wv_sim.append(0)
            x_axis.append(sy)
    
    # 绘制相似度随时间变化的图
    plot_list(wv_sim, x_axis, title=f'Similarity between {word1} and {word2} over time',
              xlabel='Year', ylabel='Similarity')

if __name__ == '__main__':
    # 训练多个年份区间的模型（已注释，可根据需要运行）
    # train_model()
    
    # 分析两个词在不同年份的相似度变化
    word1 = '江泽民'
    word2 = '伟大'
    analyze_similarity(word1, word2)
