import pickle
import networkx as nx
from node2vec import Node2Vec

from util import image_to_graph, haptic_to_graph, sdne


def node2vec_embeddings(G):

    # 使用node2vec生成节点嵌入
    node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
    # 训练模型（可以使用其他参数进行自定义）
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    # 获取节点的嵌入向量
    embeddings = model.wv
    return embeddings
