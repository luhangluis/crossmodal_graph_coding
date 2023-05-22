import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from util import graph_transformer

import pickle
import networkx as nx
from util import my_node2vec

from util import graph_transformer, image_to_graph

# 图像数据生成图结构
G, data = image_to_graph.image2graph("./数据/3.jpg")

embeddings = my_node2vec.node2vec_embeddings(G)

# 前向传播
output = graph_transformer.encoding(embeddings)

print(output.shape)  # 输出: torch.Size([10, 32])，即10个图嵌入向量经过Transformer编码器和数据压缩后得到的输出

# 原始数据字节大小
size_data = len(pickle.dumps(data))
# G的字节大小
size_G = len(pickle.dumps(G))
# G的节点字节大小
size_Gnodes = len(pickle.dumps(G.nodes))
# G的邻接矩阵字节大小
size_adjacency_matrix = len(pickle.dumps(nx.adjacency_matrix(G)))
# G的图嵌入字节大小
size_embeddings = len(pickle.dumps(embeddings))
# G的图嵌入字节大小 + transformer后的结果
size_output = len(pickle.dumps(output))

print("原始数据字节大小: {}".format(size_data))
print("G的字节大小: {}".format(size_G))
print("G的节点字节大小: {}".format(size_Gnodes))
print("G的邻接矩阵字节大小: {}".format(size_adjacency_matrix))
print("G的图嵌入字节大小: {}".format(size_embeddings))
print("G的图嵌入字节大小 + transformer后的结果: {}".format(size_output))
