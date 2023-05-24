import pickle
import networkx as nx
from util import my_node2vec

from util import image_to_graph, haptic_to_graph, sdne

# 图像数据生成图结构
# G, data = image_to_graph.image2graph("./数据/3.jpg")
# 触觉数据生成图结构
G, data = haptic_to_graph.haptic2graph("./数据/G1EpoxyRasterPlate_Movement_X_test1.txt", 100, 300)

embeddings = my_node2vec.node2vec_embeddings(G)

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

print("原始数据字节大小: {}".format(size_data))
print("G的字节大小: {}".format(size_G))
print("G的节点字节大小: {}".format(size_Gnodes))
print("G的邻接矩阵字节大小: {}".format(size_adjacency_matrix))
print("G的图嵌入字节大小: {}".format(size_embeddings))
