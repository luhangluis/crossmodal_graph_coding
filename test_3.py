import random
import networkx as nx
import matplotlib.pyplot as plt

from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from karateclub import Graph2Vec

# 图像生成图结构
import networkx as nx
from matplotlib import pyplot as plt
import sys
import pickle
import networkx as nx

from util import image_to_graph, haptic_to_graph, sdne

# # 可视化图结构
# img.show()
# plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
# plt.rcParams['font.size'] = 12  # 字体大小
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# # # 提取节点坐标信息
# node_positions = nx.get_node_attributes(G, 'pos')
# edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 5]  # 控制边的权重大于阈值的进行显示
# nx.draw(G, pos=node_positions, labels={n: str(d['label']) for n, d in G.nodes(data=True)})
# # 显示图形
# plt.show()


# # 可视化图结构
# plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
# plt.rcParams['font.size'] = 12  # 字体大小
# plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# pos = nx.spring_layout(G, k=2, iterations=100, scale=2, seed=0)  # 定义节点布局
# edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 5]  # 控制边的权重大于阈值的进行显示
# nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='r')  # 绘制边
# nx.draw_networkx_nodes(G, pos)  # 绘制节点
# # plt.savefig('./数据/graph.jpeg', dpi=500)
# # 显示
# plt.show()

# 图像数据生成图结构
G, data = image_to_graph.image2graph("./数据/1.png")
# 触觉数据生成图结构
# plt.style.use("seaborn")
# G, data = haptic_to_graph.haptic2graph("./数据/G1EpoxyRasterPlate_Movement_X_test1.txt", 1000, 5000)

# 获取图的嵌入
embeddings = sdne.SDNE_embeddings(G)


# node embeddings
def run_n2v(G, dimensions=64, walk_length=80, num_walks=10, p=1, q=1, window=10):
    """
    Given a graph G, this method will run the Node2Vec algorithm trained with the
    appropriate parameters passed in.

    Args:
        G (Graph) : The network you want to run node2vec on

    Returns:
        This method will return a model

    Example:
        G = np.barbell_graph(m1=5, m2=3)
        mdl = run_n2v(G)
    """

    mdl = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q
    )
    mdl = mdl.fit(window=window)
    return mdl


mdl = run_n2v(G)

# visualize node embeddings
x_coord = [mdl.wv.get_vector(str(x))[0] for x in G.nodes()]
y_coord = [mdl.wv.get_vector(str(x))[1] for x in G.nodes()]

# plt.clf()
# plt.scatter(x_coord, y_coord)
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.title("2 Dimensional Representation of Node2Vec Algorithm on Barbell Network")
# plt.show()
data_size = sys.getsizeof(data)


edges_embs = HadamardEmbedder(
     keyed_vectors=mdl.wv
 )


print("原始数据字节大小:", data_size)

x_coord_size = sys.getsizeof(x_coord)
print("x_coord字节大小:", x_coord_size)

y_coord_size = sys.getsizeof(y_coord)
print("y_coord字节大小:", y_coord_size)

edges_embs_size = sys.getsizeof(edges_embs)
print("edges_embs字节大小:", edges_embs_size)

edges_embs_data = pickle.dumps(edges_embs)
edges_embs_in_bytes_data = sys.getsizeof(edges_embs_data)
print("edges_embs_in_bytes字节大小:", edges_embs_in_bytes_data)


G_size = sys.getsizeof(G)
print("G字节大小:", G_size)

# 计算节点的字节大小
node_size = sys.getsizeof(G.nodes())
print("节点字节大小:", node_size)

# 计算邻接矩阵的字节大小
adj_matrix = nx.to_numpy_matrix(G)
adj_matrix_size = sys.getsizeof(adj_matrix)
print("邻接矩阵字节大小:", adj_matrix_size)

embeddings_size = sys.getsizeof(embeddings)
print("图嵌入字节大小:", embeddings_size)

# 序列化原始数据并获取字节大小
serialized_data = pickle.dumps(data)
size_in_bytes_data = sys.getsizeof(serialized_data)
# 序列化图并获取字节大小
serialized_G = pickle.dumps(G)
size_in_bytes_G = sys.getsizeof(serialized_G)
# 序列化图嵌入并获取字节大小
serialized_dict = pickle.dumps(embeddings)
size_in_bytes_dict = sys.getsizeof(serialized_dict)

print(size_in_bytes_data)
print(size_in_bytes_G)
print(size_in_bytes_dict)
