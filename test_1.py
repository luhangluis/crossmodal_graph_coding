import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np

# 初始化张量的几种方法
# 可用python自带列表创建一个张量
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([3, 2, 3])
# print(t.dtype)
# t.dtype
# t = torch.floattensor([1.1, 2, 3])  # float32
# t = torch.longtensor([1.1, 2, 3])  # int64
#
# # 也可从array转化为tensor
# np_array = np.arange(12).reshape(3, 4)
# t = torch.from_numpy(np_array)
#
# t = torch.tensor([1, 2, 3], dtype=torch.float32)  # 也可指定类型

G = nx.Graph()
G.add_edge(t1, t2, weight=4)

nx.draw(G)
plt.show()
