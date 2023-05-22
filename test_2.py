# -*- coding: utf-8 -*-
"""
目标：
1.绘制出无向图；
2.在图中标注出目标路径
"""
import networkx as nx
import matplotlib.pyplot as plt


def make_s(length):  # 标签生成函数
    temp_s = dict()
    for i in range(1, length):
        temp_s[i] = temp_s.get(0, i)
    return temp_s


nodes = [(1, 2), (1, 25), (2, 3), (3, 4), (3, 25), (4, 5), (4, 24), (5, 6), \
         (5, 24), (6, 7), (6, 23), (6, 24), (7, 8), (7, 22), (8, 9), (8, 22), \
         (9, 10), (9, 15), (9, 16), (9, 17), (9, 21), (9, 22), (10, 11), \
         (10, 13), (10, 15), (11, 12), (11, 13), (12, 13), (12, 14), (13, 14), \
         (13, 15), (14, 15), (14, 16), (15, 16), (16, 17), (16, 18), (17, 18), \
         (17, 21), (18, 19), (18, 20), (20, 21), (21, 22), (21, 23), (21, 27), \
         (22, 23), (23, 24), (23, 26), (24, 25), (24, 26), (25, 26), (26, 27)]  # 绘制点
'''
G1:无向图
'''
G1 = nx.Graph()  # 创建无向图
G1.add_edges_from(nodes)  # 加边
pos = nx.spring_layout(G1)  # 设置图的布局
'''
图的几种布局如下：
nx.spring_layout()
nx.circular_layout()
nx.spectral_layout()
nx.shell_layout()
'''
s1 = make_s(28)  # 生成标签
nx.draw(G1, pos, font_size=14, labels=s1, node_color='y')  # 绘图，设置布局，字体大小，点的颜色
path = [(1, 25), (25, 26), (26, 27)]  # 设置标注的路
nx.draw_networkx_edges(G1, pos, edgelist=path, width=3, edge_color='r')  # 绘制标注的边
plt.show()
