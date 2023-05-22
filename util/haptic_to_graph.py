# 实现将触觉数据中的特征提取出来，先使用波峰来作为特征向量并使用networkx进行图的构建
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from util import utils
import networkx as nx

#
def haptic2graph(filename, l, h):
    """
    实现将触觉数据变为图结构
    :param filename: 触觉数据文件地址
    :param l: 从第l行读取触觉文件
    :param h: 到第l行结束
    :return: 图结构G，触觉原始数据y
    """
    # 生成示例数据
    y = utils.read_txt_data(filename)
    y = y[l:h]
    x = np.arange(len(y))

    # 平滑数据（示例使用简单的移动平均）
    window_size = 15
    smoothed_data = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    #
    # # 寻找波峰
    # peaks_index, _ = find_peaks(smoothed_data)
    # # 输出波峰值的索引和对应的数值
    # peaks = smoothed_data[peaks_index]

    # # # 寻找波峰
    peaks, _ = find_peaks(y)
    # 邻接矩阵
    s = np.zeros((len(peaks), len(peaks)))
    # 创建空的图对象
    G = nx.Graph()
    # 遍历邻接矩阵 - 二维数组
    for i in range(len(peaks)):
        for j in range(len(peaks)):
            if i == j:
                continue
            s[i, j] = abs(peaks[i] - peaks[j])
            G.add_edge(peaks[i], peaks[j], weight=s[i, j])

    return G, y

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
