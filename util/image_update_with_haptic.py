import networkx as nx
import numpy as np

from util import image_to_graph, haptic_to_graph


def normalize(vector, norm=12):
    min_val = np.min(vector)  # 找到向量中的最小值
    max_val = np.max(vector)  # 找到向量中的最大值
    normalized_vector = (vector - min_val) / (max_val - min_val)  # 归一化计算
    return normalized_vector


def attention_haptic2visual(image_vec, haptic_vec):
    len_haptic = len(haptic_vec)
    len_image = len(image_vec)
    # 计算关联分数（使用softmax回归）
    image_vec = image_vec.reshape(len_image, 1)
    haptic_vec = haptic_vec.reshape(1, len_haptic)
    arr = np.matmul(image_vec, haptic_vec)

    # 计算每列的均方根，得到关联分数
    # scores = np.linalg.norm(arr, axis=0)

    squares = np.square(arr)  # 每个元素平方
    mean_squares = np.mean(squares, axis=0)  # 计算每一列平方的均值
    scores = np.sqrt(mean_squares)  # 均方根

    # 使用softmax归一化关联分数
    softmax_scores = normalize(scores)
    softmax_scores = softmax_scores.reshape(1, len_haptic)
    # softmax_scores = scores
    # softmax_scores = softmax_scores.reshape(1, len_haptic)
    # 计算要添加的值
    new_haptic_vec = haptic_vec.reshape(len_haptic, 1)
    add_value = np.matmul(softmax_scores, new_haptic_vec)
    # 将加权求和值加到向量的末尾
    result_vector = np.append(image_vec, add_value)
    return result_vector


def attention_haptic2visual_graph_update(G_image, G_haptic):
    haptic_vec = []
    for node_haptic in G_haptic.nodes():
        current_value_haptic = G_haptic.nodes[node_haptic]['value']
        haptic_vec.append(current_value_haptic)
    haptic_vec = np.array(haptic_vec)
    # 遍历所有节点
    for node_image in G_image.nodes():
        current_value_image = G_image.nodes[node_image]['value']

        new_vec = attention_haptic2visual(current_value_image, haptic_vec)
        G_image.nodes[node_image]['value'] = new_vec
    return G_image
#
#
# # 图像数据生成图结构
# G_image, data_image = image_to_graph.image2graph("./数据/3.jpg")
# # 触觉数据生成图结构
# G_haptic, data_haptic = haptic_to_graph.haptic2graph("./数据/G1EpoxyRasterPlate_Movement_X_test1.txt", 1000, 1500)
# G = attention_haptic2visual_graph_update(G_image, G_haptic)
# for node in G.nodes():
#     current_value = G.nodes[node]['value']
#     print(current_value)
