import numpy as np


def attention_haptic2visual(image_vec, haptic_vec):
    len_haptic = len(haptic_vec)
    len_image = len(image_vec)
    # 计算关联分数（使用softmax回归）
    image_vec = image_vec.reshape(len_image, 1)
    haptic_vec = haptic_vec.reshape(1, len_haptic)
    arr = np.matmul(image_vec, haptic_vec)
    # 计算每列的模，得到关联分数
    scores = np.linalg.norm(arr, axis=0)
    # 使用softmax归一化关联分数
    softmax_scores = np.exp(scores) / np.sum(np.exp(scores))
    softmax_scores = softmax_scores.reshape(1, len_haptic)
    # 计算要添加的值
    new_haptic_vec = haptic_vec.reshape(len_haptic, 1)
    add_value = np.matmul(softmax_scores, new_haptic_vec)
    print("add_value:", add_value)

    # 将加权求和值加到向量的末尾
    result_vector = np.append(image_vec, add_value)

    return result_vector


# 输入向量和浮点值
input_vector = np.random.rand(7500)
input_scores = np.random.rand(100)

# 计算加权求和后的向量
result = attention_haptic2visual(input_vector, input_scores)

print("结果向量维度:", result.shape)
print("结果向量:", result)
