import numpy as np

def softmax_regression(vector):
    exp_vector = np.exp(vector)  # 对向量进行指数运算
    softmax_output = exp_vector / np.sum(exp_vector)  # 对指数结果进行归一化
    return softmax_output

# 示例向量
vector = np.random.randn(117)  # 随机生成一个117维向量

softmax_result = softmax_regression(vector)
print(softmax_result)
