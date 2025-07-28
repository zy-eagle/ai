import numpy as np

def softmax(z):
    """计算Softmax函数的值"""
    # 为了数值稳定性，减去最大值
    z = z - np.max(z)
    print("调整后的输入:", z)

    # 计算指数和归一化
    exp_z = np.exp(z)
    print("指数值:", exp_z)

    return exp_z / np.sum(exp_z)

# 示例使用
logits = np.array([3.0, 1.0, 0.2])
probabilities = softmax(logits)
print("原始输入:", logits)
print("Softmax输出:", probabilities)
print("概率和:", np.sum(probabilities))