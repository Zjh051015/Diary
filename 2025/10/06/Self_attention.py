import numpy as np
import math

# --- 准备数据 ---
# 假设序列长度为 4, 向量维度为 3
seq_len = 4
d_model = 3

# 假设我们已经通过权重矩阵 Wq, Wk, Wv 计算出了 Q, K, V
# 我们用随机数来模拟它们
Q = np.random.rand(seq_len, d_model)
K = np.random.rand(seq_len, d_model)
V = np.random.rand(seq_len, d_model)

print("Q 的形状:", Q.shape)
print("K 的形状:", K.shape)
print("V 的形状:", V.shape)
print("-" * 20)

# --- 你的代码框架 ---

def scaled_dot_product_attention(Q, K, V):
    """
    计算缩放点积注意力。

    参数:
    Q -- 查询矩阵，形状为 (seq_len_q, d_k)
    K -- 键矩阵，形状为 (seq_len_k, d_k)
    V -- 值矩阵，形状为 (seq_len_v, d_v)
         注意: seq_len_k 必须等于 seq_len_v

    返回:
    output -- 输出矩阵
    attn_weights -- 注意力权重矩阵
    """

    # 第 1 步：计算分数 (Scores = Q * K_transpose)
    # 提示：K 的维度是 (seq_len_k, d_k)，你需要它的转置。
    #      numpy 的矩阵乘法可以使用 @ 运算符。
    #      numpy 的转置可以使用 .T 属性。
    Scores = Q @ K.T
    
    
    # 第 2 步：缩放 (Scale)
    # 提示：首先需要获取键向量的维度 d_k。
    #      d_k 可以从 K 矩阵的形状中得到。
    #      然后用分数除以 d_k 的平方根。
    # [ 在这里写你的代码 ]
    d_k = K.shape[1]
    scaled_Scores = Scores/math.sqrt(d_k)
    # 第 3 步：归一化 (Softmax)
    # 提示：Softmax(x)_i = exp(x_i) / sum(exp(x))
    #      你需要对缩放后的分数矩阵的 *每一行* 进行 softmax。
    #      np.exp() 可以计算指数。
    #      np.sum() 可以计算和，注意设置 axis 参数，保证是按行求和。
    #      为了让除法正确广播，求和时使用 keepdims=True 是个好习惯。

    # [ 在这里写你的 code ]
    exp_scaled_Scores = np.exp(scaled_Scores)
    row_sum = np.sum(exp_scaled_Scores,axis=1,keepdims=1)
    attn_weights = exp_scaled_Scores/row_sum
    
    # 第 4 步：加权求和 (Multiply by V)
    # 提示：用上一步得到的权重矩阵乘以 V 矩阵。

    # [ 在这里写你的 code ]
    output = attn_weights@V 
    return output, attn_weights


# --- 测试你的函数 ---
output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("注意力权重矩阵的形状:", attn_weights.shape)
print("最终输出的形状:", output.shape)

# 理想情况下，attn_weights 的形状应该是 (4, 4)
# output 的形状应该是 (4, 3)