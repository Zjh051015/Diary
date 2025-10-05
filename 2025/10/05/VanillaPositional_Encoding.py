import numpy as np
import matplotlib.pyplot as plt

def get_positional_encoding(max_len:int,d_model:int):
    """
    计算位置编码矩阵。

    参数:
    max_len (int): 序列的最大长度。
    d_model (int): 模型的维度 (必须是偶数)。

    返回:
    np.ndarray: 形状为 (max_len, d_model) 的位置编码矩阵。
    """
    ndarray=np.random.rand(max_len,d_model)
    for pos in range(max_len):
        for i in range(0,d_model,2):
            ndarray[pos,i]=np.sin(pos/10000**(i/d_model))
            ndarray[pos,i+1]=np.cos(pos/10000**(i/d_model)) 
    return ndarray
pe_matrix = get_positional_encoding(max_len=100, d_model=32)
print("生成的PE矩阵形状:", pe_matrix.shape)
assert pe_matrix.shape == (100, 32)
position_to_plot = [0,5,10,15,20,30]
plt.figure(figsize=(10,6))
for i,pos in enumerate(position_to_plot):
    if pos < pe_matrix.shape[0]:
        plt.subplot(3,2,i+1)
        plt.plot(pe_matrix[:,pos], label=f'Position {pos}')
        plt.legend()
plt.suptitle('Positional Encoding for Different Positions')
plt.show()
pos =10 
k=3 
i=4
d_model=32
vec_pos =np.array([pe_matrix[pos,2*i],
                  pe_matrix[pos,2*i+1]])

vec_pos_k =np.array([pe_matrix[pos+k,2*i],
                  pe_matrix[pos+k,2*i+1]])

omega = 1/10000**(2*i/d_model)
theta = k * omega

R = np.array([
    [np.cos(theta),   np.sin(theta)],
    [-np.sin(theta),   np.cos(theta)]
])

vec_transformed = R @ vec_pos 

print(f"原始:{vec_pos_k} ")
print(f"旋转后:{vec_transformed}" )

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

vec_10 = pe_matrix[10, :]
vec_11 = pe_matrix[11, :]
vec_80 = pe_matrix[80, :]

cosine_similarity_10_11 = cosine_similarity(vec_10, vec_11) 

cosine_similarity_10_80 = cosine_similarity(vec_10, vec_80)

print(f"10和11的余弦相似度: {cosine_similarity_10_11:.4f}")
print(f"10和80的余弦相似度: {cosine_similarity_10_80:.4f}")