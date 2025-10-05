# ==============================================================================
#           深入探索位置编码：“毕业”答卷
#
# 请在标记为 "# --- 在这里开始您的回答 ---" 的区域填写您的代码和思考。
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- 预备代码 (您在上一任务中已完成，无需修改) ---

def get_original_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """计算原始的香草位置编码矩阵。"""
    if d_model % 2 != 0:
        raise ValueError("d_model must be an even number.")
    
    pe = np.zeros((max_len, d_model))
    position = np.arange(max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe

print("=" * 60)
print("答卷开始！祝您好运！")
print("=" * 60)

# ==============================================================================
# 题目一：修改与对比 (Code & Analysis)
# ==============================================================================

print("\n--- 题目一：修改与对比 ---\n")

def get_flexible_positional_encoding(max_len: int, d_model: int, mode: str = 'add') -> np.ndarray:
    """
    一个更灵活的位置编码函数。

    参数:
    max_len (int): 序列的最大长度。
    d_model (int): 模型的维度。
    mode (str): 'add' 或 'concat'。

    返回:
    np.ndarray: 根据模式返回不同形状的位置编码矩阵。
    """
    # --- 在这里开始您的回答 (第一部分：代码实现) ---

    if mode == 'add':
        # 当模式为 'add' 时，行为与原始函数相同。
        # 您可以直接调用已有的函数，或者重新实现。
        # 注意：需要处理 d_model 为奇数的情况。
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number for 'add' mode.")
        return get_original_positional_encoding(max_len, d_model)
        
    elif mode == 'concat':
        # 当模式为 'concat' 时，返回 d_model/2 维的编码。
        # 您需要决定如何处理 d_model 为奇数的情况。
        # 例如，可以向上或向下取整。]
        if d_model % 2 != 0:
            raise ValueError("d_model must be an even number for 'concat' mode.")
        output_dim = d_model // 2
        # 我们需要一个d_model至少为2的偶数来生成有意义的sin/cos对
        # 为了生成 output_dim 的输出，我们需要一个至少 2*output_dim 的内部d_model
        internal_d_model = output_dim * 2

        pe_full = get_original_positional_encoding(max_len, internal_d_model)

        pe_final = np.zeros((max_len,output_dim))

        pe_final[:,0::1]=pe_full[:,0::2] + pe_full[:,1::2]
        # 返回前一半的维度
        return pe_final

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'add' or 'concat'.")

    # --- 回答结束 ---

# 测试您的函数
print("测试 'add' 模式 (d_model=64):")
pe_add = get_flexible_positional_encoding(100, 64, mode='add')
print(f"  - 输出形状: {pe_add.shape}")
assert pe_add.shape == (100, 64)

print("测试 'concat' 模式 (d_model=64 -> 输出32):")
pe_concat = get_flexible_positional_encoding(100, 64, mode='concat')
print(f"  - 输出形状: {pe_concat.shape}")
assert pe_concat.shape == (100, 32)

print("函数实现初步测试通过！\n")

print("--- 思考与分析 ---")
# --- 在这里开始您的回答 (第二部分：文字思考) ---
#
# 问题 1.1: 如果我们将 d_model/2 维的词嵌入和 d_model/2 维的位置编码 **拼接** 起来，
#           形成一个 d_model 维的最终输入，这样做有什么潜在的 **优点** 和 **缺点**？
#
# 您的回答:
# 优点:
# 1. 位置信息更鲜明: 通过拼接，位置信息不会被词嵌入信息稀释，模型可以更清晰地识别位置信息。
# 2. 不知道
#
# 缺点:
# 1. 信息减半词嵌入的维度减半，可能导致词语信息的表达能力下降。
# 2. 位置融合更割裂 拼接方式可能导致模型难以有效融合词语和位置信息。
# 3. 可能需要为此调节编码
#
#
# 问题 1.2: 与原始的 **相加** 方案相比，你认为哪种方案在理论上更优？为什么？
#
# 您的回答:
# 原始相加方案 融合更自然 
#
# --- 回答结束 ---


# ==============================================================================
# 题目二：实现可学习的位置编码 (Implementation)
# ==============================================================================

print("\n--- 题目二：实现可学习的位置编码 ---\n")

class LearnedPositionalEncoding:
    """
    一个模拟可学习位置编码的类。
    """
    def __init__(self, max_len: int, d_model: int):
        """
        初始化方法。

        参数:
        max_len (int): 序列的最大长度。
        d_model (int): 模型的维度。
        """
        # --- 在这里开始您的回答 ---
        
        # 提示: 创建一个形状为 (max_len, d_model) 的 Numpy 数组
        #       并用随机数填充它 (例如 np.random.randn)。
        self.embedding_matrix = np.random.randn(max_len,d_model)
        
        # --- 回答结束 ---

    def forward(self, pos: int) -> np.ndarray:
        """
        前向传播方法，查询特定位置的编码。

        参数:
        pos (int): 要查询的位置索引。

        返回:
        np.ndarray: 形状为 (d_model,) 的位置编码向量。
        """
        if not (0 <= pos < self.embedding_matrix.shape[0]):
            raise IndexError(f"Position {pos} is out of the valid range [0, {self.embedding_matrix.shape[0]-1}]")
            
        # --- 在这里开始您的回答 ---

        # 提示: 从 self.embedding_matrix 中返回第 pos 行。
        
        return self.embedding_matrix[pos]

        # --- 回答结束 ---

# 测试您的类
print("测试 LearnedPositionalEncoding:")
learned_pe = LearnedPositionalEncoding(max_len=50, d_model=128)

# 检查矩阵是否已创建
assert learned_pe.embedding_matrix is not None and learned_pe.embedding_matrix.shape == (50, 128)
print("  - 嵌入矩阵创建成功，形状正确。")

# 测试查询功能
pos_5_embedding = learned_pe.forward(5)
pos_10_embedding = learned_pe.forward(10)

print(f"  - 查询位置 5 的编码，形状: {pos_5_embedding.shape}")
assert pos_5_embedding.shape == (128,)

# 检查不同位置的编码是否不同
assert not np.array_equal(pos_5_embedding, pos_10_embedding)
print("  - 不同位置返回了不同的编码。")
print("类实现初步测试通过！\n")


# ==============================================================================
# 题目三：思辨与设计 (Critical Thinking)
# ==============================================================================

print("\n--- 题目三：思辨与设计 ---\n")

# --- 在这里开始您的回答 (文字思考) ---
#
# 问题 3.1: 比较 **固定位置编码 (香草)** 和 **可学习位置编码**，
#           总结它们最核心的 **三个区别**。
#
# 您的回答:
# 区别一 (例如，关于参数):
#固定位置编码无参数可学习 
#可学习位置编码含有maxlen*d_model个可学习参数 倘若训练集较大 我认为可能学习到更适合任务的编码
# 区别二 (例如，关于泛化性):
#可学习位置编码在训练数据上表现更好，但在未见过的序列长度上可能泛化较差。
#
# 区别三 (例如，关于内置结构):
#固定编码：具有强大的内置结构。它最重要的结构就是我们反复讨论的“相对位置可以通过线性变换（旋转）表示”。这个先验知识被硬编码进了模型里，帮助模型轻松学习相对位置关系。
#可学习编码：没有任何内置结构。它就是一个“白板”，完全依赖模型从数据中自己去发现位置与位置之间的关系。理论上，如果数据足够多，模型也许能学会类似固定编码那样的相对关系，但这无疑更加困难。
#
#
# 问题 3.2: 假设您正在处理一个**音频信号**的任务，输入的是一段长达几分钟的
#           原始音频波形 (序列长度可能非常非常长)。在这种场景下，
#           您会选择 **固定位置编码** 还是 **可学习位置编码**？请详细阐述您的理由。
#
# 您的回答:
# 我的选择是:固定位置编码
#
# 理由:
# 1. 训练更快 固定位置编码不需要学习额外的参数，训练过程更快。
# 2. 长度易于改变 固定位置编码可以轻松适应不同长度的输入序列。
# 3. 
#
# --- 回答结束 ---


print("\n" + "=" * 60)
print("答卷结束！感谢您的思考与努力！")
print("=" * 60)