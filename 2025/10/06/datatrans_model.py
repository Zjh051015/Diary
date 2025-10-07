import torch
import torch.nn as nn
import math

# ==============================================================================
# 编码器模块 (Encoder)
# 职责: "阅读" 输入序列，并将其压缩成一系列上下文丰富的隐藏状态。
# ==============================================================================
class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        """
        初始化编码器.
        :param input_size: 输入词汇表的大小.
        :param embedding_size: 词嵌入向量的维度.
        :param hidden_size: RNN 隐藏状态的维度.
        """
        super().__init__()
        self.hidden_size = hidden_size
        
        # 零件1: 嵌入层。将数字索引转换为向量。
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # 零件2: RNN层。处理向量序列。
        self.rnn = nn.RNN(embedding_size, hidden_size, batch_first=True)

    def forward(self, input_tensor):
        """
        定义数据如何流过编码器.
        :param input_tensor: 输入的索引张量，形状 (batch_size, seq_len).
        :return: RNN所有时间步的输出，以及最后一个时间步的隐藏状态。
        """
        # 1. 将索引转换为词嵌入向量
        #    形状变化: (batch, seq_len) -> (batch, seq_len, embedding_size)
        embedded = self.embedding(input_tensor)
        
        # 2. 将嵌入向量序列送入RNN
        #    outputs 形状: (batch, seq_len, hidden_size)  <-- "原文笔记"
        #    hidden 形状: (1, batch, hidden_size)        <-- "总结摘要"
        outputs, hidden = self.rnn(embedded)
        
        return outputs, hidden

# ==============================================================================
# 注意力模块 (Attention)
# 职责: 根据一个查询(Query)，计算出一组权重，并从一系列值(Values)中提取信息。
# ==============================================================================
class Attention(nn.Module):
    def __init__(self):
        """
        初始化注意力模块. 这是一个纯计算模块，没有可学习的权重。
        """
        super().__init__()
        # 零件1: Softmax层，用于将分数转换为概率分布。
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, keys, values):
        """
        定义注意力计算的流程.
        :param query: 查询向量, 形状 (batch, hidden_size).
        :param keys: 键矩阵 (来自编码器), 形状 (batch, seq_len, hidden_size).
        :param values: 值矩阵 (来自编码器), 形状 (batch, seq_len, hidden_size).
        :return: 上下文向量 (context) 和注意力权重 (attn_weights).
        """
        # 1. 升维 query 以便进行矩阵乘法
        #    形状变化: (batch, hidden_size) -> (batch, 1, hidden_size)
        query_unsqueezed = query.unsqueeze(1)

        # 2. 计算分数
        #    keys 转置后形状: (batch, hidden_size, seq_len)
        #    分数 scores 形状: (batch, 1, seq_len)
        scores = query_unsqueezed @ keys.transpose(1, 2)

        # 3. 缩放分数
        d_k = keys.size(-1)
        scaled_scores = scores / math.sqrt(d_k)

        # 4. 计算注意力权重
        #    形状: (batch, 1, seq_len)
        attention_weight = self.softmax(scaled_scores)

        # 5. 加权求和得到上下文向量
        #    context 形状: (batch, 1, hidden_size)
        context = attention_weight @ values

        return context, attention_weight

# ==============================================================================
# 带注意力的解码器模块 (Attention Decoder)
# 职责: 一步步生成输出序列，每一步都使用注意力机制来"回看"输入。
# ==============================================================================
class AttentionDecoderRNN(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        """
        初始化解码器.
        :param output_size: 输出词汇表的大小.
        :param embedding_size: 词嵌入向量的维度.
        :param hidden_size: RNN 隐藏状态的维度.
        """
        super().__init__()

        # 零件1: 嵌入层。
        self.embedding = nn.Embedding(output_size, embedding_size)

        # 零件2: 注意力层 (我们自己写的模块)。
        self.attention = Attention()

        # 零件3: RNN层。输入是"上一步的嵌入"和"注意力的上下文"拼接而成。
        rnn_input_size = embedding_size + hidden_size
        self.rnn = nn.RNN(rnn_input_size, hidden_size, batch_first=True)

        # 零件4: 输出层。将RNN的思考结果映射到词汇表分数。
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, decoder_hidden, encoder_outputs):
        """
        定义数据如何流过解码器 (仅一个时间步).
        :param decoder_input: 上一步输出的字符索引, 形状 (batch_size).
        :param decoder_hidden: 上一步的隐藏状态, 形状 (1, batch, hidden_size).
        :param encoder_outputs: 编码器的所有输出, 形状 (batch, seq_len, hidden_size).
        :return: 最终的词汇表分数, 新的隐藏状态, 注意力权重。
        """
        # 1. 处理输入，升维并嵌入
        #    形状变化: (batch) -> (batch, 1) -> (batch, 1, embedding_size)
        embedded = self.embedding(decoder_input.unsqueeze(1))
        
        # 2. 计算注意力
        #    Query 是解码器上一时刻的隐藏状态
        query = decoder_hidden.squeeze(0)
        context, attn_weights = self.attention(query, encoder_outputs, encoder_outputs)

        # 3. 拼接输入
        #    形状: (batch, 1, embedding_size + hidden_size)
        rnn_input = torch.cat([embedded, context], dim=2)
        
        # 4. 将拼接后的输入和隐藏状态送入RNN
        #    new_hidden 形状: (1, batch, hidden_size)
        _, new_hidden = self.rnn(rnn_input, decoder_hidden)

        # 5. 将RNN的输出(新的隐藏状态)送入线性层得到最终分数
        #    output 形状: (batch, output_size)
        output = self.out(new_hidden.squeeze(0))

        return output, new_hidden, attn_weights


# ==============================================================================
# (可选) 实例化并打印模型结构，检查是否正确
# ==============================================================================

# 假设我们已经有了上一节的数据准备代码
# (为了让文件能独立运行，我们在这里伪造一下)
INPUT_VOCAB_SIZE = 12
OUTPUT_VOCAB_SIZE = 41
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128

try:
    print("--- 编码器结构 ---")
    encoder = EncoderRNN(INPUT_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    print(encoder)
    print("\n")
    
    print("--- 解码器结构 ---")
    decoder = AttentionDecoderRNN(OUTPUT_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
    print(decoder)

except Exception as e:
    print("代码存在错误:", e)