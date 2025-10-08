import torch
import torch.nn as nn
import torch.optim as optim
import random

from datatrans_model import EncoderRNN, AttentionDecoderRNN
from build_day import INPUT_VOCAB, OUTPUT_VOCAB, input_char_to_idx, output_char_to_idx, create_dataset, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN

# --- 1. 定义超参数 ---
INPUT_VOCAB_SIZE = len(INPUT_VOCAB)
OUTPUT_VOCAB_SIZE = len(OUTPUT_VOCAB)
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 128
learning_rate = 0.001

# --- 2. 实例化组件 ---
encoder = EncoderRNN(INPUT_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)
decoder = AttentionDecoderRNN(OUTPUT_VOCAB_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE)

criterion = nn.CrossEntropyLoss()
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

def string_to_tensor(s_or_list, char_to_idx):
    """将字符串或token列表转换为张量"""
    if isinstance(s_or_list, str):
        tokens = list(s_or_list)
    else:
        tokens = s_or_list 

    indices = [char_to_idx[token] for token in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

def train_step(input_string, target_string, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # --- 1. 数据准备 ---
    # 使用我们刚刚完成的 string_to_tensor 函数，将字符串转换为张量
    input_tensor = string_to_tensor(input_string, input_char_to_idx)
    target_tokens = list(target_string) + [EOS_TOKEN]
    target_tensor = string_to_tensor(target_tokens, output_char_to_idx)
    
    # 获取目标序列的长度，解码器需要循环这么多次
    target_length = target_tensor.shape[1]

    # --- 2. 清空梯度 ---
    # 在每次训练开始前，必须清空上一次留下的梯度信息
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # --- 3. 编码器前向传播 ---
    # 将整个输入序列喂给编码器，得到所有时间步的输出和最后的隐藏状态
    encoder_outputs, encoder_hidden = encoder(input_tensor)
    
    # --- 4. 解码器前向传播 (核心部分) ---
    # a. 解码器的第一个输入是特殊的 <sos> 字符。
    #    我们需要创建一个只包含 <sos> 索引的张量。
    #    它的形状应该是 (1, 1)，因为批次大小是1，序列长度也是1。
    sos_token_idx = output_char_to_idx[SOS_TOKEN]
    decoder_input = torch.tensor([sos_token_idx], dtype=torch.long) # <-- 填充这里
    # b. 解码器的第一个隐藏状态是编码器最后的隐藏状态。
    decoder_hidden = encoder_hidden
    
    loss = 0 # 初始化损失
    
    # 我们需要一个循环，一步步地生成输出序列
    for t in range(target_length):
        # c. 让解码器工作一步
        decoder_output, decoder_hidden, _ = decoder(
            decoder_input, 
            decoder_hidden, 
            encoder_outputs
        )
            
        # d. 计算这一步的损失，并累加
        loss+= criterion(decoder_output, target_tensor[:, t])
        
        # e. "教师强制"：无论解码器自己预测了什么，下一步的输入都强制使用“标准答案”
        decoder_input = target_tensor[:, t]
    
    # --- 5. 反向传播与优化 ---
    # a. 根据总损失计算梯度
    loss.backward()
    
    # b. 让优化器根据梯度更新模型参数
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    # 返回平均损失
    return loss.item() / target_length

output_idx_to_char = {i: char for char, i in output_char_to_idx.items()}

def evaluate(encoder, decoder, input_string, max_length=30):
    """
    评估函数，用于看模型对单个输入的预测结果。
    """
    # 1. 进入评估模式 & 关闭梯度计算
    #    encoder.eval() 和 decoder.eval() 是告诉模型“现在是测试时间”。
    #    torch.no_grad() 会暂时关闭所有梯度计算，这能大大加快速度并节省内存。
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # 2. 准备输入数据 (和 train_step 一样)
        input_tensor = string_to_tensor(input_string, input_char_to_idx)
        
        # 3. 编码器前向传播 (和 train_step 一样)
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # 4. 解码器初始设置 (和 train_step 一样)
        sos_token_idx = output_char_to_idx[SOS_TOKEN]
        decoder_input = torch.tensor([sos_token_idx], dtype=torch.long)
        decoder_hidden = encoder_hidden
        
        # 用于存储解码器输出的词元
        decoded_tokens = []

        # 5. 解码器循环 (核心区别在这里)
        for _ in range(max_length):
            # a. 进行一步解码
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input,
                decoder_hidden,
                encoder_outputs
            )

            # b. 获取概率最高的那个词元的索引
            #    decoder_output 的形状是 (1, vocab_size)
            #    torch.topk(..., 1) 会返回最高分的那个值和它的索引
            _, top_index = torch.topk(decoder_output, 1)
            predicted_token_idx = top_index.item()

            # c. 检查是否是句子结束符
            if predicted_token_idx == output_char_to_idx[EOS_TOKEN]:
                break  # 如果是，就结束生成
            
            # d. 将预测的词元加入结果列表
            decoded_tokens.append(output_idx_to_char[predicted_token_idx])

            # e. 关键！将当前预测的词元作为下一步的输入 (没有教师强制)
            decoder_input = torch.tensor([predicted_token_idx], dtype=torch.long)
            
        # 6. 将 token 列表转换回字符串
        return "".join(decoded_tokens)


dataset = create_dataset(1000) # 生成1000个样本

# 定义训练超参数
n_epochs = 10 # 我们打算把整个数据集训练10遍
print_every = 100 # 每训练100个样本，我们就打印一次进度

print("开始训练...")

evaluation_samples = [
    ("1995-05-23", "may 23, 1995"),
    ("2008-11-01", "november 01, 2008"),
    ("1988-12-31", "december 31, 1988")
]


for epoch in range(1, n_epochs + 1):
    total_loss = 0 # 每个周期开始前，将总损失清零

    encoder.train()
    decoder.train()
    random.shuffle(dataset) # (可选但推荐)

    # 遍历数据集中的每一个样本
    for i in range(len(dataset)):
        # 1. 从数据集中获取当前的输入和目标字符串
        #    dataset[i] 是一个元组 (input_str, target_str)
        input_string = dataset[i][0]
        target_string =dataset[i][1]# <-- 填充这里

        # 2. 调用 train_step 函数进行单步训练，并得到损失
        step_loss = train_step(
            input_string,
            target_string,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion
        )
        
        # 3. 累加损失
        total_loss += step_loss # <-- 填充这里

        # 4. 阶段性打印进度
        if (i + 1) % print_every == 0:
            avg_loss = total_loss / (i + 1)
            print(f"    Epoch [{epoch}/{n_epochs}], Iter [{i+1}/{len(dataset)}], Avg Loss: {avg_loss:.4f}")

    # 5. 在每个周期结束后，可以再打印一次该周期的最终平均损失
    final_epoch_loss = total_loss / len(dataset)
    print(f"--- Epoch {epoch} 完成, 平均损失: {final_epoch_loss:.4f} ---")

    print(f"--- Epoch {epoch} 完成, 平均损失: {total_loss / len(dataset):.4f} ---")
    print("开始评估...")
    
    for input_str, target_str in evaluation_samples:
        prediction = evaluate(encoder, decoder, input_str)
        print(f"  输入: {input_str}")
        print(f"  标准答案: {target_str}")
        print(f"  模型预测: {prediction}")
        print("-" * 20)
        
print("训练完成！")


# --- 测试一下你的函数 ---
# test_input_str = "1999-12-31"
# input_tensor = string_to_tensor(test_input_str, input_char_to_idx)

# test_target_str = list("december 31, 1999") + [EOS_TOKEN] 
# target_tensor = string_to_tensor(test_target_str, output_char_to_idx)

# print(f"输入字符串 '{test_input_str}' 转换后的张量形状: {input_tensor.shape}")
# print(f"目标字符串转换后的张量形状: {target_tensor.shape}")