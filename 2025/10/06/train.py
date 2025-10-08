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

PAD_IDX = output_char_to_idx[PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
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

def train_step(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    target_length = target_tensor.shape[1]
    batch_size = input_tensor.shape[0]

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_tensor)

    sos_token_idx = output_char_to_idx[SOS_TOKEN]
    decoder_input = torch.tensor([sos_token_idx] * batch_size, dtype=torch.long)
    decoder_hidden = encoder_hidden

    loss = 0

    for t in range(target_length):
         decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
         loss += criterion(decoder_output, target_tensor[:, t])
         decoder_input = target_tensor[:, t]

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

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

def get_batches(dataset,batch_size):
    random.shuffle(dataset)
    for i in range (0,len(dataset),batch_size):
        batch = dataset[i:i+batch_size]
        yield batch

def batch_to_tensors(batch,input_char_to_index,output_char_to_index):
    
    #输入转化为tensor
    input_strings,target_strings= zip(*batch)
    input_indices = [[input_char_to_index[c]for c in s]for s in input_strings]
    input_max_len = max(len(indices)for indices  in input_indices)
    padded_inputs = [indices + [PAD_IDX]*(input_max_len-len(indices))for indices in input_indices]
    input_tensor = torch.tensor(padded_inputs,dtype=torch.long)

    #输出转化为tensor
    target_indices = [[output_char_to_index[c] for c in s] + [output_char_to_index[EOS_TOKEN]] for s in target_strings]
    target_max_len = max(len(indices) for indices in target_indices)
    padded_targets = [indices + [PAD_IDX] * (target_max_len - len(indices)) for indices in target_indices]
    target_tensor = torch.tensor(padded_targets, dtype=torch.long)

    return input_tensor, target_tensor


dataset = create_dataset(1000) # 生成1000个样本

# 定义训练超参数
n_epochs = 10 # 我们打算把整个数据集训练10遍
print_every = 100 # 每训练100个样本，我们就打印一次进度

batch_size = 32 

print("开始训练...")

evaluation_samples = [
    ("1995-05-23", "may 23, 1995"),
    ("2008-11-01", "november 01, 2008"),
    ("1988-12-31", "december 31, 1988")
]


for epoch in range(1, n_epochs + 1):
    total_loss = 0
    num_batches = 0

    encoder.train()
    decoder.train()

    # 使用新的批次生成器
    for batch in get_batches(dataset, batch_size):
        # 1. 将批次数据转换为张量
        input_tensor, target_tensor = batch_to_tensors(batch, input_char_to_idx, output_char_to_idx)
        
        # 2. 调用修改后的 train_step
        step_loss = train_step(
            input_tensor,
            target_tensor,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion
        )
        num_batches += 1
        # 3. 累加损失
        total_loss += step_loss # <-- 填充这里

        # 4. 阶段性打印进度
        avg_loss = total_loss / num_batches
        print(f"--- Epoch {epoch} 完成, 平均损失: {avg_loss:.4f} ---")

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