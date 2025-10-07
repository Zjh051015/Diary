import random
from datetime import date, timedelta

# --- 1. 定义词汇表 ---
# 输入字符集 (Source)
INPUT_CHARS = '0123456789-'
# 输出字符集 (Target)
OUTPUT_CHARS = 'abcdefghijklmnopqrstuvwxyz ,0123456789'

# 添加特殊标记
PAD_TOKEN = '<pad>' # 填充符
SOS_TOKEN = '<sos>' # 句子开始
EOS_TOKEN = '<eos>' # 句子结束

# 合并词汇表
INPUT_VOCAB = [PAD_TOKEN] + list(INPUT_CHARS)
OUTPUT_VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(OUTPUT_CHARS)

# 创建字符到索引的映射
input_char_to_idx = {char: i for i, char in enumerate(INPUT_VOCAB)}
output_char_to_idx = {char: i for i, char in enumerate(OUTPUT_VOCAB)}

# --- 2. 数据生成函数 ---
def generate_random_date_pair():
    """随机生成一个 (输入格式, 输出格式) 的日期对"""
    # 随机生成一个日期
    start_date = date(1980, 1, 1)
    end_date = date(2023, 12, 31)
    random_days = random.randint(0, (end_date - start_date).days)
    random_date = start_date + timedelta(days=random_days)
    
    # 输入格式: YYYY-MM-DD
    input_str = random_date.strftime("%Y-%m-%d")
    
    # 输出格式: Month Day, Year (e.g., "october 26, 2023")
    output_str = random_date.strftime("%B %d, %Y").lower()
    
    return input_str, output_str

# --- 3. 生成数据集 ---
def create_dataset(num_examples):
    dataset = []
    for _ in range(num_examples):
        dataset.append(generate_random_date_pair())
    return dataset

# 生成1000个样本
dataset = create_dataset(1000)

# 打印一些样本看看
print("--- 数据样本 ---")
for i in range(5):
    print(f"输入: '{dataset[i][0]}'  ->  输出: '{dataset[i][1]}'")

print("\n--- 词汇表示例 ---")
print(f"输入词汇表大小: {len(INPUT_VOCAB)}")
print(f"输出词汇表大小: {len(OUTPUT_VOCAB)}")
print(f"'{SOS_TOKEN}' 在输出词典中的索引是: {output_char_to_idx[SOS_TOKEN]}")