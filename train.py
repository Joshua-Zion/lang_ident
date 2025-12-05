import pandas as pd
import fasttext
import os
from sklearn.model_selection import train_test_split

# --- 配置 ---
CSV_FILE = 'data/lang_data.csv' # 确保你的CSV文件名为此
TRAIN_FILE = 'lid_train.txt'
TEST_FILE = 'lid_test.txt'
MODEL_PATH = 'model/lid_model.bin'
# FastText 关键参数 (推荐用于语种检测的设置)
FT_DIM = 50      # 词向量维度
FT_EPOCH = 25    # 训练轮数
FT_LR = 0.1      # 学习率
FT_MINN = 3      # 最小字符 n-gram
FT_MAXN = 6      # 最大字符 n-gram
TEST_SIZE = 0.1  # 10% 数据用于测试

print(f"--- 🚀 FastText 语种检测开始 ---")

# ===============================================
# 1. 数据加载、清理与格式转换
# ===============================================

print("1. 正在加载、清理和转换数据...")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"错误：找不到文件 {CSV_FILE}。请检查文件路径。")
    exit()

# 清理标签和文本两端的空格
df['labels'] = df['labels'].astype(str).str.strip()
df['text'] = df['text'].astype(str).str.strip()

# 转换为 FastText 格式：__label__language_code text
# 使用单个空格作为标签和文本之间的分隔符
# 替换文本中的换行符/回车，避免破坏文件结构
df['fasttext_format'] = '__label__' + df['labels'] + ' ' + \
                        df['text'].astype(str).str.replace(r'[\r\n]+', ' ', regex=True)

# 划分训练集和测试集
train_data, test_data = train_test_split(
    df['fasttext_format'],
    test_size=TEST_SIZE,
    random_state=42
)

# 使用 Python 内建的写入方式，确保格式准确
print(f"   - 写入训练文件: {TRAIN_FILE}")
with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(item + '\n')

print(f"   - 写入测试文件: {TEST_FILE}")
with open(TEST_FILE, 'w', encoding='utf-8') as f:
    for item in test_data:
        f.write(item + '\n')

print(f"   - 训练集大小: {len(train_data)}")
print(f"   - 测试集大小: {len(test_data)}")


# ===============================================
# 2. 模型训练与保存
# ===============================================

print("\n2. 正在训练 FastText 模型...")

# 检查训练文件是否为空，以防万一
if os.path.getsize(TRAIN_FILE) == 0:
    print(f"错误：训练文件 {TRAIN_FILE} 为空。请检查原始数据。")
    exit()

model = fasttext.train_supervised(
    input=TRAIN_FILE,
    dim=FT_DIM,
    epoch=FT_EPOCH,
    lr=FT_LR,
    minn=FT_MINN,
    maxn=FT_MAXN,
    loss='softmax'
)

# 保存模型
model.save_model(MODEL_PATH)
print(f"   - 模型训练完成并保存到 {MODEL_PATH}")

# ===============================================
# 3. 模型评估
# ===============================================

print("\n3. 正在评估模型性能...")

# FastText test 方法返回 (N, P@1, R@1)
results = model.test(TEST_FILE)

print("--- 评估结果 (Top-1) ---")
print(f"   - 测试样本数 (N): {results[0]}")
print(f"   - 准确率 (Precision @ 1): {results[1]:.4f}")
print(f"   - 召回率 (Recall @ 1): {results[2]:.4f}")
print("--------------------------")


# ===============================================
# 4. 预测演示
# ===============================================

print("\n4. 预测演示:")
texts_to_predict = [
    "人工智能是未来的方向。",
    "The best way to predict the future is to invent it.",
    "La vie est belle."
]

predictions = model.predict(texts_to_predict, k=1)

for text, (labels, probs) in zip(texts_to_predict, zip(*predictions)):
    # 增加检查，确保列表不为空，防止IndexError
    if labels:
        lang_code = labels[0].replace('__label__', '')
        probability = probs[0]
        print(f"   - 文本: '{text[:20]}...'")
        print(f"     -> 预测语言: **{lang_code}** (置信度: {probability:.4f})")
    else:
        print(f"   - 文本: '{text[:20]}...' -> 无法预测")


# 清理生成的临时文件
if os.path.exists(TRAIN_FILE):
    os.remove(TRAIN_FILE)
if os.path.exists(TEST_FILE):
    os.remove(TEST_FILE)
print(f"\n--- 临时文件 {TRAIN_FILE} 和 {TEST_FILE} 已清理。 ---")