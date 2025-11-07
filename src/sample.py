import torch
from model import Seq2SeqTransformer
from utils import load_model_and_vocab

# 加载模型和词汇表
model, vocab, inv_id = load_model_and_vocab("./results/20251106_202257/config.json", "./results/20251106_202257/best_transformer.pt")

# 示例样本
# sample = "The permanent ice is marked in red."
# target = "Das Dauereis ist mit rot markiert."
sample = "This is an annual melting river."
target = "Dies ist ein jährlicher Tauwasserfluss."

# 将样本转换为模型输入
def preprocess_sample(sample, vocab, src_len=128):
    tokens = sample.split()[:src_len]
    input_ids = torch.tensor([vocab.get(token, vocab['<unk>']) for token in tokens], dtype=torch.long).unsqueeze(0)
    return input_ids

# 预处理样本
input_ids = preprocess_sample(sample, vocab)

# 获取模型输出
model.eval()  # 切换到评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)
model = model.to(device)

# 构造解码器输入：初始解码器输入为 <s>
sos_id = vocab['<s>']
decoder_input = torch.tensor([[sos_id]], dtype=torch.long, device=device)

# 逐步生成预测
max_len = 128  # 最大生成长度
for _ in range(max_len):
    logits = model(input_ids, decoder_input)
    next_token = logits[:, -1, :].argmax(dim=-1)  # 获取最后一个时间步的预测
    decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)
    if next_token.item() == vocab['<pad>']:  # 遇到 <pad> 结束
        break

# 解码预测结果
idx_to_word = {int(idx): word for word, idx in vocab.items()}
prediction = [idx_to_word[token.item()] for token in decoder_input.squeeze()]
print("预测结果:", " ".join(prediction))
print("真实结果: ", target)