import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from model import Seq2SeqTransformer
from utils import *
import logging
import argparse
from datetime import datetime
from collections import Counter
import json

# 获取当前时间戳
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"./results/{current_time}"
log_filename = f"{results_dir}/training.log"

# 设置日志配置
os.makedirs(results_dir, exist_ok=True)

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename),  # 输出到文件
        logging.StreamHandler()            # 输出到命令行
    ]
)


def collate_fn(batch, src_lang="en", tgt_lang="de", vocab=None, src_len=128, tgt_len=128):
    """数据集的批处理函数，用于将文本转换为张量"""
    src_batch = []
    tgt_batch = []
    for example in batch:
        src = example["translation"][src_lang].split()[:src_len]
        tgt = example["translation"][tgt_lang].split()[:tgt_len]
        src_batch.append([vocab.get(word, vocab['<unk>']) for word in src])
        tgt_batch.append([vocab.get(word, vocab['<unk>']) for word in tgt])

    # 填充到相同长度
    src_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in src_batch], batch_first=True, padding_value=vocab['<pad>'])
    tgt_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in tgt_batch], batch_first=True, padding_value=vocab['<pad>'])
    return src_batch, tgt_batch


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer model dimension")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_enc_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--n_dec_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=512, help="Feedforward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducing")
    return parser.parse_args()


def train():
    args = parse_args()
    set_env_random_seed(args.seed)

    # 将配置写入日志
    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 iwslt2017 数据集
    raw_dataset = load_dataset("iwslt2017", "iwslt2017-en-de", trust_remote_code=True, cache_dir="./iwslt2017_dataset")
    src_lang = "en"
    tgt_lang = "de"
    src_len = 128
    tgt_len = 128

    # 构建词汇表
    vocab = {"<pad>": 0, "<s>": 1, "<unk>": 2}  # 初始化特殊标记
    idx = len(vocab)
    for example in raw_dataset["train"]:
        for word in example["translation"][src_lang].split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
        for word in example["translation"][tgt_lang].split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    # 基于词频构建词汇表
    # max_vocab_size = 10000  # 最大词汇表大小
    # counter = Counter()
    # for example in raw_dataset["train"]:
    #     counter.update(example["translation"][src_lang].split())
    #     counter.update(example["translation"][tgt_lang].split())
    
    # # 按词频排序并限制词汇表大小
    # most_common = counter.most_common(max_vocab_size - 3)  # 预留特殊标记
    # vocab = {"<pad>": 0, "<s>": 1, "<unk>": 2}
    # idx = len(vocab)
    # for word, _ in most_common:
    #     vocab[word] = idx
    #     idx += 1

    train_loader = DataLoader(
        raw_dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda x: collate_fn(x, src_lang, tgt_lang, vocab, src_len, tgt_len),
    )
    val_loader = DataLoader(
        raw_dataset["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, src_lang, tgt_lang, vocab, src_len, tgt_len),
    )

    logging.info(f"vocab size: {idx}")
    model = Seq2SeqTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_head=args.n_head,
        n_enc_layers=args.n_enc_layers,
        n_dec_layers=args.n_dec_layers,
        d_ff=args.d_ff,
        src_max_len=src_len,
        tgt_max_len=tgt_len,
        dropout=args.dropout,
        pad_idx=vocab["<pad>"]
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    epochs = args.epochs
    global_step = 0
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        total_samples = 0
        sos_id = vocab['<s>']
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            # 构造解码器输入：在目标序列前添加 <s> 标记，并移除最后一个 token
            decoder_input = torch.cat([torch.full((tgt.size(0), 1), sos_id, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)
            logits = model(src, decoder_input)  # (B, T, V)
            B, T, V = logits.shape
            labels = tgt
            loss = criterion(logits.view(B * T, V), labels.view(B * T))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item() * src.size(0)
            total_samples += src.size(0)
            global_step += 1

            # 每隔 500 个 batch 记录一次信息
            if (batch_idx + 1) % 500 == 0:
                avg_loss = running_loss / total_samples
                train_losses.append(avg_loss)
                logging.info(f"Epoch {ep+1}/{epochs}, Batch {batch_idx+1}, Loss: {avg_loss:.4f}")

        train_loss = running_loss / total_samples
        t1 = time.time()

        # 验证模型
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        total_val_seq = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                decoder_input = torch.cat([torch.full((tgt.size(0), 1), sos_id, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)
                logits = model(src, decoder_input)
                B, T, V = logits.shape
                labels = tgt
                loss = criterion(logits.view(B * T, V), labels.view(B * T))
                val_loss += loss.item() * labels.size(0)
                total_val_seq += labels.size(0)

                # 计算准确率
                val_accuracy += calculate_accuracy(logits, labels) * labels.size(0)

        # 计算平均损失和准确率
        val_loss /= total_val_seq
        val_losses.append(val_loss)                
        val_accuracy /= total_val_seq
        val_accuracies.append(val_accuracy)
        logging.info(f"Epoch {ep+1}/{epochs}, Train Loss: {train_loss:.4f}, "
                     f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Time: {t1-t0:.2f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = f"{results_dir}/best_transformer.pt"
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab  # 保存词汇表
            }, ckpt_path)
            logging.info(f"Saved best model to {ckpt_path}")
        
        ppl = math.exp(best_val_loss)
        logging.info(f"Best Validation Loss: {best_val_loss:.4f}, Best PPL: {ppl}")

    # 可视化训练过程
    loss_save_path = f"{results_dir}/loss_curve.png"
    visualize_training(train_losses, val_losses, loss_save_path)
    acc_save_path = f"{results_dir}/acc_curve.png"
    visualize_acc(val_accuracies, acc_save_path)

    # 保存训练参数到 JSON 文件
    params = {
        "vocab_size": len(vocab),
        "d_model": args.d_model,
        "n_head": args.n_head,
        "n_enc_layers": args.n_enc_layers,
        "n_dec_layers": args.n_dec_layers,
        "d_ff": args.d_ff,
        "src_max_len": src_len,
        "tgt_max_len": tgt_len,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr
    }
    params_path = f"{results_dir}/config.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved training parameters to {params_path}")

if __name__ == '__main__':
    train()
