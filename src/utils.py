import json
import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from model import Seq2SeqTransformer

def set_env_random_seed(seed=2020):
    """
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def visualize_training(train_losses, val_losses, save_path):
    """
    绘制训练和验证的损失曲线，支持不同粒度的数据。
    """
    # 训练集 x 轴为 batch 数
    train_x = np.arange(1, len(train_losses) + 1)
    # 验证集 x 轴为 epoch 数，直接与训练集末尾对齐
    val_x = np.arange(len(train_losses) // len(val_losses), len(train_losses) + 1, len(train_losses) // len(val_losses))

    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    plt.plot(train_x, train_losses, label="Train Loss (per 500 batches)", color='blue', alpha=0.7)
    plt.plot(val_x, val_losses, label="Validation Loss (per epoch)", color='orange', alpha=0.7)
    plt.scatter(val_x, val_losses, color='orange', s=80)
    plt.xlabel("Batch Group (Training)")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def visualize_acc(acc, save_path):
    plt.figure(figsize=(12, 8))

    # 绘制损失曲线
    plt.plot(range(1, len(acc)+1), acc, label="Validation Accuracy (per Epoch)", color='blue', alpha=0.7)
    plt.scatter(range(1, len(acc)+1), acc, color='blue', s=80)
    plt.xlabel("Epoch (Validation)")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def load_model_and_vocab(params_path, ckpt_path):
    """
    加载模型和词汇表，并根据保存的参数构建模型。
    :param params_path: 保存训练参数的 JSON 文件路径
    :param ckpt_path: 保存模型和词汇表的检查点路径
    :return: 加载的模型和词汇表
    """
    # 加载训练参数
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # 加载检查点
    ckpt = torch.load(ckpt_path)

    # 构建模型
    model = Seq2SeqTransformer(
        vocab_size=params["vocab_size"],
        d_model=params["d_model"],
        n_head=params["n_head"],
        n_enc_layers=params["n_enc_layers"],
        n_dec_layers=params["n_dec_layers"],
        d_ff=params["d_ff"],
        src_max_len=params["src_max_len"],
        tgt_max_len=params["tgt_max_len"],
        dropout=params["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 加载词汇表
    vocab = ckpt["vocab"]
    # 构建反向映射
    idx_to_word = {idx: word for word, idx in vocab.items()}

    return model, vocab, idx_to_word

def calculate_accuracy(logits, labels):
    """
    计算准确率
    :param logits: 模型输出的 logits，形状为 (batch_size, seq_len, vocab_size)
    :param labels: 目标标签，形状为 (batch_size, seq_len)
    :return: 准确率
    """
    predictions = torch.argmax(logits, dim=-1)  # 获取每个时间步的预测索引
    correct = (predictions == labels).float()  # 计算正确预测的数量
    mask = (labels != 0).float()  # 忽略填充标记 <pad>
    accuracy = (correct * mask).sum() / mask.sum()  # 计算准确率
    return accuracy.item()