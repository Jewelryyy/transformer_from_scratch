# Mini Transformer from scratch (PyTorch)

这是一个迷你可运行的 Transformer 序列到序列（Seq2Seq）模型，用于英德机器翻译任务。

## 包含文件：
```
tffs
├── iwslt2017_dataset   # IWSLT2017 数据集缓存
├── README.md           # 项目说明文档
├── requirements.txt    # 依赖包列表
├── results             # 训练结果输出目录
│   └── [time_stamp]
│       ├── acc_curve.png
│       ├── best_transformer.pt
│       ├── config.json
│       ├── loss_curve.png
│       └── training.log
├── scripts             # 运行脚本目录
│   └── run.sh
└── src                 # 源代码
    ├── model.py        # Transformer 模型定义
    ├── sample.py       # 输出样例
    ├── train.py        # 主训练脚本
    └── utils.py        # 工具函数（可视化、随机种子等）
```

## 快速开始（Linux Bash）：

```bash
conda create -n transformer python=3.10
conda activate transformer
cd tffs
pip install requirements.txt
sh scripts/run.sh
```
## 硬件配置

- **GPU**: NVIDIA TITAN XP (12GB)  
- **CPU**: Intel(R) Xeon(R) CPU E5-2609 v3 @ 1.90GHz  
- **内存**: 32GB RAM  
- **存储**: 至少 1GB 可用空间（包含数据集缓存）

## 预期运行时间

- **数据下载与预处理**: 约 5 分钟（首次运行，包含数据集下载）  
- **单轮训练**: 约 18-20 分钟（batch size = 16）  
- **完整训练（5 轮）**: 约 110 分钟  
- **验证**: 每轮训练后约 1 分钟

## 注意事项

- 如果 GPU 内存不足，可适当减小 `batch_size` 参数。  
- 首次运行会自动下载数据集，需要网络连接。  
- 训练结果会保存在 `results/[timestamp]/` 目录下，包含模型权重、训练日志和配置文件。  
- 建议在 Linux 或 macOS 系统上运行以获得最佳性能。

## 验证训练完成

训练完成后，您可以在 `results/[timestamp]/` 目录下找到以下文件：

- `training.log`: 详细的训练日志  
- `best_transformer.pt`: 在验证集上表现最好的模型权重  
- `loss_curve.png`: 训练和验证损失曲线图  
- `acc_curve.png`: 验证准确率曲线图  
- `config.json`: 训练时使用的配置参数

由于词表占用空间较大，仓库不提供`best_transformer.pt`文件（482.94MB）。