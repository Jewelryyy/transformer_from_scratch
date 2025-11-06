# Mini Transformer from scratch (PyTorch)

这是一个最小可运行的 Transformer（仅 Encoder 风格的自注意力堆栈）示例，用于在极小的数据集上快速演示训练流程。它是字符级的因果语言建模（next-char prediction）。

包含文件：
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

快速开始（Windows PowerShell）：

```powershell
python -m pip install -r "c:\Users\ljw\Desktop\大模型基础\workspace\requirements.txt"
python "c:\Users\ljw\Desktop\大模型基础\workspace\train.py"
```

说明：该示例为了简洁做了很多简化，适合教学与实验。后续可扩展：更大文本/子词 tokenizer、多 GPU、采样生成、保存训练日志等。