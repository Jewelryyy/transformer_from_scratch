import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def padding_mask(seq, pad_idx):
    """
    生成 Padding Mask
    :param seq: 输入序列张量，形状为 (B, T)
    :param pad_idx: 填充标记的索引
    :return: 掩码张量，形状为 (B, 1, 1, T)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)


def causal_mask(size, device):
    """
    生成因果掩码
    :param size: 序列长度
    :param device: 设备
    :return: 因果掩码张量，形状为 (1, 1, size, size)
    """
    return torch.tril(torch.ones((size, size), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1), :]
        return x


# # 消融实验占位类
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()

#     def forward(self, x):
#         return x


class MultiHeadAttention(nn.Module):
    """灵活的多头注意力。如果 `kv` 为 None，则对 `x` 使用自注意力。
    可用于自注意力和交叉注意力。
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, mask=None):
        # x_q: (B, Tq, D)
        # x_kv: (B, Tk, D) 或 None -> 如果为 None，则使用 x_q 作为 kv（自注意力）
        B, Tq, D = x_q.shape
        if x_kv is None:
            x_kv = x_q
        Tk = x_kv.shape[1]

        q = self.q(x_q).view(B, Tq, self.n_head, self.d_head).transpose(1, 2)  # (B,H,Tq,d)
        k = self.k(x_kv).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # (B,H,Tk,d)
        v = self.v(x_kv).view(B, Tk, self.n_head, self.d_head).transpose(1, 2)  # (B,H,Tk,d)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B,H,Tq,Tk)
        if mask is not None:
            # 掩码应广播到 (B,1,Tq,Tk) 或 (1,1,Tq,Tk)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B,H,Tq,d)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """编码器块：自注意力（默认无因果）+ 前馈网络"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), x_kv=None, mask=mask)
        x = x + self.ff(self.ln2(x))
        return x
    
    # # 消融残差连接
    # def forward(self, x, mask=None):
    #     x = self.attn(self.ln1(x), x_kv=None, mask=mask)
    #     x = self.ff(self.ln2(x))
    #     return x


class DecoderBlock(nn.Module):
    """解码器块：包含自注意力（因果）、交叉注意力（编码器->解码器）和前馈网络"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        # 自注意力（因果）
        x = x + self.self_attn(self.ln1(x), x_kv=None, mask=self_mask)
        # 交叉注意力（查询来自解码器，键/值来自编码器）
        x = x + self.cross_attn(self.ln2(x), x_kv=enc_out, mask=cross_mask)
        x = x + self.ff(self.ln3(x))
        return x
    
    # # 消融残差连接
    # def forward(self, x, enc_out, self_mask=None, cross_mask=None):
    #     # 自注意力（因果）
    #     x = self.self_attn(self.ln1(x), x_kv=None, mask=self_mask)
    #     # 交叉注意力（查询来自解码器，键/值来自编码器）
    #     x = self.cross_attn(self.ln2(x), x_kv=enc_out, mask=cross_mask)
    #     x = self.ff(self.ln3(x))
    #     return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=4, d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, src, src_mask=None, pad_idx=None):
        # src: (B, S)
        B, S = src.shape
        assert S <= self.max_seq_len
        x = self.token_emb(src)
        x = self.pos_enc(x)
        if pad_idx is not None:
            src_mask = padding_mask(src, pad_idx)  # 生成 Padding Mask
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        x = self.ln_f(x)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=4, d_ff=512, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.max_seq_len = max_seq_len

    def forward(self, tgt, enc_out, self_mask=None, cross_mask=None, pad_idx=None):
        # tgt: (B, T)
        B, T = tgt.shape
        assert T <= self.max_seq_len
        x = self.token_emb(tgt)
        x = self.pos_enc(x)
        if pad_idx is not None:
            self_mask = padding_mask(tgt, pad_idx) & causal_mask(T, device=tgt.device)  # 结合因果掩码
        for layer in self.layers:
            x = layer(x, enc_out, self_mask=self_mask, cross_mask=cross_mask)
        x = self.ln_f(x)
        return x


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_enc_layers=3, n_dec_layers=3, d_ff=512, src_max_len=128, tgt_max_len=128, dropout=0.1, pad_idx=None):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_head, n_enc_layers, d_ff, max_seq_len=src_max_len, dropout=dropout)
        self.decoder = Decoder(vocab_size, d_model, n_head, n_dec_layers, d_ff, max_seq_len=tgt_max_len, dropout=dropout)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.pad_idx = pad_idx

    def forward(self, src, tgt_input):
        # src: (B, S), tgt_input: (B, T)
        B, S = src.shape
        B2, T = tgt_input.shape
        assert B == B2
        device = src.device
        enc_out = self.encoder(src, pad_idx=self.pad_idx)  # 传递 pad_idx
        self_mask = causal_mask(T, device=device)
        dec_out = self.decoder(tgt_input, enc_out, self_mask=self_mask, cross_mask=None, pad_idx=self.pad_idx)
        logits = self.head(dec_out)
        return logits
