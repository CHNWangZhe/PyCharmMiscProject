from typing import Any

import torch
import torch.nn as nn
import math

#Transformer主要包括六个模块，按照论文中的架构图进行分别计算，首先是Embedding模块
class Embeddings(nn.Module):
    #vocab_size词汇大小；d_model要映射为几维的向量
    def __init__(se
        lf, vocab_size, d_model):
        #继承torch
        super().__init__()
        #编码过程直接继承
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    #乘以√d_model：词编码嵌入和位置嵌入比例调整
    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)

#位置编码（位置信息压缩到[-1,1]，且不重复）目前维度max_len
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        #定义Pe, 从0～max_len生成d_model向量，用于表示位置信息
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #exp为自然数e值，此式为position式1除数
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        #偶数维sin
        pe[:, 0::2] = torch.sin(position * div_term)
        #奇数维cos
        pe[:, 1::2] = torch.cos(position * div_term)

        #固定位置维度信息
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

        #正向传播
        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


#dropout用于每次训练随机屏蔽
def attention(query, key, value, mask=None, dropout=None):
    #获得Q、K、V值
    d_k = query.size(-1)

    #缩放Q、K、V值权重（点积注意力）
    #(batch, head, seq_len_q, d_k)向量形式
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        #掩码填充
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attention = dropout(attention)

    # @ 为矩阵乘法运算符
    return attention @ value, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        #巡查报错（除数为0）
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        def transform(x, linear) -> Any:
            x = linear(x)
            #第1、2维度转换
            return x.view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        query = transform(query, self.linear_q)
        key = transform(key, self.linear_k)
        value = transform(value, self.linear_v)

        #计算注意力
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)

        return self.linear_o(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.linear(d_model, d_ff),
            nn.ReLU(),
            nn.linear(d_ff, d_model),
            nn.dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class AddNorm(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #Encoder模块中调用2次
        self.sublayer = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda y: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward(x))

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, cross_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout),
            AddNorm(d_model, dropout)
        ])

    def forward(self, x, memory, src_mask, tag_mask):
        out1 = self.sublayer[0](x, lambda y: self.self_attn(x, x, x, tag_mask))
        out2 = self.sublayer[1](out1, lambda y: self.cross_attn(out1, memory, memory, src_mask))
        out3 = self.sublayer[2](out2, lambda y: self.feed_forward(out2))

        return out3

class Transformer(nn.Module):

    def __init__(self, src_vocab, tag_vocab, d_model=512, N=6, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        #元语言嵌入+位置编码
        self.src_embed = nn.Sequential(
            Embeddings(src_vocab, d_model),
            PositionalEncoding(d_model, h)
        )

        self.tag_embed = nn.Sequential(
            Embeddings(tag_vocab, d_model),
            PositionalEncoding(d_model, h)
        )

        attn = lambda: MultiHeadAttention(N, d_model, dropout)
        ff = lambda: FeedForward(d_model, d_ff, dropout)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, attn(), ff(), dropout) for _ in range(N)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, attn(), attn(), ff(), dropout) for _ in range(N)
        ])

        #定义输出层
        self.out = nn.Linear(d_model, tag_vocab)

        #实际运行
        def encoder(self, src, src_mask):
            x = self.src_embed(src)
            for layer in self.encoder:
                x = layer(x, src_mask)

            return x

        def decoder(self, tag, memory, src_mask, tag_mask):
            x = self.tag_embed(tag)
            for layer in self.decoder:
                x = layer(x, memory, src_mask, tag_mask)

            return x

        def forward(self, src, tag, src_mask=None, tag_mask=None):
            memory = self.encode(src, src_mask)
            out = self.decoder(tag, memory, src_mask, tag_mask)
            #不要忘记Linear层
            return self.out(out)


















