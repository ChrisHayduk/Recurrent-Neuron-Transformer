import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask=None):
        # Linear projections
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)

        # Split heads
        Q = Q.view(-1, self.nhead, self.head_dim)
        K = K.view(-1, self.nhead, self.head_dim)
        V = V.view(-1, self.nhead, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
        if tgt_mask is not None:
            attn_weights = attn_weights.masked_fill(tgt_mask == 0, -float('inf'))
        attn = torch.softmax(attn_weights, dim=-1)
        attn = self.dropout(attn)
        output = torch.bmm(attn, V)

        # Concatenate heads and project back
        output = output.view(-1, self.nhead * self.head_dim)
        output = self.wo(output)
        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        residual = x
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + residual)
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        return self.dropout(x)

class TransformerXLModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerXLModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.ModuleList([TransformerEncoderBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask=None):
        mems = None
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.encoder:
            x, mems = layer(x, tgt_mask=tgt_mask, mems=mems)
        x = self.dropout(x)
        return self.linear(x),
