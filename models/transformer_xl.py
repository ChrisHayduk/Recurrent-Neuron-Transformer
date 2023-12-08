import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
    return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, tgt_mask=None):
        batch_size, seq_len, _ = query.size()

        # Generate causal mask
        causal_mask = generate_causal_mask(seq_len).to(query.device)

        # Linear projections
        Q = self.wq(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.wk(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.wv(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        attn_weights += causal_mask.unsqueeze(0).unsqueeze(0)  # Adjust for batch size and number of heads

        # Apply optional target mask (if provided)
        if tgt_mask is not None:
            attn_weights = attn_weights.masked_fill(tgt_mask == 0, -float('inf'))

        attn = torch.softmax(attn_weights, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)

        # Concatenate heads and project back to [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.wo(output)

        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mem_size, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mem_size = mem_size
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mems=None, tgt_mask=None):
        input_seq_len = x.size(0)  # length of the new input

        if mems is not None:
            x_with_mems = torch.cat([mems, x], dim=0)
            x_attended = self.self_attn(x_with_mems, x_with_mems, x_with_mems, tgt_mask)
        else:
            x_attended = self.self_attn(x, x, x, tgt_mask)

        # Ensure x_attended is sliced to match the input sequence length
        x_attended = x_attended[-input_seq_len:]

        residual = x
        x = self.norm1(x_attended + residual)
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)

        # Update memory
        new_mem = x if mems is not None else None

        return self.dropout(x), new_mem

class TransformerXL(nn.Module):
    def __init__(self, vocab_size, chunk_size, max_seq_length, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerXL, self).__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.ModuleList([TransformerEncoderBlock(d_model, nhead, mem_size=chunk_size, dropout=dropout) \
                                      for _ in range(num_layers)])
        self.linear = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mems=None, tgt_mask=None):
        if mems is None:
            mems = [None] * len(self.encoder)

        new_mems = []
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        for i, layer in enumerate(self.encoder):
            x, mem = layer(x, mems=mems[i], tgt_mask=tgt_mask)
            new_mems.append(mem)

        x = self.dropout(x)
        x = self.linear(x)

        return x, new_mems
