import math
import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, inner_dim, dropout):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(model_dim, inner_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(inner_dim, model_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        return self.layer_norm(self.net(x) + x)

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, embed_dim, mem_len, num_heads, inner_dim, dropout, R, device):
        super().__init__()
        self.attn = MultiHeadAttention(model_dim, embed_dim, mem_len, num_heads, dropout, R, device)
        self.pos_ff = PositionwiseFeedForward(model_dim, inner_dim, dropout)
    
    def forward(self, x, mem):
        att_out = self.attn(x, mem)
        return self.pos_ff(att_out)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, embed_dim, mem_len, num_heads, dropout, R, device):
        super().__init__()
        
        self.R = R
        self.mem_len = mem_len
        self.embed_dim = embed_dim
        self.device = device
        
        self.u = nn.Parameter(torch.randn(1, num_heads, 1, embed_dim))
        self.t = nn.Parameter(torch.randn(1, num_heads, 1, embed_dim))
        
        self.w_q = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.w_k = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.w_v = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        self.w_r = nn.Linear(model_dim, num_heads*embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Linear(num_heads*embed_dim, model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, x, mem):
        # concat output from previous layer with "memory" from earlier segments
        h = torch.cat((mem, x), dim=1)
        
        batch_size, seg_len, _ = x.shape
        mem_len = h.shape[1] - seg_len
        total_len = h.shape[1]
        
        # compute projections of input and memory embeddings
        q = self.w_q(x).view(batch_size, seg_len, -1, self.embed_dim)
        k = self.w_k(h).view(batch_size, total_len, -1, self.embed_dim)
        v = self.w_v(h).view(batch_size, total_len, -1, self.embed_dim)
        r = self.w_r(self.R[-total_len:]).view(1, total_len, -1, self.embed_dim)
        
        # aligning matrices to (batch_size, num_heads, seg_len, embed_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        r = r.transpose(1,2)
        
        # the "XL specific" way of computing the pre-softmax attention score
        ac = torch.einsum("bhid,bhjd->bhij", q + self.u, k)
        bd = torch.einsum("bhid,bhjd->bhij", q + self.t, r)
        bd = self.circulant_shift(bd, -seg_len+1)
        
        # computing the attention scores
        att_score = ac + bd
        att_score = att_score.tril(mem_len) / self.embed_dim**0.5
        att_score[att_score == 0] = float("-inf")
        att_score = torch.softmax(att_score, dim=-1)
        
        # compute output
        att = (att_score @ v).transpose(1,2).reshape(batch_size, seg_len, -1)
        out = self.dropout(self.mlp(att))
        return self.layer_norm(out + x)
              
    def circulant_shift(self, x, shift):
        """
        Shifts top row of `x` by `shift`, second row by `shift-1`, etc. This is
        used to compute the relative positional encoding matrix in linear time
        (as opposed to quadratic time for the naive solution). Note: Right-hand
        side values are not zeroed out here.
        
        See Appendix B of the Transformer-XL paper for more details.
        """
        batch_size, num_heads, height, width = x.shape
        i = torch.arange(width).roll(shift).unsqueeze(0).to(self.device)
        i = i.flip(1).repeat(1, 2)
        i = i.unfold(dimension=1, size=width, step=1).flip(-1).unsqueeze(0)
        i = i.repeat(batch_size, num_heads, 1, 1)[:, :, :height]
        return x.gather(3, i)
    
class TransformerXL(nn.Module):
    def __init__(self, vocab_size, max_seq_length, hidden_dim, num_layers, num_heads, mem_len, dropout, device):
        super().__init__()
        self.mem = None
        self.mem_len = mem_len
        self.seg_len = max_seq_length
        self.model_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.device = device
        
        # segment length + memory length determines the size of R
        total_len = self.mem_len+self.seg_len
        R = self.get_sinusoid_pos_encoding(total_len, self.model_dim)
        R = torch.flip(R, dims=(0,)).to(device)
        
        for _ in range(num_layers):
            dec = DecoderLayer(self.model_dim,
                               self.model_dim,
                               self.mem_len,
                               num_heads,
                               4*self.model_dim,
                               dropout,
                               R,
                               device)
            self.layers.append(dec)
            
        self.out_layer = nn.Linear(self.model_dim, self.vocab_size)
        self.to(device)
    
    def forward(self, x):
        x = self.dropout(self.embed(x))
        
        # create memory tensors if they haven't been already
        if self.mem is None:
            batch_size = x.size(0)
            self.set_up_memory(batch_size)
        
        # compute model output, saving layer inputs to memory along the way
        for i, dec in enumerate(self.layers):
            x_ = x.clone()
            x = dec(x, self.mem[i])
            self.add_to_memory(x_, i)
            
        return self.out_layer(x)
    
    def get_sinusoid_pos_encoding(self, total_len, embed_dim):
        """
        Standard sinusoid positional encoding method outlined in the original
        Transformer paper. In this case, we use the encodings not to represent
        each token's position in a sequence but to represent the distance
        between two tokens (i.e. as a *relative* positional encoding).
        """
        pos = torch.arange(total_len).unsqueeze(1)
        enc = torch.arange(embed_dim).float()
        enc = enc.unsqueeze(0).repeat(total_len, 1)
        enc[:, ::2] = torch.sin(pos / 10000**(2*enc[:, ::2]/embed_dim))
        enc[:, 1::2] = torch.cos(pos / 10000**(2*enc[:, 1::2]/embed_dim))
        return enc
    
    def set_up_memory(self, batch_size):
        self.mem = [torch.zeros(batch_size, 0, self.model_dim).to(self.device)
                    for _ in range(len(self.layers))]
    
    def add_to_memory(self, x, i):
        if self.mem_len == 0: return
        self.mem[i] = torch.cat((self.mem[i], x.detach()), dim=1)[:, -self.mem_len:]
    
    def clear_memory(self):
        self.mem = None