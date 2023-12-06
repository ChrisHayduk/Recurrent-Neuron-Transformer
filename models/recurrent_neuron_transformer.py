import numpy as np
import math
import torch
from torch import nn
import random
import torch.functional as F
from dataclasses import dataclass
from models.recurrent_neuron_layer import RecurrentNeuronLayer

@dataclass
class ModelConfig:
    max_length: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    dropout: float = 0.0
    device: str = "cuda"

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = RecurrentNeuronLayer(config.hidden_dim, 4 * config.hidden_dim, config.device)
        self.gelu    = nn.GELU()
        self.c_proj  = RecurrentNeuronLayer(4 * config.hidden_dim, config.hidden_dim, config.device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, hidden_layers, layer_num=0):
        x, hidden_layers[f"c_fc_{layer_num}"] = self.c_fc(x, hidden_layers.get(f"c_fc_{layer_num}"))
        x = self.gelu(x)
        x, hidden_layers[f"c_proj_{layer_num}"] = self.c_proj(x, hidden_layers.get(f"c_proj_{layer_num}"))
        x = self.dropout(x)
        return x, hidden_layers
    
class RecurrentCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = RecurrentNeuronLayer(config.hidden_dim, 3 * config.hidden_dim, config.device)
        # output projection
        self.c_proj = RecurrentNeuronLayer(config.hidden_dim, config.hidden_dim, config.device)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.num_heads
        self.n_embd = config.hidden_dim
        self.dropout = config.dropout
        self.max_length = config.max_length

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.max_length, self.max_length))
                                        .view(1, 1, self.max_length, self.max_length))

    def forward(self, x, hidden_layers, layer_num=0):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        proj_output, hidden_layers[f"c_attn_{layer_num}"]  = self.c_attn(x, hidden_layers.get(f"c_attn_{layer_num}"))
        q, k, v = proj_output.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y, hidden_layers[f"c_proj_{layer_num}"] = self.c_proj(y, hidden_layers.get(f"c_proj_{layer_num}"))
        y = self.resid_dropout(y)
        return y, hidden_layers
    
class RecurrrentTransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = RecurrentCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(self, x, hidden_layers, layer_num = 0):
        new_x, hidden_layers = self.attn(self.ln_1(x), hidden_layers, layer_num)
        x = x + new_x
        new_x, hidden_layers = self.mlp(self.ln_2(x), hidden_layers, layer_num)
        x = x + new_x
        return x, hidden_layers

class RecurrentNeuronTransformer(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, config):
        """
        :config
        """
        super(RecurrentNeuronTransformer, self).__init__()
        assert config.hidden_dim % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.word_embedding_dim = config.hidden_dim
        self.hidden_dim = config.hidden_dim
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.device = config.device
        self.dropout = config.dropout

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, self.word_embedding_dim),
            wpe = nn.Embedding(self.max_length, self.word_embedding_dim),
            drop = nn.Dropout(self.dropout),
            h = nn.ModuleList([RecurrrentTransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(self.hidden_dim),
        ))

               
        self.lm_head = RecurrentNeuronLayer(self.hidden_dim, self.vocab_size, self.device)

        self.to(self.device)
        
        
    def forward(self, inputs, hidden_layers):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
                
        embeddings = self.embed(inputs)
        x = self.transformer.drop(embeddings)
        for idx, block in enumerate(self.transformer.h):
            x, hidden_layers = block(x, hidden_layers, idx)
        x = self.transformer.ln_f(x)
        outputs, hidden_layers["lm_output"] = self.lm_head(x, hidden_layers.get("lm_output"))

        
        return outputs, hidden_layers
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
      
        pos = torch.arange(0, self.max_length, dtype=torch.long, device=self.device) # shape (t)
        tok_emb = self.transformer.wte(inputs) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        embeddings  = tok_emb + pos_emb

        return embeddings