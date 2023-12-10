import numpy as np
import math
import torch
from torch import nn
import random
import torch.functional as F
from dataclasses import dataclass

@dataclass
class VanillaModelConfig:
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
        
        self.c_fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim)

        self.gelu = nn.GELU()

        self.c_proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x =  self.c_fc(x)
        
        x = self.gelu(x)

        x =  self.c_proj(x)
        
        x = self.dropout(x)

        return x
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.hidden_dim % config.num_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        
        # output projection
        self.c_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        proj_output = self.c_attn(x)

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
        y = self.c_proj(y)

        y = self.resid_dropout(y)
        return y
    
class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        new_x = self.attn(self.ln_1(x))
        x = x + new_x
        new_x = self.mlp(self.ln_2(x))
        x = x + new_x
        return x

class VanillaTransformer(nn.Module):
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
        super(VanillaTransformer, self).__init__()
        assert config.hidden_dim % config.num_heads == 0

        print(config)
        self.config = config
        
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
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(self.hidden_dim),
        ))

               
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_model_config(self):
        return f"{self.__class__.__name__} - {self.config}"

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
           
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        embeddings = self.embed(inputs)
        x = self.transformer.drop(embeddings)
        for idx, block in enumerate(self.transformer.h):
            x = block(x)
        x = self.transformer.ln_f(x)
        outputs = self.lm_head(x)

        
        return outputs
    
    
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
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, hidden_layers = None, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, hidden_layers = self(idx_cond, hidden_layers)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx