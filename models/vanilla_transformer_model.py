import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaTransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)

        # Transformer Decoder
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=num_decoder_layers)

        # Output layer
        self.out = nn.Linear(d_model, vocab_size)

        # Report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        return num_params

    def forward(self, tgt, tgt_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)

        # Initialize empty memory of zeros
        memory = torch.zeros(tgt.size(0), tgt.size(1), self.d_model, device=tgt.device)

        # Since it's a decoder-only model, no encoder and memory
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.out(output)
        return output

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
