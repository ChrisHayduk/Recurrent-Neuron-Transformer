import numpy as np

import torch
from torch import nn
import random

from models.recurrent_neuron_layer import RecurrentNeuronLayer

####### Do not modify these imports.

class RecurrentNeuronTransformer(nn.Module):
    """
    A single-layer Transformer which encodes a sequence of text and 
    performs binary classification.

    The model has a vocab size of V, works on
    sequences of length T, has an hidden dimension of H, uses word vectors
    also of dimension H, and operates on minibatches of size N.
    """
    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2, dim_feedforward=2048, dim_k=96, dim_v=96, dim_q=96, max_length=43):
        """
        :param input_size: the size of the input, which equals to the number of words in source language vocabulary
        :param output_size: the size of the output, which equals to the number of words in target language vocabulary
        :param hidden_dim: the dimensionality of the output embeddings that go into the final layer
        :param num_heads: the number of Transformer heads to use
        :param dim_feedforward: the dimension of the feedforward network model
        :param dim_k: the dimensionality of the key vectors
        :param dim_q: the dimensionality of the query vectors
        :param dim_v: the dimensionality of the value vectors
        """
        super(RecurrentNeuronTransformer, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q

        self.embeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)     #initialize word embedding layer
        self.posembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)    #initialize positional embedding layer
        
        # Head #1
        self.k1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_k, self.device)
        self.v1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_v, self.device)
        self.q1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_q, self.device)
        
        # Head #2
        self.k2 = RecurrentNeuronLayer(self.hidden_dim, self.dim_k, self.device) 
        self.v2 = RecurrentNeuronLayer(self.hidden_dim, self.dim_v, self.device)
        self.q2 = RecurrentNeuronLayer(self.hidden_dim, self.dim_q, self.device)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = RecurrentNeuronLayer(self.dim_v * self.num_heads, self.hidden_dim, self.device)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        self.linear1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_feedforward, self.device)
        self.linear2 = RecurrentNeuronLayer(self.dim_feedforward, self.hidden_dim, self.device)
        self.norm_linear = nn.LayerNorm(self.hidden_dim)
        
        self.linear_output = RecurrentNeuronLayer(self.hidden_dim, self.output_size, self.device)

        self.to(device)
        
        
    def forward(self, inputs, hidden_layers):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """
        inputs = inputs.to(self.device)
        outputs = self.embed(inputs)
        outputs, hidden_layers = self.multi_head_attention(outputs, hidden_layers)
        outputs, hidden_layers = self.feedforward_layer(outputs, hidden_layers)
        outputs, hidden_layers = self.final_layer(outputs, hidden_layers)
        
        return outputs, hidden_layers
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
      
        embeddings = self.embeddingL(inputs)
        pos_indices = torch.arange(0, inputs.size(1), device=self.device).unsqueeze(0).repeat(inputs.size(0), 1)
        pos_embeddings = self.posembeddingL(pos_indices)
        embeddings  = embeddings + pos_embeddings

        return embeddings
        
    def multi_head_attention(self, inputs, hidden_layers):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        N, T, _ = inputs.shape
        mask = torch.triu(torch.ones((T, T), device=self.device), diagonal=1).bool()

        # Head #1

        k1, hidden_layers["k1"] = self.k1(inputs, hidden_layers.get("k1"))
        v1, hidden_layers["v1"] = self.v1(inputs, hidden_layers.get("v1"))
        q1, hidden_layers["q1"] = self.q1(inputs, hidden_layers.get("q1"))
        
        # N x T x H -> N x T x T. Gives a distribution of similarity comparing each token to all other tokens in the sequence
        term1 = self.softmax(torch.bmm(q1, k1.permute(0,2,1))/((self.dim_k)**0.5))
        term1 = term1.masked_fill(mask, float('-inf'))  # Mask future tokens

        # N x T x T -> N x T x H. Uses distribution in term1 to take a weighted sum of v1
        head1 = torch.bmm(term1, v1)

        # Head #2
        k2, hidden_layers["k2"] = self.k2(inputs, hidden_layers.get("k2"))
        v2, hidden_layers["v2"] = self.v2(inputs, hidden_layers.get("v2"))
        q2, hidden_layers["q2"] = self.q2(inputs, hidden_layers.get("q2"))

        # N x T x H -> N x T x T. Gives a distribution of similarity comparing each token to all other tokens in the sequence
        term2 = self.softmax(torch.bmm(q2, k2.permute(0,2,1))/((self.dim_k)**0.5))
        term2 = term2.masked_fill(mask, float('-inf'))  # Mask future tokens

        # N x T x T -> N x T x H. Uses distribution in term1 to take a weighted sum of v1
        head2 = torch.bmm(term2, v2)

        full_head = torch.cat((head1, head2), dim=-1)

        proj, hidden_layers["proj"] = self.attention_head_projection(full_head, hidden_layers.get("proj"))

        outputs = self.norm_mh(proj + inputs)
        
        return outputs, hidden_layers
    
    
    def feedforward_layer(self, inputs, hidden_layers):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        linear_1_output, hidden_layers["linear1"] = self.linear1(inputs,hidden_layers.get("linear1"))
        linear_1_output = torch.relu(linear_1_output)
        linear_2_output, hidden_layers["linear2"] = self.linear2(linear_1_output, hidden_layers.get("linear2"))
        outputs = self.norm_linear(inputs + linear_2_output)
        
        return outputs, hidden_layers
        
    
    def final_layer(self, inputs, hidden_layers):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        outputs, hidden_layers["linear_output"] = self.linear_output(inputs, hidden_layers.get("linear_output"))
                
        return outputs, hidden_layers