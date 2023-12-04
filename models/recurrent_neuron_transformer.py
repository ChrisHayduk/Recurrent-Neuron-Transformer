import numpy as np

import torch
from torch import nn
import random

from recurrent_neuron_layer import RecurrentNeuronLayer

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

        self.to(device)
        
        seed_torch(0)
        
        self.embeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)     #initialize word embedding layer
        self.posembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)    #initialize positional embedding layer
        
        # Head #1
        self.k1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_k)
        self.v1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_v)
        self.q1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_q)
        
        # Head #2
        self.k2 = RecurrentNeuronLayer(self.hidden_dim, self.dim_k) 
        self.v2 = RecurrentNeuronLayer(self.hidden_dim, self.dim_v)
        self.q2 = RecurrentNeuronLayer(self.hidden_dim, self.dim_q)
        
        self.softmax = nn.Softmax(dim=2)
        self.attention_head_projection = nn.Linear(self.dim_v * self.num_heads, self.hidden_dim)
        self.norm_mh = nn.LayerNorm(self.hidden_dim)

        self.linear1 = RecurrentNeuronLayer(self.hidden_dim, self.dim_feedforward).to(self.device) 
        self.linear2 = RecurrentNeuronLayer(self.dim_feedforward, self.hidden_dim).to(self.device) 
        self.norm_linear = nn.LayerNorm(self.hidden_dim).to(self.device) 
        
        self.linear_output = RecurrentNeuronLayer(self.hidden_dim, self.output_size)
        
        
    def forward(self, inputs):
        """
        This function computes the full Transformer forward pass.
        Put together all of the layers you've developed in the correct order.

        :param inputs: a PyTorch tensor of shape (N,T). These are integer lookups.

        :returns: the model outputs. Should be scores of shape (N,T,output_size).
        """

        outputs = self.embed(inputs)
        outputs = self.multi_head_attention(outputs)
        outputs = self.feedforward_layer(outputs)
        outputs = self.final_layer(outputs)
        
        return outputs
    
    
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
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        # Head #1
        k1 = self.k1(inputs)
        v1 = self.v1(inputs)
        q1 = self.q1(inputs)
        
        # N x T x H -> N x T x T. Gives a distribution of similarity comparing each token to all other tokens in the sequence
        term1 = self.softmax(torch.bmm(q1, k1.permute(0,2,1))/((self.dim_k)**0.5))

        # N x T x T -> N x T x H. Uses distribution in term1 to take a weighted sum of v1
        head1 = torch.bmm(term1, v1)

        # Head #2
        k2 = self.k2(inputs)
        v2 = self.v2(inputs)
        q2 = self.q2(inputs)

        # N x T x H -> N x T x T. Gives a distribution of similarity comparing each token to all other tokens in the sequence
        term2 = self.softmax(torch.bmm(q2, k2.permute(0,2,1))/((self.dim_k)**0.5))

        # N x T x T -> N x T x H. Uses distribution in term1 to take a weighted sum of v1
        head2 = torch.bmm(term2, v2)

        full_head = torch.cat((head1, head2), dim=-1)

        proj = self.attention_head_projection(full_head)

        outputs = self.norm_mh(proj + inputs)
        
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        outputs = self.norm_linear(inputs + self.linear2(torch.relu(self.linear1(inputs))))
        
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        outputs = self.linear_output(inputs)
                
        return outputs