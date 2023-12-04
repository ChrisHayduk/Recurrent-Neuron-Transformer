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

        #############################################################################
        # TODO:
        # Deliverable 5: Implement the full Transformer stack for the forward pass. #
        # You will need to use all of the methods you have previously defined above.#
        # You should only be calling TransformerTranslator class methods here.      #
        #############################################################################
        outputs = self.embed(inputs)
        outputs = self.multi_head_attention(outputs)
        outputs = self.feedforward_layer(outputs)
        outputs = self.final_layer(outputs)
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def embed(self, inputs):
        """
        :param inputs: intTensor of shape (N,T)
        :returns embeddings: floatTensor of shape (N,T,H)
        """
        #############################################################################
        # TODO:
        # Deliverable 1: Implement the embedding lookup.                            #
        # Note: word_to_ix has keys from 0 to self.vocab_size - 1                   #
        # This will take a few lines.                                               #
        #############################################################################
      
        embeddings = self.embeddingL(inputs)
        pos_indices = torch.arange(0, inputs.size(1), device=self.device).unsqueeze(0).repeat(inputs.size(0), 1)
        pos_embeddings = self.posembeddingL(pos_indices)
        embeddings  = embeddings + pos_embeddings
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return embeddings
        
    def multi_head_attention(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        
        Traditionally we'd include a padding mask here, so that pads are ignored.
        This is a simplified implementation.
        """
        
        
        #############################################################################
        # TODO:
        # Deliverable 2: Implement multi-head self-attention followed by add + norm.#
        # Use the provided 'Deliverable 2' layers initialized in the constructor.   #
        #############################################################################
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
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
    
    
    def feedforward_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,H)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 3: Implement the feedforward layer followed by add + norm.    #
        # Use a ReLU activation and apply the linear layers in the order you        #
        # initialized them.                                                         #
        # This should not take more than 3-5 lines of code.                         #
        #############################################################################
        outputs = self.norm_linear(inputs + self.linear2(torch.relu(self.linear1(inputs))))
        
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        
    
    def final_layer(self, inputs):
        """
        :param inputs: float32 Tensor of shape (N,T,H)
        :returns outputs: float32 Tensor of shape (N,T,V)
        """
        
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the final layer for the Transformer Translator.  #
        # This should only take about 1 line of code.                               #
        #############################################################################
        outputs = self.linear_output(inputs)
                
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs
        

class FullTransformerTranslator(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_dim=128, num_heads=2,
                 dim_feedforward=2048, num_layers_enc=2, num_layers_dec=2, dropout=0.2, max_length=43, ignore_index=1):
        super(FullTransformerTranslator, self).__init__()

        self.num_heads = num_heads
        self.word_embedding_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.max_length = max_length
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.pad_idx=ignore_index

        seed_torch(0)

        ##############################################################################
        # TODO:
        # Deliverable 1: Initialize what you need for the Transformer Layer          #
        # You should use nn.Transformer                                              #
        ##############################################################################

        self.transformer = nn.Transformer(d_model=self.hidden_dim, nhead=self.num_heads, num_encoder_layers=num_layers_enc, num_decoder_layers=num_layers_dec, dim_feedforward=self.dim_feedforward, dropout=dropout, batch_first=True, device=self.device)

        ##############################################################################
        # TODO:
        # Deliverable 2: Initialize what you need for the embedding lookup.          #
        # You will need to use the max_length parameter above.                       #
        # Initialize embeddings in order shown below.                                #
        # Don’t worry about sine/cosine encodings- use positional encodings.         #
        ##############################################################################
        self.srcembeddingL = nn.Embedding(self.input_size, self.word_embedding_dim)     
        self.tgtembeddingL = nn.Embedding(self.input_size, self.word_embedding_dim) 
        
        
         
        self.srcposembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)     
        self.tgtposembeddingL = nn.Embedding(self.max_length, self.word_embedding_dim)

        ##############################################################################
        # TODO:
        # Deliverable 3: Initialize what you need for the final layer.               #
        ##############################################################################
        self.final_layer = nn.Linear(self.hidden_dim, self.output_size)


        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, src, tgt):
        """
         This function computes the full Transformer forward pass used during training.
         Put together all of the layers you've developed in the correct order.

         :param src: a PyTorch tensor of shape (N,T) these are tokenized input sentences
                tgt: a PyTorch tensor of shape (N,T) these are tokenized translations
         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 4: Implement the full Transformer stack for the forward pass. #
        #############################################################################

        # shift tgt to right, add one <sos> to the beginning and shift the other tokens to right
        tgt = self.add_start_token(tgt)

        # embed src and tgt for processing by transformer
        src_embed = self.srcembeddingL(src) 
        tgt_embed = self.tgtembeddingL(tgt)

        src_positions = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0),1).to(self.device)
        tgt_positions = torch.arange(0, tgt.size(1)).unsqueeze(0).repeat(tgt.size(0),1).to(self.device)
        
        src_embed += self.srcposembeddingL(src_positions)
        tgt_embed += self.tgtposembeddingL(tgt_positions)

        # create target mask and target key padding mask for decoder
        tgt_mask = (self.transformer.generate_square_subsequent_mask(tgt.shape[1], device=self.device)).bool()
        tgt_key_padding_mask = (tgt == self.pad_idx)
        
        # invoke transformer to generate output
        outputs = self.final_layer(self.transformer.forward(src_embed, tgt_embed, tgt_mask = tgt_mask, tgt_key_padding_mask = tgt_key_padding_mask))

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return outputs

    def generate_translation(self, src):
        """
         This function generates the output of the transformer taking src as its input
         it is assumed that the model is trained. The output would be the translation
         of the input

         :param src: a PyTorch tensor of shape (N,T)

         :returns: the model outputs. Should be scores of shape (N,T,output_size).
         """
        #############################################################################
        # TODO:
        # Deliverable 5: You will be calling the transformer forward function to    #
        # generate the translation for the input.                                   #
        #############################################################################
        # Initialize the target tensor with the start token
        tgt = torch.zeros((src.shape[0], src.shape[1]), dtype=torch.long) + self.pad_idx
        print(tgt.shape)

        outputs = torch.zeros((src.shape[0], src.shape[1], self.output_size))

        tgt[:, 0] = src[:, 0]

        for i in range(src.shape[1]):
            # Run the forward pass
            output = self.forward(src, tgt)

            # Get the next token from the output
            next_token = output.argmax(dim=-1)[:, i]

            if i+1 < tgt.shape[1]:
                # Append the next token to the tgt sequence
                tgt[:, i+1] = next_token

            outputs[:, i, :] = output[:,i,:]


        return outputs

    def add_start_token(self, batch_sequences, start_token=2):
        """
            add start_token to the beginning of batch_sequence and shift other tokens to the right
            if batch_sequences starts with two consequtive <sos> tokens, return the original batch_sequence

            example1:
            batch_sequence = [[<sos>, 5,6,7]]
            returns:
                [[<sos>,<sos>, 5,6]]

            example2:
            batch_sequence = [[<sos>, <sos>, 5,6,7]]
            returns:
                [[<sos>, <sos>, 5,6,7]]
        """
        def has_consecutive_start_tokens(tensor, start_token):
            """
                return True if the tensor has two consecutive start tokens
            """
            consecutive_start_tokens = torch.tensor([start_token, start_token], dtype=tensor.dtype,
                                                    device=tensor.device)

            # Check if the first two tokens in each sequence are equal to consecutive start tokens
            is_consecutive_start_tokens = torch.all(tensor[:, :2] == consecutive_start_tokens, dim=1)

            # Return True if all sequences have two consecutive start tokens at the beginning
            return torch.all(is_consecutive_start_tokens).item()

        if has_consecutive_start_tokens(batch_sequences, start_token):
            return batch_sequences

        # Clone the input tensor to avoid modifying the original data
        modified_sequences = batch_sequences.clone()

        # Create a tensor with the start token and reshape it to match the shape of the input tensor
        start_token_tensor = torch.tensor(start_token, dtype=modified_sequences.dtype, device=modified_sequences.device)
        start_token_tensor = start_token_tensor.view(1, -1)

        # Shift the words to the right
        modified_sequences[:, 1:] = batch_sequences[:, :-1]

        # Add the start token to the first word in each sequence
        modified_sequences[:, 0] = start_token_tensor

        return modified_sequences

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True