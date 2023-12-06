import torch
from torch import nn

class Neurons(nn.Module):
    def __init__(self, n_neurons, device):
        super(Neurons, self).__init__()
        self.device = device

        # Initialize matrix neuron parameters and number of neurons to create
        self.n_neurons = n_neurons
        self.params = nn.Parameter(torch.rand(n_neurons, 3, 3) * 2 - 1)   
    
    def forward(self, inputs, hidden_state=None):
        if hidden_state is not None:
            hidden_state = hidden_state.detach()
        else:
            hidden_state = torch.zeros(1, self.n_neurons, 1, device=self.device)

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        hidden_batch = hidden_state.expand(batch_size, seq_len, self.n_neurons, 1)
        inputs = inputs.view(batch_size, seq_len, -1, 1)
        ones = torch.ones_like(inputs)


        # Concatenate along the last dimension
        stacked = torch.cat((inputs, hidden_batch, ones), dim=3)

        # Reshape stacked for matrix multiplication: [batch_size, seq_len, n_neurons, 3]
        stacked = stacked.view(batch_size, seq_len, self.n_neurons, 3)

        # Perform matrix multiplication
        dot = torch.relu(torch.matmul(self.params, stacked.unsqueeze(4)).squeeze(4))

        # Update hidden state without in-place operation
        new_hidden = dot[:, :, :, 1].unsqueeze(3).detach()
        
        return dot[:, :, :, 0], new_hidden

class RecurrentNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(RecurrentNeuronLayer, self).__init__()
        self.neurons = Neurons(output_size, device)
        self.weights = nn.Linear(input_size, output_size)
        self.device = device

    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.weights(x)
        x, updated_hidden_state = self.neurons(x, hidden_state)
        
        # Reshape the output to ensure it has the shape [batch_size, n_classes]
        final_output = x.view(batch_size, seq_len, -1)

        return final_output, updated_hidden_state