import torch
from torch import nn

class Neurons(nn.Module):
    def __init__(self, n_neurons, device):
        super(Neurons, self).__init__()
        self.device = device

        # Initialize matrix neuron parameters and number of neurons to create
        self.n_neurons = n_neurons
        self.params = nn.Parameter(torch.rand(n_neurons, 3, 3) * 2 - 1).to(self.device)

        # Initialize hidden state for batch processing
        self.register_buffer('hidden', torch.zeros(1, n_neurons, 1).to(self.device))
        

        self.to(self.device)
    
    def forward(self, inputs, hidden_state=None):
        if hidden_state is not None:
            self.hidden_state = hidden_state.detach()
        else:
            self.hidden_state = torch.zeros(1, self.n_neurons, 1).to(self.device)

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        # Expand hidden to match batch size and sequence length
        hidden_batch = self.hidden.expand(batch_size, seq_len, self.n_neurons, 1).to(self.device)
        
        # Ensure inputs is 3D: (batch_size, seq_len, n_neurons)
        inputs = inputs.view(batch_size, seq_len, -1, 1)
        ones = torch.ones_like(inputs).to(self.device)

        # Concatenate along the second dimension
        stacked = torch.cat((inputs, hidden_batch, ones), dim=1).to(self.device)

        # Reshape stacked for matrix multiplication: [batch_size, seq_len, n_neurons, 3]
        stacked = stacked.view(batch_size, seq_len, self.n_neurons, 3)

        # Perform matrix multiplication
        dot = torch.relu(torch.matmul(self.params, stacked.unsqueeze(4)).squeeze(4)).to(self.device)

        # Update hidden state without in-place operation
        new_hidden = dot[:, :, :, -1].unsqueeze(3).detach()

        self.hidden = new_hidden

        return dot[:, :, :, 0], self.hidden

class RecurrentNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(RecurrentNeuronLayer, self).__init__()
        self.neurons = Neurons(output_size, device)
        self.weights = nn.Linear(input_size, output_size)
        self.device = device

        self.to(self.device)
    def forward(self, x, hidden_state=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.weights(x)
        x, updated_hidden_state = self.neurons(x, hidden_state)
        
        # Reshape the output to ensure it has the shape [batch_size, n_classes]
        final_output = x.view(batch_size, seq_len, -1)

        return final_output, updated_hidden_state