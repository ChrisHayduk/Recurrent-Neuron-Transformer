import torch
from torch import nn

class Neurons(nn.Module):
    def __init__(self, n_neurons):
        super(Neurons, self).__init__()

        # Initialize matrix neuron parameters and number of neurons to create
        self.n_neurons = n_neurons
        self.params = nn.Parameter(torch.rand(n_neurons, 3, 3) * 2 - 1)

        # Initialize hidden state for batch processing
        self.hidden = nn.Parameter(torch.zeros(1, n_neurons, 1), requires_grad=False)
    
    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # Expand hidden to match batch size
        hidden_batch = self.hidden.expand(batch_size, -1, -1)

        # Ensure inputs is 2D: (batch_size, n_neurons)
        inputs = inputs.view(batch_size, -1, 1)
        ones = torch.ones_like(inputs)

        # Concatenate along the second dimension
        stacked = torch.cat((inputs, hidden_batch, ones), dim=1)

        # Reshape stacked for matrix multiplication: [batch_size, n_neurons, 3]
        stacked = stacked.view(batch_size, self.n_neurons, 3)

        # Perform matrix multiplication
        dot = torch.tanh(torch.matmul(self.params, stacked.unsqueeze(3)).squeeze(3))

        # Update hidden state
        self.hidden = nn.Parameter(dot[:, :, -1].unsqueeze(2).detach(), requires_grad=False)

        return dot[:, :, 0]

class RecurrentNeuronLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(RecurrentNeuronLayer, self).__init__()
        self.neurons = Neurons(output_size)
        self.weights = nn.Linear(input_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        x = self.weights(x)
        x = self.neurons(x)
        
        # Reshape the output to ensure it has the shape [batch_size, n_classes]
        final_output = x.view(batch_size, seq_len, -1)

        return final_output