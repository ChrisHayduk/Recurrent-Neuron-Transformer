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
    
    def neuron_fn(self, inputs):
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

        return dot[:, :, 0], dot

class RecurrentNeuronLayer(nn.Module):
    def __init__(self, sizes):
        super(RecurrentNeuronLayer, self).__init__()
        self.neurons = nn.ModuleList([Neurons(size) for size in sizes])
        self.weights = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])

    def forward(self, x):
        batch_size = x.shape[0]
        for i, neuron in enumerate(self.neurons[:-1]):  # Process through all but last layer
            send, _ = neuron.neuron_fn(x if i == 0 else pre)
            pre = self.weights[i](send)

        # Process the last layer
        final_output, _ = self.neurons[-1].neuron_fn(pre)

        # Reshape the output to ensure it has the shape [batch_size, n_classes]
        final_output = final_output.view(batch_size, -1)

        return final_output