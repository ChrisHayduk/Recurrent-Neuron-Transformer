import torch
import torch.nn as nn

"""
This is an implementation of the stateful neuron that works on the MNIST dataset with batch size of 1.
It is a Torch-ified version of the author's original numpy implementation.

class Neurons(nn.Module):
    def __init__(self, n_neurons):
        super(Neurons, self).__init__()
        self.n_neurons = n_neurons
        self.params = nn.Parameter(torch.rand(n_neurons, 3, 3) * 2 - 1)  # Uniform initialization
        self.hidden = torch.zeros((n_neurons, 1))
        self.bias = torch.ones((n_neurons, 1))

    def neuron_fn(self, inputs):
        # Ensure inputs is 2D: (n_neurons, 1)
        inputs = inputs.view(-1, 1)

        # Create a tensor of ones with the same batch size and number of neurons
        ones = torch.ones_like(inputs)

        # Send the inputs, ones, and hidden state to the device
        inputs = inputs.to(inputs.device)
        ones = ones.to(inputs.device)
        self.hidden = self.hidden.to(inputs.device)

        # Concatenate along the second dimension
        stacked = torch.cat((inputs, self.hidden, self.bias), dim=1)  # Shape: [1, n_neurons, 3]

        # Multiply the neuron parameters by the stacked inputs
        dot = torch.tanh(torch.matmul(self.params, stacked.unsqueeze(2)).squeeze(2))  # Shape after squeeze: [n_neurons, 3]
        
        # Update hidden state
        self.hidden = dot[:, -1].unsqueeze(1).detach()

        # Return the first column and the dot product
        return dot[:, 0], dot

class NeuralDiverseNet(nn.Module):
    def __init__(self, sizes):
        super(NeuralDiverseNet, self).__init__()
        self.neurons = nn.ModuleList([Neurons(size) for size in sizes])
        self.weights = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])

    def forward(self, x):
        # Define batch size
        batch_size = x.shape[0]

        # Process through all but last layer
        for i, neuron in enumerate(self.neurons[:-1]):  # Process through all but last layer
            send, _ = neuron.neuron_fn(x if i == 0 else pre)
            pre = self.weights[i](send)

        # Process the last layer
        final_output, _ = self.neurons[-1].neuron_fn(pre)
        final_output = final_output.view(batch_size, -1)
        return final_output

"""

"""
This is an implementation of the stateful neuron that works on the MNIST dataset, with an arbitrary 
batch size. 


"""


class StatefulNeuronLayer2D(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StatefulNeuronLayer2D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Create a parameter for each neuron's 3x2 matrix
        self.weights = nn.Parameter(torch.randn(input_size, 3, 2))
        # Create a parameter for each neuron's bias
        self.biases = nn.Parameter(torch.randn(input_size, 2))

    def forward(self, x, h):
        x = x.unsqueeze(-1)  # Shape: [seq_len, input_size, 1]
        h = h.unsqueeze(-1)  # Shape: [seq_len, input_size, 1]

        # Concatenate x, h, and biases
        concatenated = torch.cat((x, h, self.biases.expand(x.shape[0], -1, -1)), dim=2)

        # Apply the weights
        combined = torch.tanh(torch.einsum('bij,ijk->bik', concatenated, self.weights))

        # Split the output into new output and new hidden state
        new_output, new_h = combined.split(1, dim=2)

        return new_output.squeeze(-1), new_h.squeeze(-1)

class StatefulNeuronLayer1D(nn.Module):
    def __init__(self, flattened_input_size, hidden_size):
        super(StatefulNeuronLayer1D, self).__init__()
        self.flattened_input_size = flattened_input_size
        self.hidden_size = hidden_size

        # Create a parameter for each neuron's 3x2 matrix
        self.weights = nn.Parameter(torch.randn(flattened_input_size, 3, 2))
        # Create a parameter for each neuron's bias
        self.biases = nn.Parameter(torch.randn(flattened_input_size, 2))

    def forward(self, flattened_x, h):
        # Reshape h to match the flattened input
        h = h.repeat_interleave(self.flattened_input_size // h.size(1), dim=1)

        # Concatenate flattened_x, h, and biases
        concatenated = torch.cat((flattened_x.unsqueeze(-1), h.unsqueeze(-1), self.biases.expand(flattened_x.shape[0], -1, -1)), dim=2)

        # Apply the weights
        combined = torch.tanh(torch.einsum('bij,ijk->bik', concatenated, self.weights))

        # Split the output into new output and new hidden state
        new_output, new_h = combined.split(1, dim=2)

        return new_output.squeeze(-1), new_h.squeeze(-1)

# Example usage
seq_len = 128     # Sequence length
input_size = 512  # num_heads * d_v. This is equivalent to dmodel
hidden_size = 1
flattened_input_size = seq_len * input_size  # Flattened input size

# Initialize the layers
custom_layer_2d = StatefulNeuronLayer2D(input_size, hidden_size)
custom_layer_flat = StatefulNeuronLayer1D(flattened_input_size, hidden_size)

# Example input (concatenated head outputs)
concatenated_head_outputs = torch.rand(seq_len, input_size)  # Shape: [seq_len, num_heads * d_v]
flattened_inputs = concatenated_head_outputs.flatten()       # Flatten the inputs for the 1D version

# Initial hidden states (one per neuron, replicated for each sequence position)
initial_hidden_states = torch.zeros(seq_len, input_size, hidden_size)
initial_hidden_states_flat = torch.zeros(1, hidden_size)

# Pass through the custom layers
output_2d, new_hidden_states_2d = custom_layer_2d(concatenated_head_outputs, initial_hidden_states)
output_flat, new_hidden_states_flat = custom_layer_flat(flattened_inputs, initial_hidden_states_flat)

# Reshape the flattened output
output_flat = output_flat.view(seq_len, input_size)