import torch
from torch import nn

def surrogate_gradient(x):
    alpha = 10  # The steepness of the surrogate gradient
    return torch.sigmoid(alpha * x)

class SpikingNeuronLayer(nn.Module):
    def __init__(self, size_in, size_out, device):
        super(SpikingNeuronLayer, self).__init__()
        self.device = device
        self.synaptic_weights = nn.Parameter(torch.randn(size_in, size_out, device=device) * 0.01)

    def forward(self, x):
        x = x.to(self.device)
        pre_synaptic = torch.matmul(x, self.synaptic_weights)
        post_synaptic = surrogate_gradient(pre_synaptic - 1)
        return post_synaptic

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, MNIST_LAYER_SIZES, device):
        super(SpikingNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.device = device
        for i in range(len(MNIST_LAYER_SIZES) - 1):
            self.layers.append(SpikingNeuronLayer(MNIST_LAYER_SIZES[i], MNIST_LAYER_SIZES[i + 1], device))

    def forward(self, x):
        x = x.to(self.device)  # Ensure input tensor is on the correct device
        for layer in self.layers:
            x = layer(x)
        return x