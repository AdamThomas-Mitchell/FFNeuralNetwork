import numpy as np
from activation import sigmoid
    
    
class FeedForward:
    def __init__(self, x, num_nodes, activation_fn):
        self.input = x
        self.num_nodes = num_nodes
        self.activation = activation_fn
        self.weights = None
        self.bias = None
        
    def init_weights(self):
        return np.random.random_sample((self.num_nodes, self.input.shape[-1]))
    
    def init_bias(self):
        return np.random.random_sample((self.num_nodes, 1))
    
    def forward(self):
        z = np.matmul(self.weights, self.input.T) + self.bias
        return self.activation(z)
    
    
    