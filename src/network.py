import numpy as np
from layer import FeedForward
from activation import sigmoid




net_struct = {
    'layer_1': {'num_nodes': 32, 'activation': sigmoid},
    'layer_2': {'num_nodes': 16, 'activation': sigmoid}
}

net_struct = {
    'layer_1': (32, sigmoid),
    'layer_2': (16, sigmoid)
}


class FeedForwardNet:
    def __init__(self, input_dim, output_dim, net_struct):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_struct = net_struct
                
        self.weights_dict = None
        self.bias_dict = None
        
        self.init_net()
            
    def init_net(self):
        for key, value in self.net_struct.items():
            print(f'key: {key}')
            setattr(self, key, None)
    
    def get_weights_dict(self):
        pass
    
    def get_bias_dict(self):
        pass
    
    
if __name__=='__main__':
    net = FeedForwardNet(4, 2, net_struct)
    print(net.__dict__)