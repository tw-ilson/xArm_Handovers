import torch
import torch.nn as nn
import numpy as np

class ContinuousActorNetwork(nn.Module):
    '''Actor network with a gaussian policy. Network predicts 8 values, a mean and variance for each of the 4 action spaces.
    '''

    def __init__(self, cnn) -> None:
        super().__init__()

        self.input_dim = state_space + 6
        # (mean, variance) for 4 action spaces
        self.output_dim = 8

        #import convolutional network feature extractor for state images
        self.conv = cnn
        
          
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(400, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict mean and variance of each action possibility
            # NOTE: should variance be fixed?
            torch.nn.Linear(256, 8)
        )

    def compute_score(self, state, action_sample):
        action_dist = self.forward(state)

        action_mean = action_dist[::2]
        action_var = action_dist[1::2]

        t_val = lambda a, mu, var: (a - mu)/(var**2) 







if __name__ == "__main__":
    import conv_nets
    inputs = torch.zeros((2, 3, 128, 128), dtype=torch.float32)
    conv = conv_nets.CNN(128)
    net = ContinuousActorNetwork(inputs.shape)  
    print(net.predict(inputs))
