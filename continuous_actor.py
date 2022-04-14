import torch
import torch.nn as nn
import numpy as np

class ContinuousActorNetwork(nn.Module):
    '''Actor network with a gaussian policy. Network predicts 8 values, a mean and variance for each of the 4 action spaces.
    '''

    def __init__(self, cnn) -> None:
        super().__init__()

        # (mean, variance) for 4 action spaces
        self.output_dim = 8

        #import convolutional network feature extractor for state images
        self.conv = cnn

        self.latent_dim = self.conv.output_size + 6
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict mean and variance of each action possibility
            torch.nn.Linear(256, 8)
        )

    def forward(self, x):
        #TODO: implement
        hs = self.conv(x)
        y = self.mlp(hs, 1)
        return y

    def compute_score(self, state, action, Q):
        action_dist = self.forward(state)

        action_mean = action_dist[::2]
        action_var = action_dist[1::2]

        t_val = lambda a, mu, var: (a - mu)/(var**2) 
        score = t_val(action, action_mean, action_var).sum()
        return score * Q

if __name__ == "__main__":
    import conv_nets
    inputs = torch.zeros((2, 3, 128, 128), dtype=torch.float32)
    conv = conv_nets.CNN(128)
    net = ContinuousActorNetwork(inputs.shape)  
    print(net.predict(inputs))
