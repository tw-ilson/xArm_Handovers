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

        self.latent_dim = self.conv.output_size 
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict mean and variance of each action possibility
            torch.nn.Linear(256, self.output_dim)
        )

    def forward(self, obs):
        hs = self.conv(obs)
        # print(f"hidden: {hs.shape}, latent_dim: {self.latent_dim}")
        y = self.mlp(hs)
        return y

    @torch.no_grad()
    def predict(self, obs):
        hs = self.conv(obs)
        # print(f"hidden: {hs.shape}, latent_dim: {self.latent_dim}")
        y = self.mlp(hs)
        return y

    def compute_score(self, state, action, Q):
        action_dist = self.forward(state)

        action_mean = action_dist[:, ::2]
        action_var = action_dist[:, 1::2]

        t_val = lambda a, mu, var: (a - mu)/(var**2) 
        assert action.shape == action_mean.shape, f"action: {action.shape}, \naction_mean:{action_mean.shape}"
        score = t_val(action, action_mean, action_var)
        return score * Q

if __name__ == "__main__":
    import conv_nets
    inputs = torch.zeros((3, 128, 128), dtype=torch.float32)
    conv = conv_nets.CNN(inputs.shape)
    net = ContinuousActorNetwork(conv)
    inputs = inputs.unsqueeze(0)
    inputs = torch.cat([inputs, inputs, inputs, inputs], 0)
    # print(inputs.shape)
    y = net.forward(inputs)
    # print(y)
