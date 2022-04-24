# pass through equivaraint CNN, PointwiseAvgPool, feed into MLP which predicts (xyz)
import torch
import torch.nn as nn

import conv_nets

# TODO experiment with kernel sizes for average pooling

class DeltaQNetwork(torch.nn.Module):
    def __init__(self, input_shape, equivariant=False)  -> None:
        """Creates equivariant X, Y, Z prediction network for input image

        Args:
            input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
            N (int, optional): Number of discrete rotations, or -1 for continuous. Defaults to 8.
            NOTES: have 2 predictions of direct joint values: (base and gripper), two predictions
            of combinations of motors: (forward z values, and pitch up down)
        """
        super().__init__()
        # (B, C, H, W)
        assert input_shape[1] == input_shape[2], "Input image should be square"
        self.input_shape = input_shape

        if equivariant:
            self.cnn = conv_nets.R2EquiCNN(input_shape)
        else:
            self.cnn = conv_nets.CNN(input_shape)

        self.mlp = torch.nn.Sequential(
            # torch.nn.Linear(self.cnn.output_size+5, 256),
            torch.nn.Linear(self.cnn.output_size, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict q values of each action possibility
            torch.nn.Linear(256, 12)
            # torch.nn.Linear(256, 81)
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, jpos) -> torch.Tensor:
        """Creates equivariant pose prediction from observation

        Args:
            x (torch.Tensor): Observation of image with block in it
            jpos (torch.Tensor): Observation of current arm joint positions
        Returns:
            x (float): the x position to move arm to
            y (float): the y position to move arm to
            z (float): the z position to move arm to
            theta (float): gripper roll
        """
        assert x.shape[1:
                       ] == self.input_shape, f"Observation shape must be {self.input_shape}, current is {x.shape[1:]}"
        assert jpos.shape[1] == 5, f"invalid joint positions: \n{jpos}"

        batch_size = x.shape[0]
        conv_out = self.cnn(x)
        state = conv_out
        # state = torch.cat((conv_out, jpos), dim=1)

        mlp_out = self.mlp(state)

        return mlp_out

    @torch.no_grad()
    def predict(self, x: torch.Tensor, jpos) -> torch.Tensor:
        "Predicts 4d action for an input state"
        mlp_out = self.forward(x, jpos)

        # concatenated argmaxes of each window of possible actions, for each batch element
        actions = torch.cat([torch.max(mlp_out[:, i:i+3], dim=1)[1].unsqueeze(1)
                            for i in range(0, 12, 3)], dim=1) - 1

        # actions = torch.max(mlp_out, dim=1)[1]
        return actions

    def compute_loss(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(q_pred, q_target)


if __name__ == "__main__":
    inputs = torch.zeros((2, 3, 128, 128), dtype=torch.float32)
    net = DeltaQNetwork((3, 128, 128))
    print(net.predict(inputs))
