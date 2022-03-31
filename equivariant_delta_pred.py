# pass through equivaraint CNN, PointwiseAvgPool, feed into MLP which predicts (xyz)
import torch
import torch.nn as tnn

from e2cnn import gspaces
from e2cnn import nn

# TODO experiment with kernel sizes for average pooling


class EquivariantDeltaNetwork(torch.nn.Module):
    def __init__(self, input_shape, N=8) -> None:
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

        self.N = N
        self.actions = [-1, 0, 1]

        self.input_shape = input_shape
        self.conv_out_channels = 16

        self.r2_act = gspaces.Rot2dOnR2(N)

        self.conv = tnn.Sequential(
            tnn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
            tnn.ReLU(True),
            tnn.MaxPool2d(kernel_size=3),
            tnn.Conv2d(16, 16, kernel_size=3, padding=1),
            tnn.ReLU(True),
            tnn.MaxPool2d(kernel_size=2),
            tnn.Conv2d(16, 16, kernel_size=3, padding=1),
            tnn.ReLU(True),
            tnn.MaxPool2d(kernel_size=2),
            tnn.Conv2d(16, self.conv_out_channels, kernel_size=3, padding=1),
            tnn.ReLU(True),
            tnn.AvgPool2d(kernel_size=2)
        )
        # self.conv = torch.nn.Sequential(
        #     # 128x128
        #     nn.R2Conv(nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr]),
        #               nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
        #               kernel_size=3, padding=1),
        #     nn.ReLU(nn.FieldType(self.r2_act, 16 * \
        #             [self.r2_act.regular_repr]), inplace=True),
        #     nn.PointwiseMaxPool(nn.FieldType(
        #         self.r2_act, 16*[self.r2_act.regular_repr]), 3),
        #     # 64x64
        #     nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
        #               nn.FieldType(self.r2_act, 16 * \
        #                            [self.r2_act.regular_repr]),
        #               kernel_size=3, padding=1),
        #     nn.ReLU(nn.FieldType(self.r2_act, 16 * \
        #             [self.r2_act.regular_repr]), inplace=True),
        #     nn.PointwiseMaxPool(nn.FieldType(
        #         self.r2_act, 16 * [self.r2_act.regular_repr]), 2),
        #     # 32x32
        #     nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
        #               nn.FieldType(self.r2_act, 16 * \
        #                            [self.r2_act.regular_repr]),
        #               kernel_size=3, padding=1),
        #     nn.ReLU(nn.FieldType(self.r2_act, 16 * \
        #             [self.r2_act.regular_repr]), inplace=True),
        #     nn.PointwiseMaxPool(nn.FieldType(
        #         self.r2_act, 16 * [self.r2_act.regular_repr]), 2),
        #     # 16x16
        #     nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
        #               nn.FieldType(
        #                   self.r2_act, self.conv_out_channels * [self.r2_act.regular_repr]),
        #               kernel_size=3, padding=1),
        #     nn.ReLU(nn.FieldType(self.r2_act, self.conv_out_channels * \
        #             [self.r2_act.regular_repr])),
        #     # get equivariant feature vector
        #     nn.PointwiseAvgPool(nn.FieldType(
        #         self.r2_act, self.conv_out_channels * [self.r2_act.regular_repr]), 4)
        # )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(400, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict q values of each action possibility
            torch.nn.Linear(256, 12)
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x) -> torch.Tensor:
        """Creates equivariant pose prediction from observation

        Args:
            x (torch.Tensor): Observation of image with block in it
        Returns:
            x (float): the x position to move arm to
            y (float): the y position to move arm to
            z (float): the z position to move arm to
            theta (float): gripper roll
        """
        assert x.shape[1:
                       ] == self.input_shape, f"Observation shape must be {self.input_shape}, current is {x.shape[1:]}"

        batch_size = x.shape[0]
        # inp = nn.GeometricTensor(x, nn.FieldType(
        #     self.r2_act, self.input_shape[0]*[self.r2_act.trivial_repr]))
        conv_out = self.conv(x).flatten(1)
        # assert conv_out.shape == torch.Size((batch_size, self.N * self.conv_out_channels)
        #                                     ), f"Conv size: {conv_out.shape} != required: {torch.Size((batch_size, self.N * self.conv_out_channels))}"

        mlp_out = self.mlp(conv_out)

        return mlp_out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        "Predicts 4d action for an input state"
        mlp_out = self.forward(x)

        # concatenated argmaxes of each window of possible actions, for each batch element
        actions = torch.cat([torch.max(mlp_out[:, i:i+3], dim=1)[1].unsqueeze(1)
                            for i in range(0, 12, 3)], dim=1) - 1

        return actions

    def compute_loss(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(q_pred, q_target)


if __name__ == "__main__":
    inputs = torch.zeros((2, 3, 128, 128), dtype=torch.float32)
    net = EquivariantDeltaNetwork((3, 128, 128))
    print(net.predict(inputs))
