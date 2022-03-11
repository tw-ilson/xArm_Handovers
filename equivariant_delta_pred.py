# pass through equivaraint CNN, PointwiseAvgPool, feed into MLP which predicts (xyz)
import torch

from e2cnn import gspaces
from e2cnn import nn

# TODO experiment with kernel sizes for average pooling


class EquivariantDeltas(torch.nn.Module):
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
        assert (
            input_shape[1] == input_shape[2],
            "Input image should be square"
        )
        self.N = N
        self.actions = []
        for _ in range(4):
            self.actions.extend([-1, 0, 1])

        self.input_shape = input_shape
        self.conv_out_channels = 8

        self.r2_rot = gspaces.Rot2dOnR2(N)
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 16*[self.r2_act.regular_repr]), 2),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 32 * \
                                   [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 32 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 32 * [self.r2_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r2_act, 32 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 64 * \
                                   [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 64 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 64 * [self.r2_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r2_act, 64 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 128 * \
                                   [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 128 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 128 * [self.r2_act.regular_repr]), 2),
            # 8x8
            nn.R2Conv(nn.FieldType(self.r2_act, 128 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 256 * \
                                   [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 256 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 256 * [self.r2_act.regular_repr]), 2),

            nn.R2Conv(nn.FieldType(self.r2_act, 256 * [self.r2_act.regular_repr]),
                      nn.FieldType(
                          self.r2_act, self.conv_out_channels * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            # get equivariant feature vector, should be [B, conv_out_channels * N, 1, 1]
            nn.PointwiseAvgPool(nn.FieldType(
                self.r2_act, 256 * [self.r2_act.regular_repr]), 11)

        )
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.conv_out_channels * self.N, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict q values of each action possibility
            torch.nn.Linear(256, 12)
        )

    def forward(self, x) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
        """Creates equivariant pose prediction from observation

        Args:
            x (torch.Tensor): Observation of image with block in it
        Returns:
            x (float): the x position to move arm to
            y (float): the y position to move arm to
            z (float): the z position to move arm to
            theta (float): gripper roll
        """
        assert(
            x.shape[1:] == self.input_shape,
            f"Observation shape must be {self.input_shape}, current is {x.shape[1:]}"
        )
        batch_size = x.shape[0]
        inp = nn.GeometricTensor(x, nn.FieldType(
            self.r2_rot, self.input_shape[0]*[self.r2_act.trivial_repr]))
        conv_out = self.conv(inp).tensor.squeeze().reshape(batch_size, -1)
        assert (
            conv_out.size == torch.Size(
                (batch_size, self.N * self.conv_out_channels)),
            f"Conv size: {conv_out.size} != required: {torch.Size((batch_size, self.N * self.conv_out_channels))}"
        )
        # TODO see if permute to conv_out is required (shouldn't be though)
        mlp_out = self.mlp(conv_out)

        x_act = torch.max(mlp_out[0:3], dim=1)[1]
        y_act = 3 + torch.max(mlp_out[3:6], dim=1)[1]
        z_act = 6 + torch.max(mlp_out[6:9], dim=1)[1]
        theta_act = 9 + torch.max(mlp_out[9:12], dim=1)[1]

        return torch.Tensor(self.actions[x_act], self.actions[y_act],
                            self.actions[z_act], self.actions[theta_act],
                            dtype=torch.float32)
