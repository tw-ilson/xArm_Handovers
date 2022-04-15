# pass through equivaraint CNN, PointwiseAvgPool, feed into MLP which predicts (xyz)
import torch
import torch.nn as tnn

from e2cnn import gspaces
from e2cnn import nn

# TODO experiment with kernel sizes for average pooling

from augmentations import Preprocess


class ValueNetwork(torch.nn.Module):
    def __init__(self, cnn) -> None:
        """Produces value estimation for state-action pairs 

        Args:
            cnn: the feature extractor to apply to the image
            input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
            NOTES: have 2 predictions of direct joint values: (base and gripper), two predictions
            of combinations of motors: (forward z values, and vertical position adjust)
        """
        super().__init__()
        # (B, C, H, W)

        #Constructed with convolutional neural network feature extractor provided.
        self.cnn = cnn

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.output_size, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            # predict values of state
            torch.nn.Linear(256, 1)
        )

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, state) -> torch.Tensor:
        """Creates equivariant pose prediction from observation

        Args:
            state (torch.Tensor): Observation of image with block in it
        Returns:
            q-value estimation associated with the state-action pair
        """
        # assert state.shape[1:] == self.input_shape, f"Observation shape must be {self.input_shape}, current is {x.shape[1:]}"

        batch_size = state.shape[0]
        # inp = nn.GeometricTensor(x, nn.FieldType(
        #     self.r2_act, self.input_shape[0]*[self.r2_act.trivial_repr]))
        conv_out = self.cnn(state)
        # assert conv_out.shape == torch.Size((batch_size, self.N * self.cnn_out_channels)
        #                                     ), f"Conv size: {conv_out.shape} != required: {torch.Size((batch_size, self.N * self.cnn_out_channels))}"

        mlp_out = self.mlp(conv_out)

        return mlp_out

    @torch.no_grad()
    def predict(self, state: torch.Tensor) -> torch.Tensor:
        "Predicts 4d action for an input state"
        V = self.forward(state)

        # # concatenated argmaxes of each window of possible actions, for each batch element
        # actions = torch.cat([torch.max(mlp_out[:, i:i+3], dim=1)[1].unsqueeze(1)
        #                     for i in range(0, 12, 3)], dim=1) - 1

        return V

    def compute_loss(self, q_pred: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(q_pred, q_target)


# if __name__ == "__main__":
#     inputs = torch.zeros((2, 3, 128, 128), dtype=torch.float32)
#     net = EquivariantDeltaNetwork((3, 128, 128))
#     print(net.predict(inputs))
