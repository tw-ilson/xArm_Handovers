import torch
import torch.nn as tnn
from e2cnn import gspaces
import e2cnn.nn as nn

class CNN(tnn.Module):

    def __init__(self, input_shape):
        '''
        Args:
            input_size int, optional): width dimension of square image with 3 channels
            
        '''
        super().__init__()

        self.input_shape = input_shape
        self.conv_out_channels = 4 
        
        self.conv = tnn.Sequential(
            tnn.Conv2d(3, 16, kernel_size=3, padding=1),
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

        self.output_size = self.forward(torch.zeros(*input_shape).unsqueeze(0)).shape[1]

    def forward(self, x):
        x = self.conv(x)
        # print(x.shape)
        return x.flatten(1)

class R2EquiCNN(tnn.Module):

    def __init__(self, input_shape, N=8):
        '''
        Args:
            input_shape (tuple, optional): Shape of the image (C, H, W). Should be square.
            N: number of discrete rotations 
        '''
        super().__init__()

        # (B, C, H, W)
        assert input_shape[1] == input_shape[2], "Input image should be square"

        self.N = N

        self.input_shape = input_shape
        self.conv_out_channels = 4

        self.r2_act = gspaces.Rot2dOnR2(N)

        self.conv = torch.nn.Sequential(
            # 128x128
            nn.R2Conv(nn.FieldType(self.r2_act, input_shape[0]*[self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 16*[self.r2_act.regular_repr]), 3),
            # 64x64
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16 * \
                                   [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 16 * [self.r2_act.regular_repr]), 2),
            # 32x32
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(self.r2_act, 16 * \
                                   [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, 16 * \
                    [self.r2_act.regular_repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(
                self.r2_act, 16 * [self.r2_act.regular_repr]), 2),
            # 16x16
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.regular_repr]),
                      nn.FieldType(
                          self.r2_act, self.conv_out_channels * [self.r2_act.regular_repr]),
                      kernel_size=3, padding=1),
            nn.ReLU(nn.FieldType(self.r2_act, self.conv_out_channels * \
                    [self.r2_act.regular_repr])),
            # get equivariant feature vector
            nn.PointwiseAvgPool(nn.FieldType(
                self.r2_act, self.conv_out_channels * [self.r2_act.regular_repr]), 4)
        )

    def forward(self, x):
        return self.conv(x).flatten(1)

if __name__ == '__main__':
    conv = CNN(84)
    print(conv.out_size)
