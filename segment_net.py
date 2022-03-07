import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import segmentation_models_pytorch as smp
import albumentations as aug
import h5py
import cv2

import pybullet as pb

from typing import Dict, Tuple, List

import os

"""
Image segmentation network used to pre-process the image obtained from wrist mounted camera. We will take a pre-trained network and tune it to produce a segmentation mask on the object if it is in frame.
"""

class ObjectMaskNetwork(torch.Module) :
    """
    Convolutional Auto-encoder pre-trained on ImageNet
    """

    def __init__(self, arch="FPN", encoder="resnet", in_channels=3, out_classes=1):
        self.model = smp.create_model( \
            arch, encoder_name=encoder, in_channels=in_channels, classes=out_classes) 

         # preprocessing parameteres for image
        self.params = smp.encoders.get_preprocessing_params(encoder)

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        """
        parameters
        ===========
        x:
            image tensor batch from wrist camera of shape (B, 3, H, W)

        returns
        ===========
        image_mask:
            image mask tensor batch of shape (B, H, W)
        """

        image_mask = self.model(x)
        return image_mask

    def train(self, x):
        pass


class ObjectMaskDataset(Dataset):
    """
    Datesets and augmentations for object segmentation task.

    Args:
        path_to_hdf5 (str): path to images folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    """ 

    def __init__(self, 
            path_to_hdf5,
            augmentation: aug.Compose=None, 
            preprocessing: aug.Compose=None):
        
        with h5py.File(path_to_hdf5, 'r') as hf:
            self.imgs = np.array(hf['imgs'])
            self.masks = np.array(hf['masks'])

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        """
        gets a singular sample from the data
        """
        image =  ToTensor()(self.imgs[i])

        mask = torch.tensor(self.masks[i], dtype=torch.int8)

        sample = {'img': image, 'mask': mask}

        if self.augmentation:
            sample = self.augment(sample)

        return sample

    def __len__(self):
        return len(self.imgs)

    def augment(self, images: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        """
        applies augmentations to a batch of image-mask mappings

        parameters
        ===========
        images:
        2-element dictionary containing training samples
            dict { 
                'img': image tensor, 
                'mask': mask tensor
                }
        returns
        ==========
        2-element dictionary containing augmented training samples
        """

        raise NotImplemented
