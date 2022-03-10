import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import segmentation_models_pytorch as smp
import albumentations as aug
import cv2
import h5py

from typing import Dict, Tuple, List


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

    def compute_loss(self,
            y_pred: torch.Tensor,
            y_target: torch.Tensor):
        return self.loss_fn(y_pred, y_target)


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
            augmentations: Tuple[str]=None):
        
        with h5py.File(path_to_hdf5, 'r') as hf:
            self.imgs = np.array(hf['images'])
            self.masks = np.array(hf['masks'])

        if augmentations:
            self.transform = self.makeTranform(augmentations)
        #self.preprocessing = preprocessing

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        gets a singular sample from the data
        """
        image =  ToTensor()(self.imgs[i])

        mask = torch.tensor(self.masks[i], dtype=torch.int8)


        if self.augmentation:
            sample = self.transform(image=image, mask=mask)
        else:
            sample = {'image': image, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.imgs)

    def makeTransform(self, augmentations: Tuple[str]):
        """
        Initializes the augmentation pipeline with the list of provided augments
        """

        pipeline = []

        if 'blur' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.IAASharpen(p=1),
                            aug.Blur(blur_limit=3, p=1),
                            aug.MedianBlur(blur_limit=3, p=1),
                            aug.MotionBlur(blur_limit=3, p=1),
                        ],
                        p=0.9,
                    )  
            )

        if 'brightness' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.CLAHE(p=1),
                            aug.RandomBrightness(p=1),
                            aug.RandomGamma(p=1),
                        ],
                        p=0.9,
                    )
            )

        if 'contrast' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.RandomContrast(p=1),
                            aug.HueSaturationValue(p=1),
                        ]
                    )
            )

        return aug.Compose(pipeline)
