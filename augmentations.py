
import albumentations as aug
import cv2
from torchvision.transforms import ToTensor

from typing import Tuple, Optional

class Preprocess:

    def __init__(self, augmentations:Optional[Tuple[str]], bkrd_path: Optional[str]) -> None:
        if augmentations:
            self.transform = self.makeTransform(augmentations)
        if bkrd_path:
            self.bkrd_img = cv2.imread(bkrd_path)

    def __call__(self, obs, obs_mask) :
        '''Takes an observation from the simulator and replaces the background, applies augmentations, and returns the transformed image with the channel dimension moved to the first dimension.
        '''
        img = self.replaceBackground(obs, obs_mask)
        img = self.transform(img)
        return ToTensor()(img)

    def replaceBackground(self, obs, obs_mask):
        """ Replaces the background for the current observation with the image provided
        """
        for pix in range( len(obs_mask)):
            if obs_mask[pix] == 0:
                obs[pix, :] = self.bkrd_img[pix, :] #replace rgb channels 
        return obs

    def makeTransform(self, augmentations: Tuple[str]):
        """
        Initializes the augmentation pipeline with the list of provided augments
        """

        pipeline = []

        #TODO: Include crop?

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
