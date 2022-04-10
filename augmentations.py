
import numpy as np
import albumentations as aug
import cv2
from torchvision.transforms import ToTensor

from typing import Tuple, Optional

class Preprocess:

    def __init__(self, *augmentations, bkrd_path: Optional[str]=None) -> None:
        
        if augmentations:
            self.transform = self.makeTransform(*augmentations)
        if bkrd_path:
            self.bkrd_img = cv2.imread(bkrd_path)

    def __call__(self, obs, obs_mask = None) :
        '''Takes an observation from the simulator and replaces the background, applies augmentations, and returns the transformed image with the channel dimension moved to the first dimension.
        '''
        if obs_mask:
            obs = self.replaceBackground(obs, obs_mask)
        obs = self.transform(image=obs)['image']
        return obs#ToTensor()(obs)

    def replaceBackground(self, obs, obs_mask):
        """ Replaces the background for the current observation with the image provided
        """
        # add virtual background for augmentation purposes (untested)
        if obs_mask is not None:
            def map_fn(pix, bkrd_pix, mask_i,
                       ): return pix if mask_i != 0 else bkrd_pix
            obs = np.vectorize(map_fn)(zip(obs[:2], self.bkrd_img[:2], obs_mask))
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
                            aug.Blur(blur_limit=30, p=1),
                            aug.MedianBlur(blur_limit=31, p=1),
                            aug.MotionBlur(blur_limit=30, p=1),
                        ],
                        p=1,
                    )  
            )

        if 'brightness' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.CLAHE(tile_grid_size=(3, 3), p=1), # same size of kernel
                            aug.RandomBrightness(0.5, p=1),
                            aug.RandomGamma([0, 120], p=1),
                            aug.RandomShadow(p=1)
                        ],
                        p=1,
                    )
            )

        if 'noise' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.GaussNoise(var_limit=(20, 80), p=1),
                            aug.ISONoise(intensity=[.2, .8], p=1)
                        ],
                        p=1
                    )
            )

        if 'contrast' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.RandomContrast(p=1),
                            aug.HueSaturationValue( p=1),
                        ],
                        p=1
                    )
            )

        return aug.Compose(transforms=pipeline)
