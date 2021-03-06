import os
import random

import numpy as np
from matplotlib import pyplot as plt
# import albumentations as aug
import cv2
from torchvision.transforms import ToTensor

from typing import Tuple, Optional


class Preprocess:
    ''' The preprocessing for a robot arm observation
    '''

    def __init__(self, augmentations:Optional[Tuple[str]]=("",), bkrd_dir: Optional[str]=None) -> None:
        self.transform = self.makeTransform(augmentations)
        if bkrd_dir:
            filename = os.path.join(bkrd_dir, random.choice(os.listdir(bkrd_dir)))
            self.bkrd_img = cv2.imread(filename)
            print(filename)

    def __call__(self, obs:np.ndarray, obs_mask:np.ndarray = None) :
        '''Takes an observation from the simulator and replaces the background, applies augmentations, and returns the transformed image with the channel dimension moved to the first dimension.
        '''
        if obs_mask is not None:
            obs = self.replaceBackground(obs, obs_mask)
        # print(type(obs))
        obs = self.transform(image=obs)['image']
        return obs#ToTensor()(obs)

    def replaceBackground(self, obs, obs_mask):
        """ Replaces the background for the current observation with the image provided
        """
        bkrd = self.bkrd_img
        def map_fn(pix, bkrd_pix, mask_i): 
            return pix if mask_i > 0 else bkrd_pix
        
        obs = [[map_fn(obs[i, j], bkrd[i, j], obs_mask[i, j]) 
                for j in range(64)] 
                for i in range(64)]
        return np.array(obs, dtype=np.uint8)

    def makeTransform(self, augmentations: Tuple[str]):
        """
        Initializes the augmentation pipeline with the list of provided augments
        """

        pipeline = []

        #NOTE: Include crop?

        if 'blur' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.Sharpen(p=1),
                            aug.Blur(blur_limit=3, p=1),
                            aug.MedianBlur(blur_limit=3, p=1),
                            aug.MotionBlur(blur_limit=3, p=1),
                        ],
                        p=1,
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
                        p=1,
                    )
            )

        if 'contrast' in augmentations:
            pipeline.append(
                    aug.OneOf(
                        [ 
                            aug.RandomContrast(p=1),
                            aug.HueSaturationValue(p=1),
                        ],
                        p=1
                    )
            )

        return aug.Compose(transforms=pipeline)

if __name__ == '__main__':


    print(filename)

    prepro = Preprocess(bkrd_path=filename)
    # print(prepro.bkrd_img)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32:, 32:] = 1
    # print(mask)

    print(prepro.bkrd_img.shape)
    plt.imshow(prepro.bkrd_img)
    plt.show()

    obs = prepro.replaceBackground(img, mask)
    
    plt.imshow(obs)
    plt.show()


