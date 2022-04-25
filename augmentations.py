import os
import random

import numpy as np
from matplotlib import pyplot as plt
import albumentations as aug
import cv2
from torchvision.transforms import ToTensor
from scipy.ndimage import rotate

from typing import Tuple, Optional


class Preprocess:
    ''' The preprocessing for a robot arm observation
    '''

    def __init__(self, augmentations:Optional[Tuple[str]]=("",), bkrd_dir: Optional[str]=None) -> None:
        self.transform = self.makeTransform(augmentations)
        self.bkrd_img = None
        if bkrd_dir is not None:
            filename = os.path.join(bkrd_dir, random.choice(os.listdir(bkrd_dir)))
            self.bkrd_img = cv2.imread(filename)
            print(filename)

    def __call__(self, obs:np.ndarray, obs_mask:np.ndarray = None, gripper_roll:Optional[float]=0) :
        '''Takes an observation from the simulator and replaces the background, applies augmentations, and returns the transformed image with the channel dimension moved to the first dimension.
        '''
        if obs_mask is not None and self.bkrd_img is not None:
            obs = self.replaceBackground(obs, obs_mask)
        obs = self.reverseRotation(self.circleCrop(obs), gripper_roll)
        obs = self.transform(image=obs)['image']
        return obs

    def replaceBackground(self, obs, obs_mask):
        """ Replaces the background for the current observation with the image provided
        """
        bkrd = self.bkrd_img
        def map_fn(pix, bkrd_pix, mask_i): 
            return pix if mask_i > 0 else bkrd_pix
        
        rgb = [[map_fn(obs[i, j], bkrd[i, j], obs_mask[i, j]) 
                for j in range(64)] 
                for i in range(64)]
        return np.array(rgb, dtype=np.uint8)

    def circleCrop(self, rgb):
        r = len(rgb)//2
        center_coord = (r -1, r -1)
        mask = np.zeros(shape= rgb.shape, dtype=np.uint8)
        #fill in cirle of ones in center of mask
        cv2.circle(mask, center_coord, r, (255, 255, 255), -1)
        rgb = cv2.bitwise_and(rgb, mask)
        return rgb

    def reverseRotation(self, img, gripper_roll): 
        deg = gripper_roll * (180/np.pi) - 90
        return rotate(img, deg, reshape=False)

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
    
    bkrd_dir = '/Users/tom/Documents/tiny-imagenet-200/val/images/'
    
    prepro = Preprocess(bkrd_dir=bkrd_dir)

    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    # print(mask)
    


    obs = prepro.replaceBackground(img, mask)
    # 
    obs = prepro.circleCrop(obs)
    obs = prepro.reverseRotation(obs, np.pi/4)
    plt.imshow(obs)
    plt.show()


