from typing import Tuple, List, Optional, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import numpy as np
import h5py
from tqdm import tqdm

from grasping_env import HandoverGraspingEnv

def create_label_pair(n_pairs, img_size, seed,
        render=False, show_progress=False):
    '''Runs env to collect samples, for parallelization'''
    np.random.seed(seed)
    env = HandoverGraspingEnv(img_size=img_size, render=render)

    imgs = []
    masks = []

    if show_progress:
        pbar = tqdm(total=n_pairs)

    while len(masks) < n_pairs:

        env.reset_object_position()

        img, mask= env.get_obs()

        imgs.append(img)
        masks.append(mask)

        if show_progress:
            pbar.update(1)

    return imgs, masks

def replace_obs_background(original: np.ndarray, bkrd_path: str):
    '''
    takes a picture observed from the camera in the simulator and replaces white background with image (from subset of ImageNet)
    Params
    ------
    original: 
        original rgb image from simulator observation. matrix shape (H, W, 3)
    bkrd_path:
        path to image which will be used as background
    Returns
    ------
    new_img:
        image with replaced background shape (H, W, 3)
    '''





def collect_dataset(dataset_name: str,
                    size: int,
                    success_only: bool=False,
                    img_size: int=42,
                    render: bool=True,
                    show_progress: bool=True,
                    seed: int=0,
                    n_processes: int=1,
                   ) -> None:
    """ watch dataset collection """

    with ProcessPoolExecutor() as executor:
        bg_futures = []
        for i in range(1, n_processes):
            new_future = executor.submit(create_label_pair, size//n_processes, 
                                         img_size, seed=seed+1000*i)
            bg_futures.append(new_future)

        size_left = size - (n_processes-1) * (size//n_processes)
        imgs, masks = create_label_pair(size_left, img_size, seed,
                                    render, show_progress)

        for f in as_completed(bg_futures):
            _imgs, _masks = f.result()
            imgs.extend(_imgs)
            masks.extend(_masks)
            

    # save it
    with h5py.File(dataset_name, 'w') as hf:
        hf.create_dataset('images', data=np.array(imgs))
        hf.create_dataset('masks', data=np.array(masks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='File path where data will be saved')
    parser.add_argument('--size', '-s', type=int, default=10000,
                        help='Number of image-label pairs in dataset')
    parser.add_argument('--render', action='store_true',
                        help='If true, render gui during dataset collection')
    parser.add_argument('--seed', type=int, default=0,
                        help='Numpy random seed')
    parser.add_argument('--n-processes', '-np', type=int, default=4,
                        help='Number of parallel processes')
    args = parser.parse_args()

    collect_dataset(dataset_name= args.name,
                    size= args.size,
                    render= args.render,
                    seed= args.seed,
                    n_processes= args.n_processes,
                   )
