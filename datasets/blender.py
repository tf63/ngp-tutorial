import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class BlenderDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "train", "transforms.json"), 'r') as f:
            meta = json.load(f)

        w = h = int(512 * self.downsample)
        fx = fy = 0.5 * w / np.tan(0.5 * meta['camera_angle_x']) * self.downsample

        K = np.float32([[fx, 0, w / 2],
                        [0, fy, h / 2],
                        [0, 0, 1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'trainval':
            with open(os.path.join(self.root_dir, "train", "transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]
            with open(os.path.join(self.root_dir, "val", "transforms.json"), 'r') as f:
                frames += json.load(f)["frames"]
        else:
            with open(os.path.join(self.root_dir, split, "transforms.json"), 'r') as f:
                frames = json.load(f)["frames"]

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            c2w = np.array(frame['transform_matrix'])[:3, :4]

            c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            pose_radius_scale = 1.5
            c2w[:, 3] /= np.linalg.norm(c2w[:, 3]) / pose_radius_scale
            self.poses += [c2w]

            try:
                img_path = os.path.join(self.root_dir, split, f"{frame['file_path']}")
                img = read_image(img_path, self.img_wh)
                self.rays += [img]
            except BaseException:
                pass

        if len(self.rays) > 0:
            self.rays = torch.FloatTensor(np.stack(self.rays))  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
