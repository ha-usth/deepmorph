"""
datasets/landmark_dataset.py - Labeled landmark dataset

Used for Stage 2: Few-shot fine-tuning.
Format matches iMorph: each image has a .txt file with the same name
containing landmark coordinates.
"""
import os
import math
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_heatmap


def _random_affine_augment(img, landmarks, image_size,
                            scale_range=(-0.05, 0.05),
                            translate_range=(-0.05, 0.05),
                            rotate_range=(-22.0, 22.0)):
    """
    Apply random scale, translation, and rotation to a PIL image and its landmarks.
    All transforms are composed into a single affine warp applied around the image center.
    """
    W, H = img.size  # both equal image_size after resize
    cx, cy = W / 2.0, H / 2.0

    scale = 1.0 + np.random.uniform(*scale_range)
    tx    = np.random.uniform(*translate_range) * W
    ty    = np.random.uniform(*translate_range) * H
    theta = math.radians(np.random.uniform(*rotate_range))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Forward 3x3 affine: scale+rotate around center, then translate
    M = np.array([
        [scale*cos_t, -scale*sin_t, cx*(1 - scale*cos_t) + cy*scale*sin_t + tx],
        [scale*sin_t,  scale*cos_t, cy*(1 - scale*cos_t) - cx*scale*sin_t + ty],
        [0.0,          0.0,          1.0],
    ], dtype=np.float64)

    # PIL AFFINE maps output → input, so pass the inverse
    M_inv = np.linalg.inv(M)
    img = img.transform(
        img.size, Image.AFFINE,
        M_inv[:2].flatten().tolist(),
        resample=Image.BILINEAR,
    )

    # Apply forward transform to landmark coordinates
    K = len(landmarks)
    pts = np.concatenate([landmarks, np.ones((K, 1))], axis=1).T  # (3, K)
    landmarks = (M @ pts).T[:, :2]                                  # (K, 2)

    return img, landmarks


class LandmarkDataset(Dataset):
    """
    Labeled landmark dataset.

    Directory structure (iMorph format):
        data_dir/
            001.bmp
            001.txt    <- landmark coordinates, one "x y" per line
            002.bmp
            002.txt
            ...
    """

    def __init__(self,
                 data_dir,
                 image_list=None,        # specific image list (for few-shot)
                 num_landmarks=15,
                 image_size=224,
                 heatmap_size=56,
                 sigma=2.0,
                 augment=True):
        """
        Args:
            data_dir: directory containing images + .txt files
            image_list: specific image filenames; if None, uses all available.
                        Important for few-shot: select N random images.
            num_landmarks: number of landmark classes
            image_size: model input image size
            heatmap_size: output heatmap size
            sigma: Gaussian heatmap width
            augment: whether to apply augmentation
        """
        self.data_dir = Path(data_dir)
        self.num_landmarks = num_landmarks
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment

        # Find images that have a matching .txt file
        if image_list is None:
            image_list = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                for img_path in self.data_dir.glob(ext):
                    txt_path = img_path.with_suffix('.txt')
                    if txt_path.exists():
                        image_list.append(img_path.name)

        self.image_names = image_list
        print(f"Dataset {data_dir}: {len(self.image_names)} labeled images.")

        # Transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_names)

    def _load_landmarks(self, txt_path):
        """Read iMorph .txt format: one 'x y' per line."""
        landmarks = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    landmarks.append([x, y])
        return np.array(landmarks)  # (K, 2)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.data_dir / img_name
        txt_path = img_path.with_suffix('.txt')

        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Load landmarks
        landmarks = self._load_landmarks(txt_path)  # (K, 2) in original coords

        # Resize image + scale landmarks accordingly
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        landmarks_scaled = landmarks.copy()
        landmarks_scaled[:, 0] *= scale_x
        landmarks_scaled[:, 1] *= scale_y

        if self.augment:
            # Horizontal flip with probability 0.5
            # if np.random.rand() < 0.5:
            #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #     landmarks_scaled[:, 0] = self.image_size - landmarks_scaled[:, 0]

            # Scale / translate / rotate
            img, landmarks_scaled = _random_affine_augment(img, landmarks_scaled, self.image_size)

        # Convert to tensor + normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
        img = self.normalize(img)

        # Generate heatmaps for each landmark (at heatmap_size coords)
        scale_to_heatmap = self.heatmap_size / self.image_size
        landmarks_heatmap = landmarks_scaled * scale_to_heatmap

        heatmaps = torch.zeros(self.num_landmarks,
                               self.heatmap_size, self.heatmap_size)
        for k in range(min(self.num_landmarks, len(landmarks_heatmap))):
            x, y = landmarks_heatmap[k]
            heatmaps[k] = generate_heatmap(
                (x, y), (self.heatmap_size, self.heatmap_size), self.sigma)

        # Always return (num_landmarks, 2) to avoid shape mismatch in collate
        coords_target = torch.zeros(self.num_landmarks, 2)
        n = min(self.num_landmarks, len(landmarks_scaled))
        coords_target[:n] = torch.from_numpy(landmarks_scaled[:n]).float()

        return {
            'image': img,
            'heatmaps': heatmaps,
            'coords': coords_target,  # (K, 2) at image_size scale
            'orig_size': torch.tensor([orig_w, orig_h], dtype=torch.float32),  # (2,)
        }


def select_few_shot_subset(data_dir, n_shots, seed=42):
    """
    Randomly select N images from a directory for few-shot fine-tuning.
    The remainder becomes the test set.
    """
    data_dir = Path(data_dir)
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        for img_path in data_dir.glob(ext):
            txt_path = img_path.with_suffix('.txt')
            if txt_path.exists():
                all_images.append(img_path.name)

    if n_shots >= len(all_images):
        print(f"[WARNING] n_shots={n_shots} >= total labeled images={len(all_images)}. "
              f"Test set will be empty — evaluation not possible.")

    rng = np.random.RandomState(seed)
    rng.shuffle(all_images)

    train_list = all_images[:n_shots]
    test_list = all_images[n_shots:]

    return train_list, test_list
