"""
datasets/unlabeled_dataset.py - Unlabeled insect wing image dataset

Used for Stage 1: MAE pretraining.
"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class UnlabeledWingDataset(Dataset):
    """
    Aggregates all wing images from multiple datasets (Droso-big, mosquito, etc.)
    for self-supervised pretraining.

    No landmark labels required.
    """

    def __init__(self, root_dirs, image_size=224):
        """
        Args:
            root_dirs: list of directories containing images.
                       e.g. ['/data/droso_big', '/data/mosquito_nolte', ...]
            image_size: target size after resize
        """
        self.image_paths = []
        for root in root_dirs:
            root = Path(root)
            # Find all images (jpg, png, bmp, tiff)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                self.image_paths.extend(list(root.rglob(ext)))

        print(f"Found {len(self.image_paths)} unlabeled images for pretraining.")

        # Augmentation for pretraining
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Avoid strong augmentation — it distorts wing vein structure
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error reading {path}: {e}")
            # Return a random other image
            return self.__getitem__((idx + 1) % len(self))
