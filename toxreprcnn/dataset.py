from typing import List, Optional, Union, Dict

import albumentations as albu
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


TransformType = Union[albu.BaseCompose, albu.BasicTransform]


class ToxReprCNNDataset(Dataset):
    def __init__(
        self,
        tiles: List[str],
        labels: np.ndarray,
        image_dict: Optional[Dict[str, np.ndarray]] = None,
        transform: Optional[TransformType] = None,
        length: Optional[int] = None,
        cache_mode: bool = False
    ):
        """
        X : patchのpath
        y : finding type
        wsi : WSI情報 (HE augmentationのようなWSI単位の情報が必要なときに使う)
        labels : yのndarray化
        transform : albumentationsで可能なaugmmentation
        """
        self.tiles = tiles
        self.labels = labels
        self.wsi = [t.split("/")[-2] for t in tiles]
        self.image_dict = image_dict
        self.transform = transform
        self.epoch = 0
        if length:
            self.length = length
        else:
            self.length = len(self.tiles)
        self.cache_mode = cache_mode

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        i = self.length*self.epoch + i
        if self.image_dict:
            image = self.image_dict[self.tiles[i]]
        else:
            image = cv2.imread(self.tiles[i])
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                print(self.tiles[i])
        image = self.transform(image=image)["image"]
        if self.labels is not None:
            return image, torch.tensor(self.labels[i]).float()
        else:
            return image, self.tiles[i]
    
    def set_epoch(self, epoch):
        self.epoch = epoch


class ToxReprCNNBalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset, n_frac = None, n_samples = None):
        avg = np.mean(dataset.labels, axis=0)
        avg[avg == 0] = 0.5
        avg[avg == 1] = 0.5
        self.avg = avg
        weights = (1 / (1 - avg + 1e-8)) * (1 - dataset.labels) + (
            1 / (avg + 1e-8)
        ) * dataset.labels
        weights = np.max(weights, axis=1)
        # weights = np.ones_like(dataset.labels[:,0])
        self.weights = weights
        if n_frac:
            super().__init__(weights, int(n_frac * len(dataset)))
        elif n_samples:
            super().__init__(weights, n_samples)

def load_features(path):
    return [line.rstrip() for line in open(path, "r")]
