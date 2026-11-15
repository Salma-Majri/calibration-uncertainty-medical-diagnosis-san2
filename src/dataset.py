"""Dataset PyTorch pour FedISIC2019 : lit uniquement les indices d'un split donne."""
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AddGaussianNoise:
    """Bruit gaussien sur l'image en [0,1], AVANT normalisation (voir protocol.yaml)."""

    def __init__(self, sigma):
        self.sigma = float(sigma)

    def __call__(self, x):
        if self.sigma <= 0:
            return x
        return (x + torch.randn_like(x) * self.sigma).clamp(0.0, 1.0)


def build_transform(cfg, train, noise_sigma=0.0):
    size = cfg["model"]["image_size"]
    normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.ToTensor(),
            normalize,
        ])
    steps = [T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor()]
    if noise_sigma and noise_sigma > 0:
        steps.append(AddGaussianNoise(noise_sigma))
    steps.append(normalize)
    return T.Compose(steps)


def _image_column(hf):
    from datasets import Image

    for name, feat in hf.features.items():
        if isinstance(feat, Image):
            return name
    raise ValueError("Aucune colonne image trouvee dans le dataset.")


class ISICDataset(Dataset):
    def __init__(self, hf, indices, transform):
        self.hf = hf
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.img_col = _image_column(hf)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = self.hf[int(self.indices[i])]
        img = row[self.img_col].convert("RGB")
        return self.transform(img), int(row["label"])


def split_labels(hf, indices):
    return np.array(hf["label"])[np.asarray(indices)]