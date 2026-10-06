"""Load the CAMELS LH splits written by prepare_data.py.

Expected layout (one folder per label set):

    params_2/  {train,val,test}_LH.npy    + {train,val,test}_labels_LH_2.npy   (Omega_m, sigma_8)
    params_6/  {train,val,test}_LH_6.npy  + {train,val,test}_labels_LH.npy     (all six LH parameters)

Images on disk are in [0, 1] and are rescaled to [-1, 1]. Labels are z-scored with the
train-split mean and standard deviation.
"""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# (image file, label file) templates per label_dim; the names match the original training data.
SPLIT_FILES = {
    2: ("{split}_LH.npy", "{split}_labels_LH_2.npy"),
    6: ("{split}_LH_6.npy", "{split}_labels_LH.npy"),
}


def split_paths(data_dir, split, label_dim):
    """Return (image path, label path) for one split."""
    if label_dim not in SPLIT_FILES:
        raise ValueError(f"label_dim must be one of {sorted(SPLIT_FILES)}, got {label_dim}")
    image_name, label_name = SPLIT_FILES[label_dim]
    return (os.path.join(data_dir, image_name.format(split=split)),
            os.path.join(data_dir, label_name.format(split=split)))


def label_statistics(train_labels):
    """Train-split mean and std per column; zero-variance columns get std = 1."""
    mean = train_labels.mean(axis=0)
    std = train_labels.std(axis=0)
    return mean, np.where(std == 0, 1.0, std)


class ConditionalImageDataset(Dataset):
    def __init__(self, data_path, label_path, label_stats=None):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
        self.label_stats = label_stats
        if len(self.data) != len(self.labels):
            raise ValueError(f"{data_path} has {len(self.data)} images but {label_path} has {len(self.labels)} labels")
        print(f"Loaded {len(self.data)} images | image shape {self.data.shape[1:]} | label shape {self.labels.shape[1:]}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).float() * 2.0 - 1.0
        label = torch.from_numpy(self.labels[idx]).float()
        if self.label_stats is not None:
            label = (label - self.label_stats["mean"]) / self.label_stats["std"]
        if img.dim() == 2:
            img = img.unsqueeze(0)
        return img, label


def get_conditional_dataloaders(data_dir, label_dim, batch_size=8, num_workers=4,
                                pin_memory=True, normalize_labels=True):
    """Train (shuffled, drop_last), validation and test loaders."""
    print(f"Loading {label_dim}-parameter dataset from {data_dir}")
    train_labels = np.load(split_paths(data_dir, "train", label_dim)[1])
    if train_labels.shape[1] != label_dim:
        raise ValueError(f"train labels have {train_labels.shape[1]} columns; expected label_dim={label_dim}")

    label_stats = None
    if normalize_labels:
        mean, std = label_statistics(train_labels)
        label_stats = {"mean": torch.from_numpy(mean).float(), "std": torch.from_numpy(std).float()}
        print(f"Label normalisation: mean={mean}, std={std}")

    loaders = []
    for split in ("train", "val", "test"):
        dataset = ConditionalImageDataset(*split_paths(data_dir, split, label_dim), label_stats=label_stats)
        is_train = split == "train"
        loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers,
                                  pin_memory=pin_memory, drop_last=is_train))
    return tuple(loaders)
