"""Datasets for DSB 2018, Sunnybrook Cardiac (SCD) and generic .npz image/mask arrays."""
import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RandomCrop(transforms.RandomResizedCrop):
    """RandomResizedCrop applying the same crop to a list of PIL images."""

    def __call__(self, imgs):
        i, j, h, w = self.get_params(imgs[0], self.scale, self.ratio)
        for k in range(len(imgs)):
            imgs[k] = transforms.functional.resized_crop(imgs[k], i, j, h, w, self.size, self.interpolation)
        return imgs


class SegDataset(Dataset):
    """2D image/mask dataset from uint8 arrays (N, H, W[, C]) with flip/rotate/crop augmentation."""

    def __init__(self, images, masks, train=True, crop_size=(256, 256)):
        self.images = images
        self.masks = masks
        self.train = train
        self.crop_size = crop_size

    def __len__(self):
        return len(self.images)

    def rotate(self, image, mask, degrees=(-15, 15), p=0.5):
        if torch.rand(1) < p:
            degree = np.random.uniform(*degrees)
            image = image.rotate(degree, Image.NEAREST)
            mask = mask.rotate(degree, Image.NEAREST)
        return image, mask

    def horizontal_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def vertical_flip(self, image, mask, p=0.5):
        if torch.rand(1) < p:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return image, mask

    def random_resized_crop(self, image, mask, p=0.1):
        if torch.rand(1) < p:
            image, mask = RandomCrop(self.crop_size, scale=(0.8, 0.95))([image, mask])
        return image, mask

    def augment(self, image, mask):
        image, mask = self.random_resized_crop(image, mask)
        image, mask = self.rotate(image, mask)
        image, mask = self.horizontal_flip(image, mask)
        image, mask = self.vertical_flip(image, mask)
        return image, mask

    def __getitem__(self, idx):
        image = Image.fromarray(np.squeeze(self.images[idx]).astype(np.uint8))
        mask = Image.fromarray(np.squeeze(self.masks[idx]).astype(np.uint8))
        if self.train:
            image, mask = self.augment(image, mask)
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.asarray(mask, np.int64)[np.newaxis])
        return image, mask


def center_crop(img, crop_size=(160, 160)):
    w_in, h_in, d_in = img.shape
    img_crop = np.zeros((*crop_size, d_in))
    w_out, h_out = crop_size
    sub_w = max((w_in - w_out) // 2 - 20, 0)
    sub_h = max((h_in - h_out) // 2 - 10, 0)

    img_clone = img[sub_w:sub_w + w_out, sub_h:sub_h + h_out]
    img_crop[:img_clone.shape[0], :img_clone.shape[1]] = img_clone
    return img_crop


class SunnybrookDataset(Dataset):
    """Sunnybrook endo/epi slices (N, H, W, 1), center-cropped to 160x160."""

    def __init__(self, images, masks, train=False):
        self.images = images
        self.masks = masks
        self.transforms = None
        if train:
            import albumentations as A
            self.transforms = A.Compose([A.HorizontalFlip(p=0.2), A.VerticalFlip(p=0.2)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.images[idx], self.masks[idx]
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        image = np.transpose(center_crop(image), (2, 0, 1)).astype(np.float32)
        mask = np.transpose(center_crop(mask), (2, 0, 1)).astype(np.float32)
        return torch.from_numpy(image) / 255.0, torch.from_numpy(mask)


def _load_npz(path):
    data = np.load(path)
    return data["image"], data["mask"]


def build_datasets(dataset, data_dir, subset="endo", test_ratio=0.2, seed=312):
    """Returns (train_set, val_set, test_set, in_channels). `val_set` may be the test set."""
    if dataset == "dsb2018":
        load = lambda name: np.load(os.path.join(data_dir, name))
        train = SegDataset(load("x_train.npy"), load("y_train.npy"), train=True, crop_size=(256, 256))
        test = SegDataset(load("x_test.npy"), load("y_test.npy"), train=False)
        return train, test, test, 3

    if dataset == "sunnybrook":
        load = lambda split, xy: np.load(os.path.join(data_dir, f"{xy}{split}_{subset}.npy"))
        train = SunnybrookDataset(load("train", "X"), load("train", "Y"), train=True)
        val = SunnybrookDataset(load("val", "X"), load("val", "Y"))
        test = SunnybrookDataset(load("test", "X"), load("test", "Y"))
        return train, val, test, 1

    if dataset == "npz":
        # Generic: <data_dir>/train.npz (+ optional test.npz) with "image" and "mask" keys.
        x, y = _load_npz(os.path.join(data_dir, "train.npz"))
        test_path = os.path.join(data_dir, "test.npz")
        if os.path.exists(test_path):
            x_test, y_test = _load_npz(test_path)
        else:
            x, x_test, y, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
        crop_size = tuple(x.shape[1:3])
        in_channels = 1 if x.ndim == 3 else x.shape[-1]
        train = SegDataset(x, y, train=True, crop_size=crop_size)
        test = SegDataset(x_test, y_test, train=False)
        return train, test, test, in_channels

    raise ValueError(f"Unknown dataset: {dataset}")


def build_loader(ds, batch_size, shuffle=False, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=True, drop_last=shuffle)
