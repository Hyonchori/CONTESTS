import torch
import numpy as np
import pandas as pd
from torchvision import transforms

DATA_DIR = "235731_북극 해빙예측 AI 경진대회_data/data/"
IMAGE_SIZE = (448, 304)

class MeltDataset(torch.utils.data.Dataset):
    def __init__(self, root, is_train, year_for_train, year_for_valid,
                 feature_interval, target_interval, gap, transforms=None):
        self.root = root
        self.is_train = is_train
        self.year_for_train = year_for_train
        self.year_for_valid = year_for_valid
        self.feature_interval = feature_interval
        self.target_interval = target_interval
        self.gap = gap
        self.transforms = transforms

        if self.is_train == "train":
            self.data_pd = pd.read_csv(self.root + "weekly_train.csv"
                                       )[-52 * self.year_for_train: -52 * self.year_for_valid]
            self.data_path = self.root + "weekly_train/" + \
                             self.data_pd["week_file_nm"].values
        else:
            self.data_pd = pd.read_csv(self.root + "weekly_train.csv"
                                       )[-52 * self.year_for_valid:]
            self.data_path = self.root + "weekly_train/" + \
                             self.data_pd["week_file_nm"].values

        self.mask = self.get_mask()
        self.imgs = self.get_imgs(self.data_path)

    def get_mask(self):
        sample = np.load(self.data_path[-1])[:, :, 3]
        mask = np.where(sample==0, 1, 0)
        return mask

    def get_imgs(self, img_path):
        print("Loading {} imgs from np files ...".format(self.is_train), end=" ")
        imgs = []
        for path in img_path:
            img = np.load(path)[:, :, 0]/250.
            imgs.append(img)
        print("Done!")
        return np.array(imgs)

    def __len__(self):
        return len(self.imgs[: -(self.feature_interval + self.gap + self.target_interval)])

    def __getitem__(self, idx):
        feature_imgs = self.imgs[idx: idx + self.feature_interval]
        target_imgs = self.imgs[idx + self.feature_interval + self.gap: \
                                idx + self.feature_interval + self.gap + self.target_interval]

        return torch.from_numpy(feature_imgs).view(1, self.feature_interval, IMAGE_SIZE[0], IMAGE_SIZE[1]),\
               torch.from_numpy(target_imgs).view(1, self.feature_interval, IMAGE_SIZE[0], IMAGE_SIZE[1]),\
               torch.from_numpy(self.mask)

data_transforms = {
    "train": transforms.Compose([
        transforms.ToTensor(),
    ]),
    "valid": transforms.Compose([
        transforms.ToTensor(),
    ])
}
def get_dataloader():
    train_dataset = MeltDataset(DATA_DIR, is_train="train",
                                year_for_train=10, year_for_valid=1,
                                feature_interval=12,
                                target_interval=12,
                                gap=2,
                                transforms=None)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=5, shuffle=True,
    )

    valid_dataset = MeltDataset(DATA_DIR, is_train="valid",
                                year_for_train=10, year_for_valid=1,
                                feature_interval=12,
                                target_interval=12,
                                gap=2,
                                transforms=None)
    valid_datalaoder = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1
    )
    return train_dataloader, valid_datalaoder