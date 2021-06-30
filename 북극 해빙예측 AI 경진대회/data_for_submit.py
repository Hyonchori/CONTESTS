import torch
import numpy as np
import pandas as pd

DIR_2020 = "235731_북극 해빙예측 AI 경진대회_data/data/"
DIR_2021 = "235731_북극_해빙예측_AI_경진대회_data_v2/data_v2/"
IMAGE_SIZE = (448, 304)

class MeltDataset(torch.utils.data.Dataset):
    def __init__(self, root, year_of_interest, feature_interval, target_interval, gap):
        self.root = root
        self.year_of_interest = year_of_interest
        self.feature_interval = feature_interval
        self.target_interval = target_interval
        self.gap = gap

        self.data_pd = pd.read_csv(self.root + "weekly_train.csv")[-52 * self.year_of_interest: ]
        self.data_path = self.root + "weekly_train/" + self.data_pd["week_file_nm"].values

        self.mask = self.get_mask()
        self.imgs = self.get_imgs(self.data_path)

    def get_mask(self):
        sample = np.load(self.data_path[-1])[:, :, 3]
        mask = np.where(sample==0, 1, 0)
        return mask

    def get_imgs(self, img_path):
        imgs = []
        for path in img_path:
            img = np.load(path)[:, :, 0] / 250.
            imgs.append(img)
        return np.array(imgs)

    def __len__(self):
        return len(self.imgs[: -(self.feature_interval + self.gap + self.target_interval)])

    def __getitem__(self, idx):
        feature = self.imgs[idx: idx + self.feature_interval]
        target = self.imgs[idx + self.feature_interval + self.gap: \
                            idx + self.feature_interval + self.gap + self.target_interval]

        return torch.from_numpy(feature).view(1, self.feature_interval, IMAGE_SIZE[0], IMAGE_SIZE[1]), \
               torch.from_numpy(target).view(1, self.feature_interval, IMAGE_SIZE[0], IMAGE_SIZE[1]), \
               torch.from_numpy(self.mask)

def get_dataloader():
    train_dataset = MeltDataset(root = DIR_2021,
                                year_of_interest=10,
                                feature_interval=12,
                                target_interval=12,
                                gap=2)
    valid_dataset = MeltDataset(root = DIR_2021,
                                year_of_interest=10,
                                feature_interval=12,
                                target_interval=12,
                                gap=2)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_dataset = torch.utils.data.Subset(train_dataset, indices[: ])
    valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-52*1: ])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    return train_dataloader, valid_dataloader