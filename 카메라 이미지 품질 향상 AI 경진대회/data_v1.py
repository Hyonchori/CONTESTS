import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_size, transform=None):
        self.root = root
        self.target_size = target_size
        self.transform = transform
        self.image_pd = pd.read_csv(self.root + "train.csv")
        self.image_paths = self.root + "train_input_img/" + self.image_pd["input_img"].values
        self.label_paths = self.root + "train_label_img/" + self.image_pd["label_img"].values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = Image.open(self.label_paths[idx]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return img, label

DATA_DIR = "235746_카메라_품질_향상_AI경진대회_data/"
TARGET_SIZE = (3, 1224, 1632)
data_transforms = transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor()
])



def get_dataloader():
    train_dataset = ImageDataset(DATA_DIR, TARGET_SIZE, data_transforms)
    valid_dataset = ImageDataset(DATA_DIR, TARGET_SIZE, data_transforms)
    indices = torch.arange(0, len(train_dataset))
    train_dataset = torch.utils.data.Subset(train_dataset, indices[: -100])
    valid_dataset = torch.utils.data.Subset(valid_dataset, indices[-100: ])

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False
    )
    return train_dataloader, valid_dataloader