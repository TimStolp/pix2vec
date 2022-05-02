from PIL import Image
from torch.utils.data import Dataset
import torch


# MNIST dataloader from mnist folder found in external datasets on deepstorage.
class MNISTDataset(Dataset):
    def __init__(self, img_dir, transform=lambda x: x, target_transform=lambda x: x):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        image = Image.open(self.img_dir + f"/mnist_{idx}/image.png")
        image = self.transform(image)

        target = self.target_transform(image)
        return image, target


class CustomDataset(Dataset):
    def __init__(self, img_dir, data_len=100, true_targets=False, transform=lambda x: x, target_transform=lambda x: x):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.true_targets = true_targets
        self.data_len = data_len

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        image = Image.open(self.img_dir + f"{idx}.png")
        image = self.transform(image)

        if self.true_targets:
            x_len, y_len = image[0].size()
            target = torch.load(self.img_dir + f"{idx}_point_cloud.pt")
            # print("target size:", target.size())
            # print("target[:, 0] size:", target[:, :, 0].size())
            target[:, :, 0] = (target[:, :, 0] / x_len)
            target[:, :, 1] = (target[:, :, 1] / y_len)
            target = target
        else:
            target = self.target_transform(image)
        return image, target
