from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
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
    def __init__(self, img_dir, data_len=100, true_targets=False, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.true_targets = true_targets
        self.data_len = data_len

        if transform is None:
            # self.transform = transforms.Compose([transforms.ToTensor(),
            #                                     transforms.Normalize(0.5, 0.5)])
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(0.456, 0.225)])
        if not true_targets and target_transform is None:
            raise ValueError("target_transform must be specified when not using true_targets.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        image = Image.open(self.img_dir + f"{idx}.png")
        image = self.transform(image)
        x_len, y_len = image[0].size()

        if self.true_targets:
            target = torch.load(self.img_dir + f"{idx}_point_cloud.pt")
            # print("target size:", target.size())
            # print("target[:, 0] size:", target[:, :, 0].size())
            target[:, :, 0] = (target[:, :, 0] / x_len)
            target[:, :, 1] = (target[:, :, 1] / y_len)
        else:
            target = self.target_transform(image)
        return image, target
