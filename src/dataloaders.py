import json
import random
import math
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy import interpolate
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, img_dir, data_len=100, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_len = data_len

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                 ])

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        image = Image.open(self.img_dir + f"{idx}.png").convert("RGB")
        image = self.transform(image)
        x_len, y_len = image[0].size()

        target = torch.load(self.img_dir + f"{idx}_point_cloud.pt")
        # print("target size:", target.size())
        # print("target[:, 0] size:", target[:, :, 0].size())
        target[:, :, 0] = (target[:, :, 0] / x_len)
        target[:, :, 1] = (target[:, :, 1] / y_len)
        return image, target


class BboxDataset(Dataset):
    def __init__(self, img_dir, data_len=100, true_targets=True, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.true_targets = true_targets
        self.data_len = data_len

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 # transforms.GaussianBlur(7),
                                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                 ])
        if not true_targets and target_transform is None:
            raise ValueError("target_transform must be specified when not using true_targets.")

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        image = Image.open(self.img_dir + f"{idx}.png").convert("RGB")
        image = self.transform(image)
        h, w = image[0].size()

        if self.true_targets:
            target = torch.load(self.img_dir + f"{idx}_bbox.pt")
            # print("target size:", target.size())
            # print("target[:, 0] size:", target[:, :, 0].size())
            target /= torch.Tensor([w, h, w, h]).unsqueeze(0)
        else:
            target = self.target_transform(image)
        return image, target


class TuSimpleDataset(Dataset):
    def __init__(self, data_dir, json_files, random_transf=False):
        self.random_transf = random_transf
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

        self.lanes = []
        self.max_len = 0
        # i = 0
        for json_file in json_files:
            for line in open(data_dir + json_file, 'r'):
                # i += 1
                # if i == 16:
                #     break
                lane = json.loads(line)
                if len(lane['h_samples']) > self.max_len:
                    self.max_len = len(lane['h_samples'])
                self.lanes.append(json.loads(line))

    def __len__(self):
        return len(self.lanes)

    def __getitem__(self, idx):
        metadata = self.lanes[idx]
        im = Image.open(self.data_dir + metadata['raw_file'])
        im = self.transform(im)

        h_samples = torch.tensor(metadata['h_samples'])
        target = []
        for lane in metadata['lanes']:
            lane = torch.tensor(lane)
            indices = lane >= 0
            target.append(torch.stack((h_samples[indices], lane[indices]), dim=1))

        # oldim = im.clone()
        # oldtarget = [buh.clone() for buh in target]

        if self.random_transf:
            im, target = random_transform(im, target)

        _, h, w = im.size()

        lanes = []
        for lane in target:
            if lane.size(0) > 5:
                interp = interpolate.interp1d(lane[:, 0], lane[:, 1], kind="slinear", fill_value="extrapolate")
                y_linear = torch.linspace(lane[0, 0], lane[-1, 0], self.max_len)
                x_linear = torch.tensor(interp(y_linear), dtype=torch.float32)
                lanes.append(torch.stack((y_linear, x_linear), dim=1))
        if len(lanes) > 0:
            target = torch.stack(lanes, dim=0) / torch.tensor([h, w]).view(1, 1, -1)
        else:
            target = torch.empty(0, self.max_len, 2)

        # oldim -= oldim.amin(dim=(1, 2), keepdim=True)
        # oldim /= oldim.amax(dim=(1, 2), keepdim=True)
        #
        # im -= im.amin(dim=(1, 2), keepdim=True)
        # im /= im.amax(dim=(1, 2), keepdim=True)
        #
        # fig, ax = plt.subplots(ncols=2)
        #
        # ax[0].imshow(oldim.permute(1, 2, 0))
        # ax[1].imshow(im.permute(1, 2, 0), extent=[0, 1, 1, 0])
        # ax[1].set_title('Transformed')
        #
        # for tt in oldtarget:
        #     ax[0].scatter(tt[:, 1], tt[:, 0], s=4, color='green')
        # for tta in target:
        #     ax[1].scatter(tta[:, 1], tta[:, 0], s=4, color='green')
        # plt.show()

        return im, target


def random_transform(im, target):
    transform_dict = {'crop': MyCrop(), 'h_flip': MyHorizontalFlip(1),
                      'affine': MyAffine(10, (0.10, 0.10), (1, 1))}

    t = random.choice(list(transform_dict.keys()))

    transformed_im, transformed_target = transform_dict[t](im, target)

    return transformed_im, transformed_target


class MyCrop(nn.Module):

    def __init__(self):
        super().__init__()
        self.crop = transforms.RandomResizedCrop((0, 0))

    def forward(self, im, target):
        _, h, w = im.size()

        top, left, height, width = self.crop.get_params(im, scale=(0.5, 1.0), ratio=(1, 1))
        transformed_im = transforms.functional.resized_crop(im, top, left, height, width, (h, w))

        transformed_target = []
        for lane in target:
            cropped_lane = lane[[(top < lane[:, 0]) & (lane[:, 0] < top + height) &
                                 (left < lane[:, 1]) & (lane[:, 1] < left + width)]]
            transformed_target.append((cropped_lane - torch.tensor([top, left])) *
                                      torch.tensor([h / height, w / width]))

        return transformed_im, transformed_target


class MyVerticalFlip(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.flip = transforms.RandomVerticalFlip(p)

    def forward(self, im, target):
        transformed_im = self.flip(im)
        transformed_target = []
        for lane in target:
            lane[:, 0] -= im.size(1)
            lane[:, 0] *= -1
            transformed_target.append(lane)

        return transformed_im, transformed_target


class MyHorizontalFlip(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.flip = transforms.RandomHorizontalFlip(p)

    def forward(self, im, target):
        transformed_im = self.flip(im)
        transformed_target = []
        for lane in target:
            lane[:, 1] -= im.size(2)
            lane[:, 1] *= -1
            transformed_target.append(lane)

        return transformed_im, transformed_target


class MyAffine(nn.Module):

    def __init__(self, degree, translation, scale):
        super().__init__()
        self.affine = transforms.RandomAffine((0, 0), (0, 0))
        self.degree = (-degree, degree)
        self.translation = translation
        self.scale = scale

    def forward(self, im, target):
        _, h, w = im.size()

        phi, translation, scale, shear = self.affine.get_params(self.degree, self.translation, self.scale, (0, 0), (h, w))
        transformed_im = transforms.functional.affine(im, phi, translation, scale, shear)

        phi = torch.tensor(-phi * math.pi / 180)
        rot = torch.stack([torch.stack([scale * torch.cos(phi), scale * -torch.sin(phi), torch.tensor(translation[1])]),
                           torch.stack([scale * torch.sin(phi), scale * torch.cos(phi), torch.tensor(translation[0])]),
                           torch.tensor([0, 0, 1])])

        transformed_target = []
        for lane in target:
            lane = torch.cat((lane, torch.ones((lane.size(0), 1))), 1)

            rotated_lane = ((lane - torch.tensor([h/2, w/2, 0])) @ rot.T) + torch.tensor([h/2, w/2, 0])

            rotated_lane = (rotated_lane / rotated_lane[:, 2].unsqueeze(1))[:, [0, 1]]

            cropped_rotated_lane = rotated_lane[[(0 < rotated_lane[:, 0]) & (rotated_lane[:, 0] < h) &
                                                 (0 < rotated_lane[:, 1]) & (rotated_lane[:, 1] < w)]]
            transformed_target.append(cropped_rotated_lane)

        return transformed_im, transformed_target


def custom_collate(batch):
    images = []
    targets = []
    for im, target in batch:
        if target.size(0) > 0:
            images.append(im)
            targets.append(target)
    if len(images) == 0:
        return images, targets
    else:
        return torch.stack(images, dim=0), targets


def prepare_dataloaders(rank, world_size, train_set, val_set, test_set, batch_size, list_collate=False):
    if list_collate:
        col = custom_collate
    else:
        col = None

    # Create dataloaders with distributed processing in mind.
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    trainloader = DataLoader(train_set, batch_size=batch_size // world_size, collate_fn=col, sampler=train_sampler, drop_last=True)
    valloader = DataLoader(val_set, batch_size=batch_size // world_size, collate_fn=col, sampler=val_sampler)
    testloader = DataLoader(test_set, batch_size=batch_size // world_size, collate_fn=col)

    return trainloader, valloader, testloader