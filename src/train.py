from math import ceil
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from net import Net
from dataloaders import MNISTDataset, CustomDataset
from custom_transforms import im2pc
from loss import VectorizationLoss
from utility import plot_output_to_images
import warnings

warnings.filterwarnings("error")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def val_loop(nett, valloaderr, loss_func):
    # Calculate validation losses.
    val_total_loss_accumulator = 0
    val_distance_loss_accumulator = 0
    val_cp_loss_accumulator = 0
    with torch.no_grad():
        for val_im, val_target in valloaderr:
            val_out, val_control_points = nett(val_im)

            val_distance_loss, val_cp_loss = loss_func(val_out, val_target.to(val_out.device), val_control_points)
            val_total_loss = (val_distance_loss + val_cp_loss).mean()
            val_distance_loss = val_distance_loss.mean()
            val_cp_loss = val_cp_loss.mean()

            val_total_loss_accumulator += val_total_loss.item()
            val_distance_loss_accumulator += val_distance_loss.item()
            val_cp_loss_accumulator += val_cp_loss.item()

    return val_total_loss_accumulator, val_distance_loss_accumulator, val_cp_loss_accumulator


def setup(rank, world_size):
    # Setup distributed processing.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def prepare_dataloaders(rank, world_size, data_len, n_target_splines, batch_size):
    # Create dataloaders with distributed processing in mind.
    dataset = CustomDataset(f'/media/scratch1/stolpt/custom_data/random_{n_target_splines}_curves_',
                            data_len=data_len, true_targets=True)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(data_len * 0.80),
                                                                           data_len - int(data_len * 0.80) - 25, 25])

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    trainloader = DataLoader(train_set, batch_size=batch_size // world_size, sampler=train_sampler)
    valloader = DataLoader(val_set, batch_size=batch_size // world_size, sampler=test_sampler)
    testloader = DataLoader(test_set, batch_size=batch_size)

    return trainloader, valloader, testloader


def cleanup():
    dist.destroy_process_group()


def set_seeds(seed):
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)


def train(rank, world_size, batch_size, data_len,
          n_controlpoints, n_target_splines, n_predicted_splines,
          epochs, lr, dropout, scheduler_step_size, scheduler_gamma,
          cp_weight, loss_type, summary_writer_directory):

    # Seeding for reproducibility
    set_seeds(0)

    # Setup process groups
    setup(rank, world_size)

    # Prepare dataloaders
    trainloader, valloader, testloader = prepare_dataloaders(rank, world_size, data_len, n_target_splines, batch_size)

    # Create network
    net = Net(channels=None,
              nhead=16,
              num_encoder_layers=12,
              num_decoder_layers=12,
              dim_feedforward=2048,
              dropout=dropout,
              n_controlpoints=n_controlpoints,
              n_splines=n_predicted_splines,
              n_eval_points=100,
              batch_size=batch_size)

    # Setup network for distributed training
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net.to(rank)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])

    # Define optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    # Define loss function
    loss_func = VectorizationLoss(loss=loss_type)

    # Define logger
    writer = SummaryWriter(summary_writer_directory)

    if not rank:
        print(f"Training...")
    i = 0
    running_total_loss = 0
    running_distance_loss = 0
    running_cp_loss = 0
    for epoch in tqdm(range(epochs)):
        trainloader.sampler.set_epoch(epoch)

        for im, target in trainloader:
            # print('im_size: ', im.size())
            # print('target_size: ', target.size())

            out, control_points = net(im)

            # print('out_size: ', out.size())
            # print('control_points_out_size: ', control_points.size())

            distance_loss, cp_loss = loss_func(out, target.to(out.device), control_points, cp_weight)
            total_loss = (distance_loss + cp_loss).mean()
            distance_loss = distance_loss.mean()
            cp_loss = cp_loss.mean()

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()
            scheduler.step()

            running_total_loss += total_loss.item()
            running_distance_loss += distance_loss.item()
            running_cp_loss += cp_loss.item()

            # Create plot every n training steps.
            if i % ceil(100 / batch_size) == ceil(100 / batch_size) - 1:
                writer.add_scalar('training_total_loss', running_total_loss / ceil(100 / batch_size), i * batch_size)
                writer.add_scalar('training_distance_loss', running_distance_loss / ceil(100 / batch_size), i * batch_size)
                writer.add_scalar('training_cp_loss', running_cp_loss / ceil(100 / batch_size), i * batch_size)
                running_total_loss = 0
                running_distance_loss = 0
                running_cp_loss = 0

            if i % ceil(1000 / batch_size) == ceil(1000 / batch_size) - 1:
                net.eval()
                val_total_loss, val_distance_loss, val_cp_loss = val_loop(net, valloader, loss_func)
                writer.add_scalar('validation_total_loss', val_total_loss / len(valloader), i * batch_size)
                writer.add_scalar('validation_distance_loss', val_distance_loss / len(valloader), i * batch_size)
                writer.add_scalar('validation_cp_loss', val_cp_loss / len(valloader), i * batch_size)
                writer.add_scalar('cp_weight', cp_weight, i * batch_size)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[-1], i * batch_size)
                net.train()

            if cp_weight > 0.1:
                if i % ceil(1000 / batch_size) == ceil(1000 / batch_size) - 1:
                    cp_weight = cp_weight * 0.99
            i += 1

        # if epoch % 10 == 9:
        if epoch:
            with torch.no_grad():
                net.eval()
                test_plots = []
                for batch, (test_im, test_target) in enumerate(testloader):
                    test_out, test_control_points = net(test_im)

                    test_out = test_out.cpu()
                    test_control_points = test_control_points.cpu()
                    test_im = test_im.cpu()
                    test_target = test_target.cpu()

                    test_plot = plot_output_to_images(test_im, test_target, test_out, test_control_points)
                    test_plots += test_plot

                test_grid = make_grid(test_plots, nrow=int(len(test_plots) ** 0.5))
                writer.add_image('test_images', test_grid, i * batch_size)

                train_plots = plot_output_to_images(im[:25].cpu(), target[:25].cpu(), out[:25].detach().cpu(),
                                                    control_points[:25].detach().cpu())
                train_grid = make_grid(train_plots, nrow=int(len(train_plots) ** 0.5))
                writer.add_image('train_images', train_grid, i * batch_size)

                net.train()
    if not rank:
        print("Done.")
    cleanup()


if __name__ == "__main__":
    batch_size = 32
    data_len = 100000
    n_controlpoints = 4
    n_target_splines = 1
    n_predicted_splines = 1
    epochs = 100
    lr = 1e-4
    dropout = 0.01
    scheduler_step_size = ceil(1000 / batch_size)
    scheduler_gamma = 0.99
    cp_weight = 1
    loss_type = "sinkhorn"
    summary_writer_directory = 'runs/resnet_backbone_bs32_gpu8_100kdata_bigger_network2'
    # summary_writer_directory = 'runs/temp'

    # Create dataloaders.
    # dataloader = MNISTDataset('../../datasets_external/mnist', transform=transform, target_transform=im2pc)

    print("cuda:", torch.version.cuda)
    print("nccl:", torch.cuda.nccl.version())

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPU(s).")
    arguments = (world_size, batch_size, data_len, n_controlpoints, n_target_splines, n_predicted_splines, epochs, lr,
                 dropout, scheduler_step_size, scheduler_gamma, cp_weight, loss_type, summary_writer_directory)

    mp.spawn(train,
             args=arguments,
             nprocs=world_size)
