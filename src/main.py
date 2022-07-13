import os
import argparse
from detr.models.detr import buildLines
from dataloaders import CustomDataset, prepare_dataloaders, TuSimpleDataset
from utility import plot_output_to_images
from math import ceil

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Loss coefficients
    parser.add_argument('--endpoint_weight', default=1, type=float)
    parser.add_argument('--distance_weight', default=2, type=float)
    parser.add_argument('--class_weight', default=1, type=float)
    parser.add_argument('--no_obj_weight', default=0.1, type=float)

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_controlpoints', default=4, type=int,
                        help="Number of controlpoints per predicted line")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    return parser


def cleanup():
    # Clean up multi gpu processes
    dist.destroy_process_group()


def setup(rank, world_size):
    # Setup distributed processing.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


@torch.no_grad()
def val_loop(valmodel, valloader, valcriterion, rank):
    # Calculate loss on validation set
    val_losses_list = []
    for val_im, val_tgt in valloader:
        if len(val_im) == 0:
            print("valcontinue")
            continue
        val_target_list = []
        for val_t in val_tgt:
            val_target_list.append({'labels': torch.zeros(val_t.size(0), dtype=int, device=rank),
                                    'points': val_t.to(rank)})

        val_out = valmodel(val_im)

        val_loss_dict = valcriterion(val_out, val_target_list)
        val_weight_dict = valcriterion.weight_dict
        val_losses = sum(val_loss_dict[k] * val_weight_dict[k] for k in val_loss_dict.keys() if k in val_weight_dict)
        val_loss = val_losses.mean()

        val_losses_list.append(val_loss.item())

    return torch.tensor(val_losses_list).mean()


def train(rank, world_size, args):
    # TEMP ARGUMENTS #
    n_target_splines = 2
    data_len = 100
    summary_writer_directory = 'runs/DETRLines_tusimple_overfit2'
    ##################

    # Logging setup
    writer = SummaryWriter(summary_writer_directory)

    # Distributed setup
    setup(rank, world_size)

    # Model and criterion setup
    model, criterion = buildLines(args)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    criterion.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Dataloaders setup
    # dataset = CustomDataset(f'/media/scratch1/stolpt/custom_data/random_{n_target_splines}_curves_bbox_',
    #                         data_len=data_len)
    # dataset = TuSimpleDataset('/media/scratch1/stolpt/tusimple/train/', ['label_data_0313.json',
    #                                                                      'label_data_0531.json',
    #                                                                      'label_data_0601.json'])
    dataset = TuSimpleDataset('/media/scratch1/stolpt/tusimple/train/', ['label_data_0313.json'], random_transf=False)

    # Get train/val/test split
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9),
                                                                           len(dataset) - int(len(dataset) * 0.9) - 25,
                                                                           25])
    # train_set, val_set, test_set = torch.utils.data.random_split(dataset, [len(dataset)-8,
    #                                                                        4,
    #                                                                        4])

    # Create dataloaders
    trainloader, valloader, testloader = prepare_dataloaders(rank, world_size, train_set, val_set, test_set,
                                                             args.batch_size, list_collate=True)

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop, gamma=0.99)

    # Training loop
    i = 0
    running_total_loss = 0
    running_class_loss = 0
    running_distance_loss = 0
    running_endpoint_loss = 0
    for epoch in range(args.epochs):
        trainloader.sampler.set_epoch(epoch)

        for im, target in trainloader:
            if len(im) == 0:
                continue
            target_list = []
            for t in target:
                target_list.append({'labels': torch.zeros(t.size(0), dtype=int, device=rank),
                                    'points': t.to(rank)})

            # print("im size", im.size())
            # print("target size", target.size())

            out = model(im)

            # print("pred_controlpoints size", out['pred_controlpoints'].size())
            # print("pred_points size", out['pred_points'].size())
            # print("pred_logits size", out['pred_logits'].size())

            loss_dict = criterion(out, target_list)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # Losses for logging
            class_losses = sum(loss_dict[key] for key in ['loss_ce', 'loss_ce_0', 'loss_ce_1', 'loss_ce_2', 'loss_ce_3',
                                                          'loss_ce_4']).mean()
            distance_losses = sum(loss_dict[key] for key in ['loss_distance', 'loss_distance_0', 'loss_distance_1',
                                                             'loss_distance_2', 'loss_distance_3', 'loss_distance_4']).mean()
            endpoint_losses = sum(loss_dict[key] for key in ['loss_endpoint', 'loss_endpoint_0', 'loss_endpoint_1',
                                                             'loss_endpoint_2', 'loss_endpoint_3', 'loss_endpoint_4']).mean()

            loss = losses.mean()

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            running_total_loss += loss.item()
            running_class_loss += class_losses.item()
            running_distance_loss += distance_losses.item()
            running_endpoint_loss += endpoint_losses.item()

            if i % ceil(100 / args.batch_size) == ceil(100 / args.batch_size) - 1:
                writer.add_scalar('training_total_loss', running_total_loss / ceil(1000 / args.batch_size), i * args.batch_size)
                writer.add_scalar('training_class_loss', running_class_loss / ceil(1000 / args.batch_size), i * args.batch_size)
                writer.add_scalar('training_distance_loss', running_distance_loss / ceil(1000 / args.batch_size), i * args.batch_size)
                writer.add_scalar('training_endpoint_loss', running_endpoint_loss / ceil(1000 / args.batch_size), i * args.batch_size)
                running_total_loss = 0
                running_class_loss = 0
                running_distance_loss = 0
                running_endpoint_loss = 0

            if i % ceil(1000 / args.batch_size) == ceil(1000 / args.batch_size) - 1:
                val_losses = val_loop(model, valloader, criterion, rank)
                writer.add_scalar('validation_total_loss', val_losses, i * args.batch_size)
            i += 1

        scheduler.step()

        # Logging and plotting
        # if True:
        if epoch % 50 == 49:
            with torch.no_grad():
                model.eval()
                # Plot train images
                train_plots = plot_output_to_images(im[:25], target[:25], out['pred_points'][:25],
                                                    out['pred_controlpoints'][:25],
                                                    out['pred_logits'][:25],
                                                    loss_dict)
                train_grid = make_grid(train_plots, nrow=int(len(train_plots) ** 0.5))
                writer.add_image('train_images', train_grid, i * args.batch_size)

                test_plots = []
                for test_im, test_tgt in testloader:
                    test_out = model(test_im)

                    test_target_list = []
                    for t in test_tgt:
                        test_target_list.append({'labels': torch.zeros(t.size(0), dtype=int, device=rank),
                                                 'points': t.to(rank)})

                    test_loss_dict = criterion(test_out, test_target_list)

                    test_plots += plot_output_to_images(test_im, test_tgt, test_out['pred_points'],
                                                        test_out['pred_controlpoints'],
                                                        test_out['pred_logits'],
                                                        test_loss_dict)

                test_grid = make_grid(test_plots, nrow=int(len(test_plots) ** 0.5))
                writer.add_image('test_images', test_grid, i * args.batch_size)
                model.train()

    if not rank:
        print("Done.")
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPU(s).")
    arguments = (world_size, args)

    mp.spawn(train,
             args=arguments,
             nprocs=world_size)
