import sys
import os
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import ceil
from detr.models import build_model
from dataloaders import BboxDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
import torch
import argparse
from numpy import frombuffer
from torchvision.utils import make_grid
sys.path.append('/home/stolpt/home2/pix2vec/src/detr')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

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
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.02, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='custom')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def prepare_dataloaders(rank, world_size, data_len, n_target_splines, batch_size):
    # Create dataloaders with distributed processing in mind.
    dataset = BboxDataset(f'/media/scratch1/stolpt/custom_data/random_{n_target_splines}_curves_bbox_',
                            data_len=data_len, true_targets=True)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [int(data_len * 0.80),
                                                                           data_len - int(data_len * 0.80) - 25, 25])

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    trainloader = DataLoader(train_set, batch_size=batch_size // world_size, sampler=train_sampler, drop_last=True)
    valloader = DataLoader(val_set, batch_size=batch_size // world_size, sampler=val_sampler)
    testloader = DataLoader(test_set, batch_size=batch_size)

    return trainloader, valloader, testloader


def cleanup():
    dist.destroy_process_group()


def setup(rank, world_size):
    # Setup distributed processing.
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


@torch.no_grad()
def plot_pred_bboxes(im, output, target):
    im = im.cpu()
    pred_logits = output['pred_logits'].cpu()
    pred_boxes = output['pred_boxes'].cpu()
    target = target.cpu()
    # print("im", im.size())
    # print("pred_logits", output['pred_logits'].size())
    # print("pred_boxes", output['pred_boxes'].size())
    # print("target", target.size())

    plots = []
    for img, logits, boxes, tgt in zip(im, pred_logits, pred_boxes, target):
        imsize_x, imsize_y = img[0].size()
        fig = plt.figure(figsize=(imsize_x / 100, imsize_y / 100))

        img -= img.amin(dim=(1, 2), keepdim=True)
        img /= img.amax(dim=(1, 2), keepdim=True)
        plt.imshow(img.permute(1, 2, 0), extent=[0, 1, 1, 0])

        ax = plt.gca()
        plt.gca().invert_yaxis()
        plt.axis('off')
        plt.tight_layout(pad=0)

        valid_box_indices = logits[:, 0] > logits[:, 1]

        for t, c in zip([tgt, boxes[valid_box_indices]], ['g', 'r']):
            for b in t:
                xmid, ymid, xsize, ysize = b

                xoffset = xsize / 2
                yoffset = ysize / 2

                xmin = xmid - xoffset
                ymin = ymid - yoffset

                # Create a Rectangle patch
                rect = patches.Rectangle((xmin, ymin), xsize, ysize, linewidth=1, edgecolor=c, facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Plot middle
                plt.scatter([xmid], [ymid], c=c)

        plt.annotate(f"Predicted boxes: {valid_box_indices.sum()}", xy=(0.01, 0.95), c='yellow')

        fig.canvas.draw()

        plot = frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).transpose(2, 0, 1).copy()
        plots.append(torch.from_numpy(plot))
        plt.close()

    return plots


def val_loop(rank, nett, valloaderr, criterion, n_target_splines):
    # Calculate validation losses.
    val_total_loss_accumulator = 0
    with torch.no_grad():
        for val_im, val_target in valloaderr:
            val_im = val_im.to(rank)
            target_list = []
            for t in val_target:
                target_list.append(
                    {'labels': torch.zeros(n_target_splines, dtype=int, device=rank), 'boxes': t.to(rank)})

            # print("im_size", im.size())
            # print("target_size", target.size())

            output = nett(val_im)

            loss_dict = criterion(output, target_list)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            val_total_loss_accumulator += losses.item()

    return val_total_loss_accumulator


def train(rank, world_size, args):
    batch_size = 32
    data_len = 100
    n_target_splines = 2
    epochs = 1
    lr = 1e-4
    scheduler_step_size = ceil(1000 / batch_size)
    scheduler_gamma = 0.99
    summary_writer_directory = 'runs/temp'

    setup(rank, world_size)

    model, criterion, postprocessors = build_model(args)
    criterion.to(rank)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    trainloader, valloader, testloader = prepare_dataloaders(rank, world_size, data_len, n_target_splines, batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    writer = SummaryWriter(summary_writer_directory)

    i = 0
    running_total_loss = 0
    for epoch in range(epochs):
        trainloader.sampler.set_epoch(epoch)

        for im, target in trainloader:
            im = im.to(rank)
            target_list = []
            for t in target:
                target_list.append({'labels': torch.zeros(n_target_splines, dtype=int, device=rank), 'boxes': t.to(rank)})

            # print("im_size", im.size())
            # print("target_size", target.size())

            output = model(im)

            loss_dict = criterion(output, target_list)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # print(output['pred_logits'].size())
            # print(output['pred_boxes'].size())

            # print(loss_dict['loss_ce'])
            # print(losses)

            optimizer.zero_grad()

            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()
            scheduler.step()

            running_total_loss += losses.item()

            if i % ceil(100 / batch_size) == ceil(100 / batch_size) - 1:
                writer.add_scalar('training_total_loss', running_total_loss / ceil(100 / batch_size), i * batch_size)
                running_total_loss = 0

                # pred_probs = output['pred_logits'].softmax(dim=-1)[0]
                # print("probs of real predicted boxes", pred_probs[pred_probs[:, 0] > pred_probs[:, 1]])

                # print("probs", output['pred_logits'].softmax(dim=-1))

                # print("pred", output['pred_boxes'])
                # print("target", target)

            if i % ceil(1000 / batch_size) == ceil(1000 / batch_size) - 1:
                model.eval()
                val_loss = val_loop(rank, model, valloader, criterion, n_target_splines)
                writer.add_scalar('validation_total_loss', val_loss / len(valloader), i * batch_size)
                model.train()

            i += 1

        if not rank:
            plots = plot_pred_bboxes(im, output, target)
            train_grid = make_grid(plots, nrow=int(len(plots) ** 0.5))
            writer.add_image('train_images', train_grid, i * batch_size)

            with torch.no_grad():
                model.eval()
                test_plots = []
                for batch, (test_im, test_target) in enumerate(testloader):
                    test_im = test_im.to(rank)
                    output = model(test_im)

                    test_plot = plot_pred_bboxes(test_im, output, test_target)
                    test_plots += test_plot

                test_grid = make_grid(test_plots, nrow=int(len(test_plots) ** 0.5))
                writer.add_image('test_images', test_grid, i * batch_size)

                model.train()

            # for pred_logits in output['pred_logits'].softmax(dim=-1):
            #     print("probs of real predicted boxes", pred_logits[pred_logits[:, 0] > pred_logits[:, 1]])

        # print("probs", output['pred_logits'].softmax(dim=-1))

        # print("pred", output['pred_boxes'])
        # print("target", target)
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
