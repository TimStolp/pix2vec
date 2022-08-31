import os
import argparse
import torch
import matplotlib
import matplotlib.pyplot as plt

from detr.models.detr import buildLines
from dataloaders import CustomDataset, prepare_dataloaders, TuSimpleDataset
from torch.utils.data import DataLoader
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

import numpy as np
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter


import warnings
warnings.filterwarnings("ignore")

matplotlib.use('TkAgg')

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
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
    parser.add_argument('--distance_weight', default=1, type=float)
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

    # * Dataloader
    parser.add_argument('--crop', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--affine', action='store_true')

    # * Logging
    parser.add_argument('--log_dir', default='runs/temp', type=str,
                        help="Name of the logging directory for tensorboard.")
    parser.add_argument('--save_model_dir', default='trained_models/temp', type=str,
                        help="Name of the logging directory to save the model after training")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    return parser
    
    
    
def plot_image(im, target, pred_logits, pred_controlpoints, pred_points, accuracy):
    for b in range(len(im)):
        indices = pred_logits[b, :, 0] > pred_logits[b, :, 1]

        fig = plt.figure(figsize=(5, 5))
        img = im[b]
        img -= img.amin(dim=(1, 2), keepdim=True)
        img /= img.amax(dim=(1, 2), keepdim=True)
        plt.imshow(img.permute(1, 2, 0), extent=[0, 1, 1, 0])
          
        
        p_points = pred_points[b, indices]
        p_controlpoints = pred_controlpoints[b, indices]
        
        # colors = ['firebrick', 'gold', 'steelblue']
        # if len(p_points) == 3:
            # plt.show()
            # plt.imshow(img.permute(1, 2, 0), extent=[0, 1, 1, 0])
            # # for tt in target:
                # # plt.scatter(tt[0, :, 1], tt[0, :, 0], s=4, color='green')
            # for c, to in zip(colors, p_points):
                # plt.scatter(to[:, 1], to[:, 0], s=4, color=c)
            # for c, tcp in zip(colors, p_controlpoints):
                # plt.scatter(tcp[:, 1], tcp[:, 0], color=c)
                
            # # plt.annotate(f"Accuracy: {accuracy}", xy=(0.01, 0.05), c='yellow', backgroundcolor='black')
            # plt.show()
            
        for tt in target:
            plt.scatter(tt[0, :, 1], tt[0, :, 0], s=4, color='green')
        for to in p_points:
            plt.scatter(to[:, 1], to[:, 0], s=4, color='blue')
        for tcp in p_controlpoints:
            plt.scatter(tcp[:, 1], tcp[:, 0], color='red')
            
        # plt.annotate(f"Accuracy: {accuracy}", xy=(0.01, 0.05), c='yellow', backgroundcolor='black')
        plt.show()


def calc_accuracy(pred_points, target, pred_logits, threshold):
    indices = pred_logits[0, :, 0] > pred_logits[0, :, 1]
    pred_points = pred_points[0, indices]
    
    # print(pred_points.size()) # torch.Size([3, 100, 2])

    target_lane_distances = [] # all target lanes to all predict lanes ----- num_target_lanes x num_pred_lanes 
    for target_lane in target:
        pred_lane_distances = [] # single target lane to all predict lanes
        for pred_lane in pred_points:
            distances = [] # target points to minimum predict points
            for target_lane_point in target_lane[0]:
                point_distances = (pred_lane - target_lane_point).abs().sum(dim=1) # target point to all predict points
                min_index = point_distances.argmin()
                distances.append(point_distances[min_index])
            pred_lane_distances.append(torch.tensor(distances))
        target_lane_distances.append(pred_lane_distances)
            
    
    cost_matrix = torch.tensor([[sum(l) for l in t] for t in target_lane_distances])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
     
    correct_points = 0
    for r, c in zip(row_ind, col_ind):
        t = target[r][0]
        p = pred_points[c]
        
        for t_point in t:
            d = (t_point[0] - p[:, 0]).abs()
            min_i = d.argmin()
            closest_point = p[min_i]
            if (t_point[1] - closest_point[1]).abs() < threshold / 1280:
                correct_points += 1
    
    total_points = sum([len(t[0]) for t in target])
    return correct_points / total_points


def myplot(x, y, s, bins=1000):
    # heatmap, xedges, yedges = np.histogram2d(x, [(ys-1)*-1 for ys in y], bins=bins, range= [[0, 1], [0, 1]])
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range= [[0, 1], [0, 1]])
    heatmap = gaussian_filter(heatmap, sigma=s)

    return heatmap.T


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPU(s).")

    model, criterion = buildLines(args)
    
    
    state_dict = torch.load("trained_models/DETRLines_tusimple_flip111.pt")
    fixed_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name == "curve_eval.uspan" or name == "curve_eval.Nu":
            continue
        fixed_state_dict[name] = v
    model.load_state_dict(fixed_state_dict)
    model.cuda()
    
    dataset = TuSimpleDataset('tusimple/', ['label_data_0313.json'], args, test_mode=True)
    dataset2 = TuSimpleDataset('tusimple/', ['test_label.json'], args, test_mode=False)
    dl = DataLoader(dataset, batch_size=1)
    dl2 = DataLoader(dataset2, batch_size=1)
    
    print("min_lanes:", dataset.min_lanes)
    print("max_lanes:", dataset.max_lanes)
    
    print(len(dataset))
    
    # query_embed = fixed_state_dict['query_embed.weight'].cpu()
    
    # print(query_embed)
    # print(query_embed.size())
    # plt.imshow(query_embed, cmap='gray')
    # plt.show()
    
    # accuracy_accumulator = 0
    # for im, target in dl:
        # # print(im.size()) # torch.Size([1, 3, 720, 1280])
        # im = im.cuda()
        
        # out = model(im)
        
        # pred_logits = out["pred_logits"]
        # pred_controlpoints = out["pred_controlpoints"]
        # pred_points = out["pred_points"]
        
        # accuracy = calc_accuracy(pred_points.detach().cpu(), target, pred_logits.detach().cpu(), 20)
        # # print(accuracy)
        
        # accuracy_accumulator += accuracy
        
        # # if accuracy > 0.95:
            # # plot_image(im.cpu(), target, pred_logits.detach().cpu(), pred_controlpoints.detach().cpu(), pred_points.detach().cpu(), accuracy)
         
    
    loss_accumulator = 0
    cp_list = [[] for i in range(10)]
    ha = 0
    for im, target in dl2:
        ha += 1
        # if ha == 10:
            # break
        im = im.cuda()
        
        out = model(im)
        
        for i in range(len(out['pred_controlpoints'][0])):
            cp_list[i].append(out['pred_controlpoints'][0][i].detach().cpu())
        
        # target_list = []
        # for t in target:
            # target_list.append({'labels': torch.zeros(t.size(0), dtype=int).cuda(),
                                # 'points': t.cuda()})
                                
        # loss_dict = criterion(out, target_list)
        
        # loss_accumulator += loss_dict['loss_distance'].item()
 
    cp_list = [torch.stack(x, dim=0) for x in cp_list]
    
    for he in range(len(cp_list)):
        x = list(cp_list[he][:, :, 1].flatten())
        y = list(cp_list[he][:, :, 0].flatten())        
    
        plt.figure()

        img = myplot(x, y, 16)
        plt.imshow(img, extent=[0, 1, 0, 1], cmap=cm.jet)

        plt.tight_layout()
        plt.savefig(f'..\Images\controlpoints heatmaps\{he}.png', bbox_inches='tight', transparent=True)
        # plt.show()
    
    
    # for cps in cp_list:
        # average_cps = cps.mean(dim=0)
        
        # print(average_cps.size())
        
        # plt.figure()
        
        # xplot = [xd for xd in average_cps[:, 1]]
        # yplot = [yd for yd in average_cps[:, 0]]

        # plt.scatter(xplot, yplot)
        
        # plt.xlim(0, 1)
        # plt.ylim(1, 0)
        
        # plt.show()
    
        
    # print("Average accuracy:", accuracy_accumulator / len(dataset))
    # print("Average distance loss:", loss_accumulator / len(dataset))
    print("Done")