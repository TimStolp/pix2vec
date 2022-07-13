from NURBSDiff.curve_eval import CurveEval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from numpy import frombuffer
from torch import from_numpy
from PIL import Image
from tqdm import tqdm
import random


canvas_size = 512
n_curves = 2
randomness = 50
number_of_examples = 10000
linewidth = 5

curve_layer = CurveEval(4, dimension=2, p=3, out_dim=250, dvc='cpu')
for i in tqdm(range(number_of_examples)):
    name = f'random_{n_curves}_curves_bbox_{i}'
    control_points = []
    for n in range(n_curves):
        while True:
            first = torch.Tensor([random.randint(1, 500), random.randint(1, 500)])
            fourth = torch.Tensor([random.randint(1, 500), random.randint(1, 500)])
            if (first - fourth).pow(2).sum().sqrt() > 0.2 * canvas_size:
                break
        while True:
            second = first + ((fourth - first) * 0.33) + torch.Tensor([random.randint(-randomness, randomness), random.randint(-randomness, randomness)])
            if torch.all(second >= 0) and torch.all(second <= canvas_size-1):
                break
        while True:
            third = first + ((fourth - first) * 0.66) + torch.Tensor([random.randint(-randomness, randomness), random.randint(-randomness, randomness)])
            if torch.all(third >= 0) and torch.all(third <= canvas_size-1):
                break

        curve_control_points = torch.stack((first, second, third, fourth))
        control_points.append(curve_control_points)

    control_points = torch.stack(control_points)
    curve_weights = torch.ones(control_points.size(0), 4, 1)

    control_points = torch.cat((control_points, curve_weights), axis=-1)

    # print("controlpoints size:", control_points.size())

    pc = curve_layer(control_points).detach()

    # print("pc size:", pc.size())

    fig = plt.figure(figsize=(canvas_size/100, canvas_size/100), facecolor=(0, 0, 0))

    plt.xlim([0, canvas_size])
    plt.ylim([0, canvas_size])
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.gca().invert_yaxis()

    for p in pc:
        plt.plot(p[:, 1], p[:, 0], 'white', linewidth=linewidth, solid_capstyle='round')

    fig.canvas.draw()

    plot = frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    im = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,)).copy()

    # plt.show()
    plt.close()

    bboxes = []
    for p in pc:
        ymin = torch.min(p[:, 0], dim=-1)[0] - linewidth * 0.7
        ymax = torch.max(p[:, 0], dim=-1)[0] + linewidth * 0.7
        xmin = torch.min(p[:, 1], dim=-1)[0] - linewidth * 0.7
        xmax = torch.max(p[:, 1], dim=-1)[0] + linewidth * 0.7

        ymin = 0 if ymin < 0 else ymin
        ymax = 512 if ymax > 512 else ymax
        xmin = 0 if xmin < 0 else xmin
        xmax = 512 if xmax > 512 else xmax

        xsize = (xmax - xmin)
        ysize = (ymax - ymin)
        xmid = xmax - xsize / 2
        ymid = ymax - ysize / 2

        bbox = torch.Tensor([xmid, ymid, xsize, ysize])
        bboxes.append(bbox)

    bboxes = torch.stack(bboxes)

    # fig = plt.figure(figsize=(canvas_size / 100, canvas_size / 100))
    #
    # plt.imshow(im)
    # ax = plt.gca()
    # plt.gca().invert_yaxis()
    #
    # for b in bboxes:
    #     xmid, ymid, xsize, ysize = b
    #
    #     xoffset = xsize / 2
    #     yoffset = ysize / 2
    #
    #     xmin = xmid - xoffset
    #     ymin = ymid - yoffset
    #     xmax = xmid + xoffset
    #     ymax = ymid + yoffset
    #
    #     # Create a Rectangle patch
    #
    #     rect = patches.Rectangle((xmin, ymin), xsize, ysize, linewidth=1, edgecolor='r', facecolor='none')
    #
    #     # Add the patch to the Axes
    #     ax.add_patch(rect)
    #
    #     plt.scatter([xmid, xmin, xmax], [ymid, ymin, ymax])
    #
    # plt.show()

    im = Image.fromarray(im)
    im.save(f'./custom_data/{name}.png')
    torch.save(control_points, f'./custom_data/{name}_control_points.pt')
    torch.save(pc, f'./custom_data/{name}_point_cloud.pt')
    torch.save(bboxes, f'./custom_data/{name}_bbox.pt')
