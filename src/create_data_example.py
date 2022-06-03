from NURBSDiff.curve_eval import CurveEval
import matplotlib.pyplot as plt
import torch
from PIL import Image
import random


canvas_size = 128
n_curves = 2
randomness = 30
number_of_examples = 10000

curve_layer = CurveEval(4, dimension=2, p=3, out_dim=250, dvc='cpu')
for i in range(number_of_examples):
    name = f'random_{n_curves}_curves_{i}'
    control_points = []
    for n in range(n_curves):
        while True:
            first = torch.Tensor([random.randint(1, 125), random.randint(1, 125)])
            fourth = torch.Tensor([random.randint(1, 125), random.randint(1, 125)])
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

    im = torch.zeros((canvas_size, canvas_size), dtype=torch.uint8)

    pixels = pc.round().int()
    for p in pixels:
        for y, x in p:
            im[y, x] = 255

    # plt.imshow(im)
    # plt.show()

    im = Image.fromarray(im.numpy())
    im.save(f'./custom_data/{name}.png')
    torch.save(control_points, f'./custom_data/{name}_control_points.pt')
    torch.save(pc, f'./custom_data/{name}_point_cloud.pt')
