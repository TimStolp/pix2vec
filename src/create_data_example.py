from NURBSDiff.curve_eval import CurveEval
import matplotlib.pyplot as plt
import torch
from PIL import Image

# one line
# name = 'one_line'
# control_points = [[[12,   40],
#                   [50,   60],
#                   [80,  60],
#                   [120, 80]]]

# one curve
# name = 'one_curve'
# control_points = [[[10,   120],
#                   [10,   65],
#                   [120,  65],
#                   [120, 10]]]

# one curve
# name = 'one_curve_2'
# control_points = [[[24,   12],
#                   [24,   65],
#                   [120,  65],
#                   [120, 120]]]

# name = 'one_curve_3'
# control_points = [[[24,   120],
#                   [24,   35],
#                   [120,  35],
#                   [120, 120]]]

# two lines
# name = 'two_lines'
# control_points = [[[2,   12],
#                   [10,   12],
#                   [18,  12],
#                   [26, 12]],
#                   [[2, 18],
#                    [10, 18],
#                    [18, 18],
#                    [26, 18]]]

# two curves
name = 'two_curves'
control_points = [[[64,   22],
                  [64,   64],
                  [110,  64],
                  [110, 34]],
                  [[24, 35],
                   [24, 100],
                   [120, 68],
                   [120, 120]]]

control_points = torch.Tensor(control_points).cuda()
curve_weights = torch.ones(control_points.size()[0], 4, 1).cuda()

control_points = torch.cat((control_points, curve_weights), axis=-1)

print(control_points)
print(control_points.size())

curve_layer = CurveEval(4, dimension=2, p=3, out_dim=250)

pc = curve_layer(control_points).detach().cpu()
control_points = control_points.cpu()

print(pc)
print(pc.size())

im = torch.zeros((128, 128), dtype=torch.uint8)

pixels = pc.round().int()
for p in pixels:
    for y, x in p:
        im[y, x] = 255
print(pixels)

plt.imshow(im)
plt.show()

im = Image.fromarray(im.numpy())
im.save(f'./custom_data/{name}.png')
torch.save(control_points, f'./custom_data/{name}_control_points.pt')
torch.save(pc, f'./custom_data/{name}_point_cloud.pt')
