from NURBSDiff.curve_eval import CurveEval
import matplotlib.pyplot as plt
import torch
from PIL import Image

# one line
# name = 'one_line'
# control_points = [[[2,   12],
#                   [10,   12],
#                   [18,  12],
#                   [26, 12]]]

# one curve
# name = 'one_curve'
# control_points = [[[2,   26],
#                   [2,   12],
#                   [26,  12],
#                   [26, 2]]]

# one curve
# name = 'one_curve_2'
# control_points = [[[8,   2],
#                   [8,   12],
#                   [26,  12],
#                   [26, 26]]]

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
control_points = [[[6,   1],
                  [6,   6],
                  [24,  6],
                  [24, 12]],
                  [[6, 13],
                   [6, 18],
                   [24, 18],
                   [24, 26]]]

control_points = torch.Tensor(control_points).cuda()
curve_weights = torch.ones(control_points.size()[0], 4, 1).cuda()

control_points = torch.cat((control_points, curve_weights), axis=-1)

print(control_points)
print(control_points.size())

curve_layer = CurveEval(4, dimension=2, p=3, out_dim=50)

pc = curve_layer(control_points).detach().cpu()
control_points = control_points.cpu()

print(pc)
print(pc.size())

im = torch.zeros((28, 28), dtype=torch.uint8)

pixels = pc.round().int()
for p in pixels:
    for y, x in p:
        im[y, x] = 255
print(pixels)

# plt.imshow(im)
# # for p in pc:
# #     plt.scatter(p[:, 1], p[:, 0])
# #
# # for c in control_points:
# #     plt.scatter(c[:, 1], c[:, 0], color='r')
# # plt.savefig(f'./outputImages/data_example.png')

im = Image.fromarray(im.numpy())
im.save(f'./custom_data/{name}.png')
torch.save(control_points, f'./custom_data/{name}_control_points.pt')
torch.save(pc, f'./custom_data/{name}_point_cloud.pt')
