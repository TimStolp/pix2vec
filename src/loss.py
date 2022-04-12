import torch
from pytorch3d.loss import chamfer_distance


def pointcloud_loss(out, target):
    loss, _ = chamfer_distance(out.flatten(1, 2), target, point_reduction='mean')

    reg_loss = 0
    if out.size(1) > 1:
        reg_losses = []
        for k in range(out.size(1)):
            reg_loss, _ = chamfer_distance(out[:, k, :, :], out[:, torch.arange(out.size(1)) != k, :, :].flatten(1, 2),
                                           point_reduction='mean')
            reg_losses.append(reg_loss)

        reg_loss = 1 / (sum(reg_losses) / len(reg_losses) + 1e-6)

    return loss, reg_loss