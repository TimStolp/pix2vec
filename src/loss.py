import torch
import torch.nn as nn
import geomloss
from pytorch3d.loss import chamfer_distance


def pointcloud_loss(out, target, controlpoints=None):
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


class VectorizationLoss(nn.Module):
    def __init__(self, loss='sinkhorn', p=2, blur=0.05):
        super().__init__()
        self.loss_func = geomloss.SamplesLoss(loss, p, blur)
        self.p = p

    def forward(self, out, target, controlpoints=None):
        loss = self.loss_func(out.flatten(1, 2), target)

        if controlpoints is not None:
            batch_size, n_splines, n_cp, xy = controlpoints.size()
            cp_losses = []
            for b in range(batch_size):
                cp_loss = sum((controlpoints[b, 0, 0] - target[b, 0]) ** self.p / self.p +
                              (controlpoints[b, 0, -1] - target[b, -1]) ** self.p / self.p)
                cp_losses.append(cp_loss)

            loss += sum(cp_losses) / batch_size
        return loss
