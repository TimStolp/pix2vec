import torch
import torch.nn as nn
import geomloss
from pytorch3d.loss import chamfer_distance


class VectorizationLoss(nn.Module):
    def __init__(self, loss='sinkhorn', p=2, blur=0.05):
        super().__init__()
        self.loss_func = geomloss.SamplesLoss(loss, p, blur)
        self.p = p
        self.loss = loss

    def forward(self, out, target, controlpoints=None):
        if self.loss == 'chamfer':
            loss, _ = chamfer_distance(out.flatten(1, 2), target, point_reduction='mean')
        else:
            loss = self.loss_func(out.flatten(1, 2), target)
            loss = sum(loss) / len(loss)

        if controlpoints is not None:
            batch_size, n_splines, n_cp, xy = controlpoints.size()
            cp_losses = []
            for b in range(batch_size):
                zero_zero = (controlpoints[b, 0, 0] - target[b, 0]) ** self.p / self.p
                zero_one = (controlpoints[b, 0, 0] - target[b, -1]) ** self.p / self.p
                one_zero = (controlpoints[b, 0, -1] - target[b, 0]) ** self.p / self.p
                one_one = (controlpoints[b, 0, -1] - target[b, -1]) ** self.p / self.p

                cp_loss = min(sum(zero_zero + one_one), sum(zero_one + one_zero))

                # cp_loss = sum((controlpoints[b, 0, 0] - target[b, 0]) ** self.p / self.p +
                #               (controlpoints[b, 0, -1] - target[b, -1]) ** self.p / self.p)
                cp_losses.append(cp_loss)

            loss += sum(cp_losses) / batch_size
        return loss
