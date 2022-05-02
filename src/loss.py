import torch
import torch.nn as nn
import geomloss
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment


class VectorizationLoss(nn.Module):
    def __init__(self, loss='sinkhorn', p=2, blur=0.01):
        super().__init__()
        self.loss_func = geomloss.SamplesLoss(loss, p, blur)
        self.p = p
        self.loss = loss

    def forward(self, out, target, controlpoints=None):
        batch_size, n_splines_out, n_points_out, dim = out.size()
        batch_size, n_splines_target, n_points_target, dim = target.size()
        batch_losses = []
        # For every example in batch
        for i in range(len(out)):
            costs = []
            # For every target spline
            for t in target[i]:
                # Calculate losses from predicted splines to that target spline
                repeated_target = t.unsqueeze(0).repeat(n_splines_out, 1, 1)
                if self.loss == 'chamfer':
                    cost, _ = chamfer_distance(out[i], repeated_target, batch_reduction=None, point_reduction='mean')
                else:
                    cost = self.loss_func(out[i], repeated_target)

                first_cp_cost = ((controlpoints[i, :, 0] - repeated_target[:, 0])**2).sum(dim=1).sqrt()
                last_cp_cost = ((controlpoints[i, :, 1] - repeated_target[:, 1])**2).sum(dim=1).sqrt()
                cost = cost + ((first_cp_cost + last_cp_cost) / 2)

                costs.append(cost.unsqueeze(1))
            # Create cost matrix and find optimal bipartite matching
            cost_matrix = torch.cat(costs, 1)
            row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu())
            # Sum losses of each matching
            batch_losses.append(cost_matrix[row_ind, col_ind].sum())
        # Mean losses across batch
        loss = sum(batch_losses) / batch_size
        return loss
