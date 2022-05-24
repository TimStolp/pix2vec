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

    def forward(self, out, target, controlpoints=None, cp_weight=1):
        batch_size, n_splines_out, n_points_out, dim = out.size()
        batch_cp_losses = []
        batch_distance_losses = []
        # For every example in batch
        for i in range(len(out)):
            cp_costs = []
            distance_costs = []
            # For every target spline
            for t in target[i]:
                # Calculate losses from predicted splines to that target spline
                repeated_target = t.unsqueeze(0).repeat(n_splines_out, 1, 1)
                if self.loss == 'chamfer':
                    distance_cost, _ = chamfer_distance(out[i], repeated_target, batch_reduction=None, point_reduction='mean')
                else:
                    distance_cost = self.loss_func(out[i], repeated_target)

                zero_zero = ((controlpoints[i, :, 0] - repeated_target[:, 0]) ** self.p / self.p).sum(dim=1)
                zero_one = ((controlpoints[i, :, 0] - repeated_target[:, -1]) ** self.p / self.p).sum(dim=1)
                one_zero = ((controlpoints[i, :, -1] - repeated_target[:, 0]) ** self.p / self.p).sum(dim=1)
                one_one = ((controlpoints[i, :, -1] - repeated_target[:, -1]) ** self.p / self.p).sum(dim=1)

                cp_cost = cp_weight * torch.minimum(zero_zero + one_one, zero_one + one_zero)

                cp_costs.append(cp_cost.unsqueeze(1))
                distance_costs.append(distance_cost.unsqueeze(1))

            # Create cost matrices
            cp_cost_matrix = torch.cat(cp_costs, 1)
            distance_cost_matrix = torch.cat(distance_costs, 1)

            # find optimal bipartite matching
            # total_cost_matrix = distance_cost_matrix + distance_cost_matrix
            row_ind, col_ind = linear_sum_assignment(cp_cost_matrix.detach().cpu())

            # Sum losses of each matching
            batch_cp_losses.append(cp_cost_matrix[row_ind, col_ind].sum())
            batch_distance_losses.append(distance_cost_matrix[row_ind, col_ind].sum())
        # Mean losses across batch
        return torch.stack(batch_distance_losses), torch.stack(batch_cp_losses)
