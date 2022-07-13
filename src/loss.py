import torch
import torch.nn as nn
import geomloss
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment


class VectorizationLoss(nn.Module):
    def __init__(self, weights, loss='sinkhorn', p=2, blur=0.01):
        super().__init__()
        self.loss_func = geomloss.SamplesLoss(loss, p, blur)
        self.p = p
        self.loss = loss
        self.cp_weight = weights['cp_weight']
        self.distace_weight = weights['distance_weight']
        self.class_weight = weights['class_weight']
        self.no_obj_weight = weights['no_obj_weight']

    def forward(self, out, target, controlpoints, spline_logits):
        # print("out", out)
        # print("target", target)
        # print("spline_logits", spline_logits)

        batch_size, n_splines_out, n_points_out, dim = out.size()
        batch_cp_losses = []
        batch_distance_losses = []
        batch_class_losses = []
        batch_total_losses = []

        # For every example in batch
        for i in range(len(out)):
            cp_costs = []
            distance_costs = []
            # For every target spline
            # print("target_size: ", target.size())
            for t in target[i]:
                # Calculate losses from predicted splines to target spline
                repeated_target = t.unsqueeze(0).repeat(n_splines_out, 1, 1)
                # print("repeated_target_size: ", repeated_target.size())
                if self.loss == 'chamfer':
                    distance_cost, _ = chamfer_distance(out[i], repeated_target, batch_reduction=None, point_reduction='mean')
                else:
                    distance_cost = self.loss_func(out[i], repeated_target)

                zero_zero = ((controlpoints[i, :, 0] - repeated_target[:, 0]) ** self.p / self.p).sum(dim=1)
                zero_one = ((controlpoints[i, :, 0] - repeated_target[:, -1]) ** self.p / self.p).sum(dim=1)
                one_zero = ((controlpoints[i, :, -1] - repeated_target[:, 0]) ** self.p / self.p).sum(dim=1)
                one_one = ((controlpoints[i, :, -1] - repeated_target[:, -1]) ** self.p / self.p).sum(dim=1)

                cp_cost = torch.minimum(zero_zero + one_one, zero_one + one_zero)

                # print("cp_cost_size: ", cp_cost.size())
                # print("distance_cost_size: ", distance_cost.size())

                cp_costs.append(cp_cost.unsqueeze(1))
                distance_costs.append(distance_cost.unsqueeze(1))

            # Create cost matrices
            cp_cost_matrix = torch.cat(cp_costs, 1)
            distance_cost_matrix = torch.cat(distance_costs, 1)
            class_cost_matrix = spline_logits[i].softmax(-1)[:, torch.zeros(target.size(1), dtype=torch.int64, device=spline_logits.device)]

            total_cost_matrix = self.cp_weight * cp_cost_matrix + self.distace_weight * distance_cost_matrix + self.class_weight * class_cost_matrix

            # print("cp_cost_matrix_size: ", cp_cost_matrix.size())
            # print("distance_cost_matrix_size: ", distance_cost_matrix.size())
            # print("class_cost_matrix_size: ", class_cost_matrix.size())
            # print("total_cost_matrix_size: ", total_cost_matrix.size())

            # Find optimal bipartite matching
            row_ind, col_ind = linear_sum_assignment(total_cost_matrix.detach().cpu())
            # print("distance_cost_matrix", distance_cost_matrix)
            # print("row_ind", row_ind)
            # print("col_ind", col_ind)


            # Calculate losses
            # print("cp_cost_matrix_size", cp_cost_matrix.size())
            # print("cp_cost_matrix[row_ind, col_ind]_size", cp_cost_matrix[row_ind, col_ind].size())

            tgt = torch.ones((1, n_splines_out), dtype=torch.int64, device=spline_logits.device)
            tgt[col_ind, row_ind] = 0
            pred_classes = spline_logits[i].unsqueeze(0).transpose(1, 2)
            entropy_weights = torch.tensor([1, self.no_obj_weight], device=spline_logits.device)

            class_loss = nn.functional.cross_entropy(pred_classes, tgt, entropy_weights)
            cp_loss = cp_cost_matrix[row_ind, col_ind].sum()
            distance_loss = distance_cost_matrix[row_ind, col_ind].sum()

            # Log losses
            batch_cp_losses.append(cp_loss)
            batch_distance_losses.append(distance_loss)
            batch_class_losses.append(class_loss)
            batch_total_losses.append(self.cp_weight * cp_loss + self.distace_weight * distance_loss + self.class_weight * class_loss)
        # Return all losses in a batch.
        return torch.stack(batch_total_losses), torch.stack(batch_distance_losses), torch.stack(batch_cp_losses), torch.stack(batch_class_losses)
