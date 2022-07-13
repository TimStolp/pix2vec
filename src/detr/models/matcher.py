# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import geomloss
from scipy.optimize import linear_sum_assignment
from torch import nn

from ..util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        print("out_prob size", out_prob.size())
        print("out_bbox size", out_bbox.size())

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        print("tgt_ids size", tgt_ids.size())
        print("tgt_bbox size", tgt_bbox.size())

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        print("bbox size", cost_bbox.size())
        print("class size", cost_class.size())
        print("giou size", cost_giou.size())
        #
        # print("bbox", cost_bbox)
        # print("class", cost_class)
        # print("giou", cost_giou)

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()
        #
        print("C size", C.size())
        print()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)


class HungarianMatcherLines(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, class_weight: float = 1, endpoint_weight: float = 1, distance_weight: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_weight = class_weight
        self.endpoint_weight = endpoint_weight
        self.distance_weight = distance_weight
        self.distance_loss_func = geomloss.SamplesLoss('sinkhorn', p=1, blur=0.01)
        assert class_weight != 0 or endpoint_weight != 0 or distance_weight != 0, "all costs cant be 0"

    @torch.no_grad()
    def distance_cost(self, out_points, tgt_points):
        # Repeat output and target for batch calculation
        repeat_int_n = out_points.size(0)
        repeat_n = tgt_points.size(0)
        out_points = out_points.repeat(repeat_n, 1, 1)
        tgt_points = tgt_points.repeat_interleave(repeat_int_n, dim=0)
        # print("out_points after size", out_points.size())
        # print("tgt_points after size", tgt_points.size())

        # Compute the sinkhorn pointcloud distance
        cost_distance = self.distance_loss_func(out_points, tgt_points).reshape(repeat_int_n, repeat_n)
        return cost_distance

    @torch.no_grad()
    def endpoint_cost(self, out_controlpoints, tgt_points):
        out_controlpoints_zero_one = out_controlpoints[:, [0, -1], :].flatten(1, 2)
        out_controlpoints_one_zero = out_controlpoints[:, [-1, 0], :].flatten(1, 2)
        tgt_points = tgt_points[:, [0, -1], :].flatten(1, 2)
        # print("out_controlpoints after size", out_controlpoints.size())
        # print("tgt_points after size", tgt_points.size())

        cost_endpoint_zero_one = torch.cdist(out_controlpoints_zero_one, tgt_points)
        cost_endpoint_one_zero = torch.cdist(out_controlpoints_one_zero, tgt_points)
        cost_endpoint = torch.min(cost_endpoint_zero_one, cost_endpoint_one_zero)

        # print("cost_endpoint size", cost_endpoint.size())

        return cost_endpoint

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_controlpoints = outputs["pred_controlpoints"].flatten(0, 1)
        out_points = outputs["pred_points"].flatten(0, 1)

        # print("out_prob size", out_prob.size())
        # print("out_controlpoints size", out_controlpoints.size())
        # print("out_points size", out_points.size())

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["points"] for v in targets])

        # print("tgt_ids size", tgt_ids.size())
        # print("tgt_points size", tgt_points.size())

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be omitted.
        cost_class = -out_prob[:, tgt_ids]
        # print("class size", cost_class.size())

        # Compute the sinkhorn pointcloud distance

        try:
            cost_distance = self.distance_cost(out_points, tgt_points)
        except:
            print("out_points size", out_points.size())
            print("tgt_points size", tgt_points.size())
            raise ValueError

        # print("cost_distance size", cost_distance.size())

        # Compute endpoint distance
        cost_endpoint = self.endpoint_cost(out_controlpoints, tgt_points)

        # Final cost matrix
        C = self.endpoint_weight * cost_endpoint + self.class_weight * cost_class + self.distance_weight * cost_distance

        C = C.view(bs, num_queries, -1).cpu()
        #
        # print("C size", C.size())
        # print()

        sizes = [len(v["points"]) for v in targets]

        # print(sizes)

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher_lines(args):
    return HungarianMatcherLines(class_weight=args.class_weight, endpoint_weight=args.endpoint_weight, distance_weight=args.distance_weight)
