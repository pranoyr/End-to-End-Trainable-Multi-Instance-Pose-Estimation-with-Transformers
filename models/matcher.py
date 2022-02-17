# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
from typing import Sized
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


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
        self.l_deltas = 0.5
        self.l_vis = 0.2
        self.l_ctr = 0.5
        self.l_abs = 4
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, num_boxes):
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
        out_keypoints = outputs["pred_keypoints"].flatten(0, 1)  # [batch_size * num_queries, (3 * 17) + 1]

        C_pred = out_keypoints[:, :2]
        Z_pred = out_keypoints[:, 2:36]
        V_pred = out_keypoints[:, 36:]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
    
        tgt_keypoints = torch.cat([v["keypoints"] for v in targets])
        
        C_gt = tgt_keypoints[:, :2]
        Z_gt = tgt_keypoints[:, 2:36]
        V_gt = tgt_keypoints[:, 36:]

        C_gt_expand = torch.repeat_interleave(C_gt.unsqueeze(1), 17, dim=1).view(-1,34)
        A_gt = C_gt_expand + Z_gt

        C_pred_expand = torch.repeat_interleave(C_pred.unsqueeze(1), 17, dim=1).view(-1,34)
        A_pred = C_pred_expand + Z_pred

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        
        cost_class = -out_prob[:, tgt_ids]

        Vgt_ = torch.repeat_interleave(V_gt , 2, dim=1)
        offset_loss = [torch.cdist(Z_pred * v_gt_single.unsqueeze(0), z_gt_single.unsqueeze(0) * v_gt_single.unsqueeze(0), p=1) for v_gt_single, z_gt_single in zip(Vgt_, Z_gt)] 
        offset_loss = torch.cat(offset_loss, dim=1) / num_boxes
        viz_loss  =  torch.cdist(V_pred, V_gt, p=2).square() / num_boxes
        center_loss =  torch.cdist(C_pred ,C_gt, p=2).square() / num_boxes
        abs_loss = [torch.cdist(A_pred * v_gt_single.unsqueeze(0), a_gt_single.unsqueeze(0) * v_gt_single.unsqueeze(0), p=1) for v_gt_single, a_gt_single in zip(Vgt_, A_gt)] 
        abs_loss = torch.cat(abs_loss, dim=1) / num_boxes

        C =  self.cost_class * cost_class +  self.l_deltas * offset_loss + self.l_vis * viz_loss + self.l_ctr * center_loss + self.l_abs * abs_loss
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["keypoints"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
