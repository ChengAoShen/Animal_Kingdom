# Copyright (c) UD_lab. All rights reserved.
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from mmaction.registry import MODELS

from .base import BaseWeightedLoss


@MODELS.register_module()
class FocalLossWithLogits(BaseWeightedLoss):
    """Focal Loss for Dense Object Detection.

    This class implements the Focal Loss function, which is designed to address class imbalance by down-weighting easy examples and focusing training on hard negatives.

    Args:
        alpha (float): Scaling factor for positive examples, default = 1.
        gamma (float): Focusing parameter to scale the loss, default = 0.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
                         'none': no reduction will be applied.
                         'mean': the sum of the output will be divided by the number of elements in the output.
                         'sum': the output will be summed. Defaults to 'mean'.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def _forward(
        self, cls_score: torch.Tensor, label: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward function.

        Calculates the Focal Loss.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The calculated Focal Loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(
            cls_score, label, reduction="none"
        )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        elif self.reduction == "sum":
            return torch.sum(F_loss)
        else:  # 'none'
            return F_loss


@MODELS.register_module()
class LDAMLossWithLogits(BaseWeightedLoss):
    """Label-Distribution-Aware Margin Loss.

    This class implements the LDAM loss, which is designed to address class imbalance by applying different margins to different classes based on their label distribution.

    Args:
        class_counts (List[float]): Number of samples for each class.
        max_m (float): Maximum margin applied to the minority class. Defaults to 0.5.
        s (float): Scale parameter for the logits. Defaults to 30.
        step_epoch (int): Number of epochs after which the class weights are recalibrated. Defaults to 80.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
    """

    def __init__(
        self,
        class_counts: List[float],
        max_m: float = 0.5,
        s: float = 30,
        step_epoch: int = 80,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.class_counts = class_counts
        m_list = 1.0 / np.sqrt(np.sqrt(class_counts))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list).cuda()
        self.s = s
        self.step_epoch = step_epoch
        self.weight = None

    def reset_epoch(self, epoch):
        """Adjust class weights based on the current epoch."""
        idx = epoch // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.class_counts)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = (
            per_cls_weights / np.sum(per_cls_weights) * len(self.class_counts)
        )
        self.weight = torch.FloatTensor(per_cls_weights).cuda()

    def _forward(
        self, cls_score: torch.Tensor, label: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward function.

        Calculates the LDAM Loss.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The calculated LDAM Loss.
        """
        label = label.to(torch.float32)
        batch_m = torch.matmul(self.m_list[None, :], label.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        cls_score_m = cls_score - batch_m

        output = torch.where(label.type(torch.uint8), cls_score_m, cls_score)
        loss = F.binary_cross_entropy_with_logits(
            self.s * output, label, reduction=self.reduction, weight=self.weight
        )
        return loss


@MODELS.register_module()
class EQLLossWithLogits(BaseWeightedLoss):
    """Equilibrium Loss for Long-Tail Distribution.

    EQL is designed to alleviate the impact of the easy negative samples on the tail classes by down-weighting their losses.

    Args:
        class_counts (List[float]): Number of samples for each class.
        max_tail_num (int): The threshold for defining tail classes based on the number of samples. Defaults to 100.
        gamma (float): The suppression factor for down-weighting the loss of easy negative samples. Defaults to 1.76e-3.
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
                         'none': no reduction will be applied,
                         'mean': the sum of the output will be divided by the number of elements in the output,
                         'sum': the output will be summed. Defaults to 'mean'.
    """

    def __init__(
        self,
        class_counts: List[float],
        max_tail_num: int = 100,
        gamma: float = 1.76e-3,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.class_counts = class_counts
        self.max_tail_num = max_tail_num
        self.gamma = gamma
        self.reduction = reduction
        self.tail_flags = None  # This will be initialized in the first forward pass

    def _forward(
        self, cls_score: torch.Tensor, label: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.

        Returns:
            torch.Tensor: The calculated EQL loss.
        """
        if self.tail_flags is None:
            self.tail_flags = torch.tensor(
                [num <= self.max_tail_num for num in self.class_counts]
            ).to(cls_score.device)

        n_i, n_c = cls_score.size()
        weight = cls_score.new_zeros(n_c)
        weight[self.tail_flags] = 1
        weight = weight.view(1, n_c).expand(n_i, n_c)

        rand = torch.rand((n_i, n_c)).to(cls_score.device)
        rand[rand < 1 - self.gamma] = 0
        rand[rand >= 1 - self.gamma] = 1

        eql_w = 1 - rand * weight * (1 - label)
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_score, label, weight=eql_w, reduction=self.reduction
        )

        return loss_cls * self.loss_weight
