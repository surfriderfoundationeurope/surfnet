"""The ``losses`` submodule provides several loss functions to evaluate models.

This submodule contains the following class:

- ``FocalLoss`` : This class allows to compute the focal loss of a object detection model with several loss functions (train, test, focal).

This submodule contains the following functions:

- ``l1_loss(pred:torch.Tensor, gt:torch.Tensor, pos_inds:torch.Tensor)`` : Evaluate and compute the L1 loss on pred and gt based on pos_inds.
- ``sigmoid(x:torch.Tensor)`` : Compute the input Tensor with the sigmoid activation function.

"""

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss as torch_l1_loss


class FocalLoss(nn.Module):

    """This class allows to compute the focal loss of a object detection model.

    Args:
        alpha (int): alpha coefficient for focal loss. Set as default to ``2``.
        beta (int): beta coefficient for focal loss. Set as default to ``4``.
        train (bool): ``True`` if we compute the focal loss for train phasis. In this case we use the ``_focal_loss`` method to compute the loss. Set as default to ``True``.
        centernet_output (bool): ``True`` if the model used for predict outputs is Centernet. Set as default to ``True``.
    """

    def __init__(
        self,
        alpha: int = 2,
        beta: int = 4,
        train: bool = True,
        centernet_output: bool = True,
    ):
        super().__init__()
        self.alpha = int(alpha)
        self.beta = int(beta)
        self.focal_loss = self._focal_loss if train else self._focal_loss_class_wise
        self.loss = self._train_loss if train else self._test_loss
        self.centernet_output = centernet_output

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> float:

        """Compute the loss between predictions ``pred`` and the true values ``gt`` (ground truths).

        Args:
            pred (torch.Tensor): model predictions
            gt (torch.Tensor): ground truths

        Returns:
            The loss between predictions and ground truths.
        """

        return self.loss(pred, gt)

    def train_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> float:

        """Evaluate and compute the training loss.

        Args:
            pred (torch.Tensor): model predictions
            gt (torch.Tensor): ground truths

        Returns:
            The training loss between predictions and ground truths.
        """

        if self.centernet_output:
            pred = pred[0]

        pred_centers = pred["hm"]
        gt_centers = gt[:, :-2, :, :]

        # pred_wh = pred['wh']
        # gt_wh = gt[:,-2:,:,:]

        pos_inds = gt_centers.eq(1).float()
        neg_inds = gt_centers.lt(1).float()

        return self.focal_loss(
            pred_centers, gt_centers, pos_inds, neg_inds
        )  # + 0.1 * _l1_loss(pred_wh, gt_wh, pos_inds)

    def test_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> float:

        """Evaluate and compute the test loss.

        Args:
            pred (torch.Tensor): model predictions
            gt (torch.Tensor): ground truths

        Returns:
            The test loss between predictions and ground truths.
        """

        if self.centernet_output:
            pred = pred[0]

        pred_centers = pred["hm"]
        gt_centers = gt[:, :-2, :, :]

        pos_inds = gt_centers.eq(1).float()
        neg_inds = gt_centers.lt(1).float()

        return self._focal_loss_class_wise(pred_centers, gt_centers, pos_inds, neg_inds)

    def focal_loss(
        self,
        pred_hm: torch.Tensor,
        gt_hm: torch.Tensor,
        pos_inds: torch.Tensor,
        neg_inds: torch.Tensor,
    ) -> float:

        """Evaluate and compute the test loss. Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory

        Args:
            pred_hm (torch.Tensor): predictions with size (batch x c x h x w)
            gt_hm (torch.Tensor): ground truths with size (batch x c x h x w)
            pos_inds (torch.Tensor): positive indices
            neg_inds (torch.Tensor): negative indices

        Returns:
            The focal loss between predictions and gt.
        """

        pred_hm = sigmoid(pred_hm)

        neg_weights = torch.pow(1 - gt_hm, self.beta)

        loss = 0.0

        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred_hm)
            * torch.pow(pred_hm, self.alpha)
            * neg_weights
            * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    def focal_loss_class_wise(
        self,
        pred_hm: torch.Tensor,
        gt_hm: torch.Tensor,
        pos_inds: torch.Tensor,
        neg_inds: torch.Tensor,
    ) -> torch.Tensor:

        """Evaluate and compute the test loss. Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory.

        Args:
            pred_hm (torch.Tensor): predictions with size (batch x c x h x w)
            gt_hm (torch.Tensor): ground truths with size (batch x c x h x w)
            pos_inds (torch.Tensor): positive indices
            neg_inds (torch.Tensor): negative indices

        Returns:
            The focal loss Tensor between predictions and gt.
        """

        pred_hm = sigmoid(pred_hm)

        neg_weights = torch.pow(1 - gt_hm, self.beta)

        pos_loss = torch.log(pred_hm) * torch.pow(1 - pred_hm, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred_hm)
            * torch.pow(pred_hm, self.alpha)
            * neg_weights
            * neg_inds
        )

        num_pos = torch.sum(pos_inds.float(), dim=(0, 2, 3))
        pos_loss = torch.sum(pos_loss, dim=(0, 2, 3))
        neg_loss = torch.sum(neg_loss, dim=(0, 2, 3))

        loss = torch.zeros(size=(len(num_pos),))

        for i, num_poss_class in enumerate(num_pos):

            if num_poss_class == 0:
                loss[i] = loss[i] - neg_loss[i]
            else:
                loss[i] = loss[i] - (pos_loss[i] + neg_loss[i]) / num_poss_class

        return loss


def sigmoid(x: torch.Tensor) -> torch.Tensor:

    """Compute the input Tensor with the sigmoid activation function.

    Args:
        x (torch.Tensor): the input Tensor

    Returns:
        y (torch.Tensor): the output Tensor which corresponds to the application of the sigmoid function on the input
    """

    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

    return y


def l1_loss(
    pred: torch.Tensor, gt: torch.Tensor, pos_inds: torch.Tensor
) -> torch.Tensor:

    """Evaluate and compute the L1 loss on predictions `pred` and ground truths `gt` based on ``pos_inds``. L1 Loss Function is used to minimize the error which is
        the sum of the all the absolute differences between the true value and the predicted value.

    Args:
        pred (torch.Tensor): predictions with size (batch x c x h x w)
        gt (torch.Tensor): ground truths with size (batch x c x h x w)
        pos_inds (torch.Tensor): positive indices

    Returns:
        The L1 Loss Tensor between predictions and ground truths.
    """

    pos_inds = torch.sum(pos_inds, axis=1).ge(1).unsqueeze(1).expand_as(pred)

    return torch_l1_loss(pred[pos_inds], gt[pos_inds], size_average=False) / (
        pos_inds.float().sum() + 1e-4
    )
