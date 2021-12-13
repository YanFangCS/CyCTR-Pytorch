import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['weighted_dice_loss']


def weighted_dice_loss(
    prediction,
    target_seg,
    weighted_val: float = 1.0,
    reduction: str = "sum",
    eps: float = 1e-8,
):
    """
    Weighted version of Dice Loss

    Args:
        prediction: prediction
        target_seg: segmentation target
        weighted_val: values of k positives,
        reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum' : The output will be summed.
        eps: the minimum eps,
    """
    target_seg_fg = target_seg==1
    target_seg_bg = target_seg==0
    target_seg = torch.stack([target_seg_bg, target_seg_fg], dim=1).float()
    
    n, _, h, w = target_seg.shape

    prediction = prediction.reshape(-1, h, w)
    target_seg = target_seg.reshape(-1, h, w)
    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h*w)
    target_seg = target_seg.reshape(-1, h*w)

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum()/n
    elif reduction == "mean":
        loss = loss.mean()
    return loss

class WeightedDiceLoss(nn.Module):
    def __init__(
            self,
            weighted_val: float = 1.0,
            reduction: str = "sum",
        ):
        super(WeightedDiceLoss, self).__init__()
        self.weighted_val = weighted_val
        self.reduction = reduction
        
    def forward(self, 
                prediction,
                target_seg,):
        return weighted_dice_loss(
            prediction,
            target_seg,
            self.weighted_val,
            self.reduction,
        )

