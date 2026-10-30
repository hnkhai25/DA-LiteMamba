"""Segmentation metrics. All functions take logits as `y_pred`."""
import torch


def dice_score(y_pred, y_true, smooth=0.):
    """Soft Dice on sigmoid probabilities (used for checkpoint selection)."""
    y_pred = torch.sigmoid(y_pred).reshape(-1)
    y_true = y_true.reshape(-1).float()
    intersection = (y_pred * y_true).sum()
    return (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def iou_score(y_pred, y_true, smooth=1e-5):
    """IoU on binarized predictions (threshold 0.5)."""
    y_pred = torch.sigmoid(y_pred) > 0.5
    y_true = y_true > 0.5
    intersection = (y_pred & y_true).sum().float()
    union = (y_pred | y_true).sum().float()
    return (intersection + smooth) / (union + smooth)


def binary_stats(y_pred, y_true, eps=1e-8):
    """Per-batch mean precision, recall and F1 on binarized predictions."""
    y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    y_true = (y_true > 0.5).float()
    dims = tuple(range(1, y_pred.dim()))

    TP = (y_pred * y_true).sum(dim=dims)
    FP = (y_pred * (1 - y_true)).sum(dim=dims)
    FN = ((1 - y_pred) * y_true).sum(dim=dims)

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision.mean(), recall.mean(), f1.mean()
