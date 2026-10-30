"""MCC loss and the proposed Tan-hyperbolic Complement Matthews (TCM) loss — Sec. 3.3."""
import torch

DEEP_SUPERVISION_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


def mcc_loss(y_pred, y_true, smooth=1e-5):
    """1 - MCC, computed on soft predictions. `y_pred` are logits."""
    y_pred = torch.sigmoid(y_pred).reshape(-1)
    y_true = y_true.reshape(-1).float()

    TP = torch.sum(y_true * y_pred)
    FN = torch.sum(y_true * (1 - y_pred))
    FP = torch.sum((1 - y_true) * y_pred)
    TN = torch.sum((1 - y_true) * (1 - y_pred))

    numer = TP * TN - FP * FN
    deno = torch.sqrt((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP)) + smooth
    return 1 - numer / deno


def tcm_loss(y_pred, y_true):
    """TCM loss: tanh(1 - MCC)."""
    return torch.tanh(mcc_loss(y_pred, y_true))


LOSSES = {"mcc": mcc_loss, "tcm": tcm_loss}


def deep_supervision_loss(preds, y_true, loss_fn=tcm_loss, weights=DEEP_SUPERVISION_WEIGHTS):
    """Weighted sum over the 4 prediction heads, Eq. (9)."""
    return sum(w * loss_fn(p, y_true) for w, p in zip(weights, preds))
