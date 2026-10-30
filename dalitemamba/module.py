"""PyTorch Lightning wrapper for training / evaluating DA-LiteMamba."""
import pytorch_lightning as pl
import torch

from .losses import LOSSES, deep_supervision_loss
from .metrics import binary_stats, dice_score, iou_score
from .optim import Nadam


class Segmentor(pl.LightningModule):
    def __init__(self, model, loss="tcm", lr=1e-3):
        super().__init__()
        self.model = model
        self.loss_fn = LOSSES[loss]
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch
        preds = self.model(image)
        loss = deep_supervision_loss(preds, y_true, self.loss_fn)
        return loss, preds[0], y_true

    def training_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice_score(y_pred, y_true), "train_iou": iou_score(y_pred, y_true)}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._step(batch)
        metrics = {"val_loss": loss, "val_dice": dice_score(y_pred, y_true), "val_iou": iou_score(y_pred, y_true)}
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_true = self._step(batch)
        precision, recall, f1 = binary_stats(y_pred, y_true)
        metrics = {
            "test_loss": loss,
            "test_dice": dice_score(y_pred, y_true),
            "test_iou": iou_score(y_pred, y_true),
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
        }
        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = Nadam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
        return [optimizer], {"scheduler": scheduler, "monitor": "val_dice"}


def load_weights(model, ckpt_path):
    """Load a Lightning checkpoint (or plain state_dict) into a bare model."""
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    return model
