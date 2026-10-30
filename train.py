import argparse
import os

import pytorch_lightning as pl

from dalitemamba import DALiteMamba
from dalitemamba.data import build_datasets, build_loader
from dalitemamba.module import Segmentor, load_weights

DEFAULT_BATCH_SIZE = {"dsb2018": 16, "sunnybrook": 8, "npz": 4}


def parse_args():
    p = argparse.ArgumentParser(description="Train DA-LiteMamba")
    p.add_argument("--dataset", required=True, choices=["dsb2018", "sunnybrook", "npz"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subset", default="endo", choices=["endo", "epi"], help="Sunnybrook subset")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="tcm", choices=["tcm", "mcc"])
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--resume-weights", default=None, help="Initialize from a checkpoint")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    train_set, val_set, _, in_channels = build_datasets(args.dataset, args.data_dir, args.subset)
    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.dataset]
    train_loader = build_loader(train_set, batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_loader(val_set, 1 if args.dataset != "sunnybrook" else batch_size,
                              num_workers=args.num_workers)

    model = DALiteMamba(in_channels=in_channels, num_classes=1)
    if args.resume_weights:
        load_weights(model, args.resume_weights)

    name = args.dataset + (f"_{args.subset}" if args.dataset == "sunnybrook" else "")
    ckpt_dir = os.path.join(args.out_dir, name)
    checkpoint = pl.callbacks.ModelCheckpoint(
        ckpt_dir, filename="dalitemamba-{val_dice:.4f}", monitor="val_dice", mode="max",
        save_top_k=1, save_weights_only=True, auto_insert_metric_name=False, verbose=True,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision=args.precision,
        benchmark=True,
        logger=pl.loggers.CSVLogger(args.out_dir, name=name),
        callbacks=[checkpoint, pl.callbacks.TQDMProgressBar()],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    trainer.fit(Segmentor(model, loss=args.loss, lr=args.lr), train_loader, val_loader)
    print(f"Best checkpoint: {checkpoint.best_model_path} (val_dice={checkpoint.best_model_score})")


if __name__ == "__main__":
    main()
