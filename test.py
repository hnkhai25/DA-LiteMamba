import argparse

import pytorch_lightning as pl

from dalitemamba import DALiteMamba
from dalitemamba.data import build_datasets, build_loader
from dalitemamba.module import Segmentor, load_weights


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DA-LiteMamba")
    p.add_argument("--dataset", required=True, choices=["dsb2018", "sunnybrook", "npz"])
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subset", default="endo", choices=["endo", "epi"], help="Sunnybrook subset")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--loss", default="tcm", choices=["tcm", "mcc"])
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    _, _, test_set, in_channels = build_datasets(args.dataset, args.data_dir, args.subset)
    test_loader = build_loader(test_set, args.batch_size, num_workers=args.num_workers)

    model = load_weights(DALiteMamba(in_channels=in_channels, num_classes=1), args.ckpt)
    pl.Trainer(logger=False).test(Segmentor(model, loss=args.loss), test_loader)


if __name__ == "__main__":
    main()
