"""Print parameter count and FLOPs of DA-LiteMamba (requires a CUDA GPU for mamba_ssm)."""
import argparse
import os
import sys

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dalitemamba import DALiteMamba  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    args = p.parse_args()

    model = DALiteMamba(in_channels=args.in_channels).cuda().eval()
    x = torch.rand(1, args.in_channels, *args.size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.3f}M")
    # Note: fvcore does not count the selective-scan CUDA kernel.
    print(flop_count_table(FlopCountAnalysis(model, x)))


if __name__ == "__main__":
    main()
