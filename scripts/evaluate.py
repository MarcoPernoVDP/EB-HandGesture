from __future__ import annotations

import argparse
import sys
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets.dataset import EBHandGestureDataset
from models.losses import build_loss
from models.snn import SNNBaseline
from training.trainer import evaluate
from utils.config import load_config
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_root = Path(cfg["paths"]["data_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    out_dir = Path(cfg["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    t_min = cfg["dataset"]["t_min"]
    t_max = cfg["dataset"]["t_max"]
    t_target = cfg["dataset"]["t_target"]
    splits = cfg["dataset"]["valid_splits"]

    eval_ds = EBHandGestureDataset(data_root, args.split, cache_dir, t_min, t_max, t_target, splits)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    num_classes = len(eval_ds.label_map)
    model = SNNBaseline(
        in_channels=cfg["model"]["in_channels"],
        channels=cfg["model"]["channels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=num_classes,
        drop=cfg["model"]["drop"],
        lif_params=cfg["model"]["lif"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)

    loss_fn = build_loss()
    metrics = evaluate(model, eval_loader, device, loss_fn)
    save_json(out_dir / f"metrics_{args.split}.json", metrics)


if __name__ == "__main__":
    main()
