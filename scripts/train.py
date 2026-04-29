from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import EBHandGestureDataset
from models.losses import build_loss
from models.snn import SNNBaseline
from training.trainer import TrainState, evaluate, train_one_epoch
from utils.config import load_config
from utils.seed import set_seed
from utils.io import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["training"]["seed"])

    data_root = Path(cfg["paths"]["data_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    out_dir = Path(cfg["paths"]["outputs"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    t_min = cfg["dataset"]["t_min"]
    t_max = cfg["dataset"]["t_max"]
    t_target = cfg["dataset"]["t_target"]
    splits = cfg["dataset"]["valid_splits"]

    train_ds = EBHandGestureDataset(data_root, "train", cache_dir, t_min, t_max, t_target, splits)
    val_ds = EBHandGestureDataset(data_root, "val", cache_dir, t_min, t_max, t_target, splits)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    num_classes = len(train_ds.label_map)
    model = SNNBaseline(
        in_channels=cfg["model"]["in_channels"],
        channels=cfg["model"]["channels"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_classes=num_classes,
        drop=cfg["model"]["drop"],
        lif_params=cfg["model"]["lif"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    scheduler_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_cfg["t_max"]
    )

    loss_fn = build_loss()
    use_amp = bool(cfg["training"]["amp"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler("cuda") if use_amp else torch.cuda.amp.GradScaler(enabled=False)

    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
    state = TrainState(epoch=0, best_val_acc=0.0)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        state.epoch = epoch
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            loss_fn,
            cfg["training"].get("grad_clip"),
            use_amp,
        )
        metrics = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", metrics["loss"], epoch)
        writer.add_scalar("acc/val", metrics["accuracy"], epoch)
        writer.add_scalar("f1/val", metrics["macro_f1"], epoch)

        if metrics["accuracy"] > state.best_val_acc:
            state.best_val_acc = metrics["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")

        if epoch % cfg["training"]["save_every"] == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / f"epoch_{epoch}.pt")

        save_json(out_dir / "last_metrics.json", {"epoch": epoch, **metrics})

    writer.close()


if __name__ == "__main__":
    main()
