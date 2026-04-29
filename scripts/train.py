from __future__ import annotations

import argparse
import logging
import sys
import time
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")

    set_seed(cfg["training"]["seed"])

    data_root = Path(cfg["paths"]["data_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    out_dir = Path(cfg["paths"]["outputs"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    log.info("Device: %s | AMP: %s", device, cfg["training"]["amp"])

    t_min = cfg["dataset"]["t_min"]
    t_max = cfg["dataset"]["t_max"]
    t_target = cfg["dataset"]["t_target"]
    splits = cfg["dataset"]["valid_splits"]

    spatial_size = cfg["dataset"].get("spatial_size")
    spatial_mode = cfg["dataset"].get("spatial_mode", "area")
    temporal_mode = cfg["dataset"].get("temporal_mode", "sum")

    train_ds = EBHandGestureDataset(
        data_root,
        "train",
        cache_dir,
        t_min,
        t_max,
        t_target,
        splits,
        spatial_size=spatial_size,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
    )
    val_ds = EBHandGestureDataset(
        data_root,
        "val",
        cache_dir,
        t_min,
        t_max,
        t_target,
        splits,
        spatial_size=spatial_size,
        spatial_mode=spatial_mode,
        temporal_mode=temporal_mode,
    )

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
    log.info("Train samples: %d | Val samples: %d | Classes: %d", len(train_ds), len(val_ds), num_classes)
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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    tb_dir = out_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir), flush_secs=5)
    log.info("TensorBoard logdir: %s", tb_dir)
    state = TrainState(epoch=0, best_val_acc=0.0)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        state.epoch = epoch
        t0 = time.time()
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
        dt = time.time() - t0

        log.info(
            "Epoch %d/%d | time %.1fs | train_loss %.4f | val_loss %.4f | val_acc %.4f | val_f1 %.4f",
            epoch,
            cfg["training"]["epochs"],
            dt,
            train_loss,
            metrics["loss"],
            metrics["accuracy"],
            metrics["macro_f1"],
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", metrics["loss"], epoch)
        writer.add_scalar("acc/val", metrics["accuracy"], epoch)
        writer.add_scalar("f1/val", metrics["macro_f1"], epoch)
        writer.flush()

        if metrics["accuracy"] > state.best_val_acc:
            state.best_val_acc = metrics["accuracy"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")

        if epoch % cfg["training"]["save_every"] == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / f"epoch_{epoch}.pt")

        save_json(out_dir / "last_metrics.json", {"epoch": epoch, **metrics})

    writer.close()


if __name__ == "__main__":
    main()
