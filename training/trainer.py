from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.metrics import accuracy, confusion_matrix, macro_f1_from_cm


@dataclass
class TrainState:
    epoch: int
    best_val_acc: float


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    loss_fn: torch.nn.Module,
    grad_clip: float | None,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
    return total_loss / max(1, len(loader.dataset))


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> Dict[str, float | np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    pred_arr = np.concatenate(preds) if preds else np.array([])
    target_arr = np.concatenate(targets) if targets else np.array([])
    num_classes = int(target_arr.max() + 1) if target_arr.size else 0
    cm = confusion_matrix(pred_arr, target_arr, num_classes) if num_classes else np.zeros((0, 0))

    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy(pred_arr, target_arr),
        "macro_f1": macro_f1_from_cm(cm) if num_classes else 0.0,
        "confusion_matrix": cm,
    }
