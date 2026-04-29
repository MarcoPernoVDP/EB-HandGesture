from __future__ import annotations

import torch.nn as nn


def build_loss(name: str = "cross_entropy") -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unknown loss: {name}")
