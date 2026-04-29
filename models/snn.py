from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

try:
    import sinabs.layers as sl
except Exception:  # pragma: no cover
    sl = None


class SNNBaseline(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: Iterable[int],
        hidden_dim: int,
        num_classes: int,
        drop: float = 0.0,
        lif_params: dict | None = None,
    ) -> None:
        super().__init__()
        ch1, ch2 = list(channels)
        lif_params = lif_params or {"tau_mem": 20.0, "tau_syn": 5.0, "v_th": 1.0}

        self.conv1 = nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop)

        if sl is not None:
            self.lif1 = sl.LIF(**lif_params)
            self.lif2 = sl.LIF(**lif_params)
        else:
            self.lif1 = nn.ReLU()
            self.lif2 = nn.ReLU()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, num_classes),
        )

    def reset_states(self) -> None:
        for module in self.modules():
            if hasattr(module, "reset_states"):
                module.reset_states()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,T,C,H,W], got {x.shape}")
        self.reset_states()
        b, t, _, _, _ = x.shape
        logits = None
        for step in range(t):
            xt = x[:, step]
            z = self.pool(self.lif1(self.conv1(xt)))
            z = self.pool(self.lif2(self.conv2(z)))
            z = self.drop(self.gap(z))
            out = self.head(z)
            logits = out if logits is None else logits + out
        return logits / float(t)
