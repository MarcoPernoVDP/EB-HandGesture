from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FILENAME_RE = r"^(?P<subject>[^_]+)_(?P<distance>[^_]+)_(?P<class_idx>.+)$"
CLASS_RE = r"^(?P<class>[A-Za-z_]+?)(?P<index>\d+)$"


def _normalize_name(value: Optional[str]) -> Optional[str]:
    return value.strip().lower() if value else None


def parse_base_name(base_name: str) -> Dict[str, Optional[str]]:
    import re
    m = re.match(FILENAME_RE, base_name)
    if not m:
        return {
            "subject": None,
            "distance": None,
            "gesture_class": None,
            "sample_index": None,
            "parse_ok": False,
        }
    class_idx = m.group("class_idx")
    c = re.match(CLASS_RE, class_idx)
    if c:
        gesture_class = _normalize_name(c.group("class"))
        sample_index = int(c.group("index"))
    else:
        gesture_class = _normalize_name(class_idx)
        sample_index = None
    return {
        "subject": _normalize_name(m.group("subject")),
        "distance": _normalize_name(m.group("distance")),
        "gesture_class": gesture_class,
        "sample_index": sample_index,
        "parse_ok": True,
    }


def _read_T(h5_path: Path) -> Optional[int]:
    try:
        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                return None
            dset = f["data"]
            if dset.ndim < 1:
                return None
            return int(dset.shape[0])
    except Exception:
        return None


def build_valid_index(
    data_root: Path,
    split: str,
    cache_dir: Path,
    t_min: int,
    t_max: int,
) -> Dict[str, List[Dict[str, str]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"valid_index_{split}_T{t_min}-{t_max}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    split_dir = data_root / split
    h5_files = sorted(split_dir.glob("*.h5"))
    bbox_files = sorted(split_dir.glob("*_bbox.npy"))

    h5_map = {p.stem: p for p in h5_files}
    bbox_map = {p.stem.replace("_bbox", ""): p for p in bbox_files}

    all_keys = sorted(set(h5_map) | set(bbox_map))
    valid = []
    rejected = []

    for key in all_keys:
        meta = parse_base_name(key)
        if not meta.get("parse_ok"):
            rejected.append({"sample_id": key, "reason": "parse_failed"})
            continue
        if key not in h5_map or key not in bbox_map:
            rejected.append({"sample_id": key, "reason": "missing_pair"})
            continue
        t_val = _read_T(h5_map[key])
        if t_val is None or not (t_min <= t_val <= t_max):
            rejected.append({"sample_id": key, "reason": f"T_outside_range:{t_val}"})
            continue

        valid.append({
            "split": split,
            "sample_id": key,
            "h5_path": str(h5_map[key]),
            "bbox_path": str(bbox_map[key]),
            "gesture_class": meta.get("gesture_class"),
            "distance": meta.get("distance"),
            "T": int(t_val),
        })

    payload = {
        "split": split,
        "t_min": t_min,
        "t_max": t_max,
        "valid": valid,
        "rejected": rejected,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def build_label_mapping(cache_dir: Path, splits: List[str], t_min: int, t_max: int) -> Dict[str, int]:
    labels_path = cache_dir / f"labels_T{t_min}-{t_max}.json"
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    labels = set()
    for split in splits:
        index_path = cache_dir / f"valid_index_{split}_T{t_min}-{t_max}.json"
        if not index_path.exists():
            continue
        with index_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("valid", []):
            if row.get("gesture_class"):
                labels.add(row["gesture_class"])

    label_map = {name: i for i, name in enumerate(sorted(labels))}
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    return label_map


def _pad_or_crop_T(x: np.ndarray, t_target: int) -> np.ndarray:
    t = x.shape[0]
    if t == t_target:
        return x
    if t > t_target:
        return x[:t_target]
    pad = np.zeros((t_target - t, *x.shape[1:]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


class EBHandGestureDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        cache_dir: str | Path,
        t_min: int,
        t_max: int,
        t_target: int,
        splits_for_labels: List[str],
    ) -> None:
        self.data_root = Path(data_root)
        self.cache_dir = Path(cache_dir)
        payload = build_valid_index(self.data_root, split, self.cache_dir, t_min, t_max)
        self.rows = payload.get("valid", [])
        self.label_map = build_label_mapping(self.cache_dir, splits_for_labels, t_min, t_max)
        self.t_target = t_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        h5_path = Path(row["h5_path"])
        with h5py.File(h5_path, "r") as f:
            data = np.asarray(f["data"])
        if data.ndim == 3:
            data = data[:, None, :, :]
        elif data.ndim != 4:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        data = _pad_or_crop_T(data, self.t_target)
        x = torch.from_numpy(data.astype(np.float32))
        label_name = row.get("gesture_class")
        y = self.label_map[label_name]
        return x, y
