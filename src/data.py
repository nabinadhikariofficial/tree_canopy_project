from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from common import discover_year_paths, get_split_mask, normalize_channels, sliding_windows, stack_inputs


@dataclass
class PatchIndex:
    year: int
    top: int
    left: int
    split: str


class MultiYearRasterStore:
    def __init__(self, data_root: str, years: Sequence[int], stats: dict,
                 aligned_dir: Optional[str] = None, ref_path: Optional[str] = None):
        self.data_root = data_root
        self.stats = stats
        self.years = list(years)
        self.ref_path = ref_path
        self.inputs: Dict[int, np.ndarray] = {}
        self.targets: Dict[int, Optional[np.ndarray]] = {}
        self.meta: Dict[int, dict] = {}
        for year in years:
            paths = discover_year_paths(data_root, year)
            x, y, meta = stack_inputs(
                paths, label_align_dir=aligned_dir, ref_path=ref_path)
            self.inputs[year] = normalize_channels(x, stats)
            self.targets[year] = y
            self.meta[year] = meta

    def shape(self, year: int) -> Tuple[int, int]:
        arr = self.inputs[year]
        return int(arr.shape[1]), int(arr.shape[2])


class PatchDataset(Dataset):
    def __init__(self, store: MultiYearRasterStore, years: Sequence[int], split: str,
                 patch_size: int = 128, stride: int = 128, block_size: int = 256,
                 require_labels: bool = True, seed: int = 42, augment: bool = False):
        self.store = store
        self.years = list(years)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.block_size = block_size
        self.require_labels = require_labels
        self.seed = seed
        self.augment = augment
        self.items: List[PatchIndex] = []
        self._build_index()

    def _build_index(self):
        for year in self.years:
            h, w = self.store.shape(year)
            split_mask = get_split_mask(
                h, w, self.split, self.block_size, seed=self.seed)
            y = self.store.targets[year]
            for top, left in sliding_windows(h, w, self.patch_size, self.stride):
                patch_mask = split_mask[top: top +
                                        self.patch_size, left: left + self.patch_size]
                if patch_mask.mean() < 0.95:
                    continue
                x_patch = self.store.inputs[year][:, top: top +
                                                  self.patch_size, left: left + self.patch_size]
                if not np.isfinite(x_patch).all():
                    continue
                if self.require_labels:
                    if y is None:
                        continue
                    y_patch = y[top: top + self.patch_size,
                                left: left + self.patch_size]
                    if np.isfinite(y_patch).mean() < 0.95:
                        continue
                self.items.append(PatchIndex(
                    year=year, top=top, left=left, split=self.split))

    def __len__(self):
        return len(self.items)

    def _apply_augmentation(self, x_t: torch.Tensor, y_t: Optional[torch.Tensor]):
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x_t = torch.rot90(x_t, k, dims=(-2, -1))
            if y_t is not None:
                y_t = torch.rot90(y_t, k, dims=(-2, -1))
        if bool(torch.rand(1).item() < 0.5):
            x_t = torch.flip(x_t, dims=(-1,))
            if y_t is not None:
                y_t = torch.flip(y_t, dims=(-1,))
        if bool(torch.rand(1).item() < 0.5):
            x_t = torch.flip(x_t, dims=(-2,))
            if y_t is not None:
                y_t = torch.flip(y_t, dims=(-2,))
        return x_t, y_t

    def __getitem__(self, idx: int):
        item = self.items[idx]
        x = self.store.inputs[item.year][:, item.top:item.top +
                                         self.patch_size, item.left:item.left + self.patch_size]
        y = self.store.targets[item.year]

        x_t = torch.from_numpy(x).float()
        y_t = None
        out = {
            "x": x_t,
            "year": torch.tensor(item.year, dtype=torch.long),
            "top": torch.tensor(item.top, dtype=torch.long),
            "left": torch.tensor(item.left, dtype=torch.long),
        }
        if y is not None:
            y = y / 100.0
            y_patch = y[item.top:item.top + self.patch_size,
                        item.left:item.left + self.patch_size]
            y_t = torch.from_numpy(y_patch[None, ...]).float()
        if self.augment and self.split == "train":
            x_t, y_t = self._apply_augmentation(x_t, y_t)
        out["x"] = x_t
        if y_t is not None:
            out["y"] = y_t
        return out


class PairConsistencyDataset(Dataset):
    def __init__(self, store: MultiYearRasterStore, year_a: int, year_b: int, split: str,
                 patch_size: int = 128, stride: int = 128, block_size: int = 256,
                 seed: int = 42):
        self.store = store
        self.year_a = year_a
        self.year_b = year_b
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.block_size = block_size
        self.seed = seed
        self.items: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self):
        ha, wa = self.store.shape(self.year_a)
        hb, wb = self.store.shape(self.year_b)
        ma = self.store.meta[self.year_a]
        mb = self.store.meta[self.year_b]
        if (ha, wa) != (hb, wb) or ma["transform"] != mb["transform"] or ma["crs"] != mb["crs"]:
            raise ValueError(
                "Temporal consistency requires all years to be aligned to the same reference grid")

        h, w = ha, wa
        split_mask = get_split_mask(
            h, w, self.split, self.block_size, seed=self.seed)
        ya = self.store.targets[self.year_a]
        yb = self.store.targets[self.year_b]
        for top, left in sliding_windows(h, w, self.patch_size, self.stride):
            patch_mask = split_mask[top: top +
                                    self.patch_size, left: left + self.patch_size]
            if patch_mask.mean() < 0.95:
                continue
            xa = self.store.inputs[self.year_a][:, top: top +
                                                self.patch_size, left: left + self.patch_size]
            xb = self.store.inputs[self.year_b][:, top: top +
                                                self.patch_size, left: left + self.patch_size]
            if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
                continue
            if ya is not None and yb is not None:
                ya_patch = ya[top: top + self.patch_size,
                              left: left + self.patch_size]
                yb_patch = yb[top: top + self.patch_size,
                              left: left + self.patch_size]
                if np.isfinite(ya_patch).mean() < 0.95 or np.isfinite(yb_patch).mean() < 0.95:
                    continue
            self.items.append((top, left))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        top, left = self.items[idx]
        xa = self.store.inputs[self.year_a][:, top: top +
                                            self.patch_size, left: left + self.patch_size]
        xb = self.store.inputs[self.year_b][:, top: top +
                                            self.patch_size, left: left + self.patch_size]
        return {
            "x1": torch.from_numpy(xa).float(),
            "x2": torch.from_numpy(xb).float(),
            "top": torch.tensor(top, dtype=torch.long),
            "left": torch.tensor(left, dtype=torch.long),
        }


class MultiPairConsistencyDataset(Dataset):
    def __init__(self, store: MultiYearRasterStore, year_pairs: Sequence[Tuple[int, int]], split: str,
                 patch_size: int = 128, stride: int = 128, block_size: int = 256,
                 seed: int = 42):
        self.store = store
        self.year_pairs = list(year_pairs)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.block_size = block_size
        self.seed = seed
        self.items: List[Tuple[int, int, int, int]] = []
        self._build_index()

    def _build_index(self):
        for year_a, year_b in self.year_pairs:
            ha, wa = self.store.shape(year_a)
            hb, wb = self.store.shape(year_b)
            ma = self.store.meta[year_a]
            mb = self.store.meta[year_b]
            if (ha, wa) != (hb, wb) or ma["transform"] != mb["transform"] or ma["crs"] != mb["crs"]:
                raise ValueError(
                    "Temporal consistency requires all years to be aligned to the same reference grid")
            h, w = ha, wa
            split_mask = get_split_mask(
                h, w, self.split, self.block_size, seed=self.seed)
            for top, left in sliding_windows(h, w, self.patch_size, self.stride):
                patch_mask = split_mask[top: top +
                                        self.patch_size, left: left + self.patch_size]
                if patch_mask.mean() < 0.95:
                    continue
                xa = self.store.inputs[year_a][:, top: top +
                                               self.patch_size, left: left + self.patch_size]
                xb = self.store.inputs[year_b][:, top: top +
                                               self.patch_size, left: left + self.patch_size]
                if not (np.isfinite(xa).all() and np.isfinite(xb).all()):
                    continue
                self.items.append((year_a, year_b, top, left))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        year_a, year_b, top, left = self.items[idx]
        xa = self.store.inputs[year_a][:, top: top +
                                       self.patch_size, left: left + self.patch_size]
        xb = self.store.inputs[year_b][:, top: top +
                                       self.patch_size, left: left + self.patch_size]
        return {
            "x1": torch.from_numpy(xa).float(),
            "x2": torch.from_numpy(xb).float(),
            "year1": torch.tensor(year_a, dtype=torch.long),
            "year2": torch.tensor(year_b, dtype=torch.long),
            "top": torch.tensor(top, dtype=torch.long),
            "left": torch.tensor(left, dtype=torch.long),
        }
