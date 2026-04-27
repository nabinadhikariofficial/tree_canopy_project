from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

PRISM_KEY = "tmean"
LANDCOVER_KEY = "landcover"
CHANNEL_ORDER = ["nlcd", LANDCOVER_KEY, "ndvi", "lst", PRISM_KEY]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_torch_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_pin_memory(device) -> bool:
    return getattr(device, "type", str(device)) == "cuda"


@dataclass
class YearPaths:
    year: int
    nlcd: str
    landcover: str
    ndvi: str
    lst: str
    tmean: str
    ppt: Optional[str] = None
    tmin: Optional[str] = None
    tmax: Optional[str] = None
    label: Optional[str] = None


def _first_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def discover_year_paths(data_root: str | Path, year: int) -> YearPaths:
    root = Path(data_root) / str(year)
    if not root.exists():
        raise FileNotFoundError(f"Year folder missing: {root}")

    prism_root = root / "Annual Prism Data"
    nlcd = _first_existing([
        root / f"{year}_Tree_NLCD.tif",
        root / f"{year}_tree_nlcd.tif",
    ])
    landcover = _first_existing([
        root / f"{year}_NLCD_LandCover.tif",
        root / f"{year}_nlcd_landcover.tif",
        root / f"{year}_NLCD_land_cover.tif",
    ])
    ndvi = _first_existing([root / f"NDVI_{year}_Albers_clip_resampled.tif"])
    lst = _first_existing([root / f"LST_{year}_Albers_clip_resampled.tif"])
    ppt = _first_existing(
        [prism_root / f"prism_ppt_us_30s_{year}_clip_resampled.tif"])
    tmin = _first_existing(
        [prism_root / f"prism_tmin_us_30s_{year}_clip_resampled.tif"])
    tmean = _first_existing(
        [prism_root / f"prism_tmean_us_30s_{year}_clip_resampled.tif"])
    tmax = _first_existing(
        [prism_root / f"prism_tmax_us_30s_{year}_clip_resampled.tif"])
    label = _first_existing([
        root / f"Tree_{year}.tif",
        root / f"tree_{year}.tif",
    ])

    required = {
        "nlcd": nlcd,
        "landcover": landcover,
        "ndvi": ndvi,
        "lst": lst,
        "tmean": tmean,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise FileNotFoundError(f"Year {year} missing files: {missing}")

    return YearPaths(
        year=year,
        nlcd=str(nlcd),
        landcover=str(landcover),
        ndvi=str(ndvi),
        lst=str(lst),
        tmean=str(tmean),
        ppt=str(ppt) if ppt else None,
        tmin=str(tmin) if tmin else None,
        tmax=str(tmax) if tmax else None,
        label=str(label) if label else None,
    )


def resolve_reference_path(data_root: str | Path, reference_year: int) -> str:
    return discover_year_paths(data_root, reference_year).nlcd


def read_profile(path: str | Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": tuple(src.transform),
            "shape": (src.height, src.width),
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "count": src.count,
            "bounds": tuple(src.bounds),
            "res": tuple(src.res),
        }


def read_raster(path: str | Path, out_dtype=np.float32) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None and np.isfinite(nodata):
            arr[arr == nodata] = np.nan
        arr[np.abs(arr) > 1e20] = np.nan
        arr = arr.astype(out_dtype)
        profile = src.profile.copy()
        return arr, profile


def reproject_to_match(src_path: str | Path, ref_path: str | Path, dst_path: str | Path,
                       resampling: Resampling = Resampling.bilinear) -> str:
    dst_path = str(dst_path)
    with rasterio.open(ref_path) as ref, rasterio.open(src_path) as src:
        dst_profile = ref.profile.copy()
        dst_profile.update(count=1, dtype="float32", compress="lzw")
        dst_arr = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        os.makedirs(Path(dst_path).parent, exist_ok=True)
        with rasterio.open(dst_path, "w", **dst_profile) as dst:
            dst.write(dst_arr, 1)
    return dst_path


def stack_inputs(paths: YearPaths, label_align_dir: Optional[str] = None,
                 ref_path: Optional[str | Path] = None) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    arrays = []
    meta = None

    ref_path = str(ref_path or paths.nlcd)

    for key in CHANNEL_ORDER:
        path = getattr(paths, key)

        if not raster_matches(path, ref_path):
            align_dir = Path(label_align_dir or Path(
                path).parent / "_aligned_inputs")
            align_dir.mkdir(parents=True, exist_ok=True)

            aligned = align_dir / Path(path).name

            if not aligned.exists():
                reproject_to_match(path, ref_path, aligned,
                                   resampling=Resampling.bilinear)

            path = str(aligned)

        arr, profile = read_raster(path)
        arrays.append(arr)

        if meta is None:
            meta = profile

    x = np.stack(arrays, axis=0)

    y = None
    if paths.label:
        label_path = paths.label

        if not raster_matches(paths.label, ref_path):
            align_dir = Path(label_align_dir or Path(
                paths.label).parent / "_aligned_labels")
            align_dir.mkdir(parents=True, exist_ok=True)

            aligned = align_dir / Path(paths.label).name

            if not aligned.exists():
                print("[ALIGN LABEL] -> NLCD grid")
                reproject_to_match(paths.label, ref_path,
                                   aligned, resampling=Resampling.bilinear)

            label_path = str(aligned)

        y, _ = read_raster(label_path)

    return x, y, meta


def raster_matches(path_a: str | Path, path_b: str | Path) -> bool:
    pa = read_profile(path_a)
    pb = read_profile(path_b)
    return pa["shape"] == pb["shape"] and pa["crs"] == pb["crs"] and pa["transform"] == pb["transform"]


def compute_channel_stats(xs: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
    stats = {}
    stacked = np.concatenate([x.reshape(x.shape[0], -1) for x in xs], axis=1)
    for i, name in enumerate(CHANNEL_ORDER):
        vals = stacked[i]
        vals = vals[np.isfinite(vals)]
        stats[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std() + 1e-6),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return stats


def normalize_channels(x: np.ndarray, stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    for i, name in enumerate(CHANNEL_ORDER):
        x[i] = (x[i] - stats[name]["mean"]) / stats[name]["std"]
        invalid = ~np.isfinite(x[i])
        if np.any(invalid):
            # After normalization, zero corresponds to the training-set mean.
            x[i][invalid] = 0.0
    return x


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def canopy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask].astype(np.float64)
    yp = y_pred[mask].astype(np.float64)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    bias = float(np.mean(yp - yt))
    denom = np.sum((yt - np.mean(yt)) ** 2)
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / (denom + 1e-12))
    return {"mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def block_ids(height: int, width: int, block_size: int) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    by = yy // block_size
    bx = xx // block_size
    return by * int(math.ceil(width / block_size)) + bx


def spatial_block_split(height: int, width: int, block_size: int,
                        train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    blocks = np.unique(block_ids(height, width, block_size))
    rng = np.random.default_rng(seed)
    blocks = rng.permutation(blocks)
    n = len(blocks)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_blocks = set(blocks[:n_train].tolist())
    val_blocks = set(blocks[n_train:n_train + n_val].tolist())
    test_blocks = set(blocks[n_train + n_val:].tolist())
    bid = block_ids(height, width, block_size)
    train_mask = np.isin(bid, list(train_blocks))
    val_mask = np.isin(bid, list(val_blocks))
    test_mask = np.isin(bid, list(test_blocks))
    return train_mask, val_mask, test_mask


def get_split_mask(height: int, width: int, split: str, block_size: int,
                   seed: int = 42) -> np.ndarray:
    train_mask, val_mask, test_mask = spatial_block_split(
        height, width, block_size, seed=seed)
    return {"train": train_mask, "val": val_mask, "test": test_mask}[split]


def sliding_windows(height: int, width: int, patch: int, stride: int) -> Iterator[Tuple[int, int]]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive integers")
    if patch > height or patch > width:
        raise ValueError(
            f"Patch size {patch} exceeds raster dimensions {(height, width)}")

    ys = list(range(0, height - patch + 1, stride))
    xs = list(range(0, width - patch + 1, stride))
    if ys[-1] != height - patch:
        ys.append(height - patch)
    if xs[-1] != width - patch:
        xs.append(width - patch)

    for y in ys:
        for x in xs:
            yield y, x
