from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from common import canopy_metrics, reproject_to_match


def read_arr(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def plot_bias_by_bin(y_true, y_pred, ref, outdir: Path):
    bins = np.array([0, 10, 30, 60, 100], dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])
    model_bias, ref_bias = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_true >= lo) & (y_true < hi)
        model_bias.append(float(np.mean(y_pred[m] - y_true[m])) if m.any() else np.nan)
        ref_bias.append(float(np.mean(ref[m] - y_true[m])) if m.any() else np.nan)
    plt.figure(figsize=(7, 4))
    plt.plot(centers, model_bias, marker="o", label="Model")
    plt.plot(centers, ref_bias, marker="o", label="NLCD")
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Reference canopy-cover bin (%)")
    plt.ylabel("Bias (prediction - reference)")
    plt.title("Bias by canopy-cover bin")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bias_by_bin.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--target-path", type=str, required=True)
    parser.add_argument("--reference-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pred, pred_profile = read_arr(args.pred_path)
    target_path = Path(args.target_path)
    ref_path = Path(args.reference_path)

    aligned_target = outdir / target_path.name
    aligned_ref = outdir / ref_path.name
    if not aligned_target.exists():
        reproject_to_match(target_path, args.pred_path, aligned_target)
    if not aligned_ref.exists():
        reproject_to_match(ref_path, args.pred_path, aligned_ref)

    target, _ = read_arr(aligned_target)
    ref, _ = read_arr(aligned_ref)

    mask = np.isfinite(pred) & np.isfinite(target) & np.isfinite(ref)
    y_true = target[mask]
    y_pred = pred[mask]
    y_ref = ref[mask]

    metrics = {
        "model": canopy_metrics(y_true, y_pred),
        "nlcd": canopy_metrics(y_true, y_ref),
    }
    print(metrics)

    with open(outdir / "metrics.txt", "w") as f:
        f.write(str(metrics))

    plot_bias_by_bin(y_true, y_pred, y_ref, outdir)

    resid = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=50)
    plt.xlabel("Residual (prediction - reference)")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(outdir / "residual_hist.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
