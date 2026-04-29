from __future__ import annotations

import ast
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colors, patches

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import (  # noqa: E402
    CHANNEL_ORDER,
    discover_year_paths,
    resolve_reference_path,
    stack_inputs,
)


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
ALIGN_DIR = RESULTS_DIR / "_aligned"

RUN_OLD = ROOT / "runs" / "runs 04_24"
RUN_MID = ROOT / "runs" / "runs 04_27"
RUN_4CH = ROOT / "runs" / "20260427_134448"
RUN_5CH = ROOT / "runs" / "20260427_143154"
RUN_LATEST = ROOT / "runs" / "20260427_152647"

SUPERVISED_YEARS = {2015, 2021}
TRAIN_YEARS = list(range(2015, 2023))
TEST_YEAR = 2023
REFERENCE_YEAR = 2015


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, FIGURES_DIR, TABLES_DIR, ALIGN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_eval_metrics(model_dir: Path) -> dict | None:
    path = model_dir / "eval_2023" / "metrics.txt"
    if not path.exists():
        return None
    return ast.literal_eval(path.read_text())


def load_model_record(run_name: str, model_dir: Path) -> dict:
    record = {"run": run_name, "model": model_dir.name}
    test_path = model_dir / "test_metrics.json"
    rf_path = model_dir / "metrics.json"
    if test_path.exists():
        record.update({f"test_{k}": v for k, v in load_json(test_path).items() if k in {"rmse", "mae", "r2", "bias"}})
    if rf_path.exists():
        rf_data = load_json(rf_path)
        record.update({f"test_{k}": v for k, v in rf_data.items() if k in {"rmse", "mae", "r2", "bias"}})
        record["feature_names"] = rf_data.get("feature_names")
    eval_data = load_eval_metrics(model_dir)
    if eval_data is not None:
        for prefix in ["model", "nlcd"]:
            if prefix in eval_data:
                for key, value in eval_data[prefix].items():
                    record[f"{prefix}_{key}"] = value
                    if prefix == "model":
                        record[f"eval_{key}"] = value
    ds_path = model_dir / "dataset_summary.json"
    if ds_path.exists():
        record["dataset_summary"] = load_json(ds_path)
    return record


def load_run_records(run_dir: Path) -> list[dict]:
    records = []
    for model_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        rec = load_model_record(run_dir.name, model_dir)
        if len(rec) > 2:
            records.append(rec)
    return records


def model_label(model_name: str) -> str:
    mapping = {
        "rf_2023": "RF",
        "unet_2023": "U-Net",
        "unet_consistency_2023": "U-Net + Consistency",
        "prithvi_100m_2023": "Prithvi 100M",
        "prithvi_100m_consistency_2023": "Prithvi 100M + Consistency",
        "prithvi_consistency_2023": "Prithvi 100M + Consistency",
        "prithvi_300m_2023": "Prithvi 300M",
        "prithvi_300m_consistency_2023": "Prithvi 300M + Consistency",
        "prithvi_tiny_2023": "Prithvi Tiny",
    }
    return mapping.get(model_name, model_name)


def write_csv_and_markdown(stem: str, headers: list[str], rows: list[list[object]]) -> None:
    csv_path = TABLES_DIR / f"{stem}.csv"
    md_path = TABLES_DIR / f"{stem}.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    def fmt(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    with md_path.open("w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(v) for v in row) + " |\n")


def robust_limits(arr: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(valid, lower))
    hi = float(np.percentile(valid, upper))
    if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi


def select_informative_crop(target: np.ndarray, size: int = 320, step: int = 64) -> tuple[int, int, int]:
    h, w = target.shape
    best_score = -np.inf
    best = (max(0, h // 2 - size // 2), max(0, w // 2 - size // 2), size)
    for top in range(0, max(1, h - size + 1), step):
        for left in range(0, max(1, w - size + 1), step):
            window = target[top: top + size, left: left + size]
            valid = np.isfinite(window)
            valid_frac = float(valid.mean())
            if valid_frac < 0.98:
                continue
            valid_vals = window[valid]
            score = float(np.std(valid_vals) + 0.15 * np.mean(valid_vals))
            if score > best_score:
                best_score = score
                best = (top, left, size)
    return best


def crop(arr: np.ndarray, top: int, left: int, size: int) -> np.ndarray:
    return arr[top: top + size, left: left + size]


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_prediction(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def load_aligned_2023_data() -> tuple[np.ndarray, np.ndarray]:
    year_paths = discover_year_paths(ROOT / "data", TEST_YEAR)
    reference_path = resolve_reference_path(ROOT / "data", REFERENCE_YEAR)
    x, y, _ = stack_inputs(year_paths, label_align_dir=str(ALIGN_DIR), ref_path=reference_path)
    if y is None:
        raise RuntimeError("Aligned 2023 target label could not be loaded")
    return x, y


def create_workflow_figure() -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_axis_off()

    boxes = [
        ((0.04, 0.58), 0.18, 0.22, "1. Data Preparation\n\nDiscover yearly rasters\nAlign labels to NLCD grid\nCompute channel statistics"),
        ((0.28, 0.58), 0.18, 0.22, "2. Supervised Training\n\nRF baseline\nU-Net baseline\nPrithvi baselines"),
        ((0.52, 0.58), 0.18, 0.22, "3. Temporal Extension\n\nYear-pair consistency\n2015-2022\nSemi-supervised loss"),
        ((0.76, 0.58), 0.18, 0.22, "4. Full-Raster Inference\n\nReflect padding\nWeighted overlap blending\nTTA + calibration"),
        ((0.28, 0.16), 0.18, 0.20, "5. Evaluation\n\nPatch-level test metrics\nFull-raster metrics\nBias / RMSE / R2"),
        ((0.52, 0.16), 0.18, 0.20, "6. Final Deliverables\n\nBest valid model\nPrediction maps\nReport figures and tables"),
    ]

    face_colors = ["#e8f1ff", "#eaf7ea", "#fff4dd", "#fbe8e7", "#eee8ff", "#eef5f5"]
    for (xy, w, h, text), fc in zip(boxes, face_colors):
        rect = patches.FancyBboxPatch(
            xy,
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.6,
            edgecolor="#2f3b52",
            facecolor=fc,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )

    arrow_kw = dict(arrowstyle="->", linewidth=2.0, color="#34495e")
    ax.annotate("", xy=(0.28, 0.69), xytext=(0.22, 0.69), xycoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=(0.52, 0.69), xytext=(0.46, 0.69), xycoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=(0.76, 0.69), xytext=(0.70, 0.69), xycoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=(0.37, 0.36), xytext=(0.37, 0.58), xycoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=(0.61, 0.36), xytext=(0.61, 0.58), xycoords="axes fraction", arrowprops=arrow_kw)
    ax.annotate("", xy=(0.52, 0.26), xytext=(0.46, 0.26), xycoords="axes fraction", arrowprops=arrow_kw)

    ax.set_title("End-to-End Tree Canopy Prediction Workflow", fontsize=16, pad=18)
    save_figure(fig, "figure_01_workflow_diagram.png")


def create_input_stack_figure(x: np.ndarray, crop_spec: tuple[int, int, int]) -> None:
    top, left, size = crop_spec
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()

    titles = ["NLCD Canopy", "NLCD Land Cover", "NDVI", "LST", "PRISM tmean"]
    for idx, ax in enumerate(axes[:5]):
        panel = crop(x[idx], top, left, size)
        if CHANNEL_ORDER[idx] == "landcover":
            values = np.unique(panel[np.isfinite(panel)]).astype(int)
            if values.size == 0:
                values = np.array([0], dtype=int)
            display = np.full(panel.shape, np.nan, dtype=np.float32)
            for mapped_idx, raw_value in enumerate(values):
                display[np.isclose(panel, raw_value)] = mapped_idx
            cmap = plt.get_cmap("tab20", max(len(values), 1))
            norm = colors.BoundaryNorm(np.arange(-0.5, len(values) + 0.5, 1.0), cmap.N)
            im = ax.imshow(display, cmap=cmap, norm=norm)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_ticks(np.arange(len(values)))
            cbar.set_ticklabels([str(v) for v in values])
            cbar.ax.tick_params(labelsize=8)
        else:
            vmin, vmax = robust_limits(panel)
            im = ax.imshow(panel, cmap="viridis", vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(titles[idx], fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.9,
        "Sample crop metadata\n\n"
        f"Year: {TEST_YEAR}\n"
        f"Reference year: {REFERENCE_YEAR}\n"
        f"Crop size: {size} x {size}\n"
        f"Top-left pixel: ({top}, {left})\n\n"
        "The same crop is reused in the\nqualitative comparison figure.",
        fontsize=12,
        va="top",
    )

    fig.suptitle("Sample 2023 Input Stack Crop", fontsize=16, y=0.98)
    save_figure(fig, "figure_02_input_stack_sample.png")


def create_map_comparison_figure(x: np.ndarray, y: np.ndarray, crop_spec: tuple[int, int, int]) -> None:
    top, left, size = crop_spec
    latest_records = {rec["model"]: rec for rec in load_run_records(RUN_LATEST)}
    prithvi_candidates = [m for m in latest_records if m.startswith("prithvi_") and "eval_rmse" in latest_records[m]]
    best_prithvi = min(prithvi_candidates, key=lambda m: latest_records[m]["eval_rmse"])

    panels = [
        ("Target Canopy", y),
        ("NLCD Baseline", x[0]),
        ("RF Prediction", load_prediction(RUN_LATEST / "rf_2023" / "pred_2023.tif")),
        ("Final U-Net", load_prediction(RUN_LATEST / "unet_2023" / "pred_2023.tif")),
        (f"{model_label(best_prithvi)}", load_prediction(RUN_LATEST / best_prithvi / "pred_2023.tif")),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(20, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("YlGn")
    cmap = cmap.copy()
    cmap.set_bad(color="black")

    ims = []
    for ax, (title, arr) in zip(axes, panels):
        panel = crop(arr, top, left, size)
        im = ax.imshow(panel, cmap=cmap, vmin=0, vmax=100)
        ims.append(im)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(ims[0], ax=axes, location="right", fraction=0.028, pad=0.02)
    cbar.set_label("Canopy cover (%)")
    fig.suptitle("Qualitative 2023 Map Comparison on a Representative Crop", fontsize=16, y=1.02)
    save_figure(fig, "figure_03_map_comparison_2023.png")


def create_edge_artifact_figure(y: np.ndarray) -> None:
    old_pred = load_prediction(RUN_OLD / "unet_2023" / "pred_2023.tif")
    fixed_pred = load_prediction(RUN_LATEST / "unet_2023" / "pred_2023.tif")
    size = 256
    panels = [
        ("Old U-Net\n(04_24 run)", crop(old_pred, 0, 0, size)),
        ("Corrected U-Net\n(final valid run)", crop(fixed_pred, 0, 0, size)),
        ("Aligned 2023\nTarget", crop(y, 0, 0, size)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8))
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad(color="black")
    for ax, (title, panel) in zip(axes, panels):
        im = ax.imshow(panel, cmap=cmap, vmin=0, vmax=100)
        ax.set_xlabel(title, fontsize=10, labelpad=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(top=0.78, bottom=0.16, left=0.04, right=0.92, wspace=0.12)
    fig.colorbar(im, ax=axes, location="right", fraction=0.04, pad=0.02, label="Canopy cover (%)")
    fig.suptitle("Top-Left Border Artifact Before and After the Inference Fix", fontsize=14, y=0.95)
    save_figure(fig, "figure_04_edge_artifact_fix.png")


def create_metrics_bar_chart() -> None:
    latest = {rec["model"]: rec for rec in load_run_records(RUN_LATEST)}
    order = [
        "rf_2023",
        "unet_2023",
        "unet_consistency_2023",
        "prithvi_100m_2023",
        "prithvi_100m_consistency_2023",
        "prithvi_300m_2023",
        "prithvi_300m_consistency_2023",
    ]
    labels = ["NLCD"] + [model_label(name) for name in order]
    rmse_vals = [latest["unet_2023"]["nlcd_rmse"]] + [latest[name]["eval_rmse"] for name in order]
    r2_vals = [latest["unet_2023"]["nlcd_r2"]] + [latest[name]["eval_r2"] for name in order]
    colors_list = ["#7f8c8d", "#95a5a6", "#2e86de", "#5dade2", "#c0392b", "#d35400", "#e67e22", "#f1948a"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    x = np.arange(len(labels))
    axes[0].bar(x, rmse_vals, color=colors_list)
    axes[0].set_title("Full-Raster RMSE")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    for i, val in enumerate(rmse_vals):
        axes[0].text(i, val + 0.3, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    axes[1].bar(x, r2_vals, color=colors_list)
    axes[1].set_title("Full-Raster R2")
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    for i, val in enumerate(r2_vals):
        axes[1].text(i, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Metric Comparison Across Model Families (`20260427_152647`)", fontsize=16, y=1.02)
    save_figure(fig, "figure_05_metrics_bar_chart.png")


def create_scatter_figure(y: np.ndarray) -> None:
    pred = load_prediction(RUN_LATEST / "unet_2023" / "pred_2023.tif")
    valid = np.isfinite(y) & np.isfinite(pred)
    y_true = y[valid]
    y_pred = pred[valid]
    rng = np.random.default_rng(42)
    sample_size = min(50000, y_true.size)
    idx = rng.choice(y_true.size, size=sample_size, replace=False)
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    eval_metrics = load_eval_metrics(RUN_LATEST / "unet_2023")["model"]
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.scatter(y_true, y_pred, s=6, alpha=0.15, color="#2166ac", edgecolors="none")
    ax.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Target canopy (%)")
    ax.set_ylabel("Predicted canopy (%)")
    ax.set_title("Final U-Net Predicted vs. Target Canopy")
    annotation = (
        f"Sampled pixels: {sample_size:,}\n"
        f"RMSE: {eval_metrics['rmse']:.3f}\n"
        f"MAE: {eval_metrics['mae']:.3f}\n"
        f"R2: {eval_metrics['r2']:.3f}\n"
        f"Bias: {eval_metrics['bias']:.3f}"
    )
    ax.text(
        0.05,
        0.95,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#888"),
    )
    save_figure(fig, "figure_06_final_unet_scatter.png")


def create_learning_evolution_figure() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    years = list(range(2015, 2024))
    y0 = 0.0
    box_h = 0.72

    role_specs = {
        "reference + supervised": {"fc": "#2e86de", "ec": "#1b4f72", "text": "Reference +\nSupervised"},
        "supervised": {"fc": "#5dade2", "ec": "#1b4f72", "text": "Supervised"},
        "temporal-only input": {"fc": "#f8c471", "ec": "#935116", "text": "Temporal-only\ninput"},
        "held-out test": {"fc": "#ec7063", "ec": "#7b241c", "text": "Held-out\n2023 test"},
    }

    for idx, year in enumerate(years):
        if year == TEST_YEAR:
            role = "held-out test"
        elif year == REFERENCE_YEAR and year in SUPERVISED_YEARS:
            role = "reference + supervised"
        elif year in SUPERVISED_YEARS:
            role = "supervised"
        else:
            role = "temporal-only input"
        spec = role_specs[role]
        rect = patches.FancyBboxPatch(
            (idx - 0.42, y0 - box_h / 2),
            0.84,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=spec["fc"],
            edgecolor=spec["ec"],
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(idx, y0, f"{year}\n{spec['text']}", ha="center", va="center", fontsize=10)

    for idx in range(len(years) - 2):
        start_x = idx + 0.42
        end_x = idx + 1 - 0.42
        ax.annotate(
            "",
            xy=(end_x, y0 + 0.18),
            xytext=(start_x, y0 + 0.18),
            arrowprops=dict(arrowstyle="->", linewidth=1.8, color="#7d6608"),
        )

    ax.annotate(
        "",
        xy=(8, y0 + 0.55),
        xytext=(0, y0 + 0.55),
        arrowprops=dict(arrowstyle="->", linewidth=2.2, color="#117a65"),
    )
    ax.text(
        4.0,
        y0 + 0.72,
        "Model context expands through annual inputs from 2015 to 2022,\nwhile direct canopy supervision is available only in 2015 and 2021.",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#0b5345",
    )

    legend_handles = [
        patches.Patch(facecolor=spec["fc"], edgecolor=spec["ec"], label=role.title())
        for role, spec in role_specs.items()
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, frameon=False)
    ax.set_xlim(-0.7, len(years) - 0.3)
    ax.set_ylim(-0.95, 1.15)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=10)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Learning Evolution Across Years: Supervision, Temporal Context, and Held-Out Testing", fontsize=16, pad=16)
    save_figure(fig, "figure_07_learning_evolution_2015_2023.png")


def create_run_progression_figure() -> None:
    run_order = ["runs 04_24", "runs 04_27", "20260427_134448", "20260427_143154", "20260427_152647"]
    run_dirs = {
        "runs 04_24": RUN_OLD,
        "runs 04_27": RUN_MID,
        "20260427_134448": RUN_4CH,
        "20260427_143154": RUN_5CH,
        "20260427_152647": RUN_LATEST,
    }
    labels = ["Initial", "Post-fix", "Dense 4-ch", "Add landcover", "Latest full"]

    best_eval_rmse = []
    best_eval_r2 = []
    nlcd_rmse = []
    nlcd_r2 = []
    best_model_labels = []

    for run_name in run_order:
        records = load_run_records(run_dirs[run_name])
        deep_records = [rec for rec in records if rec["model"] != "rf_2023" and "eval_rmse" in rec]
        best = min(deep_records, key=lambda rec: rec["eval_rmse"])
        best_eval_rmse.append(best["eval_rmse"])
        best_eval_r2.append(best["eval_r2"])
        nlcd_rmse.append(best["nlcd_rmse"])
        nlcd_r2.append(best["nlcd_r2"])
        best_model_labels.append(model_label(best["model"]))

    x = np.arange(len(run_order))
    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.4))

    axes[0].plot(x, best_eval_rmse, marker="o", linewidth=2.5, color="#1f77b4", label="Best model in run")
    axes[0].plot(x, nlcd_rmse, marker="s", linewidth=2.0, linestyle="--", color="#7f8c8d", label="NLCD baseline")
    axes[0].set_title("Best Full-Raster RMSE by Run", fontsize=11, pad=8)
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(alpha=0.25, axis="y")
    for i, val in enumerate(best_eval_rmse):
        axes[0].text(i, val + 0.28, f"{val:.2f}\n{best_model_labels[i]}", ha="center", va="bottom", fontsize=8)

    axes[1].plot(x, best_eval_r2, marker="o", linewidth=2.5, color="#2ca02c", label="Best model in run")
    axes[1].plot(x, nlcd_r2, marker="s", linewidth=2.0, linestyle="--", color="#7f8c8d", label="NLCD baseline")
    axes[1].set_title("Best Full-Raster R2 by Run", fontsize=11, pad=8)
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].grid(alpha=0.25, axis="y")
    for i, val in enumerate(best_eval_r2):
        axes[1].text(i, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.06, right=0.98, wspace=0.20)
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Performance Evolution Across Major Run Generations", fontsize=14, y=0.95)
    save_figure(fig, "figure_08_run_progression.png")


def create_yearly_canopy_evolution_figure(crop_spec: tuple[int, int, int]) -> None:
    top, left, size = crop_spec
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.0))
    axes = axes.ravel()
    cmap = plt.get_cmap("YlGn").copy()
    cmap.set_bad(color="black")

    for ax, year in zip(axes, range(2015, 2024)):
        yp = discover_year_paths(ROOT / "data", year)
        x_year, _, _ = stack_inputs(yp, label_align_dir=str(ALIGN_DIR), ref_path=resolve_reference_path(ROOT / "data", REFERENCE_YEAR))
        panel = crop(x_year[0], top, left, size)
        im = ax.imshow(panel, cmap=cmap, vmin=0, vmax=100)
        if year == TEST_YEAR:
            role = "held-out test"
            color = "#c0392b"
        elif year in SUPERVISED_YEARS:
            role = "supervised"
            color = "#1f618d"
        else:
            role = "temporal context"
            color = "#9c640c"
        ax.set_title(f"{year}", fontsize=12, pad=6, color=color)
        ax.set_xlabel(role, fontsize=9, labelpad=6, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.04, right=0.92, hspace=0.22, wspace=0.08)
    fig.colorbar(im, ax=axes, location="right", fraction=0.025, pad=0.02, label="NLCD canopy (%)")
    fig.suptitle("Year-by-Year Canopy Context from 2015 to 2023 on the Same Spatial Crop", fontsize=15, y=0.97)
    save_figure(fig, "figure_09_yearly_canopy_evolution_2015_2023.png")


def create_training_history_figure() -> None:
    unet_hist = load_json(RUN_LATEST / "unet_2023" / "history.json")["history"]
    cons_hist = load_json(RUN_LATEST / "unet_consistency_2023" / "history.json")["history"]

    epochs = [row["epoch"] for row in unet_hist]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    axes[0].plot(epochs, [row["train_loss"] for row in unet_hist], label="U-Net train loss", color="#1f77b4", linewidth=2.2)
    axes[0].plot(epochs, [row["train_loss"] for row in cons_hist], label="U-Net + consistency train loss", color="#ff7f0e", linewidth=2.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Optimization Progress")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [row["rmse"] for row in unet_hist], label="U-Net validation RMSE", color="#1f77b4", linewidth=2.2)
    axes[1].plot(epochs, [row["rmse"] for row in cons_hist], label="U-Net + consistency validation RMSE", color="#ff7f0e", linewidth=2.2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation RMSE")
    axes[1].set_title("Validation Performance Over Epochs")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.07, right=0.98, wspace=0.18)
    fig.suptitle("Training Progress of the Final U-Net Models", fontsize=15, y=0.96)
    save_figure(fig, "figure_10_training_history_final_unet.png")


def create_dataset_table() -> None:
    headers = ["year", "nlcd", "landcover", "ndvi", "lst", "tmean", "label_available", "role"]
    rows: list[list[object]] = []
    for year in range(2015, 2024):
        yp = discover_year_paths(ROOT / "data", year)
        if year == TEST_YEAR:
            role = "held-out test"
        elif year == REFERENCE_YEAR and year in SUPERVISED_YEARS:
            role = "reference + supervised"
        elif year in SUPERVISED_YEARS:
            role = "supervised"
        else:
            role = "temporal-only input"
        rows.append([
            year,
            Path(yp.nlcd).name,
            Path(yp.landcover).name,
            Path(yp.ndvi).name,
            Path(yp.lst).name,
            Path(yp.tmean).name,
            "yes" if yp.label else "no",
            role,
        ])
    write_csv_and_markdown("table_01_dataset_year_availability", headers, rows)


def create_model_configuration_table() -> None:
    headers = [
        "model",
        "family",
        "features",
        "pretraining",
        "consistency_weight",
        "consistency_years",
        "train_stride",
        "eval_stride",
        "train_patches",
        "augmentation",
        "tta_inference",
        "calibration",
    ]
    latest_records = {rec["model"]: rec for rec in load_run_records(RUN_LATEST)}
    rows = []
    specs = [
        ("rf_2023", "Random Forest", "none", "n/a"),
        ("unet_2023", "U-Net", "none", "2015, 2021"),
        ("unet_consistency_2023", "U-Net", "none", "2015-2022"),
        ("prithvi_100m_2023", "Prithvi Geo-ViT", "Prithvi 100M", "2015-2022"),
        ("prithvi_100m_consistency_2023", "Prithvi Geo-ViT", "Prithvi 100M", "2015-2022"),
        ("prithvi_300m_2023", "Prithvi Geo-ViT", "Prithvi 300M", "2015-2022"),
        ("prithvi_300m_consistency_2023", "Prithvi Geo-ViT", "Prithvi 300M", "2015-2022"),
    ]
    for model_name, family, pretraining, consistency_years in specs:
        rec = latest_records[model_name]
        summary = rec.get("dataset_summary", {})
        if model_name == "rf_2023":
            rows.append([
                model_label(model_name),
                family,
                ", ".join(rec.get("feature_names", CHANNEL_ORDER)),
                pretraining,
                0.0,
                "n/a",
                "n/a",
                "n/a",
                "sample_fraction=0.05",
                "n/a",
                "no",
                "no",
            ])
        else:
            consistency_weight = 0.2 if "consistency" in model_name else 0.0
            rows.append([
                model_label(model_name),
                family,
                ", ".join(CHANNEL_ORDER),
                pretraining,
                consistency_weight,
                consistency_years,
                summary.get("train_stride"),
                summary.get("eval_stride"),
                summary.get("num_train_patches"),
                summary.get("augment"),
                "yes",
                "yes",
            ])
    write_csv_and_markdown("table_02_model_configuration_summary", headers, rows)


def create_historical_progression_table() -> None:
    headers = [
        "run",
        "channels",
        "train_patches",
        "val_patches",
        "test_patches",
        "best_model",
        "best_eval_rmse",
        "best_eval_r2",
        "artifact_fixed",
        "notable_change",
    ]
    run_info = {
        "runs 04_24": (RUN_OLD, 4, "legacy 4-channel baseline; raw best metrics but invalid border coverage"),
        "runs 04_27": (RUN_MID, 4, "post-fix rerun; artifact removed but quality regressed"),
        "20260427_134448": (RUN_4CH, 4, "dense patches, augmentation, calibration"),
        "20260427_143154": (RUN_5CH, 5, "added NLCD land cover channel"),
        "20260427_152647": (RUN_LATEST, 5, "latest full comparison including Prithvi 100M and 300M"),
    }
    rows = []
    for run_name, (run_dir, channels, note) in run_info.items():
        records = load_run_records(run_dir)
        best = min(
            (rec for rec in records if rec["model"] != "rf_2023" and "eval_rmse" in rec),
            key=lambda rec: rec["eval_rmse"],
        )
        ds = next(rec for rec in records if rec["model"] == "unet_2023").get("dataset_summary", {})
        rows.append([
            run_name,
            channels,
            ds.get("num_train_patches"),
            ds.get("num_val_patches"),
            ds.get("num_test_patches"),
            model_label(best["model"]),
            best["eval_rmse"],
            best["eval_r2"],
            "yes" if run_name != "runs 04_24" else "no",
            note,
        ])
    write_csv_and_markdown("table_03_historical_run_progression", headers, rows)


def create_final_metrics_table() -> None:
    latest = {rec["model"]: rec for rec in load_run_records(RUN_LATEST)}
    headers = [
        "model",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_bias",
        "eval_rmse",
        "eval_mae",
        "eval_r2",
        "eval_bias",
    ]
    ordered_models = [
        "rf_2023",
        "unet_2023",
        "unet_consistency_2023",
        "prithvi_100m_2023",
        "prithvi_100m_consistency_2023",
        "prithvi_300m_2023",
        "prithvi_300m_consistency_2023",
    ]
    rows = []
    nlcd_row = [
        "NLCD baseline",
        None,
        None,
        None,
        None,
        latest["unet_2023"]["nlcd_rmse"],
        latest["unet_2023"]["nlcd_mae"],
        latest["unet_2023"]["nlcd_r2"],
        latest["unet_2023"]["nlcd_bias"],
    ]
    rows.append(nlcd_row)
    for model_name in ordered_models:
        rec = latest[model_name]
        rows.append([
            model_label(model_name),
            rec.get("test_rmse"),
            rec.get("test_mae"),
            rec.get("test_r2"),
            rec.get("test_bias"),
            rec.get("eval_rmse"),
            rec.get("eval_mae"),
            rec.get("eval_r2"),
            rec.get("eval_bias"),
        ])
    write_csv_and_markdown("table_04_final_consolidated_metrics_20260427_152647", headers, rows)


def create_landcover_ablation_table() -> None:
    base = {rec["model"]: rec for rec in load_run_records(RUN_4CH)}
    improved = {rec["model"]: rec for rec in load_run_records(RUN_5CH)}
    headers = [
        "model",
        "channels_before",
        "channels_after",
        "eval_rmse_before",
        "eval_rmse_after",
        "delta_eval_rmse",
        "eval_r2_before",
        "eval_r2_after",
        "delta_eval_r2",
    ]
    comparable = [
        "rf_2023",
        "unet_2023",
        "unet_consistency_2023",
        "prithvi_100m_2023",
        "prithvi_consistency_2023",
    ]
    rows = []
    for model_name in comparable:
        before = base[model_name]
        after = improved[model_name]
        rows.append([
            model_label(model_name),
            "nlcd, ndvi, lst, tmean",
            ", ".join(CHANNEL_ORDER),
            before["eval_rmse"],
            after["eval_rmse"],
            after["eval_rmse"] - before["eval_rmse"],
            before["eval_r2"],
            after["eval_r2"],
            after["eval_r2"] - before["eval_r2"],
        ])
    write_csv_and_markdown("table_05_landcover_ablation", headers, rows)


def create_limitations_table() -> None:
    headers = ["category", "current_limitation", "impact_on_results", "recommended_future_work"]
    rows = [
        [
            "Supervision",
            "Only a small number of years have canopy labels for direct supervision.",
            "Limits model capacity, especially for large transformers.",
            "Add more labeled years or use pseudo-labeling / semi-supervised expansion.",
        ],
        [
            "Spatial generalization",
            "Evaluation is limited to a held-out year within the same study domain.",
            "Does not prove transfer across regions or ecosystems.",
            "Run cross-region and cross-domain validation experiments.",
        ],
        [
            "Temporal prior",
            "Consistency loss assumes smooth year-to-year evolution except for weighted change areas.",
            "Can suppress real abrupt canopy changes.",
            "Use uncertainty-aware or change-detection-guided consistency losses.",
        ],
        [
            "Categorical encoding",
            "NLCD canopy and land cover are passed as numeric channels.",
            "May underuse semantic categorical structure.",
            "Try one-hot or embedding-based categorical encodings.",
        ],
        [
            "Transformer transfer",
            "Prithvi pretraining does not match the downstream canopy regression setup closely.",
            "Large pretrained models underperform despite higher capacity.",
            "Test alternate adapters, multi-scale heads, or domain-specific fine-tuning strategies.",
        ],
        [
            "Search breadth",
            "Hyperparameter exploration was meaningful but not exhaustive.",
            "Some performance gains may remain undiscovered.",
            "Run a focused sweep on the U-Net family only.",
        ],
        [
            "Evaluation scope",
            "Final claims depend heavily on 2023 as the held-out year.",
            "Robustness across years is not fully characterized.",
            "Add cross-year validation and uncertainty analysis.",
        ],
    ]
    write_csv_and_markdown("table_06_limitations_and_future_work", headers, rows)


def create_results_readme() -> None:
    lines = [
        "# Results Assets",
        "",
        "This folder contains the figures and tables referenced in the technical report draft.",
        "",
        "## Figures",
        "",
        "- `figures/figure_01_workflow_diagram.png`: end-to-end workflow diagram.",
        "- `figures/figure_02_input_stack_sample.png`: sample 2023 crop of the five input channels.",
        "- `figures/figure_03_map_comparison_2023.png`: qualitative 2023 target / baseline / model comparison.",
        "- `figures/figure_04_edge_artifact_fix.png`: old border artifact versus corrected inference output.",
        "- `figures/figure_05_metrics_bar_chart.png`: RMSE and R2 comparison across model families for `20260427_152647`.",
        "- `figures/figure_06_final_unet_scatter.png`: scatter plot of final U-Net predictions versus target canopy.",
        "- `figures/figure_07_learning_evolution_2015_2023.png`: yearly timeline showing supervised years, temporal-only context years, and the held-out 2023 test year.",
        "- `figures/figure_08_run_progression.png`: performance progression across the major run generations from `runs 04_24` to `20260427_152647`.",
        "- `figures/figure_09_yearly_canopy_evolution_2015_2023.png`: year-by-year NLCD canopy crop sequence from 2015 to 2023.",
        "- `figures/figure_10_training_history_final_unet.png`: training and validation progress across epochs for the final U-Net models.",
        "",
        "## Tables",
        "",
        "- `tables/table_01_dataset_year_availability.{csv,md}`",
        "- `tables/table_02_model_configuration_summary.{csv,md}`",
        "- `tables/table_03_historical_run_progression.{csv,md}`",
        "- `tables/table_04_final_consolidated_metrics_20260427_152647.{csv,md}`",
        "- `tables/table_05_landcover_ablation.{csv,md}`",
        "- `tables/table_06_limitations_and_future_work.{csv,md}`",
        "",
        "These assets were generated from the actual rasters and metric files in `runs/`.",
    ]
    (RESULTS_DIR / "README.md").write_text("\n".join(lines) + "\n")


def create_manifest(crop_spec: tuple[int, int, int]) -> None:
    latest_records = load_run_records(RUN_LATEST)
    best_prithvi = min(
        (rec for rec in latest_records if rec["model"].startswith("prithvi_") and "eval_rmse" in rec),
        key=lambda rec: rec["eval_rmse"],
    )
    manifest = {
        "reference_year": REFERENCE_YEAR,
        "test_year": TEST_YEAR,
        "selected_crop": {
            "top": crop_spec[0],
            "left": crop_spec[1],
            "size": crop_spec[2],
        },
        "final_model": "unet_2023",
        "best_prithvi_for_map_comparison": best_prithvi["model"],
        "generated_figures": sorted(p.name for p in FIGURES_DIR.glob("*.png")),
        "generated_tables": sorted(p.name for p in TABLES_DIR.iterdir() if p.is_file()),
    }
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    ensure_dirs()
    x, y = load_aligned_2023_data()
    crop_spec = select_informative_crop(y, size=320, step=64)

    create_workflow_figure()
    create_input_stack_figure(x, crop_spec)
    create_map_comparison_figure(x, y, crop_spec)
    create_edge_artifact_figure(y)
    create_metrics_bar_chart()
    create_scatter_figure(y)
    create_learning_evolution_figure()
    create_run_progression_figure()
    create_yearly_canopy_evolution_figure(crop_spec)
    create_training_history_figure()

    create_dataset_table()
    create_model_configuration_table()
    create_historical_progression_table()
    create_final_metrics_table()
    create_landcover_ablation_table()
    create_limitations_table()
    create_results_readme()
    create_manifest(crop_spec)
    print(f"Saved figures to {FIGURES_DIR}")
    print(f"Saved tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
