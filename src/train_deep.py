from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CHANNEL_ORDER, compute_channel_stats, discover_year_paths, get_split_mask, resolve_reference_path, save_json, set_seed, stack_inputs, canopy_metrics, get_torch_device, should_pin_memory
from data import MultiYearRasterStore, PatchDataset, PairConsistencyDataset, MultiPairConsistencyDataset
from losses import masked_regression_loss, temporal_consistency_loss
from models import build_model


def make_blend_weights(patch_size: int) -> np.ndarray:
    if patch_size <= 1:
        return np.ones((patch_size, patch_size), dtype=np.float32)
    ramp = 1.0 - np.abs(np.linspace(-1.0, 1.0, patch_size, dtype=np.float32))
    ramp = np.clip(ramp, 0.05, None)
    weight = np.outer(ramp, ramp)
    return weight.astype(np.float32)


def evaluate_model(model, loader, device):
    dataset = loader.dataset
    per_year = {}
    blend_weights = make_blend_weights(dataset.patch_size)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            pred = model(x)
            pred = pred.squeeze(1).cpu().numpy()
            years = batch["year"].cpu().numpy()
            tops = batch["top"].cpu().numpy()
            lefts = batch["left"].cpu().numpy()

            for idx, year in enumerate(years):
                year = int(year)
                top = int(tops[idx])
                left = int(lefts[idx])
                if year not in per_year:
                    h, w = dataset.store.shape(year)
                    per_year[year] = {
                        "pred_sum": np.zeros((h, w), dtype=np.float32),
                        "pred_count": np.zeros((h, w), dtype=np.float32),
                    }
                per_year[year]["pred_sum"][top: top + dataset.patch_size,
                                           left: left + dataset.patch_size] += pred[idx] * blend_weights
                per_year[year]["pred_count"][top: top + dataset.patch_size,
                                             left: left + dataset.patch_size] += blend_weights

    y_true_all = []
    y_pred_all = []
    for year, rec in per_year.items():
        target = dataset.store.targets[year]
        if target is None:
            continue
        pred = np.full(target.shape, np.nan, dtype=np.float32)
        valid = rec["pred_count"] > 0
        pred[valid] = rec["pred_sum"][valid] / rec["pred_count"][valid]
        split_mask = get_split_mask(
            target.shape[0], target.shape[1], dataset.split, dataset.block_size, dataset.seed)
        mask = valid & split_mask & np.isfinite(target)
        if not np.any(mask):
            continue
        y_true_all.append(target[mask])
        y_pred_all.append(pred[mask] * 100.0)

    if not y_true_all:
        raise RuntimeError("No valid evaluation pixels found for the requested split")

    return canopy_metrics(np.concatenate(y_true_all), np.concatenate(y_pred_all))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--train-years", type=int, nargs="+", required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--supervised-years", type=int,
                        nargs="+", default=None)
    parser.add_argument("--consistency-years", type=int,
                        nargs="+", default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument("--model", type=str, default="vit")
    parser.add_argument("--vit-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--change-gamma", type=float, default=3.0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    supervised_years = args.supervised_years if args.supervised_years is not None else args.train_years
    consistency_years = args.consistency_years if args.consistency_years is not None else args.train_years
    reference_year = args.reference_year if args.reference_year is not None else args.train_years[0]
    ref_path = resolve_reference_path(args.data_root, reference_year)
    all_store_years = list(dict.fromkeys(
        list(args.train_years)
        + list(supervised_years)
        + list(consistency_years)
        + [args.test_year]
    ))

    x_for_stats = []
    for year in args.train_years:
        yp = discover_year_paths(args.data_root, year)
        x, _, _ = stack_inputs(
            yp, label_align_dir=out / "aligned", ref_path=ref_path)
        x_for_stats.append(x)
    stats = compute_channel_stats(x_for_stats)
    save_json(stats, out / "stats.json")

    store = MultiYearRasterStore(
        args.data_root, all_store_years, stats, aligned_dir=str(out / "aligned"), ref_path=ref_path)
    train_ds = PatchDataset(store, supervised_years, split="train", patch_size=args.patch_size,
                            stride=args.stride, block_size=args.block_size, require_labels=True, seed=args.seed)
    val_ds = PatchDataset(store, supervised_years, split="val", patch_size=args.patch_size,
                          stride=args.stride, block_size=args.block_size, require_labels=True, seed=args.seed)
    test_ds = PatchDataset(store, [args.test_year], split="test", patch_size=args.patch_size,
                           stride=args.stride, block_size=args.block_size, require_labels=True, seed=args.seed)

    device = get_torch_device()
    pin_memory = should_pin_memory(device)
    print(f"Using device: {device}")

    pair_loader = None
    if len(consistency_years) >= 2 and args.consistency_weight > 0:
        if len(consistency_years) == 2:
            pair_train = PairConsistencyDataset(store, consistency_years[0], consistency_years[1], split="train",
                                                patch_size=args.patch_size, stride=args.stride,
                                                block_size=args.block_size, seed=args.seed)
        else:
            year_pairs = list(
                zip(consistency_years[:-1], consistency_years[1:]))
            pair_train = MultiPairConsistencyDataset(store, year_pairs=year_pairs, split="train",
                                                     patch_size=args.patch_size, stride=args.stride,
                                                     block_size=args.block_size, seed=args.seed)
        pair_loader = DataLoader(pair_train, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=pin_memory)

    save_json({
        "train_years_loaded": args.train_years,
        "supervised_years": supervised_years,
        "consistency_years": consistency_years,
        "test_year": args.test_year,
        "reference_year": reference_year,
        "num_train_patches": len(train_ds),
        "num_val_patches": len(val_ds),
        "num_test_patches": len(test_ds),
    }, out / "dataset_summary.json")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)
    model = build_model(
        model_name=args.model,
        in_channels=len(CHANNEL_ORDER),
        patch_size=args.patch_size,
        pretrained_checkpoint=args.pretrained_checkpoint,
        vit_name=args.vit_name,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=4, factor=0.5)

    resolved_args = vars(args).copy()
    resolved_args["reference_year"] = reference_year
    resolved_args["vit_name"] = getattr(model, "vit_name", args.vit_name)

    best_val = float("inf")
    best_path = out / "best.pt"
    pair_iter = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        if pair_loader is not None:
            pair_iter = iter(pair_loader)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred, feats = model(x, return_features=True)
            sup_loss = masked_regression_loss(pred, y)
            total_loss = sup_loss
            cons_loss_val = torch.tensor(0.0, device=device)
            if pair_iter is not None:
                try:
                    pb = next(pair_iter)
                except StopIteration:
                    pair_iter = iter(pair_loader)
                    pb = next(pair_iter)
                x1 = pb["x1"].to(device)
                x2 = pb["x2"].to(device)
                _, f1 = model(x1, return_features=True)
                _, f2 = model(x2, return_features=True)
                cons_loss_val = temporal_consistency_loss(
                    f1, f2, x1, x2, gamma=args.change_gamma)
                total_loss = total_loss + args.consistency_weight * cons_loss_val
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(total_loss.item()))
            pbar.set_postfix({
                "loss": f"{float(total_loss.item()):.4f}",
                "sup": f"{float(sup_loss.item()):.4f}",
                "cons": f"{float(cons_loss_val.item()):.4f}",
            })

        with torch.no_grad():
            val_metrics = evaluate_model(model, val_loader, device)
        scheduler.step(val_metrics["rmse"])
        history.append({"epoch": epoch, **val_metrics,
                       "train_loss": float(np.mean(epoch_losses))})
        save_json({"history": history}, out / "history.json")
        print(f"Val metrics: {val_metrics}")
        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "args": resolved_args,
                "channels": CHANNEL_ORDER,
                "reference_year": reference_year,
                "resolved_vit_name": resolved_args["vit_name"],
            }, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_model(model, test_loader, device)
    save_json(test_metrics, out / "test_metrics.json")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
