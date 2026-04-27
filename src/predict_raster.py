from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import rasterio
from tqdm import tqdm

from common import CHANNEL_ORDER, discover_year_paths, get_torch_device, load_json, normalize_channels, resolve_reference_path, sliding_windows, stack_inputs
from models import build_model, infer_vit_name_from_state_dict


def pad_for_inference(x: np.ndarray, patch_size: int) -> tuple[np.ndarray, int]:
    if patch_size <= 1:
        return x, 0
    pad = patch_size // 2
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="reflect")
    return padded, pad


def make_blend_weights(patch_size: int) -> np.ndarray:
    if patch_size <= 1:
        return np.ones((patch_size, patch_size), dtype=np.float32)
    ramp = 1.0 - np.abs(np.linspace(-1.0, 1.0, patch_size, dtype=np.float32))
    ramp = np.clip(ramp, 0.05, None)
    weight = np.outer(ramp, ramp)
    return weight.astype(np.float32)


def predict_patch(model, inp: torch.Tensor, tta: bool) -> np.ndarray:
    preds = [model(inp)]
    if tta:
        flip_dims = [(-1,), (-2,), (-2, -1)]
        for dims in flip_dims:
            aug_inp = torch.flip(inp, dims=dims)
            aug_pred = model(aug_inp)
            preds.append(torch.flip(aug_pred, dims=dims))
    pred = torch.stack(preds, dim=0).mean(dim=0)
    return pred.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument("--vit-name", type=str, default="vit_base_patch16_224")
    parser.add_argument("--tta", action="store_true",
                        help="Average predictions over horizontal/vertical flip test-time augmentations")
    args = parser.parse_args()

    stats = load_json(args.stats_path)
    device = get_torch_device()
    print(f"Using device: {device}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    state = ckpt["model_state"] if isinstance(
        ckpt, dict) and "model_state" in ckpt else ckpt
    resolved_patch_size = int(model_args.get("patch_size", args.patch_size))
    resolved_vit_name = model_args.get(
        "vit_name",
        ckpt.get("resolved_vit_name", args.vit_name) if isinstance(ckpt, dict) else args.vit_name,
    )
    resolved_vit_name = infer_vit_name_from_state_dict(
        state, resolved_vit_name)
    reference_year = args.reference_year if args.reference_year is not None else model_args.get(
        "reference_year", ckpt.get("reference_year", args.year) if isinstance(ckpt, dict) else args.year)
    ref_path = resolve_reference_path(args.data_root, int(reference_year))

    yp = replace(discover_year_paths(args.data_root, args.year), label=None)
    x, _, meta = stack_inputs(
        yp,
        label_align_dir=str(Path(args.output_path).parent / "aligned"),
        ref_path=ref_path,
    )
    x = normalize_channels(x, stats)

    if resolved_patch_size != args.patch_size:
        print(
            f"Using checkpoint patch size {resolved_patch_size} instead of CLI value {args.patch_size}")
    model = build_model(
        model_name=model_args.get("model", "vit"),
        in_channels=len(CHANNEL_ORDER),
        patch_size=resolved_patch_size,
        pretrained_checkpoint=None,
        vit_name=resolved_vit_name,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    orig_h, orig_w = x.shape[1:]
    x, pad = pad_for_inference(x, resolved_patch_size)
    h, w = x.shape[1:]
    pred_sum = np.zeros((h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.float32)
    blend_weights = make_blend_weights(resolved_patch_size)

    with torch.no_grad():
        for top, left in tqdm(sliding_windows(h, w, resolved_patch_size, args.stride)):
            patch = x[:, top: top + resolved_patch_size, left: left + resolved_patch_size]
            if patch.shape[1] != resolved_patch_size or patch.shape[2] != resolved_patch_size:
                continue
            inp = torch.from_numpy(patch[None]).float().to(device)
            out = predict_patch(model, inp, args.tta)
            pred_sum[top: top + resolved_patch_size, left: left + resolved_patch_size] += out * blend_weights
            pred_count[top: top + resolved_patch_size, left: left + resolved_patch_size] += blend_weights

    pred = np.full((h, w), np.nan, dtype=np.float32)
    valid = pred_count > 0
    pred[valid] = pred_sum[valid] / pred_count[valid]
    if pad > 0:
        pred = pred[pad: pad + orig_h, pad: pad + orig_w]
        valid = valid[pad: pad + orig_h, pad: pad + orig_w]
    pred[valid] = np.clip(pred[valid] * 100.0, 0.0, 100.0)

    profile = meta.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_path, "w", **profile) as dst:
        dst.write(pred.astype(np.float32), 1)
    print(f"Saved prediction to {args.output_path}")


if __name__ == "__main__":
    main()
