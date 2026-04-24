from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import rasterio

from common import discover_year_paths, resolve_reference_path, stack_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--reference-year", type=int, default=None)
    args = parser.parse_args()

    reference_year = args.reference_year if args.reference_year is not None else args.year
    ref_path = resolve_reference_path(args.data_root, int(reference_year))

    year_paths = replace(discover_year_paths(args.data_root, args.year), label=None)
    x, _, meta = stack_inputs(
        year_paths,
        label_align_dir=str(Path(args.output_path).parent / "aligned"),
        ref_path=ref_path,
    )

    model = joblib.load(args.model_path)

    h, w = x.shape[1:]
    pred = np.full((h, w), np.nan, dtype=np.float32)
    valid = np.all(np.isfinite(x), axis=0)
    if np.any(valid):
        features = x[:, valid].T
        pred[valid] = model.predict(features).astype(np.float32)

    pred = np.clip(pred, 0.0, 100.0)

    profile = meta.copy()
    profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.output_path, "w", **profile) as dst:
        dst.write(pred, 1)

    print(f"Saved RF prediction to {args.output_path}")


if __name__ == "__main__":
    main()
