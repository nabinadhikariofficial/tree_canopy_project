from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from common import CHANNEL_ORDER, compute_channel_stats, discover_year_paths, get_split_mask, resolve_reference_path, save_json, stack_inputs


def sample_pixels(x: np.ndarray, y: np.ndarray, fraction: float, seed: int = 42):
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=0)
    coords = np.argwhere(mask)
    rng = np.random.default_rng(seed)
    n = max(1000, int(len(coords) * fraction))
    idx = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
    coords = coords[idx]
    feats = x[:, coords[:, 0], coords[:, 1]].T
    labels = y[coords[:, 0], coords[:, 1]]
    return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--train-years", type=int, nargs="+", required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.05)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference_year = args.reference_year if args.reference_year is not None else args.train_years[0]
    ref_path = resolve_reference_path(args.data_root, reference_year)

    train_xs, train_ys, x_for_stats = [], [], []
    for year in args.train_years:
        yp = discover_year_paths(args.data_root, year)
        x, y, _ = stack_inputs(yp, label_align_dir=out / "aligned", ref_path=ref_path)
        fx, fy = sample_pixels(x, y, args.sample_fraction)
        train_xs.append(fx)
        train_ys.append(fy)
        x_for_stats.append(x)

    X_train = np.concatenate(train_xs, axis=0)
    y_train = np.concatenate(train_ys, axis=0)
    stats = compute_channel_stats(x_for_stats)
    save_json(stats, out / "stats.json")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )),
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, out / "rf.joblib")

    test_paths = discover_year_paths(args.data_root, args.test_year)
    x_test, y_test, _ = stack_inputs(
        test_paths, label_align_dir=out / "aligned", ref_path=ref_path)
    test_mask = get_split_mask(
        y_test.shape[0], y_test.shape[1], "test", args.block_size, args.seed)
    mask = np.isfinite(y_test) & np.all(np.isfinite(x_test), axis=0) & test_mask
    X_eval = x_test[:, mask].T
    y_eval = y_test[mask]
    y_pred = model.predict(X_eval)

    metrics = {
        "r2": float(r2_score(y_eval, y_pred)),
        "mae": float(mean_absolute_error(y_eval, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_eval, y_pred))),
        "feature_names": CHANNEL_ORDER,
        "reference_year": reference_year,
        "num_eval_pixels": int(mask.sum()),
    }
    save_json(metrics, out / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
