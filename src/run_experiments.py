from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: list[str]):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-years", nargs="+", type=int, required=True)
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--supervised-years", nargs="+", type=int, default=None)
    parser.add_argument("--consistency-years", nargs="+", type=int, default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--train-stride", type=int, default=64)
    parser.add_argument("--eval-stride", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-year", type=int, default=None)
    parser.add_argument("--prithvi-checkpoint", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_root) / run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving this run under {out}")

    supervised_years = args.supervised_years if args.supervised_years is not None else args.train_years
    consistency_years = args.consistency_years if args.consistency_years is not None else args.train_years

    common = [
        "--data-root", args.data_root,
        "--train-years", *[str(y) for y in args.train_years],
        "--test-year", str(args.test_year),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--patch-size", str(args.patch_size),
        "--stride", str(args.stride),
        "--train-stride", str(args.train_stride),
        "--eval-stride", str(args.eval_stride),
        "--block-size", str(args.block_size),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--supervised-years", *[str(y) for y in supervised_years],
        "--consistency-years", *[str(y) for y in consistency_years],
    ]
    if args.reference_year is not None:
        common += ["--reference-year", str(args.reference_year)]

    rf_cmd = [
        sys.executable, "src/train_rf.py",
        "--data-root", args.data_root,
        "--train-years", *[str(y) for y in supervised_years],
        "--test-year", str(args.test_year),
        "--output-dir", str(out / "rf"),
        "--sample-fraction", "0.05",
        "--block-size", str(args.block_size),
        "--seed", str(args.seed),
    ]
    if args.reference_year is not None:
        rf_cmd += ["--reference-year", str(args.reference_year)]
    run(rf_cmd)

    run([
        sys.executable, "src/train_deep.py", *common,
        "--output-dir", str(out / "unet_baseline"),
        "--model", "unet",
        "--consistency-weight", "0.0",
    ])

    run([
        sys.executable, "src/train_deep.py", *common,
        "--output-dir", str(out / "unet_consistency"),
        "--model", "unet",
        "--consistency-weight", "0.2",
        "--change-gamma", "3.0",
    ])

    vit_cmd = [
        sys.executable, "src/train_deep.py", *common,
        "--output-dir", str(out / "vit_baseline"),
        "--model", "geo_vit",
        "--consistency-weight", "0.0",
    ]
    if args.prithvi_checkpoint:
        vit_cmd += ["--pretrained-checkpoint", args.prithvi_checkpoint]
    run(vit_cmd)

    vit_cons_cmd = [
        sys.executable, "src/train_deep.py", *common,
        "--output-dir", str(out / "vit_consistency"),
        "--model", "geo_vit",
        "--consistency-weight", "0.2",
        "--change-gamma", "3.0",
    ]
    if args.prithvi_checkpoint:
        vit_cons_cmd += ["--pretrained-checkpoint", args.prithvi_checkpoint]
    run(vit_cons_cmd)

    summary = {}
    for name in ["rf", "unet_baseline", "unet_consistency", "vit_baseline", "vit_consistency"]:
        metric_path = out / name / ("metrics.json" if name == "rf" else "test_metrics.json")
        if metric_path.exists():
            summary[name] = json.loads(metric_path.read_text())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved experiment summary to", out / "summary.json")


if __name__ == "__main__":
    main()
