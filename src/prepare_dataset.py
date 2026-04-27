from __future__ import annotations

import argparse
from pathlib import Path

from common import compute_channel_stats, discover_year_paths, read_profile, resolve_reference_path, save_json, stack_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-years", type=int, nargs="*", default=None)
    parser.add_argument("--reference-year", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reference_year = args.reference_year if args.reference_year is not None else args.years[0]
    ref_path = resolve_reference_path(args.data_root, reference_year)

    manifest = {
        "years": {},
        "train_years": args.train_years or args.years,
        "reference_year": reference_year,
        "reference_profile": read_profile(ref_path),
    }
    train_xs = []
    for year in args.years:
        yp = discover_year_paths(args.data_root, year)
        year_rec = {}
        for key in ["nlcd", "landcover", "ndvi", "lst", "ppt", "tmin", "tmean", "tmax"]:
            path = getattr(yp, key)
            year_rec[key] = read_profile(path) if path else None
        if yp.label:
            year_rec["label"] = read_profile(yp.label)
        manifest["years"][str(year)] = year_rec
        if year in (args.train_years or args.years):
            x, _, _ = stack_inputs(
                yp, label_align_dir=out / "aligned", ref_path=ref_path)
            train_xs.append(x)
    stats = compute_channel_stats(train_xs)
    save_json(manifest, out / "manifest.json")
    save_json(stats, out / "stats.json")
    print(f"Saved manifest to {out / 'manifest.json'}")
    print(f"Saved stats to {out / 'stats.json'}")


if __name__ == "__main__":
    main()
