"""Evaluate CFA detections on the FITS set produced by make_test_set.py.

    python scripts/eval_detection.py --test-set test_set_d56 \\
        --weights cfa-unet_final.pt --freq-order 90 150 220

Add --catalog clusters.fits for catalogue recall; otherwise only ACT-detector
agreement is measured. This script does not sample or train diffusion models.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astropy.table import Table

from bbdm.metrics.detection import evaluate_detection
from cfa_unet_inference import CFAUNetInference


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--freq-order", type=int, nargs=3, required=True,
                        help="ACT-frequency labels in the detector's training channel order")
    parser.add_argument("--catalog", help="FITS/CSV catalogue with RA/Dec in degrees")
    parser.add_argument("--ra-column", default="RADeg")
    parser.add_argument("--dec-column", default="decDeg")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="CFA score threshold; calibrate on validation, not D56")
    parser.add_argument("--match-radius-arcmin", type=float, default=2.4,
                        help="sky matching tolerance, not a fixed pixel or cluster size")
    parser.add_argument("--edge-margin", type=int, default=8,
                        help="exclude this many pixels around map edges and common-mask holes")
    parser.add_argument("--device", default=None, help="e.g. cuda or cpu; CFA defaults to CUDA when available")
    parser.add_argument("--out-dir", help="new report directory; defaults to TEST_SET/detection")
    args = parser.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else Path(args.test_set) / "detection"
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new --out-dir to preserve earlier results")
    catalog = Table.read(args.catalog) if args.catalog else None
    detector = CFAUNetInference(args.weights, device=args.device)
    print("Evaluating exported maps one at a time; no diffusion sampling or training.", flush=True)
    report = evaluate_detection(args.test_set, detector, args.freq_order, threshold=args.threshold,
                                match_radius_arcmin=args.match_radius_arcmin, edge_margin=args.edge_margin,
                                catalog=catalog, ra_column=args.ra_column, dec_column=args.dec_column)
    report["settings"].update(test_set=str(Path(args.test_set).resolve()),
                              weights=str(Path(args.weights).resolve()), catalog=args.catalog,
                              ra_column=args.ra_column, dec_column=args.dec_column)
    out.mkdir(parents=True, exist_ok=False)
    for key, filename, fields in [
        ("windows", "per_window.csv", list(report["windows"][0])),
        ("detections", "detections.csv", ["name", "variant", "x", "y", "ra", "dec", "score"]),
    ]:
        with (out / filename).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(report.pop(key))
    with (out / "summary.json").open("w") as fh:
        json.dump(report, fh, indent=2, allow_nan=False)
    for variant, metrics in report["summary"].items():
        print(variant, json.dumps(metrics, allow_nan=False))
    if catalog is None:
        print("No catalogue: ACT recall/matched_fraction describe CFA agreement, not cluster detection accuracy.")
    print(f"Report: {out}")
    return report


if __name__ == "__main__":
    main()
