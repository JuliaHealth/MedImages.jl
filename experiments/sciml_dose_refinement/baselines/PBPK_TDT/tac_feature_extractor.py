#!/usr/bin/env python3
"""
Time‑Activity Curve (TAC) Feature Extractor (GitHub‑Ready)
=========================================================

Reads Excel workbooks containing time columns and one or more activity columns
(nmol) on each sheet, converts nmol→MBq for a chosen isotope (F‑18, Ga‑68, Cu‑64),
extracts a rich set of temporal/statistical features per column, and saves a tidy CSV.

Features
--------
- CLI + optional GUI pickers (folders + isotope dialog).
- Robust time column detection (case‑insensitive substring match for 'time').
- Skips non‑numeric columns and handles NaNs safely.
- Uses Simpson integration (AUC, MBq·min) and additional kinetics metrics.

Examples
--------
CLI:
    python tac_feature_extractor.py \
        --input_dir /path/to/xlsx \
        --out /path/to/extracted_features.csv \
        --isotope F-18

GUI:
    python tac_feature_extractor.py --use_gui

Dependencies: pandas, numpy, scipy, openpyxl (for .xlsx), xlrd (for legacy .xls)
License: MIT
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.stats import kurtosis, skew

# Optional GUI (enabled by --use_gui)
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
except Exception:  # pragma: no cover
    tk = None
    filedialog = None
    simpledialog = None
    messagebox = None

# -------------------- Constants --------------------
AVOGADRO = 6.022e23  # mol^-1
HALF_LIVES_SEC: Dict[str, float] = {
    "F-18": 6586.2,
    "Ga-68": 4071.8,
    "Cu-64": 43200.0,
}

# -------------------- Helpers --------------------

def get_decay_constant(half_life_sec: float) -> float:
    return float(np.log(2.0) / half_life_sec)


def convert_nmol_to_MBq(nmol_array: np.ndarray, isotope: str) -> np.ndarray:
    lam = get_decay_constant(HALF_LIVES_SEC[isotope])
    atoms = nmol_array * 1e-9 * AVOGADRO  # nmol → mol → atoms
    return (lam * atoms) / 1e6  # MBq


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if isinstance(c, str) and "time" in c.lower():
            return c
    return None


def extract_features(time_series: np.ndarray, activity_series: np.ndarray) -> Dict[str, float]:
    t = np.asarray(time_series, dtype=float)
    a = np.asarray(activity_series, dtype=float)

    # Clean/validate
    mask = np.isfinite(t) & np.isfinite(a)
    t, a = t[mask], a[mask]
    if t.size < 2:
        return {}

    # Enforce monotonic time for integrals and threshold searches
    order = np.argsort(t)
    t, a = t[order], a[order]

    # Basic stats
    amax_idx = int(np.argmax(a))
    amin_idx = int(np.argmin(a))

    features: Dict[str, float] = {
        "Tmax": float(t[amax_idx]),
        "AMax": float(a[amax_idx]),
        "TimeToPeak": float(t[amax_idx] - t[0]),
        "Tmin": float(t[amin_idx]),
        "AMin": float(a[amin_idx]),
        "TimeFromPeakToMin": float(t[amin_idx] - t[amax_idx]),
        "AMean": float(np.mean(a)),
        "AStdev": float(np.std(a, ddof=0)),
        "AMedian": float(np.median(a)),
        "Skewness": float(skew(a, nan_policy="omit")),
        "Kurtosis": float(kurtosis(a, nan_policy="omit")),
        # Shannon entropy (discrete approx, normalize a to probabilities if positive)
        "Entropy": float(_entropy(a)),
        "Energy": float(np.sum(a ** 2)),
        "AUC": float(simpson(a, t)),  # MBq·min if t is minutes
    }

    # Half‑life approximation relative to AMax
    half_max = features["AMax"] / 2.0 if features["AMax"] > 0 else np.nan
    if np.isfinite(half_max):
        lt_mask = a <= half_max
        features["HalfLife"] = float(t[np.argmax(lt_mask)] - t[amax_idx]) if np.any(lt_mask) else np.nan
    else:
        features["HalfLife"] = np.nan

    features["Clearance"] = float(features["AMax"] / features["AUC"]) if features["AUC"] > 0 else np.nan

    # Percentiles
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        features[f"P{p}"] = float(np.percentile(a, p))

    # Times reaching ≥ X% of AMax (rising or anywhere)
    max_a = features["AMax"]
    if max_a > 0:
        for th in range(10, 101, 10):
            thr = (th / 100.0) * max_a
            mask_ge = a >= thr
            features[f"T{th}"] = float(t[np.argmax(mask_ge)]) if np.any(mask_ge) else np.nan

        # Times reaching ≤ X% of AMax (post‑peak)
        t_post = t[amax_idx:]
        a_post = a[amax_idx:]
        for th in range(90, 0, -10):
            thr = (th / 100.0) * max_a
            mask_le = a_post <= thr
            features[f"T-{th}"] = float(t_post[np.argmax(mask_le)]) if np.any(mask_le) else np.nan
    else:
        for th in range(10, 101, 10):
            features[f"T{th}"] = np.nan
        for th in range(90, 0, -10):
            features[f"T-{th}"] = np.nan

    # Slope features
    dt = np.diff(t)
    da = np.diff(a)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = da / np.where(dt == 0, np.nan, dt)
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size:
        features["IncreasingSlope"] = float(np.max(slopes[slopes > 0])) if np.any(slopes > 0) else 0.0
        features["DecreasingSlope"] = float(np.min(slopes[slopes < 0])) if np.any(slopes < 0) else 0.0
        features["MaxSlope"] = float(np.max(slopes))
        features["MinSlope"] = float(np.min(slopes))
    else:
        features["IncreasingSlope"] = 0.0
        features["DecreasingSlope"] = 0.0
        features["MaxSlope"] = 0.0
        features["MinSlope"] = 0.0

    return features


def _entropy(a: np.ndarray) -> float:
    """Shannon entropy of a non‑negative vector (adds small eps to avoid log(0))."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    a = np.clip(a, 0.0, None)
    s = a.sum()
    if s <= 0:
        return float("nan")
    p = a / s
    eps = 1e-12
    return float(-np.sum(p * np.log(p + eps)))


# -------------------- File processing --------------------

def process_excel_file(file_path: Path, isotope: str) -> List[dict]:
    try:
        xls = pd.ExcelFile(file_path)
        out: List[dict] = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            time_col = detect_time_column(df)
            if time_col is None:
                logging.debug("%s: sheet '%s' has no time column; skipped", file_path.name, sheet_name)
                continue
            t = df[time_col].to_numpy(dtype=float)

            for col in df.columns:
                if col == time_col or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                a_nmol = df[col].to_numpy(dtype=float)
                a_MBq = convert_nmol_to_MBq(a_nmol, isotope)
                feats = extract_features(t, a_MBq)
                if not feats:
                    continue
                feats.update({
                    "Sheet": sheet_name,
                    "File": file_path.name,
                    "ActivityColumn": col,
                    "Isotope": isotope,
                })
                out.append(feats)
        return out
    except Exception as e:
        logging.exception("Error processing %s: %s", file_path, e)
        return []


# -------------------- CLI / GUI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract features from TAC Excel workbooks.")
    p.add_argument("--input_dir", type=Path, help="Folder containing Excel files")
    p.add_argument("--out", type=Path, help="Output CSV path")
    p.add_argument("--isotope", choices=list(HALF_LIVES_SEC.keys()), help="Isotope for nmol→MBq conversion")
    p.add_argument("--use_gui", action="store_true", help="Use GUI pickers instead of CLI paths")
    p.add_argument("--log_level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    return p.parse_args()


def pick_with_gui(args: argparse.Namespace) -> argparse.Namespace:
    if tk is None or filedialog is None or simpledialog is None or messagebox is None:
        raise RuntimeError("Tkinter GUI is not available on this system.")
    root = tk.Tk(); root.withdraw()
    input_dir = filedialog.askdirectory(title="Select folder containing Excel files")
    if not input_dir:
        messagebox.showwarning("No folder", "No input folder selected.")
        raise SystemExit(1)
    out_path = filedialog.asksaveasfilename(title="Save results as", defaultextension=".csv", filetypes=[("CSV", ".csv")], initialfile="extracted_features.csv")
    if not out_path:
        messagebox.showwarning("Canceled", "Save operation canceled.")
        raise SystemExit(2)
    isotope = simpledialog.askstring("Radiopharmaceutical", f"Enter isotope: {', '.join(HALF_LIVES_SEC.keys())}")
    if isotope not in HALF_LIVES_SEC:
        messagebox.showerror("Invalid Isotope", f"{isotope} is not supported.")
        raise SystemExit(3)
    args.input_dir = Path(input_dir)
    args.out = Path(out_path)
    args.isotope = isotope
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if args.use_gui:
        args = pick_with_gui(args)

    if not args.input_dir or not Path(args.input_dir).is_dir():
        logging.error("Invalid --input_dir: %s", args.input_dir)
        return 2
    if not args.out:
        logging.error("Invalid --out path.")
        return 2

    rows: List[dict] = []
    for fn in os.listdir(args.input_dir):
        if fn.lower().endswith((".xlsx", ".xls")):
            fp = Path(args.input_dir) / fn
            logging.info("Processing %s", fn)
            rows.extend(process_excel_file(fp, args.isotope))

    if not rows:
        logging.warning("No valid data found in %s", args.input_dir)
        return 0

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path.with_suffix(".csv")
    out_df.to_csv(out_path, index=False)
    logging.info("Saved features to %s", out_path)
    print(f"✅ Success! Results saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
