#!/usr/bin/env python3
"""
Add Log‑Normal Noise to Excel Sheets (GitHub‑Ready)
==================================================

• Opens one or more Excel workbooks and applies log‑normal noise to numeric
  columns for sheets 2..N (leaves the first sheet unchanged), then saves a new
  workbook with a "noisy_" prefix next to the original.

Noise model
-----------
For each numeric value X>0 in a column (excluding a column named "Time" case‑insensitively):
    mean = ln(X^2 / sqrt(X^2 + X * alpha^2))
    sd   = sqrt( ln( 1 + (alpha^2 / X) ) )
    Y    = exp( Normal(mean, sd) )
We also clip/replace non‑positive values to a tiny epsilon before computing parameters.

Usage (CLI)
-----------
    python add_lognormal_noise_excel.py --files a.xlsx b.xlsx --alpha 1e-3

Or select via GUI dialogs:
    python add_lognormal_noise_excel.py --use_gui --alpha 1e-3

Options:
    --alpha <float>       Log‑normal noise scale parameter (default 1e-3)
    --seed <int>          RNG seed for reproducibility (default 42)
    --include_first       Also add noise to the first sheet (default: skip it)
    --suffix <str>        Filename suffix instead of "noisy_" prefix (mutually exclusive)
    --overwrite           Overwrite if the output file exists (default: keep both)

Dependencies: pandas, numpy, openpyxl (for writing .xlsx)
License: MIT
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Optional GUI (enabled with --use_gui)
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def add_lognorm_to_dataframe(df: pd.DataFrame, alpha: float, exclude_cols: Iterable[str]) -> pd.DataFrame:
    """Return a new DataFrame with log‑normal noise applied to numeric columns not in exclude_cols.
    Non‑positive values are replaced by a small epsilon before parameterization.
    """
    out = df.copy()
    exclude_lower = {c.lower() for c in exclude_cols}

    for col in out.columns:
        if col.lower() in exclude_lower:
            continue
        if not is_numeric_series(out[col]):
            continue

        x = out[col].to_numpy(dtype=float, copy=True)
        # Handle NaNs gracefully: keep NaNs as NaNs (don’t perturb)
        mask_nan = ~np.isfinite(x)
        x_work = x.copy()
        x_work[mask_nan] = np.nan

        # Replace non‑positive or NaN with epsilon for parameter calculation, but keep original NaNs unmodified
        eps = 1e-6
        pos_mask = (x_work > 0) & np.isfinite(x_work)
        x_safe = np.where(pos_mask, x_work, eps)

        # Parameters per provided formulation
        mean = np.log(x_safe ** 2 / np.sqrt(x_safe ** 2 + x_safe * (alpha ** 2)))
        sd = np.sqrt(np.log(1.0 + ((alpha ** 2) / x_safe)))

        # Sample; for non‑positive original entries we’ll still produce a noisy value based on epsilon params
        noisy = np.exp(np.random.normal(mean, sd))

        # Restore NaNs where they originally existed
        noisy[mask_nan] = np.nan
        out[col] = noisy

    return out


def process_workbook(path: Path, alpha: float, include_first: bool, suffix: str | None, overwrite: bool) -> Path:
    logging.info("Processing: %s", path)
    all_sheets: Dict[str, pd.DataFrame] = pd.read_excel(path, sheet_name=None)

    names: List[str] = list(all_sheets.keys())
    start_idx = 0 if include_first else 1

    for idx, sheet_name in enumerate(names):
        df = all_sheets[sheet_name]
        if idx < start_idx:
            logging.debug("Skipping first sheet: %s", sheet_name)
            continue
        all_sheets[sheet_name] = add_lognorm_to_dataframe(df, alpha=alpha, exclude_cols=["Time"])  # case‑insensitive

    # Build output path
    if suffix and suffix.strip():
        out_name = f"{Path(path).stem}{suffix}.xlsx"
        out_path = path.with_name(out_name)
    else:
        out_path = path.with_name(f"noisy_{path.name}")

    if out_path.exists() and not overwrite:
        # Find a non‑clobbering filename
        i = 1
        candidate = out_path
        while candidate.exists():
            candidate = out_path.with_name(out_path.stem + f"_{i}" + out_path.suffix)
            i += 1
        out_path = candidate

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in all_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    logging.info("Saved: %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Add log‑normal noise to Excel sheets (2..N).")
    p.add_argument("--files", nargs="*", type=Path, help="Excel .xlsx files to process")
    p.add_argument("--alpha", type=float, default=1e-3, help="Noise alpha parameter (default 1e-3)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--include_first", action="store_true", help="Also perturb the first sheet")
    p.add_argument("--suffix", type=str, default=None, help="Filename suffix instead of 'noisy_' prefix")
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist")
    p.add_argument("--use_gui", action="store_true", help="Pick files with a GUI dialog")
    p.add_argument("--log_level", default="INFO", help="Logging level: INFO, DEBUG, ...")
    return p.parse_args()


def pick_files_gui() -> Tuple[List[Path], bool]:
    if tk is None or filedialog is None or messagebox is None:
        raise RuntimeError("Tkinter GUI not available on this system.")
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("Select Excel Files", "Please select one or more Excel files (.xlsx)")
    file_paths = filedialog.askopenfilenames(title="Select Excel Files", filetypes=[("Excel files", "*.xlsx")])
    if not file_paths:
        messagebox.showwarning("No file selected", "No files were selected.")
        return [], False
    return [Path(p) for p in file_paths], True


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    np.random.seed(args.seed)

    files: List[Path] = list(args.files) if args.files else []
    if args.use_gui and not files:
        files, ok = pick_files_gui()
        if not ok:
            return 1

    if not files:
        logging.error("No files provided. Use --files or --use_gui.")
        return 2

    for f in files:
        try:
            process_workbook(
                path=f,
                alpha=args.alpha,
                include_first=args.include_first,
                suffix=args.suffix,
                overwrite=args.overwrite,
            )
        except Exception as e:
            logging.exception("Failed on %s: %s", f, e)

    print("✅ All files processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
