#!/usr/bin/env python3
"""
Lu-177 Dosimetry from TACs (GitHub-Ready)
=========================================

Reads an Excel workbook containing organ sheets with time-activity columns (A*),
computes AUC (MBq·min), Dose (Gy), G(T), BED_G, and EQD2 for each A* column, and
saves a tidy results table to Excel.

Features
--------
- Supports GUI pickers (optional) or full CLI.
- Validates tumor volume for S-value lookup (1, 0.1, 0.01, 0.001 L).
- Skips sheets without the required columns and rows with invalid/empty data.
- Uses Simpson/Trapz integration consistently (time assumed in minutes).

Example (CLI)
-------------
python dosimetry_from_TACs_Lu177.py \
  --excel /path/data.xlsx \
  --volume 0.1 \
  --out NEW_DOSE_SINGLEmc.xlsx

Or with GUI pickers:
python dosimetry_from_TACs_Lu177.py --use_gui

Dependencies: pandas, numpy, scipy, openpyxl (for Excel I/O)
License: MIT
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import simpson

# Optional GUI imports (only if --use_gui)
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
except Exception:  # pragma: no cover
    tk = None
    filedialog = None
    simpledialog = None
    messagebox = None

# -------------------- Constants --------------------
AVOGADRO = 6.022e23
HALF_LIFE_MIN = 9573.68  # minutes
HALF_LIFE_SEC = HALF_LIFE_MIN * 60.0
DECAY_CONSTANT = np.log(2.0) / HALF_LIFE_SEC

# Radiobiology / material params (example values, domain-specific)
mu_values: Dict[str, float] = {"Tumor": 0.0231, "Salivary_gland": 0.0077, "Kidney": 0.004125}
AB_values: Dict[str, float] = {"Tumor": 3.9, "Salivary_gland": 4.5, "Kidney": 2.5}

# S-values for Lu-177 (Gy/MBq·min) — Tumor depends on volume
S_values_Lu177: Dict[str, object] = {
    "Tumor": {
        "L1": 1.4600e-06,
        "L0_1": 1.4158e-05,
        "L0_01": 1.3730e-04,
        "L0_001": 1.3315e-03,
    },
    "Salivary_gland": 6.90e-05,
    "Kidney": 4.82e-06,
    "Liver": 8.22e-07,
    "Red_marrow": 7.14e-07,
    "Spleen": 6.68e-06,
}

# -------------------- Helpers --------------------

def format_volume_key(volume_liters: float) -> str:
    """1 -> 'L1', 0.1 -> 'L0_1', 0.01 -> 'L0_01', 0.001 -> 'L0_001'"""
    return f"L{volume_liters:.3f}".rstrip("0").rstrip(".").replace(".", "_")


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def calculate_G(T_min: float, mu: float, dotD: np.ndarray, time_min: np.ndarray) -> float:
    """Calculate Lea-Catcheside G(T) factor via nested integration.

    Parameters
    ----------
    T_min : float
        Overall time window in minutes (for clarity; not explicitly used in this formulation).
    mu : float
        Repair rate parameter (1/min) — domain-specific.
    dotD : np.ndarray
        Dose-rate proxy; here proportional to activity(t) (e.g., MBq vs. time).
    time_min : np.ndarray
        Time samples in minutes (monotonic, >= 2 points).
    """
    dotD = np.asarray(dotD, dtype=float)
    t = np.asarray(time_min, dtype=float)
    if dotD.size < 2 or t.size < 2:
        return float("nan")

    # Ensure monotonic time to avoid negative/NaN integrals
    order = np.argsort(t)
    t = t[order]
    dotD = dotD[order]

    inner = np.zeros_like(dotD)
    for i in range(len(t)):
        if i == 0:
            inner[i] = dotD[i] * (1.0 - np.exp(-mu * t[i]))
        else:
            # ∫_0^{t_i} dotD(τ) e^{-μ (t_i - τ)} dτ
            inner[i] = simpson(dotD[: i + 1] * np.exp(-mu * (t[i] - t[: i + 1])), t[: i + 1])

    outer = simpson(dotD * inner, t)
    denom = np.trapz(dotD, t)
    if denom == 0.0:
        return float("nan")
    G_T = (2.0 / (denom ** 2)) * outer
    return float(np.clip(G_T, 0.0, 1.0))


def calculate_BED_G(Dose_Gy: float, G: float, alpha_over_beta: float) -> float:
    return float(Dose_Gy * (1.0 + G * Dose_Gy / alpha_over_beta))


# -------------------- Core --------------------

def process_excel(
    excel_path: Path,
    tumor_volume_L: float,
    T_min: float,
) -> pd.DataFrame:
    """Process an Excel workbook and return a tidy results DataFrame."""
    xls = pd.ExcelFile(excel_path)

    all_res: List[List[object]] = []
    organs_in_book = [s for s in xls.sheet_names if s in S_values_Lu177.keys()]

    vol_key = format_volume_key(tumor_volume_L)

    for organ in organs_in_book:
        df = xls.parse(organ)
        if "Time" not in df.columns or not is_numeric_series(df["Time"]):
            logging.debug("Sheet %s skipped: missing/invalid 'Time' column", organ)
            continue
        time = df["Time"].dropna().to_numpy(dtype=float)
        if time.size == 0 or np.all(time == 0):
            logging.debug("Sheet %s skipped: empty or zero time", organ)
            continue

        # Find activity columns (A*)
        a_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("A") and is_numeric_series(df[c])]
        if not a_cols:
            logging.debug("Sheet %s skipped: no A* numeric columns", organ)
            continue

        # S-value (Gy / MBq·min)
        try:
            if organ == "Tumor":
                S_value = float(S_values_Lu177["Tumor"][vol_key])  # type: ignore[index]
            else:
                S_value = float(S_values_Lu177[organ])  # type: ignore[index]
        except Exception as e:
            logging.warning("Missing S-value for %s or volume %s: %s", organ, vol_key, e)
            continue

        mu = float(mu_values.get(organ, 0.0))
        AB = float(AB_values.get(organ, 1.0))

        for col in a_cols:
            A_nmol = df[col].to_numpy(dtype=float)
            A_nmol = A_nmol[np.isfinite(A_nmol)]
            if A_nmol.size == 0 or np.all(A_nmol == 0):
                continue

            # Convert nmol → mol → atoms → activity (Bq) using λ; then MBq
            A_mol = A_nmol * 1e-9
            N_atoms = A_mol * AVOGADRO
            activity_Bq = DECAY_CONSTANT * N_atoms  # Bq
            activity_MBq = activity_Bq / 1e6

            # Align time length (truncate/pad if needed)
            n = min(activity_MBq.size, time.size)
            t_use = time[:n]
            a_use = activity_MBq[:n]

            # Integrals and radiobiology metrics
            AUC = float(np.trapz(a_use, t_use))  # MBq·min
            Dose = float(AUC * S_value)          # Gy
            G_T = calculate_G(T_min=T_min, mu=mu, dotD=a_use, time_min=t_use)
            BED_G = calculate_BED_G(Dose_Gy=Dose, G=G_T, alpha_over_beta=AB)
            EQD2 = float(BED_G / (1.0 + 2.0 / AB))

            all_res.append([excel_path.name, organ, col, AUC, Dose, G_T, BED_G, EQD2])

    columns = ["File", "Organ", "Case", "AUC", "Dose", "G_T", "BED_G", "EQD2"]
    return pd.DataFrame(all_res, columns=columns) if all_res else pd.DataFrame(columns=columns)


# -------------------- CLI / GUI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Lu-177 dosimetry metrics from Excel TACs.")
    p.add_argument("--excel", type=Path, help="Input Excel workbook (.xlsx)")
    p.add_argument("--out", type=Path, help="Output Excel path (.xlsx)")
    p.add_argument("--volume", type=float, choices=[1.0, 0.1, 0.01, 0.001], help="Tumor volume (L)")
    p.add_argument("--T", type=float, default=20000.0, help="G(T) horizon T in minutes (default 20000)")
    p.add_argument("--use_gui", action="store_true", help="Use GUI pickers for paths and volume")
    p.add_argument("--log_level", default="INFO", help="Logging level, e.g., INFO, DEBUG")
    return p.parse_args()


def pick_with_gui(args: argparse.Namespace) -> argparse.Namespace:
    if tk is None or filedialog is None or simpledialog is None or messagebox is None:
        raise RuntimeError("Tkinter GUI not available on this system.")
    root = tk.Tk(); root.withdraw()
    excel_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
    if not excel_path:
        messagebox.showwarning("No file", "No Excel file selected.")
        raise SystemExit(1)
    volume = simpledialog.askfloat("Volume Input", "Enter tumor volume (1, 0.1, 0.01, 0.001 liters):")
    if volume not in (1.0, 0.1, 0.01, 0.001):
        messagebox.showerror("Invalid volume", "Volume must be 1, 0.1, 0.01, or 0.001 L.")
        raise SystemExit(2)
    out_path = filedialog.asksaveasfilename(
        title="Save Results As",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialfile="NEW_DOSE_SINGLEmc.xlsx",
    )
    if not out_path:
        messagebox.showwarning("Canceled", "Save operation canceled.")
        raise SystemExit(3)

    args.excel = Path(excel_path)
    args.out = Path(out_path)
    args.volume = float(volume)
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if args.use_gui:
        args = pick_with_gui(args)

    # Validate CLI paths
    if not args.excel or not Path(args.excel).is_file():
        logging.error("Invalid --excel path: %s", args.excel)
        return 2
    if not args.out:
        logging.error("Invalid --out path.")
        return 2
    if args.volume not in (1.0, 0.1, 0.01, 0.001):
        logging.error("--volume must be one of 1, 0.1, 0.01, 0.001 L")
        return 2

    df = process_excel(Path(args.excel), tumor_volume_L=args.volume, T_min=args.T)

    if df.empty:
        logging.warning("No valid data processed. No output file created.")
        return 0

    # Ensure .xlsx extension
    out_path = Path(args.out)
    if out_path.suffix.lower() != ".xlsx":
        out_path = out_path.with_suffix(".xlsx")

    df.to_excel(out_path, index=False)
    logging.info("Saved results to: %s", out_path)
    print(f"✅ Results saved at: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
