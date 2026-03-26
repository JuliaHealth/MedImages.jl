#!/usr/bin/env python3
"""
PBPK / Predictive Dosimetry ML Pipeline (Clean GitHub Version)
==============================================================

• Reads multiple Excel feature files (folder) and one target Excel file.
• Splits runs into two groups based on filename: "Reference" vs everything else ("Noise").
• Performs feature cleaning (drop sparse cols, remove highly correlated), feature selection (RFE / LASSO / RF importance),
  scaling, and nested cross‑validation for multiple regressors.
• Saves summary tables to Excel and (optionally) scatter plots to PNG.

Usage (CLI):
    python pbpk_ml_pipeline.py \
        --features_dir /path/to/features \
        --target_file /path/to/targets.xlsx \
        --out_dir /path/to/save \
        --plot                   # optional: save scatter plots
        --noise_limit 0          # optional: 0 means use all noise files; default 0
        --k_features 5           # optional: number of selected features; default 5
        --corr_thresh 0.90       # optional: correlation threshold; default 0.90
        --seed 42                # optional: random seed; default 42

Alternatively (GUI pickers):
    python pbpk_ml_pipeline.py --use_gui --plot

Author: (Your Name)
License: MIT
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Optional GUI imports (only used when --use_gui is provided)
try:  # don't hard‑require Tk for headless servers/CI
    from tkinter import Tk, filedialog  # type: ignore
except Exception:  # pragma: no cover
    Tk = None  # type: ignore
    filedialog = None  # type: ignore

warnings.filterwarnings("ignore")

# -------------------------------
# Config
# -------------------------------
ALL_METHODS: Sequence[str] = ("rfe", "lasso", "rf_importance")
ALL_MODELS: Dict[str, object] = {
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=None, random_state=0, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=400, max_depth=None, random_state=0, n_jobs=-1),
    "Ridge": Ridge(alpha=1.0, random_state=0),
    "SVR": "SVR_PLACEHOLDER",  # initialized later to avoid warnings about probability etc.
}
TARGETS: Sequence[str] = ("Dose", "BED_G", "EQD2", "AUC")
FORBIDDEN_COLS_BASE: Sequence[str] = (
    "ActivityColumn", "Case", "File", "Sheet", "Curve", "Organ", "G_T", "AUC_x", "AUC_y",
)

# -------------------------------
# Data structures
# -------------------------------
@dataclass
class CVResult:
    metrics_mean: Dict[str, float]
    metrics_std: Dict[str, float]
    top_features: List[str]
    y_true: List[float]
    y_pred: List[float]


# -------------------------------
# Helpers
# -------------------------------

def evaluate(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    """Compute standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def remove_correlated_features(X: pd.DataFrame, threshold: float = 0.90) -> Tuple[pd.DataFrame, List[str]]:
    """Remove columns that are highly correlated with others (absolute Pearson > threshold)."""
    if X.shape[1] <= 1:
        return X, []
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X.drop(columns=to_drop, errors="ignore"), to_drop


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    forbidden_features: Sequence[str],
    method: str = "rfe",
    k: int = 5,
    seed: int = 42,
) -> List[str]:
    """Select up to k features using the specified method, excluding forbidden columns."""
    cols = [c for c in X_train.columns if c not in forbidden_features]
    Xf = X_train[cols]
    if Xf.empty:
        return []

    k = max(1, min(k, Xf.shape[1]))

    if method == "rfe":
        base = LinearRegression()
        selector = RFE(base, n_features_to_select=k)
        selector.fit(Xf, y_train)
        return list(Xf.columns[selector.support_])

    if method == "lasso":
        lasso = LassoCV(cv=3, max_iter=2000, random_state=seed).fit(Xf, y_train)
        selector = SelectFromModel(lasso, prefit=True)
        selected = list(Xf.columns[selector.get_support()])
        if not selected:  # fallback: pick top |coef| from lasso
            coef = pd.Series(np.abs(lasso.coef_), index=Xf.columns)
            selected = list(coef.sort_values(ascending=False).head(k).index)
        return selected[:k]

    if method == "rf_importance":
        rf = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(Xf, y_train)
        importances = pd.Series(rf.feature_importances_, index=Xf.columns)
        return list(importances.nlargest(k).index)

    return []


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    feature_method: str,
    k_features: int = 5,
    forbidden_features: Sequence[str] = (),
    seed: int = 42,
) -> CVResult:
    """5‑fold outer CV. Inner selection is done by the selector itself; no param tuning here."""
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    metrics_all: List[Dict[str, float]] = []
    preds: List[float] = []
    truth: List[float] = []
    feats_runs: List[List[str]] = []

    for train_idx, test_idx in outer.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        selected = select_features(Xtr, ytr, forbidden_features, method=feature_method, k=k_features, seed=seed)
        if not selected:
            logging.warning("Feature selection returned 0 features; skipping fold.")
            continue

        feats_runs.append(selected)
        Xtr_sel, Xte_sel = Xtr[selected], Xte[selected]

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(Xtr_sel)
        Xte_sc = scaler.transform(Xte_sel)

        m = clone(model)
        m.fit(Xtr_sc, ytr)
        yhat = m.predict(Xte_sc)

        preds.extend(map(float, yhat))
        truth.extend(map(float, yte))
        metrics_all.append(evaluate(yte, yhat))

    if not metrics_all:
        return CVResult(metrics_mean={}, metrics_std={}, top_features=[], y_true=[], y_pred=[])

    dfm = pd.DataFrame(metrics_all)
    mean = dfm.mean(numeric_only=True).to_dict()
    std = dfm.std(numeric_only=True).to_dict()

    # Most frequent features across folds (tie‑broken by first appearance order via pandas value_counts)
    most_common = (
        pd.Series([f for sub in feats_runs for f in sub])
        .value_counts()
        .head(k_features)
        .index
        .tolist()
    )

    return CVResult(metrics_mean=mean, metrics_std=std, top_features=most_common, y_true=truth, y_pred=preds)


# -------------------------------
# Core pipeline
# -------------------------------

def process_file(
    feature_path: Path,
    target_df: pd.DataFrame,
    forbidden_cols: Sequence[str],
    targets: Sequence[str],
    models: Dict[str, object],
    methods: Sequence[str],
    k_features: int,
    corr_thresh: float,
    seed: int,
    save_plots: bool,
    out_dir: Path,
    group: str,
    run_idx: int,
) -> List[Dict[str, object]]:
    """Process a single feature Excel file across targets/models/methods."""
    results: List[Dict[str, object]] = []
    df = pd.read_excel(feature_path)

    # Merge on ActivityColumn ↔ Case (only if both columns exist)
    if "ActivityColumn" in df.columns and "Case" in target_df.columns:
        merged = df.merge(target_df, left_on="ActivityColumn", right_on="Case", how="inner")
    else:
        logging.warning("Expected keys not found; using raw feature DF aligned rows with target DF.")
        merged = pd.concat([df.reset_index(drop=True), target_df.reset_index(drop=True)], axis=1)

    # Drop forbidden
    drop_cols = [c for c in forbidden_cols if c in merged.columns]
    X = merged.drop(columns=drop_cols, errors="ignore")

    # Basic cleaning: drop columns with too many NaNs and drop rows with any remaining NaNs
    X = X.dropna(axis=1, thresh=int(0.8 * len(X)))
    X = X.dropna(axis=0)

    # Remove highly correlated features
    X, dropped_corr = remove_correlated_features(X, threshold=corr_thresh)
    if dropped_corr:
        logging.info("Removed %d highly correlated features.", len(dropped_corr))

    for target_col in targets:
        if target_col not in merged.columns:
            continue
        # Keep alignment with X after row drops
        y = merged.loc[X.index, target_col]

        for model_name, model in models.items():
            if model_name == "SVR":
                from sklearn.svm import SVR  # local import to avoid mypy confusion
                model = SVR(C=1.0, epsilon=0.2)

            for method in methods:
                tag = f"{group} | Run {run_idx} | Target: {target_col}, Model: {model_name}, Method: {method}"
                logging.info(tag)

                try:
                    cvres = nested_cv(
                        X=X,
                        y=y,
                        model=model,
                        feature_method=method,
                        k_features=k_features,
                        forbidden_features=FORBIDDEN_COLS_BASE + tuple(targets),
                        seed=seed,
                    )
                    if not cvres.metrics_mean:
                        logging.warning("No metrics computed for %s", tag)
                        continue

                    row = {
                        "Group": group,
                        "Run": run_idx,
                        "Model": model_name,
                        "Target": target_col,
                        "FeatureMethod": method,
                        "FeatureFile": feature_path.name,
                        "Features": ", ".join(cvres.top_features),
                        **{f"{k}": v for k, v in cvres.metrics_mean.items()},
                        **{f"{k}_STD": v for k, v in cvres.metrics_std.items()},
                    }
                    results.append(row)

                    if save_plots and cvres.y_true and cvres.y_pred:
                        plt.figure()
                        plt.scatter(cvres.y_true, cvres.y_pred, alpha=0.6)
                        plt.xlabel("True")
                        plt.ylabel("Predicted")
                        plt.title(f"{group} - {model_name} - {target_col}")
                        plt.grid(True)
                        fname = f"{group.lower()}_scatter_{target_col}_{model_name}_run{run_idx}.png"
                        plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
                        plt.close()

                except Exception as e:  # keep going on single‑run failures
                    logging.exception("Failure in %s: %s", tag, e)

    return results


def composite_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a composite score (lower is better) across metrics for quick model picking."""
    work = df.copy()
    for col in ("MAE", "MSE", "RMSE", "MedAE", "MAPE"):
        if col in work:
            work[f"rank_{col}"] = work[col].rank(method="min")
    if "R2" in work:
        work["rank_R2"] = work["R2"].rank(ascending=False, method="min")
    rank_cols = [c for c in work.columns if c.startswith("rank_")]
    if rank_cols:
        work["CompositeScore"] = work[rank_cols].sum(axis=1)
    return work


def save_results_excel(df_all: pd.DataFrame, out_path: Path) -> None:
    """Save per‑target tabs and best models per group."""
    targets_present = [t for t in TARGETS if t in df_all["Target"].unique()]

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        # Per‑target sheets
        for t in targets_present:
            sub = df_all[df_all["Target"] == t]
            if not sub.empty:
                sub.to_excel(writer, sheet_name=t[:31], index=False)

        # Best per group/target using composite rank
        best_ref, best_noise = [], []
        for t in targets_present:
            for g in ("Reference", "Noise"):
                sub = df_all[(df_all["Target"] == t) & (df_all["Group"] == g)]
                if sub.empty:
                    continue
                ranked = composite_rank(sub)
                if "CompositeScore" in ranked:
                    best_row = ranked.loc[ranked["CompositeScore"].idxmin()]
                    (best_ref if g == "Reference" else best_noise).append(best_row.to_dict())

        if best_ref:
            pd.DataFrame(best_ref).to_excel(writer, sheet_name="BestModels_Reference", index=False)
        if best_noise:
            pd.DataFrame(best_noise).to_excel(writer, sheet_name="BestModels_Noise", index=False)


# -------------------------------
# CLI / Main
# -------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PBPK / Predictive Dosimetry ML Pipeline")
    p.add_argument("--features_dir", type=Path, help="Folder with feature Excel files (.xlsx)")
    p.add_argument("--target_file", type=Path, help="Target Excel file (.xlsx)")
    p.add_argument("--out_dir", type=Path, help="Output folder for results")
    p.add_argument("--plot", action="store_true", help="Save scatter plots")
    p.add_argument("--noise_limit", type=int, default=0, help="Use first N noise files (0=all)")
    p.add_argument("--k_features", type=int, default=5, help="Number of selected features per run")
    p.add_argument("--corr_thresh", type=float, default=0.90, help="Correlation removal threshold [0..1]")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--use_gui", action="store_true", help="Pick paths with GUI dialogs (Tkinter)")
    p.add_argument("--log_level", default="INFO", help="Logging level (e.g., INFO, DEBUG)")
    return p.parse_args(argv)


def pick_paths_with_gui(args: argparse.Namespace) -> argparse.Namespace:
    if Tk is None or filedialog is None:
        logging.error("Tkinter is not available in this environment.")
        sys.exit(2)
    root = Tk(); root.withdraw()
    if not args.features_dir:
        print("📁 Select folder with feature Excel files...")
        args.features_dir = Path(filedialog.askdirectory())
    if not args.target_file:
        print("📄 Select the target Excel file...")
        args.target_file = Path(filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")]))
    if not args.out_dir:
        print("💾 Select folder to save results...")
        args.out_dir = Path(filedialog.askdirectory())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if args.use_gui:
        args = pick_paths_with_gui(args)

    # Validate paths
    if not args.features_dir or not Path(args.features_dir).is_dir():
        logging.error("Invalid --features_dir: %s", args.features_dir)
        return 2
    if not args.target_file or not Path(args.target_file).is_file():
        logging.error("Invalid --target_file: %s", args.target_file)
        return 2
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load targets
    target_df = pd.read_excel(args.target_file)

    # Enumerate feature files
    feature_files = [p for p in Path(args.features_dir).glob("*.xlsx")]
    if not feature_files:
        logging.error("No .xlsx files found under %s", args.features_dir)
        return 2

    # Partition into Reference vs Noise
    reference_files = [p for p in feature_files if "Reference" in p.name]
    noise_files = [p for p in feature_files if p not in reference_files]
    if args.noise_limit and args.noise_limit > 0:
        noise_files = noise_files[: args.noise_limit]

    logging.info("Found %d Reference and %d Noise files", len(reference_files), len(noise_files))

    forbidden_cols = list(FORBIDDEN_COLS_BASE) + list(TARGETS)

    all_rows: List[Dict[str, object]] = []

    # Process Reference group (run_idx=0)
    for f in reference_files:
        all_rows.extend(
            process_file(
                feature_path=f,
                target_df=target_df,
                forbidden_cols=forbidden_cols,
                targets=TARGETS,
                models=ALL_MODELS,
                methods=ALL_METHODS,
                k_features=args.k_features,
                corr_thresh=args.corr_thresh,
                seed=args.seed,
                save_plots=args.plot,
                out_dir=Path(args.out_dir),
                group="Reference",
                run_idx=0,
            )
        )

    # Process Noise group (run_idx increments)
    for idx, f in enumerate(noise_files, start=1):
        all_rows.extend(
            process_file(
                feature_path=f,
                target_df=target_df,
                forbidden_cols=forbidden_cols,
                targets=TARGETS,
                models=ALL_MODELS,
                methods=ALL_METHODS,
                k_features=args.k_features,
                corr_thresh=args.corr_thresh,
                seed=args.seed,
                save_plots=args.plot,
                out_dir=Path(args.out_dir),
                group="Noise",
                run_idx=idx,
            )
        )

    if not all_rows:
        logging.warning("No results computed.")
        return 0

    df_all = pd.DataFrame(all_rows)

    # Save combined and Excel breakdown
    csv_path = Path(args.out_dir) / "all_model_results.csv"
    xlsx_path = Path(args.out_dir) / "all_model_results_by_target.xlsx"
    df_all.to_csv(csv_path, index=False)
    save_results_excel(df_all, xlsx_path)

    logging.info("Saved: %s", csv_path)
    logging.info("Saved: %s", xlsx_path)
    print("✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
