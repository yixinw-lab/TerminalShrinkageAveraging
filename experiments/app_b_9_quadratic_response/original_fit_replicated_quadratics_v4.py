#!/usr/bin/env python3
"""Fit g(alpha)=b*alpha+c*alpha^2 to replicated TSA shrinkage curves.

The script is deliberately conservative:
  * it excludes development/calibration paths by default;
  * it keeps only the documented dense scalar grid alpha={0,.05,...,1};
  * it requires alpha=0 for every optimizer/floor/seed so gains are paired to raw;
  * it requires the expected number of independent seeds before writing paper outputs.

It can consume either:
  1. a tidy CSV with columns optimizer,floor,seed,alpha,bpb; or
  2. a D12 result-root directory, which it scans recursively for compatible CSVs.

Examples
--------
python fit_replicated_quadratics.py \
  --root "$STRUCT_ROOT" \
  --output-dir "$STRUCT_ROOT/figures/quadratic_fits"

python fit_replicated_quadratics.py \
  --tidy-csv scalar_grid_confirmation.csv \
  --output-dir quadratic_fits
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "optimizer": ("optimizer", "optim", "optimizer_name", "optimizer_regime"),
    "floor": ("floor", "terminal_floor", "terminal_lr_floor", "lr_floor"),
    "seed": ("stream_seed", "experiment_seed", "seed", "data_seed"),
    "alpha": ("alpha", "scalar_alpha", "shrinkage_alpha", "tsa_alpha"),
    "alpha_group": ("alpha_group", "alpha_a", "alpha_matrix", "alpha_m"),
    "alpha_other": ("alpha_other", "alpha_b", "alpha_complement", "alpha_o"),
    "bpb": ("bpb", "val_bpb", "validation_bpb", "canonical_bpb", "mean_bpb"),
    "family": ("family", "estimator_family", "estimator", "candidate_family"),
    "split": ("split", "stage", "eval_split", "evaluation_split", "subset"),
}

DEFAULT_EXCLUDE_RE = re.compile(r"(?:^|/)(?:development|dev)(?:/|$)|calibrat", re.I)
DEFAULT_CONFIRM_RE = re.compile(r"confirm|holdout", re.I)
EXPECTED_ALPHA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 10)


@dataclass(frozen=True)
class FitRow:
    optimizer: str
    floor_pct: float
    b: float
    c: float
    best_tested_alpha: float
    fitted_vertex: float
    fitted_vertex_clipped: float
    r2: float
    n_seeds: int
    n_alpha: int


def pick_column(columns: Iterable[str], canonical: str) -> Optional[str]:
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in COLUMN_ALIASES[canonical]:
        if alias in lookup:
            return lookup[alias]
    return None


def normalize_optimizer(value: object, path: str = "") -> Optional[str]:
    text = f"{value if value is not None else ''} {path}".lower()
    if "pure_adamw" in text or "pure-adamw" in text or re.search(r"\bpure\s+adamw\b", text):
        return "Pure AdamW"
    if "muon+adamw" in text or "muon_adamw" in text or "native" in text:
        return "Muon+AdamW"
    if re.search(r"\badamw\b", text) and "muon" not in text:
        return "Pure AdamW"
    if "muon" in text:
        return "Muon+AdamW"
    return None


def normalize_floor(value: object, path: str = "") -> Optional[float]:
    candidates = []
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        candidates.append(str(value))
    candidates.append(path)
    for candidate_index, text in enumerate(candidates):
        s = text.lower().strip()
        # Explicit percent form, e.g. 5%, floor_15pct.
        m = re.search(r"(?:floor[_\- ]*)?(\d+(?:\.\d+)?)\s*(?:%|pct)", s)
        if m:
            return float(m.group(1))
        # Decimal form in a column or path, e.g. floor_0.05.
        m = re.search(r"(?:^|floor[_\- ]*)(0\.\d+|1\.0+)(?:$|[^0-9])", s)
        if m:
            x = float(m.group(1))
            return 100.0 * x if x <= 1.0 else x
        # Bare numeric column value.
        if candidate_index == 0:
            try:
                x = float(s)
                return 100.0 * x if x <= 1.0 else x
            except ValueError:
                pass
        # Common directory names floor_5, floor_10, floor_15.
        m = re.search(r"floor[_\- ]*(5|10|15)(?:$|[^0-9])", s)
        if m:
            return float(m.group(1))
    return None


def infer_seed(value: object, path: str = "") -> Optional[int]:
    if value is not None and not (isinstance(value, float) and math.isnan(value)):
        try:
            return int(float(str(value)))
        except ValueError:
            pass
    m = re.search(r"seed[_\- ]*(\d+)", path.lower())
    return int(m.group(1)) if m else None


def read_candidate_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    cols = list(header.columns)
    bpb_col = pick_column(cols, "bpb")
    alpha_col = pick_column(cols, "alpha")
    ag_col = pick_column(cols, "alpha_group")
    ao_col = pick_column(cols, "alpha_other")
    if bpb_col is None or (alpha_col is None and (ag_col is None or ao_col is None)):
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}: {exc}", file=sys.stderr)
        return None


def standardize_frame(df: pd.DataFrame, path: Path, require_confirmation: bool) -> pd.DataFrame:
    cols = list(df.columns)
    opt_col = pick_column(cols, "optimizer")
    floor_col = pick_column(cols, "floor")
    seed_col = pick_column(cols, "seed")
    alpha_col = pick_column(cols, "alpha")
    ag_col = pick_column(cols, "alpha_group")
    ao_col = pick_column(cols, "alpha_other")
    bpb_col = pick_column(cols, "bpb")
    family_col = pick_column(cols, "family")
    split_col = pick_column(cols, "split")

    if bpb_col is None:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["optimizer"] = [
        normalize_optimizer(v, str(path)) for v in (df[opt_col] if opt_col else [None] * len(df))
    ]
    out["floor_pct"] = [
        normalize_floor(v, str(path)) for v in (df[floor_col] if floor_col else [None] * len(df))
    ]
    out["seed"] = [
        infer_seed(v, str(path)) for v in (df[seed_col] if seed_col else [None] * len(df))
    ]

    if alpha_col:
        out["alpha"] = pd.to_numeric(df[alpha_col], errors="coerce")
        out["diagonal"] = True
    else:
        ag = pd.to_numeric(df[ag_col], errors="coerce")
        ao = pd.to_numeric(df[ao_col], errors="coerce")
        out["alpha"] = ag
        out["diagonal"] = np.isclose(ag, ao, atol=1e-10, rtol=0)

    out["bpb"] = pd.to_numeric(df[bpb_col], errors="coerce")
    out["source"] = str(path)

    if family_col:
        fam = df[family_col].astype(str).str.strip().str.lower()
        # Paper curves use the *dense scalar grid* only.  Do not use a substring
        # match here: confirmation files can also contain structured-grid diagonal
        # diagnostics at alpha={0,.125,...,1}.  Mixing those with the 0.05 scalar
        # grid produces 25 unique alpha values and changes the apparent grid optimum
        # (e.g. 0.375 instead of the paper's 0.40 at the native 5% floor).
        scalar_mask = fam.eq("scalar_grid")
        if scalar_mask.any():
            out = out[scalar_mask]
        else:
            # A file with an explicit family column but no scalar_grid rows is not a
            # source for the paper's scalar response.
            return pd.DataFrame()
    else:
        # Fallback only for tidy/legacy files without a family column.
        out = out[out["diagonal"]]

    if split_col and require_confirmation:
        split = df.loc[out.index, split_col].astype(str).str.lower()
        confirm_mask = split.str.contains("confirm|holdout", regex=True)
        if confirm_mask.any():
            out = out[confirm_mask]

    return out.dropna(subset=["optimizer", "floor_pct", "seed", "alpha", "bpb"])


def load_from_root(root: Path, include_regex: Optional[str], require_confirmation: bool) -> tuple[pd.DataFrame, list[Path]]:
    include_re = re.compile(include_regex, re.I) if include_regex else None
    frames: list[pd.DataFrame] = []
    used: list[Path] = []
    for path in sorted(root.rglob("*.csv")):
        pstr = str(path)
        if DEFAULT_EXCLUDE_RE.search(pstr):
            continue
        if include_re and not include_re.search(pstr):
            continue
        df = read_candidate_csv(path)
        if df is None:
            continue
        std = standardize_frame(df, path, require_confirmation=require_confirmation)
        if std.empty:
            continue
        frames.append(std)
        used.append(path)

    if not frames and include_regex is None:
        # Second pass: some suites call the confirmation stage "evaluation" without
        # putting "confirmation" in the path. We still exclude development/calibration.
        for path in sorted(root.rglob("*.csv")):
            pstr = str(path)
            if DEFAULT_EXCLUDE_RE.search(pstr):
                continue
            df = read_candidate_csv(path)
            if df is None:
                continue
            std = standardize_frame(df, path, require_confirmation=require_confirmation)
            if std.empty:
                continue
            frames.append(std)
            used.append(path)

    if not frames:
        raise RuntimeError(
            "No compatible scalar-grid CSV rows found. Run with --list-candidates, "
            "or provide --tidy-csv with columns optimizer,floor,seed,alpha,bpb."
        )
    return pd.concat(frames, ignore_index=True), used


def load_tidy(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"optimizer", "floor", "seed", "alpha", "bpb"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tidy CSV is missing columns: {sorted(missing)}")
    out = pd.DataFrame(
        {
            "optimizer": [normalize_optimizer(x) for x in df["optimizer"]],
            "floor_pct": [normalize_floor(x) for x in df["floor"]],
            "seed": [infer_seed(x) for x in df["seed"]],
            "alpha": pd.to_numeric(df["alpha"], errors="coerce"),
            "bpb": pd.to_numeric(df["bpb"], errors="coerce"),
            "source": str(path),
        }
    )
    return out.dropna()



def retain_expected_scalar_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the paper's documented dense scalar grid {0,.05,...,1}.

    Some holdout_results.csv files contain extra scalar-like diagnostic rows at
    eighth-grid values such as .125/.375/.625/.875.  Those rows are not part of
    the post-freeze dense scalar grid reported in the paper, so exclude them
    explicitly rather than relying on the family label alone.
    """
    out = df.copy()
    a = out["alpha"].to_numpy(dtype=float)
    keep = np.zeros(len(out), dtype=bool)
    for x in EXPECTED_ALPHA_GRID:
        keep |= np.isclose(a, x, atol=1e-10, rtol=0)
    out = out.loc[keep].copy()
    # Snap tiny floating errors exactly onto the documented grid.
    snapped = []
    for x in out["alpha"].to_numpy(dtype=float):
        idx = int(np.argmin(np.abs(EXPECTED_ALPHA_GRID - x)))
        snapped.append(float(EXPECTED_ALPHA_GRID[idx]))
    out["alpha"] = snapped
    return out


def collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    key = ["optimizer", "floor_pct", "seed", "alpha"]
    grouped = df.groupby(key, as_index=False).agg(
        bpb=("bpb", "mean"),
        bpb_min=("bpb", "min"),
        bpb_max=("bpb", "max"),
        n_rows=("bpb", "size"),
    )

    # Duplicates are acceptable only if they are numerically identical.  Do not
    # silently average distinct evaluations, because that would change the paper
    # curve.
    conflict = grouped[
        (grouped["n_rows"] > 1)
        & ((grouped["bpb_max"] - grouped["bpb_min"]).abs() > 1e-12)
    ]
    if not conflict.empty:
        raise RuntimeError(
            "Non-identical duplicate scalar-grid rows remain after restricting "
            "to {0,.05,...,1}. Inspect these keys:\n"
            + conflict[key + ["n_rows", "bpb_min", "bpb_max"]].to_string(index=False)
        )

    return grouped[key + ["bpb"]]


def paired_gain_table(df: pd.DataFrame) -> pd.DataFrame:
    key = ["optimizer", "floor_pct", "seed"]
    raw = (
        df[np.isclose(df["alpha"], 0.0)]
        .groupby(key, as_index=False)["bpb"]
        .mean()
        .rename(columns={"bpb": "raw_bpb"})
    )
    out = df.merge(raw, on=key, how="left", validate="many_to_one")
    if out["raw_bpb"].isna().any():
        bad = out[out["raw_bpb"].isna()][key].drop_duplicates()
        raise RuntimeError(f"Missing alpha=0 raw row for:\n{bad.to_string(index=False)}")
    out["gain_vs_raw"] = out["raw_bpb"] - out["bpb"]
    return out


def fit_curve(alpha: np.ndarray, gain: np.ndarray) -> tuple[float, float, float, float, float]:
    x = np.asarray(alpha, dtype=float)
    y = np.asarray(gain, dtype=float)
    design = np.column_stack([x, x * x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    b, c = map(float, coef)
    pred = design @ coef
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if sst <= 0 else 1.0 - sse / sst
    vertex = float("nan") if abs(c) < 1e-18 else -b / (2.0 * c)
    clipped = float(np.clip(vertex, 0.0, 1.0)) if np.isfinite(vertex) else float("nan")
    return b, c, vertex, clipped, r2


def tcrit95(n: int) -> float:
    lookup = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    return lookup.get(n, 1.96)


def summarize(gains: pd.DataFrame, expected_seeds: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    curve = (
        gains.groupby(["optimizer", "floor_pct", "alpha"], as_index=False)
        .agg(mean_gain=("gain_vs_raw", "mean"), sd_gain=("gain_vs_raw", "std"), n=("seed", "nunique"))
        .sort_values(["optimizer", "floor_pct", "alpha"])
    )
    curve["ci95_halfwidth"] = [
        tcrit95(int(n)) * (sd / math.sqrt(n)) if n > 1 and np.isfinite(sd) else 0.0
        for sd, n in zip(curve["sd_gain"], curve["n"])
    ]

    fits: list[FitRow] = []
    per_seed_rows: list[dict[str, object]] = []
    for (optimizer, floor_pct), block in gains.groupby(["optimizer", "floor_pct"]):
        n_seeds = int(block["seed"].nunique())
        if n_seeds != expected_seeds:
            raise RuntimeError(
                f"Expected {expected_seeds} seeds for {optimizer}, floor={floor_pct:g}%, found {n_seeds}. "
                "This usually means development rows, incomplete confirmation outputs, or duplicate summaries were mixed in."
            )
        mean_block = curve[(curve["optimizer"] == optimizer) & np.isclose(curve["floor_pct"], floor_pct)]

        # The paper's post-freeze scalar response is alpha in {0, .05, ..., 1}:
        # exactly 21 points.  Fail loudly if structured-diagonal or selected-rule
        # rows have leaked into the curve.
        expected_alpha = EXPECTED_ALPHA_GRID
        actual_alpha = np.round(np.sort(mean_block["alpha"].unique()), 10)
        if len(actual_alpha) != len(expected_alpha) or not np.allclose(actual_alpha, expected_alpha, atol=1e-10, rtol=0):
            raise RuntimeError(
                f"Unexpected alpha grid for {optimizer}, floor={floor_pct:g}%: "
                f"found {len(actual_alpha)} points {actual_alpha.tolist()}. "
                "Expected the 21-point scalar_grid {0,.05,...,1}."
            )
        b, c, vertex, clipped, r2 = fit_curve(mean_block["alpha"].to_numpy(), mean_block["mean_gain"].to_numpy())
        best_idx = mean_block["mean_gain"].idxmax()
        best_alpha = float(mean_block.loc[best_idx, "alpha"])
        fits.append(
            FitRow(
                optimizer=str(optimizer),
                floor_pct=float(floor_pct),
                b=b,
                c=c,
                best_tested_alpha=best_alpha,
                fitted_vertex=vertex,
                fitted_vertex_clipped=clipped,
                r2=r2,
                n_seeds=n_seeds,
                n_alpha=int(mean_block["alpha"].nunique()),
            )
        )

        for seed, seed_block in block.groupby("seed"):
            b_s, c_s, v_s, vc_s, r2_s = fit_curve(seed_block["alpha"].to_numpy(), seed_block["gain_vs_raw"].to_numpy())
            per_seed_rows.append(
                {
                    "optimizer": optimizer,
                    "floor_pct": floor_pct,
                    "seed": int(seed),
                    "b": b_s,
                    "c": c_s,
                    "fitted_vertex": v_s,
                    "fitted_vertex_clipped": vc_s,
                    "r2": r2_s,
                }
            )

    fit_df = pd.DataFrame([row.__dict__ for row in fits]).sort_values(["optimizer", "floor_pct"])
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["optimizer", "floor_pct", "seed"])
    return curve, fit_df, per_seed_df


def write_latex_table(fit_df: pd.DataFrame, path: Path) -> None:
    order = ["Muon+AdamW", "Pure AdamW"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Quadratic fits to the replicated shrinkage responses.}",
        r"For each optimizer and terminal floor, we fit",
        r"$g(\alpha)=b\alpha+c\alpha^2$ with $g(0)=0$ to the mean paired BPB",
        r"improvement over raw. The fitted vertex is",
        r"$\widehat{\alpha}^{\star}=-b/(2c)$.}",
        r"\label{tab:app-quadratic-fits}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Optimizer & Floor & Best tested $\alpha$ & $\widehat{\alpha}^{\star}$ & $R^2$ \\",
        r"\midrule",
    ]
    first_opt = True
    for opt in order:
        block = fit_df[fit_df["optimizer"] == opt].sort_values("floor_pct")
        if block.empty:
            continue
        if not first_opt:
            lines.append(r"\midrule")
        first_opt = False
        for j, row in enumerate(block.itertuples(index=False)):
            opt_cell = opt if j == 0 else ""
            lines.append(
                f"{opt_cell} & {row.floor_pct:g}\\% & {row.best_tested_alpha:.2f} & "
                f"{row.fitted_vertex:.3f} & {row.r2:.5f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plot(curve: pd.DataFrame, fit_df: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping PDF figure", file=sys.stderr)
        return

    optimizers = [x for x in ["Muon+AdamW", "Pure AdamW"] if x in set(curve["optimizer"])]
    floors = [5.0, 10.0, 15.0]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.1), sharex=True)
    dense = np.linspace(0.0, 1.0, 501)

    for i, opt in enumerate(optimizers):
        for j, floor in enumerate(floors):
            ax = axes[i, j]
            block = curve[(curve["optimizer"] == opt) & np.isclose(curve["floor_pct"], floor)].copy()
            fit = fit_df[(fit_df["optimizer"] == opt) & np.isclose(fit_df["floor_pct"], floor)].iloc[0]

            ax.errorbar(
                block["alpha"], block["mean_gain"], yerr=block["ci95_halfwidth"],
                fmt="o", markersize=2.8, linewidth=0.8, capsize=1.5,
                label="Observed mean ± 95% t-CI",
            )
            ax.plot(
                dense, fit["b"] * dense + fit["c"] * dense**2,
                linewidth=1.25, label="Quadratic fit",
            )
            # Mark the actual tested-grid optimum and the continuous fitted vertex.
            best_row = block.loc[block["mean_gain"].idxmax()]
            ax.plot(
                [best_row["alpha"]], [best_row["mean_gain"]],
                marker="*", markersize=8, linestyle="None", label="Best tested α",
            )
            ax.axvline(fit["fitted_vertex_clipped"], linestyle="--", linewidth=0.8)
            ax.axhline(0.0, linewidth=0.7)

            ax.set_title(f"{opt} · {floor:g}% floor")
            ax.text(
                0.04, 0.94,
                f"grid={fit['best_tested_alpha']:.2f}\nfit={fit['fitted_vertex']:.3f}\n$R^2$={fit['r2']:.5f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
            )
            if i == 1:
                ax.set_xlabel(r"Shrinkage coefficient $\alpha$")
            if j == 0:
                ax.set_ylabel("BPB improvement over raw")

    # One shared legend, deduplicated.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab); h2.append(h); l2.append(lab)
    # Keep the legend away from panel titles.  The LaTeX caption already
    # supplies the figure-level title, so do not add a suptitle here.
    fig.legend(
        h2, l2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    # Explicit spacing is more reliable than tight_layout for a figure-level
    # legend.  Leave a dedicated strip at the bottom for the shared legend.
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.955,
        bottom=0.155,
        wspace=0.34,
        hspace=0.34,
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def list_candidates(root: Path) -> None:
    for path in sorted(root.rglob("*.csv")):
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if pick_column(cols, "bpb") and (pick_column(cols, "alpha") or pick_column(cols, "alpha_group")):
            print(f"{path}\n  columns={cols}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", type=Path, help="D12 result root to scan recursively")
    source.add_argument("--tidy-csv", type=Path, help="CSV with optimizer,floor,seed,alpha,bpb")
    p.add_argument("--output-dir", type=Path, default=Path("quadratic_fits"))
    p.add_argument("--expected-seeds", type=int, default=5)
    p.add_argument(
        "--include-regex",
        default=None,
        help="Optional regex that candidate CSV paths must match (for example 'confirmation|holdout')",
    )
    p.add_argument(
        "--allow-nonconfirmation",
        action="store_true",
        help="Disable confirmation/holdout safeguards. Do not use this for the paper table unless intentional.",
    )
    p.add_argument("--list-candidates", action="store_true", help="Print compatible CSV headers and exit")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_candidates:
        if args.root is None:
            raise SystemExit("--list-candidates requires --root")
        list_candidates(args.root)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.tidy_csv:
        raw = load_tidy(args.tidy_csv)
        used = [args.tidy_csv]
    else:
        raw, used = load_from_root(
            args.root,
            include_regex=args.include_regex,
            require_confirmation=not args.allow_nonconfirmation,
        )

    raw = retain_expected_scalar_grid(raw)
    collapsed = collapse_duplicates(raw)
    gains = paired_gain_table(collapsed)
    curve, fit_df, per_seed_df = summarize(gains, expected_seeds=args.expected_seeds)

    collapsed.to_csv(args.output_dir / "extracted_scalar_grid.csv", index=False)
    gains.to_csv(args.output_dir / "paired_scalar_grid.csv", index=False)
    curve.to_csv(args.output_dir / "quadratic_curve_summary.csv", index=False)
    fit_df.to_csv(args.output_dir / "quadratic_fit_summary.csv", index=False)
    per_seed_df.to_csv(args.output_dir / "quadratic_fit_per_seed.csv", index=False)
    (args.output_dir / "fit_sources.txt").write_text("\n".join(map(str, used)) + "\n", encoding="utf-8")
    write_latex_table(fit_df, args.output_dir / "tab_quadratic_fits.tex")
    write_plot(curve, fit_df, args.output_dir / "fig_quadratic_fits.pdf")

    print("\n=== PAPER FIT SUMMARY ===")
    print(
        fit_df[
            ["optimizer", "floor_pct", "best_tested_alpha", "fitted_vertex", "r2", "n_seeds", "n_alpha"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )
    print(f"\nWrote outputs to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
