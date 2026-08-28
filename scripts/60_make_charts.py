#!/usr/bin/env python3
"""
Generate reporting charts for the Armenia liquidity stress-testing project.

The script discovers available CSV outputs and produces charts for:

    baseline full-engine results
    bank-level liquidity paths
    CBA support and LoLR mechanics
    loss decomposition
    breach counts
    policy sensitivity
    Monte Carlo tail-risk diagnostics
    currency-aware stress outputs, if available

The chart generator is defensive: if an optional dataset is missing, the
corresponding chart is skipped with a clear warning instead of failing.

Run from project root:

    PYTHONPATH=src python scripts/60_make_charts.py
"""

from __future__ import annotations

import math
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =============================================================================
# Paper typography scaling
# =============================================================================
# LaTeX shrinks figures inside subfigures, so chart text must be larger
# inside the exported image itself. Change FONT_SCALE only if needed:
#   1.40 = moderate
#   1.60 = good for side-by-side paper figures
#   1.80 = very large / presentation-like
FONT_SCALE = 1.60
MIN_FONT_SIZE = 12.0

from matplotlib.text import Text as _MplText

if not getattr(_MplText, "_paper_font_scaling_applied", False):
    _ORIG_SET_FONTSIZE = _MplText.set_fontsize

    def _paper_scaled_set_fontsize(self, size):
        if isinstance(size, (int, float, np.integer, np.floating)):
            size = max(float(size) * FONT_SCALE, MIN_FONT_SIZE)
        return _ORIG_SET_FONTSIZE(self, size)

    _MplText.set_fontsize = _paper_scaled_set_fontsize
    _MplText._paper_font_scaling_applied = True



# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()

SEARCH_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "output",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "result",
    PROJECT_ROOT / "artifacts",
    PROJECT_ROOT / "reports",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "outputs",
    PROJECT_ROOT / "data" / "results",
]

CHART_DIR = PROJECT_ROOT / "outputs" / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300
FIGSIZE_WIDE = (14.5, 8.0)
FIGSIZE_MEDIUM = (12.5, 7.0)
FIGSIZE_TALL = (13.5, 10.0)
FIGSIZE_GRID = (15.5, 10.5)

# Palette chosen to match the charts you already liked.
NAVY = "#06183D"
BLUE = "#1F5A82"
LIGHT_BLUE = "#74A6BF"
BURGUNDY = "#8C0030"
GOLD = "#D8A719"
OLIVE = "#4A5D23"
GREY = "#9E9E9E"
DARK_GREY = "#4A4A4A"
BG = "#E6E6E6"
GRID = "#D6D6D6"

STRESS_ORDER = ["mild", "moderate", "adverse", "severe"]

BANK_LABEL_OVERRIDES = {
    "Ameria Bank": "Ameriabank",
    "Ameriabank CJSC": "Ameriabank",
    "ArmEconomBank": "ArmEconomBank",
    "ArmEconombank": "ArmEconomBank",
    "Ardshin Bank": "Ardshinbank",
    "Ardshinbank": "Ardshinbank",
    "FastBank": "Fast Bank",
    "Fast Bank": "Fast Bank",
}


# =============================================================================
# Utility helpers
# =============================================================================


def log(msg: str) -> None:
    print(f"[60_make_charts] {msg}")


def clean_col(name: str) -> str:
    return str(name).strip().replace(" ", "_").replace("-", "_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_col(c) for c in out.columns]
    return out


def normalize_bank_name(x: object) -> str:
    s = str(x).strip()
    return BANK_LABEL_OVERRIDES.get(s, s)


def safe_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a Series to numeric without using deprecated/unsupported errors='ignore'."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s, errors="coerce")


def numericize(df: pd.DataFrame, exclude: Sequence[str] = ("bank_name", "bank_id", "stress_level", "scenario", "scenario_name", "run_id", "simulation_id")) -> pd.DataFrame:
    out = df.copy()
    exclude_set = set(exclude)
    for c in out.columns:
        if c not in exclude_set:
            converted = safe_numeric_series(out[c])
            # Keep original if conversion destroys everything and original had meaningful strings.
            if converted.notna().sum() > 0 or out[c].notna().sum() == 0:
                out[c] = converted
    return out


def boolish(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0) > 0
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def require_cols(df: Optional[pd.DataFrame], cols: Sequence[str], label: str) -> bool:
    if df is None:
        log(f"SKIP {label}: dataset missing")
        return False
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log(f"SKIP {label}: missing columns {missing}")
        return False
    return True


def short_label(s: str, max_len: int = 24) -> str:
    s = str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def wrap_label(s: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))


def format_k(x: float, pos: Optional[int] = None) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax >= 1_000_000:
        return f"{x/1_000_000:.1f}m"
    if ax >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


def format_pct(x: float, pos: Optional[int] = None) -> str:
    return f"{100*x:.0f}%"


def setup_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.85, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)


def savefig(fig: plt.Figure, filename: str) -> None:
    path = CHART_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"saved {path.relative_to(PROJECT_ROOT)}")


def discover_csvs() -> list[Path]:
    roots = [p for p in SEARCH_DIRS if p.exists()]
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        for p in root.rglob("*.csv"):
            rp = p.resolve()
            if rp not in seen and "charts" not in p.parts:
                seen.add(rp)
                out.append(p)
    return sorted(out)


@dataclass
class DataStore:
    csvs: list[Path]
    base_bank_monthly: Optional[pd.DataFrame] = None
    base_monthly: Optional[pd.DataFrame] = None
    policy_summary: Optional[pd.DataFrame] = None
    mc_runs: Optional[pd.DataFrame] = None
    mc_monthly: Optional[pd.DataFrame] = None
    mc_bank: Optional[pd.DataFrame] = None
    mc_quantiles: Optional[pd.DataFrame] = None

def canonical_base_slice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one deterministic base / combined / final slice before plotting.

    This prevents charts from accidentally mixing:
        - multiple policy scenarios,
        - AMD / FX / combined rows,
        - pre / post / final transmission stages,
        - multiple run_id or simulation_id values.
    """
    if df is None:
        return df

    out = df.copy()

    # 1. Prefer base / baseline scenario rows.
    scenario_cols = [
        "scenario",
        "scenario_name",
        "policy_scenario",
        "policy_name",
        "run_name",
        "case_name",
    ]

    for c in scenario_cols:
        if c in out.columns and out[c].nunique(dropna=True) > 1:
            s = out[c].astype(str).str.lower()

            preferred = (
                s.str.contains("base", regex=False)
                | s.str.contains("baseline", regex=False)
                | s.str.contains("full_engine", regex=False)
                | s.str.contains("cash-gap", regex=False)
                | s.str.contains("cash_gap", regex=False)
            )

            if preferred.any():
                out = out.loc[preferred].copy()
            else:
                chosen = out[c].value_counts(dropna=True).index[0]
                out = out.loc[out[c] == chosen].copy()

    # 2. Prefer combined / total currency rows.
    currency_cols = [
        "currency",
        "currency_mode",
        "currency_bucket",
        "currency_type",
        "currency_name",
        "currency_code",
        "currency_segment",
        "reporting_currency",
        "balance_currency",
        "liquidity_currency",
        "sheet",
        "sheet_name",
    ]

    for c in currency_cols:
        if c in out.columns and out[c].nunique(dropna=True) > 1:
            s = out[c].astype(str).str.lower()

            preferred = (
                s.str.contains("combined", regex=False)
                | s.str.contains("all", regex=False)
                | s.str.contains("total", regex=False)
                | s.str.contains("system", regex=False)
                | s.str.contains("aggregate", regex=False)
            )

            if preferred.any():
                out = out.loc[preferred].copy()
            else:
                chosen = out[c].value_counts(dropna=True).index[0]
                out = out.loc[out[c] == chosen].copy()

    # 3. Prefer final / post-transmission balance-sheet rows.
    stage_cols = [
        "stage",
        "step",
        "state",
        "snapshot",
        "record_type",
        "balance_sheet_stage",
        "transmission_stage",
        "measurement_stage",
        "phase",
    ]

    for c in stage_cols:
        if c in out.columns and out[c].nunique(dropna=True) > 1:
            s = out[c].astype(str).str.lower()

            preferred = (
                s.str.contains("final", regex=False)
                | s.str.contains("terminal", regex=False)
                | s.str.contains("post_interbank", regex=False)
                | s.str.contains("after_interbank", regex=False)
                | s.str.contains("post-transmission", regex=False)
                | s.str.contains("post_transmission", regex=False)
                | s.str.contains("after_policy", regex=False)
                | s.str.contains("post_policy", regex=False)
                | s.str.contains("end_of_month", regex=False)
            )

            if preferred.any():
                out = out.loc[preferred].copy()

    # 4. If accidental multiple deterministic runs are present, keep one.
    # Do not use this for Monte Carlo datasets.
    if "stress_level" not in out.columns:
        for c in ["run_id", "simulation_id"]:
            if c in out.columns and out[c].nunique(dropna=True) > 1:
                chosen = sorted(out[c].dropna().unique(), key=lambda z: str(z))[0]
                out = out.loc[out[c] == chosen].copy()

    return out


def _last_valid(s: pd.Series):
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[-1]


def _collapse_duplicate_keys(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Collapse accidental duplicate rows after filtering.

    For ratios and LCR columns, use the last valid value.
    For flow / need / loss columns, use max instead of sum to avoid double-counting
    repeated stage rows.
    """
    out = df.copy()

    missing = [k for k in keys if k not in out.columns]
    if missing:
        return out

    out["_row_order"] = np.arange(len(out))
    out = out.sort_values(keys + ["_row_order"]).copy()

    duplicated = out.duplicated(keys).sum()
    if duplicated == 0:
        return out.drop(columns=["_row_order"], errors="ignore")

    log(f"WARNING: {duplicated} duplicate rows found for keys {keys}. Collapsing before plotting.")

    agg = {}

    for c in out.columns:
        if c in keys:
            continue

        if c == "_row_order":
            agg[c] = "max"
            continue

        cl = c.lower()

        if pd.api.types.is_bool_dtype(out[c]):
            agg[c] = "max"

        elif pd.api.types.is_numeric_dtype(out[c]):
            if (
                "lcr" in cl
                or "ratio" in cl
                or "percent" in cl
                or "rate" in cl
            ):
                agg[c] = _last_valid

            elif any(
                token in cl
                for token in [
                    "support",
                    "shortfall",
                    "loss",
                    "proceeds",
                    "gap",
                    "outflow",
                    "inflow",
                    "need",
                ]
            ):
                agg[c] = "max"

            else:
                agg[c] = _last_valid

        else:
            agg[c] = _last_valid

    out = out.groupby(keys, as_index=False).agg(agg)
    out = out.drop(columns=["_row_order"], errors="ignore")
    return out


def one_row_per_bank_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce one row per bank-month for bank-level charts.
    This is the key fix for the vertical-line problem.
    """
    if df is None:
        return df

    out = canonical_base_slice(df)

    required = ["bank_name", "stress_month"]
    if not set(required).issubset(out.columns):
        return out

    out = _collapse_duplicate_keys(out, required)
    out = out.sort_values(required).copy()

    return out


def one_row_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce one row per stress month for system-level charts.
    """
    if df is None:
        return df

    out = canonical_base_slice(df)

    if "stress_month" not in out.columns:
        return out

    out = _collapse_duplicate_keys(out, ["stress_month"])
    out = out.sort_values("stress_month").copy()

    return out


# =============================================================================
# Baseline t=0 helpers for path charts
# =============================================================================

def _first_stress_month_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per bank from the earliest stress month."""
    out = one_row_per_bank_month(df.copy())
    month = get_month_col(out)
    if month is None or "bank_name" not in out.columns:
        return out.iloc[0:0].copy()
    m0 = pd.to_numeric(out[month], errors="coerce").min()
    out = out.loc[pd.to_numeric(out[month], errors="coerce") == m0].copy()
    out = _collapse_duplicate_keys(out, ["bank_name"])
    return out


def _first_existing_nonnull(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def baseline_lcr_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_nonnull(
        df,
        [
            "pre_shock_LCR",
            "baseline_LCR",
            "starting_LCR",
            "initial_LCR",
            "pre_stress_LCR",
            "LCR",
        ],
    )


def baseline_hqla_assets_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_nonnull(
        df,
        [
            "pre_shock_HQLA_assets_ratio",
            "baseline_HQLA_assets_ratio",
            "starting_HQLA_assets_ratio",
            "initial_HQLA_assets_ratio",
            "HQLA_assets_ratio",
            "post_interbank_HQLA_assets_ratio",
        ],
    )


def baseline_hqla_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_nonnull(
        df,
        [
            "pre_shock_HQLA",
            "baseline_HQLA",
            "starting_HQLA",
            "initial_HQLA",
            "HQLA",
        ],
    )


def baseline_ncof_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_nonnull(
        df,
        [
            "pre_shock_NCOF",
            "baseline_NCOF",
            "starting_NCOF",
            "initial_NCOF",
            "NCOF",
        ],
    )


def baseline_assets_col(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_nonnull(
        df,
        [
            "pre_shock_total_assets",
            "baseline_total_assets",
            "starting_total_assets",
            "initial_total_assets",
            "total_assets",
            "assets",
        ],
    )


def add_baseline_row_to_monthly_lcr(
    monthly: pd.DataFrame,
    bank_monthly: Optional[pd.DataFrame],
    system_col: str,
    weakest_col: str,
) -> pd.DataFrame:
    """Prepend stress_month=0 baseline LCR row to a monthly path."""
    if bank_monthly is None:
        return monthly
    b0 = _first_stress_month_slice(bank_monthly)
    if b0.empty:
        return monthly

    lcr0 = baseline_lcr_col(b0)
    hqla0 = baseline_hqla_col(b0)
    ncof0 = baseline_ncof_col(b0)

    row = {c: np.nan for c in monthly.columns}
    row["stress_month"] = 0

    if hqla0 and ncof0 and pd.to_numeric(b0[ncof0], errors="coerce").sum() > 0:
        row[system_col] = (
            pd.to_numeric(b0[hqla0], errors="coerce").sum()
            / pd.to_numeric(b0[ncof0], errors="coerce").sum()
        )
    elif lcr0:
        row[system_col] = pd.to_numeric(b0[lcr0], errors="coerce").mean()

    if lcr0:
        row[weakest_col] = pd.to_numeric(b0[lcr0], errors="coerce").min()

    out = monthly.loc[monthly["stress_month"] != 0].copy()
    out = pd.concat([pd.DataFrame([row]), out], ignore_index=True)
    return out.sort_values("stress_month")


def add_baseline_row_to_monthly_hqla_assets(
    monthly: pd.DataFrame,
    bank_monthly: Optional[pd.DataFrame],
    system_col: str,
    weakest_col: str,
) -> pd.DataFrame:
    """Prepend stress_month=0 baseline HQLA/assets row to a monthly path."""
    if bank_monthly is None:
        return monthly
    b0 = _first_stress_month_slice(bank_monthly)
    if b0.empty:
        return monthly

    ratio0 = baseline_hqla_assets_col(b0)
    hqla0 = baseline_hqla_col(b0)
    assets0 = baseline_assets_col(b0)

    row = {c: np.nan for c in monthly.columns}
    row["stress_month"] = 0

    if hqla0 and assets0 and pd.to_numeric(b0[assets0], errors="coerce").sum() > 0:
        row[system_col] = (
            pd.to_numeric(b0[hqla0], errors="coerce").sum()
            / pd.to_numeric(b0[assets0], errors="coerce").sum()
        )
    elif ratio0:
        row[system_col] = pd.to_numeric(b0[ratio0], errors="coerce").mean()

    if ratio0:
        row[weakest_col] = pd.to_numeric(b0[ratio0], errors="coerce").min()

    out = monthly.loc[monthly["stress_month"] != 0].copy()
    out = pd.concat([pd.DataFrame([row]), out], ignore_index=True)
    return out.sort_values("stress_month")


def add_baseline_bank_rows(
    df: pd.DataFrame,
    value_col: str,
    baseline_value_col: Optional[str],
    path_col: str,
    supporting_cols: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Add one t=0 row per bank for path charts.

    value_col is the stress-period column. baseline_value_col is the baseline
    column from the earliest stress month. supporting_cols maps stress-period
    columns to baseline columns so system-weighted aggregates can be computed.
    """
    out = one_row_per_bank_month(df.copy())
    month = get_month_col(out)
    bank = get_bank_col(out)
    if month is None or bank is None:
        return out

    out[path_col] = out[value_col]

    if baseline_value_col is None or baseline_value_col not in out.columns:
        return out

    base = _first_stress_month_slice(out)
    if base.empty:
        return out

    base = base.copy()
    base[month] = 0
    base[path_col] = base[baseline_value_col]

    # Baseline rows represent the pre-stress stock state, not current-period flows.
    # Flow/loss/support diagnostics must therefore be zero at t=0.
    for c in base.columns:
        cl = str(c).lower()
        if any(token in cl for token in [
            "support", "shortfall", "loss", "proceeds", "gap",
            "outflow", "inflow", "need", "asset_sale", "fire_sale", "pledged", "contagion"
        ]):
            if pd.api.types.is_numeric_dtype(base[c]):
                base[c] = 0.0

    if supporting_cols:
        for stress_col, base_col in supporting_cols.items():
            if stress_col in base.columns and base_col in base.columns:
                base[stress_col] = base[base_col]

    combined = pd.concat([base, out.loc[out[month] != 0].copy()], ignore_index=True)
    combined = _collapse_duplicate_keys(combined, [bank, month])
    return combined.sort_values([bank, month])

# =============================================================================
# Data discovery
# =============================================================================


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
        df = normalize_columns(df)
        df = numericize(df)
        if "bank_name" in df.columns:
            df["bank_name"] = df["bank_name"].map(normalize_bank_name)
        return df
    except Exception as exc:
        log(f"could not read {path}: {exc}")
        return None


def score_base_bank_monthly(df: pd.DataFrame, path: Path) -> int:
    cols = set(df.columns)
    score = 0
    if "stress_month" in cols and "bank_name" in cols:
        score += 10
    if any(c in cols for c in ["post_interbank_LCR", "final_post_interbank_LCR", "post_policy_LCR", "post_asset_sale_LCR"]):
        score += 8
    if any(c in cols for c in ["HQLA_assets_ratio", "HQLA", "total_assets"]):
        score += 4
    name = path.name.lower()
    if "monte_carlo" in name or "mc_" in name:
        score -= 8
    if "bank" in name and "month" in name:
        score += 5
    if "baseline" in name or "base" in name or "engine" in name:
        score += 3
    return score


def score_base_monthly(df: pd.DataFrame, path: Path) -> int:
    cols = set(df.columns)
    score = 0
    if "stress_month" in cols and "bank_name" not in cols:
        score += 10
    if any(c.startswith("aggregate_") for c in cols):
        score += 8
    if any(c in cols for c in ["aggregate_cb_support_granted", "total_cb_support_granted", "aggregate_fire_sale_mtm_loss"]):
        score += 6
    name = path.name.lower()
    if "monte_carlo" in name or "mc_" in name:
        score -= 6
    if "monthly" in name:
        score += 4
    if "base" in name or "baseline" in name or "engine" in name:
        score += 4
    return score


def score_policy_summary(df: pd.DataFrame, path: Path) -> int:
    cols = set(df.columns)
    score = 0
    if any(c in cols for c in ["scenario", "scenario_name", "policy", "policy_name"]):
        score += 10
    if any(c in cols for c in ["total_cb_support_granted", "aggregate_cb_support_granted"]):
        score += 6
    if any(c in cols for c in ["total_fire_sale_mtm_loss", "aggregate_fire_sale_mtm_loss"]):
        score += 6
    name = path.name.lower()
    if "policy" in name or "scenario" in name or "regime" in name:
        score += 8
    if "monte_carlo" in name:
        score -= 2
    return score


def load_best(candidates: list[tuple[int, Path, pd.DataFrame]], label: str, min_score: int = 12) -> Optional[pd.DataFrame]:
    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
    if not candidates or candidates[0][0] < min_score:
        log(f"No reliable {label} dataset found")
        return None
    score, path, df = candidates[0]
    log(f"using {label}: {path.relative_to(PROJECT_ROOT)} score={score}, shape={df.shape}")
    return df.copy()


def find_exact_or_pattern(
    csvs: list[Path],
    exact_names: Sequence[str],
    patterns: Sequence[str],
    required_cols: Sequence[str] = (),
    exclude_name_tokens: Sequence[str] = (),
) -> Optional[pd.DataFrame]:

    # Exact filename first.
    lower_map = {p.name.lower(): p for p in csvs}
    for name in exact_names:
        p = lower_map.get(name.lower())
        if p is not None:
            df = read_csv_safe(p)
            if df is not None and all(c in df.columns for c in required_cols):
                log(f"using {p.relative_to(PROJECT_ROOT)}")
                return df
    # Pattern fallback.
    for p in csvs:
        name = p.name.lower()

        if any(token in name for token in exclude_name_tokens):
            continue

        if all(re.search(pat, name) for pat in patterns):
            df = read_csv_safe(p)
            if df is not None and all(c in df.columns for c in required_cols):
                log(f"using {p.relative_to(PROJECT_ROOT)}")
                return df
    return None

def is_deterministic_baseline_candidate(path: Path) -> bool:
    """
    Return True only for files that may safely drive deterministic baseline charts.

    Monte Carlo, sensitivity, and currency-only outputs must not be used as
    deterministic baseline monthly or bank-month datasets.
    """
    name = path.name.lower()

    blocked_tokens = [
        "monte_carlo",
        "mc_",
        "sensitivity",
        "currency_interbank",
        "_amd",
        "_fx",
        "banks_amd",
        "banks_fx",
    ]

    return not any(token in name for token in blocked_tokens)

def load_data() -> DataStore:
    csvs = discover_csvs()
    log(f"found {len(csvs)} CSV files")

    dfs: list[tuple[Path, pd.DataFrame]] = []

    for p in csvs:
        df = read_csv_safe(p)
        if df is not None and len(df.columns) > 0:
            dfs.append((p, df))

    store = DataStore(csvs=csvs)

    # -------------------------------------------------------------------------
    # Monte Carlo outputs
    # -------------------------------------------------------------------------
    store.mc_runs = find_exact_or_pattern(
        csvs,
        exact_names=[
            "monte_carlo_run_summary.csv",
            "monte_carlo_shock_run_summary.csv",
        ],
        patterns=[r"monte.*carlo", r"run.*summary"],
        required_cols=["stress_level"],
    )

    store.mc_monthly = find_exact_or_pattern(
        csvs,
        exact_names=[
            "monte_carlo_monthly_summary.csv",
            "monte_carlo_shock_monthly_summary.csv",
        ],
        patterns=[r"monte.*carlo", r"monthly.*summary"],
        required_cols=["stress_level"],
    )

    store.mc_bank = find_exact_or_pattern(
        csvs,
        exact_names=[
            "monte_carlo_bank_summary.csv",
            "monte_carlo_shock_bank_summary.csv",
        ],
        patterns=[r"monte.*carlo", r"bank.*summary"],
        required_cols=["stress_level", "bank_name"],
    )

    store.mc_quantiles = find_exact_or_pattern(
        csvs,
        exact_names=[
            "monte_carlo_quantile_summary.csv",
            "monte_carlo_shock_quantile_summary.csv",
        ],
        patterns=[r"monte.*carlo", r"quantile.*summary"],
    )

    # -------------------------------------------------------------------------
    # Canonical deterministic baseline outputs.
    #
    # IMPORTANT:
    # These exact files must be preferred over currency outputs, otherwise
    # baseline charts accidentally use currency_interbank_contagion_*.csv.
    # -------------------------------------------------------------------------
    baseline_exclude_tokens = [
        "monte_carlo",
        "mc_",
        "sensitivity",
        "currency_interbank",
        "_amd",
        "_fx",
        "banks_amd",
        "banks_fx",
    ]

    store.base_bank_monthly = find_exact_or_pattern(
        csvs,
        exact_names=["interbank_contagion_bank_results.csv"],
        patterns=[r"interbank.*contagion.*bank.*results"],
        required_cols=["bank_name", "stress_month"],
        exclude_name_tokens=baseline_exclude_tokens,
    )

    store.base_monthly = find_exact_or_pattern(
        csvs,
        exact_names=["interbank_contagion_monthly_summary.csv"],
        patterns=[r"interbank.*contagion.*monthly.*summary"],
        required_cols=["stress_month"],
        exclude_name_tokens=baseline_exclude_tokens,
    )

    # -------------------------------------------------------------------------
    # Canonical policy scenario output.
    #
    # IMPORTANT:
    # Use sensitivity_summary.csv, not sensitivity_bank_summary.csv.
    # The latter is bank-level and should not drive policy-frontier charts.
    # -------------------------------------------------------------------------
    store.policy_summary = find_exact_or_pattern(
        csvs,
        exact_names=["sensitivity_summary.csv"],
        patterns=[r"sensitivity.*summary"],
        required_cols=["scenario"],
    )

    # -------------------------------------------------------------------------
    # Fallback scoring only if exact canonical files are absent.
    # -------------------------------------------------------------------------
    base_bank_candidates = []
    base_monthly_candidates = []
    policy_candidates = []

    for p, df in dfs:
        if is_deterministic_baseline_candidate(p):
            base_bank_candidates.append((score_base_bank_monthly(df, p), p, df))
            base_monthly_candidates.append((score_base_monthly(df, p), p, df))

        policy_candidates.append((score_policy_summary(df, p), p, df))


    if store.base_bank_monthly is None:
        store.base_bank_monthly = load_best(
            base_bank_candidates,
            "baseline bank-month",
            min_score=18,
        )

    if store.base_monthly is None:
        store.base_monthly = load_best(
            base_monthly_candidates,
            "baseline monthly aggregate",
            min_score=18,
        )

    if store.policy_summary is None:
        store.policy_summary = load_best(
            policy_candidates,
            "policy scenario summary",
            min_score=20,
        )

    # Safety: do not allow Monte Carlo run summaries to be treated as policy tables.
    if (
        store.policy_summary is not None
        and "stress_level" in store.policy_summary.columns
        and first_existing(
            store.policy_summary,
            ["scenario", "scenario_name", "policy", "policy_name"],
        )
        is None
    ):
        store.policy_summary = None

    return store


# =============================================================================
# Column extraction and aggregate builders
# =============================================================================


def get_lcr_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(
        df,
        [
            "post_interbank_LCR",
            "final_post_interbank_LCR",
            "post_policy_LCR",
            "post_asset_sale_LCR",
            "post_reaction_LCR",
            "final_LCR",
            "LCR",
            "lcr",
        ],
    )


def get_pre_lcr_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["pre_reaction_LCR", "pre_stress_LCR", "starting_LCR", "initial_LCR", "baseline_LCR"])


def get_hqla_assets_col(df: pd.DataFrame) -> Optional[str]:
    c = first_existing(df, ["HQLA_assets_ratio", "hqla_assets_ratio", "HQLA_to_assets", "hqla_to_assets"])
    if c is not None:
        return c
    if "HQLA" in df.columns and "total_assets" in df.columns:
        df["HQLA_assets_ratio"] = np.where(df["total_assets"] > 0, df["HQLA"] / df["total_assets"], np.nan)
        return "HQLA_assets_ratio"
    return None


def get_month_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["stress_month", "month", "t", "period"])


def get_bank_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["bank_name", "bank", "name"])


def get_scenario_col(df: pd.DataFrame) -> Optional[str]:
    return first_existing(df, ["scenario", "scenario_name", "policy", "policy_name", "regime", "run_name"])


def aggregate_from_bank_monthly(bank_monthly: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if bank_monthly is None:
        return None

    df = one_row_per_bank_month(bank_monthly.copy())

    month = get_month_col(df)
    bank = get_bank_col(df)
    lcr = get_lcr_col(df)
    hqla_ratio = get_hqla_assets_col(df)

    if month is None or bank is None:
        return None

    hqla_col = first_existing(
        df,
        [
            "post_interbank_HQLA",
            "post_cb_support_HQLA",
            "post_asset_sale_HQLA",
            "post_first_round_HQLA",
            "HQLA",
        ],
    )

    ncof_col = first_existing(
        df,
        [
            "post_interbank_NCOF",
            "post_cb_support_NCOF",
            "post_asset_sale_NCOF",
            "post_first_round_NCOF",
            "NCOF",
        ],
    )

    assets_col = first_existing(
        df,
        [
            "total_assets",
            "post_interbank_total_assets",
            "final_total_assets",
            "assets",
        ],
    )

    aggregations: dict[str, tuple[str, str]] = {}

    if lcr:
        aggregations["simple_average_LCR"] = (lcr, "mean")
        aggregations["system_average_LCR"] = (lcr, "mean")  # backward-compatible alias
        aggregations["weakest_bank_LCR"] = (lcr, "min")
        aggregations["maximum_bank_LCR"] = (lcr, "max")

    if hqla_ratio:
        aggregations["simple_average_HQLA_assets_ratio"] = (hqla_ratio, "mean")
        aggregations["system_HQLA_assets_ratio"] = (hqla_ratio, "mean")  # backward-compatible alias
        aggregations["weakest_HQLA_assets_ratio"] = (hqla_ratio, "min")
        aggregations["maximum_HQLA_assets_ratio"] = (hqla_ratio, "max")

    if hqla_col:
        aggregations["aggregate_HQLA"] = (hqla_col, "sum")

    if ncof_col:
        aggregations["aggregate_NCOF"] = (ncof_col, "sum")

    if assets_col:
        aggregations["aggregate_system_assets"] = (assets_col, "sum")

    for new, candidates in {
        "aggregate_cash_gap_before_support": [
            "cash_shortfall_before_support",
            "cash_gap_before_support",
            "original_cash_gap",
        ],
        "aggregate_cb_support_granted": [
            "cb_support_granted",
            "support_granted",
        ],
        "aggregate_asset_sale_proceeds": [
            "asset_sale_proceeds",
            "asset_sale_cash",
            "asset_sale_proceeds_total",
        ],
        "aggregate_asset_sale_cash_retained_for_lcr": [
            "asset_sale_cash_retained_for_lcr",
        ],
        "aggregate_asset_sale_proceeds_used_for_outflows": [
            "asset_sale_proceeds_used_for_outflows",
        ],
        "aggregate_final_unresolved_shortfall": [
            "final_unresolved_shortfall_amount",
            "unresolved_shortfall_amount",
            "residual_shortfall",
        ],
        "aggregate_fire_sale_mtm_loss": [
            "fire_sale_mtm_loss",
            "asset_sale_loss",
            "fire_sale_loss",
            "total_fire_sale_mtm_loss",
        ],
        "aggregate_interbank_contagion_loss": [
            "interbank_contagion_loss",
        ],
        "aggregate_lcr_support_gap": [
            "lolr_lcr_support_gap",
            "lcr_support_gap",
            "regulatory_hqla_gap_before_reaction",
        ],
        "aggregate_support_used_for_outflows": [
            "cb_support_used_for_outflows",
            "support_used_for_outflows",
        ],
        "aggregate_support_retained_as_cash_for_lcr": [
            "cb_support_retained_as_cash_for_lcr",
            "support_retained_as_cash_for_lcr",
        ],
    }.items():
        c = first_existing(df, candidates)
        if c:
            aggregations[new] = (c, "sum")

    if not aggregations:
        return None

    out = df.groupby(month, as_index=False).agg(**aggregations)
    out = out.rename(columns={month: "stress_month"})

    if "aggregate_HQLA" in out.columns and "aggregate_NCOF" in out.columns:
        out["system_weighted_LCR"] = np.where(
            out["aggregate_NCOF"] > 0,
            out["aggregate_HQLA"] / out["aggregate_NCOF"],
            np.nan,
        )

    if "aggregate_HQLA" in out.columns and "aggregate_system_assets" in out.columns:
        out["system_weighted_HQLA_assets_ratio"] = np.where(
            out["aggregate_system_assets"] > 0,
            out["aggregate_HQLA"] / out["aggregate_system_assets"],
            np.nan,
        )

    out = one_row_per_month(out)
    return out.sort_values("stress_month")

def get_active_lcr_threshold(
    monthly: Optional[pd.DataFrame] = None,
    bank_monthly: Optional[pd.DataFrame] = None,
    default: float = 1.20,
) -> float:
    """
    Read the active LCR requirement if available.
    Falls back to Armenia's current 120% requirement.
    """
    candidates = []

    for df in [monthly, bank_monthly]:
        if df is not None and "active_lcr_requirement" in df.columns:
            s = pd.to_numeric(df["active_lcr_requirement"], errors="coerce").dropna()
            if not s.empty:
                candidates.append(float(s.median()))

    if candidates:
        return float(np.median(candidates))

    return float(default)


def ensure_system_metrics(
    monthly: Optional[pd.DataFrame],
    bank_monthly: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Ensure monthly data contains true system-weighted metrics computed from bank-month data.
    """
    if monthly is None and bank_monthly is None:
        return None

    base_agg = aggregate_from_bank_monthly(bank_monthly)

    if monthly is None:
        return base_agg

    out = monthly.copy()

    if "stress_month" not in out.columns:
        m = get_month_col(out)
        if m:
            out = out.rename(columns={m: "stress_month"})

    if base_agg is not None and "stress_month" in out.columns:
        for c in base_agg.columns:
            if c == "stress_month":
                continue

            # Prefer bank-month recomputation for path metrics.
            if c in [
                "system_weighted_LCR",
                "simple_average_LCR",
                "system_average_LCR",
                "weakest_bank_LCR",
                "system_weighted_HQLA_assets_ratio",
                "simple_average_HQLA_assets_ratio",
                "system_HQLA_assets_ratio",
                "weakest_HQLA_assets_ratio",
                "aggregate_HQLA",
                "aggregate_NCOF",
                "aggregate_system_assets",
            ]:
                out = out.drop(columns=[c], errors="ignore")

            if c not in out.columns:
                out = out.merge(base_agg[["stress_month", c]], on="stress_month", how="left")

    return one_row_per_month(out)


def plot_system_path_chart(
    df: pd.DataFrame,
    x_col: str,
    system_col: str,
    weakest_col: str,
    title: str,
    ylabel: str,
    filename: str,
    system_label: str,
    weakest_label: str,
    threshold: Optional[float] = None,
    threshold_label: Optional[str] = None,
    pct_axis: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, title, "Stress month", ylabel)

    if threshold is not None:
        ax.axhspan(0, threshold, color=BURGUNDY, alpha=0.08)
        ax.axhline(
            threshold,
            color=BURGUNDY,
            linestyle="--",
            linewidth=1.8,
            label=threshold_label or f"Threshold = {threshold:.2f}",
        )

    ax.plot(
        df[x_col],
        df[system_col],
        marker="o",
        linewidth=3.0,
        color=BLUE,
        label=system_label,
    )

    ax.plot(
        df[x_col],
        df[weakest_col],
        marker="o",
        linewidth=3.0,
        color=BURGUNDY,
        label=weakest_label,
    )

    ax.set_xticks(df[x_col])

    if pct_axis:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))

    ax.legend(frameon=False, loc="lower left", fontsize=12)
    savefig(fig, filename)

def complete_months(df: pd.DataFrame, month_col: str = "stress_month") -> pd.DataFrame:
    out = df.copy()
    if month_col not in out.columns:
        return out
    months = sorted(pd.to_numeric(out[month_col], errors="coerce").dropna().astype(int).unique())
    if not months:
        return out
    full = pd.DataFrame({month_col: list(range(min(months), max(months) + 1))})
    out = full.merge(out, on=month_col, how="left")
    for c in out.columns:
        if c != month_col and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0)
    return out


def merged_monthly(store: DataStore) -> Optional[pd.DataFrame]:
    base_agg = aggregate_from_bank_monthly(store.base_bank_monthly)
    monthly = store.base_monthly.copy() if store.base_monthly is not None else None
    if monthly is None:
        monthly = base_agg
    elif base_agg is not None:
        # Base bank-month computed columns are often more reliable for LCR/HQLA paths.
        month = get_month_col(monthly) or "stress_month"
        monthly = monthly.rename(columns={month: "stress_month"})
        for c in base_agg.columns:
            if c != "stress_month" and c not in monthly.columns:
                monthly = monthly.merge(base_agg[["stress_month", c]], on="stress_month", how="left")
    if monthly is not None:
        if "stress_month" not in monthly.columns:
            m = get_month_col(monthly)
            if m:
                monthly = monthly.rename(columns={m: "stress_month"})
        monthly = complete_months(monthly, "stress_month")
    return monthly


def scenario_order_key(name: str) -> int:
    s = str(name).lower()
    order = [
        "no cba",
        "no_cb",
        "base cash",
        "base",
        "asset purchase",
        "asset",
        "lcr req",
        "lcr low",
        "lcr*",
        "wider",
        "collateral capped",
        "full coverage",
        "lcr-cb",
        "lcr cb",
        "plus asset",
    ]
    for i, token in enumerate(order):
        if token in s:
            return i
    return 999


def tidy_policy_summary(policy: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if policy is None:
        return None
    df = policy.copy()
    scenario_col = get_scenario_col(df)
    if scenario_col is None:
        return None
    df = df.rename(columns={scenario_col: "scenario"})
    df["scenario"] = df["scenario"].astype(str).str.replace("_", " ", regex=False).str.strip()
    # Rename common columns.
    ren = {}
    for target, candidates in {
        "total_cb_support_granted": ["total_cb_support_granted", "aggregate_cb_support_granted", "cb_support_granted"],
        "total_fire_sale_mtm_loss": ["total_fire_sale_mtm_loss", "aggregate_fire_sale_mtm_loss", "fire_sale_mtm_loss", "total_asset_sale_loss"],
        "minimum_LCR": ["min_post_interbank_LCR", "minimum_final_post_interbank_LCR", "minimum_LCR", "min_LCR"],
        "unresolved_banks": ["ever_final_unresolved_shortfall_banks", "final_terminal_distress_after_interbank_banks", "terminal_distress_banks"],
    }.items():
        c = first_existing(df, candidates)
        if c and c != target:
            ren[c] = target
    df = df.rename(columns=ren)
    df = df.sort_values("scenario", key=lambda x: x.map(scenario_order_key))
    return df


# =============================================================================
# Chart functions: system and bank-level baseline
# =============================================================================

def chart_01_base_lcr_path(
    monthly: Optional[pd.DataFrame],
    bank_monthly: Optional[pd.DataFrame],
) -> None:
    """
    Headline LCR chart: use stressed_LCR, not post-interbank recovered LCR.

    post_interbank_LCR answers: what is the normal-times LCR after the balance
    sheet has already contracted and after mitigation. The flagship stress-test
    chart should instead show the forward-looking stressed measure.
    """
    df = ensure_system_metrics(monthly, bank_monthly)

    if df is None:
        log("SKIP 01: no monthly data")
        return

    if "stress_month" not in df.columns:
        log("SKIP 01: no stress_month")
        return

    # Prefer the system stressed LCR built from aggregate stressed HQLA/NCOF.
    system_col = first_existing(
        df,
        [
            "system_stressed_LCR",
            "asset_weighted_stressed_LCR",
            "average_stressed_LCR",
        ],
    )

    weakest_col = first_existing(
        df,
        [
            "minimum_stressed_LCR",
            "min_stressed_LCR",
        ],
    )

    # If the monthly summary lacks stressed fields, compute them from bank-month data.
    if (system_col is None or weakest_col is None) and bank_monthly is not None:
        bm = one_row_per_bank_month(bank_monthly.copy())
        month_col = get_month_col(bm)

        if (
            month_col is not None
            and "stressed_HQLA" in bm.columns
            and "stressed_NCOF" in bm.columns
            and "stressed_LCR" in bm.columns
        ):
            stressed = (
                bm.groupby(month_col)
                .agg(
                    aggregate_stressed_HQLA=("stressed_HQLA", "sum"),
                    aggregate_stressed_NCOF=("stressed_NCOF", "sum"),
                    minimum_stressed_LCR=("stressed_LCR", "min"),
                )
                .reset_index()
                .rename(columns={month_col: "stress_month"})
            )

            stressed["system_stressed_LCR"] = np.where(
                stressed["aggregate_stressed_NCOF"] > 0,
                stressed["aggregate_stressed_HQLA"]
                / stressed["aggregate_stressed_NCOF"],
                np.nan,
            )

            df = df.merge(
                stressed[["stress_month", "system_stressed_LCR", "minimum_stressed_LCR"]],
                on="stress_month",
                how="left",
                suffixes=("", "_from_bank"),
            )

            if system_col is None:
                system_col = "system_stressed_LCR"
            if weakest_col is None:
                weakest_col = "minimum_stressed_LCR"

    if system_col is None or weakest_col is None:
        log("SKIP 01: missing stressed system LCR or stressed weakest-bank LCR")
        return

    # Add t=0 baseline so the path starts from the pre-stress state.
    df = add_baseline_row_to_monthly_lcr(df, bank_monthly, system_col, weakest_col)

    threshold = get_active_lcr_threshold(monthly=df, bank_monthly=bank_monthly)

    plot_system_path_chart(
        df=df,
        x_col="stress_month",
        system_col=system_col,
        weakest_col=weakest_col,
        title="First-Round LCR Path: Baseline and Stressed Months",
        ylabel="LCR",
        filename="01_base_lcr_path.png",
        system_label="System first-round LCR = aggregate stressed HQLA / stressed NCOF",
        weakest_label="Weakest-bank first-round LCR",
        threshold=threshold,
        threshold_label=f"Regulatory threshold = {threshold:.1f}",
        pct_axis=False,
    )

def chart_01b_base_hqla_assets_path(
    monthly: Optional[pd.DataFrame],
    bank_monthly: Optional[pd.DataFrame],
) -> None:
    df = ensure_system_metrics(monthly, bank_monthly)

    if df is None:
        log("SKIP 01b: no monthly data")
        return

    if "stress_month" not in df.columns:
        log("SKIP 01b: no stress_month")
        return

    system_col = first_existing(
        df,
        [
            "system_weighted_HQLA_assets_ratio",
            "system_HQLA_assets_ratio",
            "average_post_interbank_HQLA_assets_ratio",
        ],
    )

    weakest_col = first_existing(
        df,
        [
            "weakest_HQLA_assets_ratio",
            "minimum_post_interbank_HQLA_assets_ratio",
            "min_HQLA_assets_ratio",
        ],
    )

    if system_col is None or weakest_col is None:
        log("SKIP 01b: missing HQLA/assets columns")
        return

    # Add t=0 baseline so the path starts from the pre-stress liquidity stock.
    df = add_baseline_row_to_monthly_hqla_assets(df, bank_monthly, system_col, weakest_col)

    plot_system_path_chart(
        df=df,
        x_col="stress_month",
        system_col=system_col,
        weakest_col=weakest_col,
        title="HQLA / Assets Path: Baseline and Full-Transmission Months",
        ylabel="HQLA / total assets",
        filename="01b_base_hqla_assets_path.png",
        system_label="System HQLA / assets = aggregate HQLA / aggregate assets",
        weakest_label="Weakest-bank HQLA / assets",
        threshold=0.10,
        threshold_label="Monitoring line = 10%",
        pct_axis=True,
    )


def chart_01c_all_bank_lcr_lines(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 01c: no bank-month data")
        return

    df = one_row_per_bank_month(bank_monthly.copy())

    month = get_month_col(df)
    bank = get_bank_col(df)
    lcr = get_lcr_col(df)

    if month is None or bank is None or lcr is None:
        log("SKIP 01c: missing month/bank/LCR columns")
        return

    hqla_col = first_existing(df, ["post_interbank_HQLA", "HQLA"])
    ncof_col = first_existing(df, ["post_interbank_NCOF", "NCOF"])

    # Plot a true path: t=0 baseline plus final post-transmission months.
    lcr_path_col = "__path_LCR"
    df = add_baseline_bank_rows(
        df,
        value_col=lcr,
        baseline_value_col=baseline_lcr_col(df),
        path_col=lcr_path_col,
        supporting_cols={
            hqla_col: baseline_hqla_col(df) if hqla_col else None,
            ncof_col: baseline_ncof_col(df) if ncof_col else None,
        } if hqla_col and ncof_col else None,
    )

    piv = pivot_bank_month(df, lcr_path_col, sort_by_min=False)

    system = None
    if hqla_col is not None and ncof_col is not None:
        system = (
            df.groupby(month)
            .apply(lambda g: g[hqla_col].sum() / g[ncof_col].sum() if g[ncof_col].sum() > 0 else np.nan)
            .reset_index(name="system_weighted_LCR")
            .sort_values(month)
        )

    simple_avg = (
        df.groupby(month)[lcr_path_col]
        .mean()
        .reset_index(name="simple_average_LCR")
        .sort_values(month)
    )

    threshold = get_active_lcr_threshold(bank_monthly=df)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(
        ax,
        "All-bank LCR Paths: Baseline and Final Post-Transmission Months",
        "Stress month",
        "LCR",
    )

    ax.axhspan(0, threshold, color=BURGUNDY, alpha=0.08)
    ax.axhline(
        threshold,
        color=BURGUNDY,
        linestyle="--",
        linewidth=1.8,
        label=f"Regulatory threshold = {threshold:.1f}",
    )

    for bank_name in piv.index:
        ax.plot(
            piv.columns,
            piv.loc[bank_name],
            linewidth=1.4,
            alpha=0.35,
            color=GREY,
        )

    if system is not None:
        ax.plot(
            system[month],
            system["system_weighted_LCR"],
            marker="o",
            linewidth=3.5,
            color=BLUE,
            label="System LCR = aggregate HQLA / aggregate NCOF",
        )

    ax.plot(
        simple_avg[month],
        simple_avg["simple_average_LCR"],
        marker="o",
        linewidth=2.6,
        linestyle=":",
        color=NAVY,
        label="Simple average bank LCR",
    )

    ax.set_xticks(sorted(df[month].dropna().astype(int).unique()))
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    savefig(fig, "01c_all_bank_lcr_lines.png")


def chart_01d_all_bank_hqla_assets_lines(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 01d: no bank-month data")
        return

    df = one_row_per_bank_month(bank_monthly.copy())

    month = get_month_col(df)
    bank = get_bank_col(df)
    ratio = get_hqla_assets_col(df)

    if month is None or bank is None or ratio is None:
        log("SKIP 01d: missing month/bank/HQLA-assets columns")
        return

    hqla_col = first_existing(df, ["post_interbank_HQLA", "HQLA"])
    assets_col = first_existing(df, ["total_assets", "final_total_assets", "assets"])

    # Plot a true path: t=0 baseline plus final post-transmission months.
    ratio_path_col = "__path_HQLA_assets_ratio"
    df = add_baseline_bank_rows(
        df,
        value_col=ratio,
        baseline_value_col=baseline_hqla_assets_col(df),
        path_col=ratio_path_col,
        supporting_cols={
            hqla_col: baseline_hqla_col(df) if hqla_col else None,
            assets_col: baseline_assets_col(df) if assets_col else None,
        } if hqla_col and assets_col else None,
    )

    piv = pivot_bank_month(df, ratio_path_col, sort_by_min=False)

    system = None
    if hqla_col is not None and assets_col is not None:
        system = (
            df.groupby(month)
            .apply(lambda g: g[hqla_col].sum() / g[assets_col].sum() if g[assets_col].sum() > 0 else np.nan)
            .reset_index(name="system_weighted_HQLA_assets_ratio")
            .sort_values(month)
        )

    simple_avg = (
        df.groupby(month)[ratio_path_col]
        .mean()
        .reset_index(name="simple_average_HQLA_assets_ratio")
        .sort_values(month)
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(
        ax,
        "All-bank HQLA / Assets Paths: Baseline and Final Months",
        "Stress month",
        "HQLA / total assets",
    )

    ax.axhspan(0, 0.10, color=BURGUNDY, alpha=0.08)
    ax.axhline(
        0.10,
        color=BURGUNDY,
        linestyle="--",
        linewidth=1.8,
        label="Monitoring line = 10%",
    )

    for bank_name in piv.index:
        ax.plot(
            piv.columns,
            piv.loc[bank_name],
            linewidth=1.4,
            alpha=0.35,
            color=GREY,
        )

    if system is not None:
        ax.plot(
            system[month],
            system["system_weighted_HQLA_assets_ratio"],
            marker="o",
            linewidth=3.5,
            color=BLUE,
            label="System HQLA / assets = aggregate HQLA / aggregate assets",
        )

    ax.plot(
        simple_avg[month],
        simple_avg["simple_average_HQLA_assets_ratio"],
        marker="o",
        linewidth=2.6,
        linestyle=":",
        color=NAVY,
        label="Simple average bank HQLA / assets",
    )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    ax.set_xticks(sorted(df[month].dropna().astype(int).unique()))
    ax.legend(frameon=False, loc="lower left", fontsize=11)
    savefig(fig, "01d_all_bank_hqla_assets_lines.png")

def pivot_bank_month(df: pd.DataFrame, value_col: str, sort_by_min: bool = True) -> pd.DataFrame:
    month_col = get_month_col(df)
    bank_col = get_bank_col(df)
    if month_col is None or bank_col is None:
        raise ValueError("bank-month pivot requires month and bank columns")
    piv = df.pivot_table(index=bank_col, columns=month_col, values=value_col, aggfunc="mean")
    piv = piv.sort_index(axis=1)
    if sort_by_min:
        piv = piv.loc[piv.min(axis=1).sort_values().index]
    return piv


def heatmap(ax: plt.Axes, data: pd.DataFrame, title: str, cbar_label: str, vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = "RdYlBu") -> None:
    ax.set_facecolor(BG)
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=17, fontweight="bold", pad=14)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([str(int(c)) if isinstance(c, (int, float, np.integer, np.floating)) else str(c) for c in data.columns], fontsize=11)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index.astype(str), fontsize=10)
    ax.set_xlabel("Stress month", fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Cell labels.
    finite = data.values[np.isfinite(data.values)]
    midpoint = np.nanmedian(finite) if len(finite) else 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.iloc[i, j]
            if pd.notna(val):
                color = "white" if val < midpoint else "#111111"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label(cbar_label, fontsize=11)


def chart_02_lcr_heatmap(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 02: no bank-month data")
        return
    lcr = get_lcr_col(bank_monthly)
    if lcr is None:
        log("SKIP 02: no LCR column")
        return
    df = add_baseline_bank_rows(
        bank_monthly,
        value_col=lcr,
        baseline_value_col=baseline_lcr_col(bank_monthly),
        path_col="__path_LCR",
    )
    piv = pivot_bank_month(df, "__path_LCR", sort_by_min=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    heatmap(ax, piv, "Bank-level LCR Heatmap: Baseline and Stress Horizon", "Final post-transmission LCR", vmin=0.0, vmax=max(1.4, np.nanpercentile(piv.values, 95)), cmap="RdYlBu")
    ax.axvline(-0.5, color="none")
    savefig(fig, "02_bank_level_lcr_heatmap.png")


def chart_03_hqla_assets_heatmap(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 03: no bank-month data")
        return
    df = bank_monthly.copy()
    ratio = get_hqla_assets_col(df)
    if ratio is None:
        log("SKIP 03: no HQLA/assets column")
        return
    df = add_baseline_bank_rows(
        df,
        value_col=ratio,
        baseline_value_col=baseline_hqla_assets_col(df),
        path_col="__path_HQLA_assets_ratio",
    )
    piv = pivot_bank_month(df, "__path_HQLA_assets_ratio", sort_by_min=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    heatmap(ax, piv, "Liquidity-stock Erosion: Baseline and Stress Horizon", "HQLA / total assets", vmin=0.0, vmax=max(0.35, np.nanpercentile(piv.values, 95)), cmap="YlGnBu")
    savefig(fig, "03_bank_level_hqla_assets_heatmap.png")


def small_multiples_paths(
    df: pd.DataFrame,
    value_col: str,
    filename: str,
    title: str,
    ylabel: str,
    threshold: Optional[float] = None,
    threshold_label: Optional[str] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    month_col = get_month_col(df)
    bank_col = get_bank_col(df)
    if month_col is None or bank_col is None:
        log(f"SKIP {filename}: missing month/bank columns")
        return
    piv = pivot_bank_month(df, value_col, sort_by_min=True)
    banks = list(piv.index)
    n = len(banks)
    if n == 0:
        log(f"SKIP {filename}: empty pivot")
        return
    ncols = 3 if n <= 12 else 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.1 * nrows), sharex=True, sharey=True)
    axes_arr = np.array(axes).reshape(-1)

    x = list(piv.columns)
    for idx, bank in enumerate(banks):
        ax = axes_arr[idx]
        setup_ax(ax)
        y = piv.loc[bank].values.astype(float)
        ax.plot(x, y, marker="o", linewidth=2.2, color=BLUE)
        if threshold is not None:
            ax.axhline(threshold, color=BURGUNDY, linestyle="--", linewidth=1.2)
        ax.set_title(str(bank), fontsize=11, fontweight="bold", pad=6)
        ax.set_xticks(x)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for j in range(n, len(axes_arr)):
        axes_arr[j].axis("off")

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.005)
    fig.supxlabel("Stress month", fontsize=13, y=0.02)
    fig.supylabel(ylabel, fontsize=13, x=0.005)
    if threshold is not None and threshold_label:
        # Use the first axis for legend.
        axes_arr[0].plot([], [], color=BURGUNDY, linestyle="--", label=threshold_label)
        axes_arr[0].legend(frameon=False, fontsize=9, loc="best")
    savefig(fig, filename)


def chart_04_bank_lcr_small_multiples(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 04: no bank-month data")
        return
    lcr = get_lcr_col(bank_monthly)
    if lcr is None:
        log("SKIP 04: no LCR column")
        return
    df = add_baseline_bank_rows(
        bank_monthly,
        value_col=lcr,
        baseline_value_col=baseline_lcr_col(bank_monthly),
        path_col="__path_LCR",
    )
    vals = pd.to_numeric(df["__path_LCR"], errors="coerce")
    upper = min(max(1.35, float(np.nanpercentile(vals, 95)) + 0.1), max(2.0, float(np.nanmax(vals)) + 0.05))
    threshold = get_active_lcr_threshold(bank_monthly=bank_monthly)
    small_multiples_paths(
        df,
        "__path_LCR",
        "04_bank_lcr_small_multiples.png",
        "How Each Bank's LCR Evolves from Baseline Through the Stress Horizon",
        "Final post-transmission LCR",
        threshold=threshold,
        threshold_label=f"LCR threshold = {threshold:.1f}",
        ylim=(0, upper),
    )


def chart_05_bank_hqla_assets_small_multiples(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 05: no bank-month data")
        return
    df = bank_monthly.copy()
    ratio = get_hqla_assets_col(df)
    if ratio is None:
        log("SKIP 05: no HQLA/assets column")
        return
    df = add_baseline_bank_rows(
        df,
        value_col=ratio,
        baseline_value_col=baseline_hqla_assets_col(df),
        path_col="__path_HQLA_assets_ratio",
    )
    vals = pd.to_numeric(df["__path_HQLA_assets_ratio"], errors="coerce")
    upper = min(max(0.20, float(np.nanpercentile(vals, 95)) + 0.03), max(0.40, float(np.nanmax(vals)) + 0.02))
    small_multiples_paths(
        df,
        "__path_HQLA_assets_ratio",
        "05_bank_hqla_assets_small_multiples.png",
        "How Each Bank's HQLA / Assets Ratio Moves from Baseline Through Stress",
        "HQLA / total assets",
        threshold=None,
        ylim=(0, upper),
    )


def chart_06_cash_gap_absorption(monthly: Optional[pd.DataFrame]) -> None:
    if monthly is None or "stress_month" not in monthly.columns:
        log("SKIP 06: no monthly data")
        return
    df = monthly.copy()
    cash_col = first_existing(df, ["aggregate_cash_gap_before_support", "aggregate_cash_shortfall_before_support", "total_cash_shortfall_before_support", "cash_shortfall_before_support"])
    cb_col = first_existing(df, ["aggregate_cb_support_granted", "total_cb_support_granted", "cb_support_granted"])
    sale_col = first_existing(df, ["aggregate_asset_sale_proceeds", "total_asset_sale_proceeds", "asset_sale_proceeds"])
    residual_col = first_existing(df, ["aggregate_final_unresolved_shortfall", "total_final_unresolved_shortfall", "residual_shortfall", "final_unresolved_shortfall_amount"])
    if cash_col is None or cb_col is None:
        log("SKIP 06: cash gap or CBA support column missing")
        return
    if sale_col is None:
        df["__asset_sales"] = 0.0
        sale_col = "__asset_sales"
    if residual_col is None:
        df["__residual"] = np.maximum(df[cash_col] - df[cb_col] - df[sale_col], 0)
        residual_col = "__residual"

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Base Scenario: Cash Gap Absorption", "Stress month", "Amount")
    x = df["stress_month"].astype(int).values
    width = 0.72
    ax.bar(x, df[cb_col], width=width, color=BLUE, alpha=0.95, label="CBA refinancing")
    ax.bar(x, df[sale_col], bottom=df[cb_col], width=width, color=GOLD, alpha=0.95, label="Bank asset-sale proceeds")
    ax.bar(x, df[residual_col], bottom=df[cb_col] + df[sale_col], width=width, color=BURGUNDY, alpha=0.95, label="Residual shortfall")
    ax.plot(x, df[cash_col], marker="o", linewidth=3, color=NAVY, label="Original cash gap")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
    savefig(fig, "06_base_cash_gap_absorption.png")


def chart_07_lolr_mechanism(monthly: Optional[pd.DataFrame]) -> None:
    if monthly is None or "stress_month" not in monthly.columns:
        log("SKIP 07: no monthly data")
        return
    df = monthly.copy()
    month = df["stress_month"].astype(int)
    cash_need = first_existing(df, ["aggregate_cash_gap_before_support", "aggregate_cash_shortfall_before_support", "total_cash_shortfall_before_support"])
    lcr_gap = first_existing(df, ["aggregate_lcr_support_gap", "regulatory_hqla_gap_before_reaction", "lcr_support_gap"])
    support = first_existing(df, ["aggregate_cb_support_granted", "total_cb_support_granted", "cb_support_granted"])
    used = first_existing(df, ["aggregate_support_used_for_outflows", "support_used_for_outflows"])
    retained = first_existing(df, ["aggregate_support_retained_as_cash_for_lcr", "support_retained_as_cash_for_lcr"])

    if support is None:
        log("SKIP 07: support column missing")
        return
    if cash_need is None:
        df["__cash_need"] = 0.0
        cash_need = "__cash_need"
    if lcr_gap is None:
        df["__lcr_gap"] = np.maximum(df[support] - df[cash_need], 0)
        lcr_gap = "__lcr_gap"
    if used is None:
        df["__used"] = np.minimum(df[support], df[cash_need])
        used = "__used"
    if retained is None:
        df["__retained"] = np.maximum(df[support] - df[used], 0)
        retained = "__retained"

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "LoLR Mechanism: Cash Gap vs LCR Gap", "Stress month", "Amount")
    ax.bar(month, df[used], color=BLUE, width=0.72, alpha=0.92, label="Support used for outflows")
    ax.bar(month, df[retained], bottom=df[used], color=LIGHT_BLUE, width=0.72, alpha=0.92, label="Support retained as cash/HQLA")
    ax.plot(month, df[support], marker="s", linewidth=3, color=NAVY, label="Required / granted support")
    ax.plot(month, df[cash_need], marker="o", linewidth=2.3, color=BURGUNDY, linestyle=":", label="Cash-shortfall need")
    ax.plot(month, df[lcr_gap], marker="o", linewidth=2.3, color=GOLD, linestyle="--", label="LCR support gap")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.set_xticks(month)
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
    savefig(fig, "07_lolr_mechanism_cash_gap_vs_lcr_gap.png")


def chart_08_loss_decomposition(monthly: Optional[pd.DataFrame]) -> None:
    if monthly is None or "stress_month" not in monthly.columns:
        log("SKIP 08: no monthly data")
        return

    df = monthly.copy()

    first_round = first_existing(
        df,
        [
            "aggregate_security_valuation_loss",
            "total_security_valuation_loss",
            "aggregate_first_round_security_valuation_loss",
        ],
    )
    asset_sale = first_existing(
        df,
        [
            "aggregate_asset_sale_loss",
            "total_asset_sale_loss",
            "asset_sale_liquidation_loss",
        ],
    )
    fire = first_existing(
        df,
        [
            "aggregate_fire_sale_mtm_loss",
            "total_fire_sale_mtm_loss",
            "fire_sale_mtm_loss",
        ],
    )
    contagion = first_existing(
        df,
        [
            "aggregate_interbank_contagion_loss",
            "total_interbank_contagion_loss",
            "interbank_contagion_loss",
        ],
    )

    loss_cols = [first_round, asset_sale, fire, contagion]
    if all(c is None for c in loss_cols):
        log("SKIP 08: no loss columns")
        return

    defaults = {
        "__first_round": first_round,
        "__asset_sale": asset_sale,
        "__fire": fire,
        "__contagion": contagion,
    }

    for fallback, col in defaults.items():
        if col is None:
            df[fallback] = 0.0

    first_round = first_round or "__first_round"
    asset_sale = asset_sale or "__asset_sale"
    fire = fire or "__fire"
    contagion = contagion or "__contagion"

    x = df["stress_month"].astype(int)
    first_vals = df[first_round].fillna(0.0)
    asset_vals = df[asset_sale].fillna(0.0)
    fire_vals = df[fire].fillna(0.0)
    contagion_vals = df[contagion].fillna(0.0)

    total = first_vals + asset_vals + fire_vals + contagion_vals

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(
        ax,
        "Loss Decomposition Across Stress Months",
        "Stress month",
        "Loss amount",
    )

    ax.bar(
        x,
        first_vals,
        color=GOLD,
        width=0.72,
        alpha=0.92,
        label="First-round securities valuation loss",
    )
    ax.bar(
        x,
        asset_vals,
        bottom=first_vals,
        color=OLIVE,
        width=0.72,
        alpha=0.92,
        label="Asset-sale liquidation loss",
    )
    ax.bar(
        x,
        fire_vals,
        bottom=first_vals + asset_vals,
        color=BURGUNDY,
        width=0.72,
        alpha=0.92,
        label="Fire-sale MTM loss",
    )
    ax.bar(
        x,
        contagion_vals,
        bottom=first_vals + asset_vals + fire_vals,
        color=BLUE,
        width=0.72,
        alpha=0.92,
        label="Interbank contagion loss",
    )
    ax.plot(x, total, marker="o", linewidth=3, color=NAVY, label="Total modeled loss")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    savefig(fig, "08_loss_decomposition_by_month.png")

def chart_09_breach_counts(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 09: no bank-month data")
        return
    df = bank_monthly.copy()
    month = get_month_col(df)
    lcr = get_lcr_col(df)
    ratio = get_hqla_assets_col(df)
    if month is None or lcr is None:
        log("SKIP 09: missing month/LCR")
        return
    tmp = df.copy()
    threshold = get_active_lcr_threshold(bank_monthly=df)
    tmp["lcr_lt_active"] = tmp[lcr] < threshold
    tmp["lcr_lt_100"] = tmp[lcr] < 1.00

    aggs = {
        f"Banks with LCR < {threshold:.2f}": ("lcr_lt_active", "sum"),
        "Banks with LCR < 1.00": ("lcr_lt_100", "sum"),
    }
    if ratio is not None:
        # Warning threshold is not regulatory, just a liquidity-stock deterioration diagnostic.
        tmp["hqla_assets_lt_10pct"] = tmp[ratio] < 0.10
        aggs["Banks with HQLA/assets < 10%"] = ("hqla_assets_lt_10pct", "sum")
    out = tmp.groupby(month, as_index=False).agg(**aggs).rename(columns={month: "stress_month"})

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    setup_ax(ax, "How Many Banks Fall into Liquidity Distress Over Time", "Stress month", "Number of banks")
    for col, color in zip([c for c in out.columns if c != "stress_month"], [BURGUNDY, GOLD, BLUE]):
        ax.plot(out["stress_month"], out[col], marker="o", linewidth=3, color=color, label=col)
    ax.set_xticks(out["stress_month"])
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    savefig(fig, "09_bank_distress_breach_counts.png")


def chart_10_lcr_distribution(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 10: no bank-month data")
        return
    df = bank_monthly.copy()
    month = get_month_col(df)
    lcr = get_lcr_col(df)
    if month is None or lcr is None:
        log("SKIP 10: missing month/LCR")
        return
    df = add_baseline_bank_rows(
        df,
        value_col=lcr,
        baseline_value_col=baseline_lcr_col(df),
        path_col="__path_LCR",
    )
    q = df.groupby(month)["__path_LCR"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack().reset_index()
    q = q.rename(columns={month: "stress_month", 0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95"})
    threshold = get_active_lcr_threshold(bank_monthly=bank_monthly)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Cross-bank LCR Distribution: Baseline and Stress Horizon", "Stress month", "Final post-transmission LCR")
    ax.axhline(threshold, color=BURGUNDY, linestyle="--", linewidth=1.8, label=f"LCR threshold = {threshold:.1f}")
    ax.fill_between(q["stress_month"], q["p05"], q["p95"], color=LIGHT_BLUE, alpha=0.25, label="p05 to p95 range")
    ax.fill_between(q["stress_month"], q["p25"], q["p75"], color=LIGHT_BLUE, alpha=0.45, label="p25 to p75 range")
    ax.plot(q["stress_month"], q["p50"], marker="o", linewidth=3, color=BLUE, label="Median bank LCR")
    ax.plot(q["stress_month"], q["p05"], marker="o", linewidth=2.5, color=BURGUNDY, label="5th percentile")
    ax.set_xticks(q["stress_month"])
    ax.legend(frameon=False, fontsize=11, loc="best")
    savefig(fig, "10_cross_bank_lcr_distribution.png")


def chart_11_hqla_assets_distribution(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 11: no bank-month data")
        return
    df = bank_monthly.copy()
    month = get_month_col(df)
    ratio = get_hqla_assets_col(df)
    if month is None or ratio is None:
        log("SKIP 11: missing month/HQLA assets")
        return
    df = add_baseline_bank_rows(
        df,
        value_col=ratio,
        baseline_value_col=baseline_hqla_assets_col(df),
        path_col="__path_HQLA_assets_ratio",
    )
    q = df.groupby(month)["__path_HQLA_assets_ratio"].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack().reset_index()
    q = q.rename(columns={month: "stress_month", 0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95"})
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Cross-bank HQLA / Assets Distribution: Baseline and Stress Horizon", "Stress month", "HQLA / total assets")
    ax.fill_between(q["stress_month"], q["p05"], q["p95"], color=LIGHT_BLUE, alpha=0.25, label="p05 to p95 range")
    ax.fill_between(q["stress_month"], q["p25"], q["p75"], color=LIGHT_BLUE, alpha=0.45, label="p25 to p75 range")
    ax.plot(q["stress_month"], q["p50"], marker="o", linewidth=3, color=BLUE, label="Median bank")
    ax.plot(q["stress_month"], q["p05"], marker="o", linewidth=2.5, color=BURGUNDY, label="5th percentile")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    ax.set_xticks(q["stress_month"])
    ax.legend(frameon=False, fontsize=11, loc="best")
    savefig(fig, "11_cross_bank_hqla_assets_distribution.png")


def chart_12_support_concentration(bank_monthly: Optional[pd.DataFrame], title_prefix: str = "Preferred policy") -> None:
    if bank_monthly is None:
        log("SKIP 12: no bank-month data")
        return
    bank = get_bank_col(bank_monthly)
    support = first_existing(bank_monthly, ["cb_support_granted", "support_granted", "total_cb_support_granted"])
    if bank is None or support is None:
        log("SKIP 12: missing bank/support columns")
        return
    out = bank_monthly.groupby(bank, as_index=False)[support].sum().sort_values(support, ascending=False)
    out = out[out[support] > 0]
    if out.empty:
        log("SKIP 12: no positive CBA support in bank-month data")
        return
    top = out.head(10).iloc[::-1]
    top3_share = out.head(3)[support].sum() / out[support].sum() if out[support].sum() > 0 else np.nan

    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    setup_ax(ax, f"{title_prefix}: CBA Support Concentration by Bank", "Cumulative CBA support granted", "Bank")
    ax.barh(top[bank], top[support], color=BLUE, alpha=0.95)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.text(0.02, 0.03, f"Top 3 banks account for {100*top3_share:.1f}% of total support.", transform=ax.transAxes,
            fontsize=12, color=NAVY, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    savefig(fig, "12_cba_support_concentration_by_bank.png")


def chart_13_normalized_support_burden(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP 13: no bank-month data")
        return
    df = bank_monthly.copy()
    bank = get_bank_col(df)
    month = get_month_col(df)
    support = first_existing(df, ["cb_support_granted", "support_granted", "total_cb_support_granted"])
    assets = first_existing(df, ["total_assets", "final_total_assets", "assets"])
    hqla = first_existing(df, ["HQLA", "hqla"])
    if bank is None or support is None or assets is None:
        log("SKIP 13: missing bank/support/assets columns")
        return
    total_support = df.groupby(bank, as_index=False)[support].sum().rename(columns={support: "support"})
    if month is not None:
        start = df.sort_values(month).groupby(bank, as_index=False).first()
    else:
        start = df.groupby(bank, as_index=False).first()
    base_cols = [bank, assets] + ([hqla] if hqla else [])
    out = total_support.merge(start[base_cols], on=bank, how="left")
    out["support_to_assets"] = np.where(out[assets] > 0, out["support"] / out[assets], np.nan)
    if hqla:
        out["support_to_hqla"] = np.where(out[hqla] > 0, out["support"] / out[hqla], np.nan)
    out = out[out["support"] > 0]
    if out.empty:
        log("SKIP 13: no positive support")
        return
    out = out.sort_values("support_to_assets", ascending=False).head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2 if hqla else 1, figsize=(15, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]
    setup_ax(ax, "CBA Support Burden Normalized by Bank Size", "Support / starting total assets", "Bank")
    ax.barh(out[bank], out["support_to_assets"], color=BLUE, alpha=0.95)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    if hqla:
        out2 = out.sort_values("support_to_hqla", ascending=False).iloc[::-1]
        ax2 = axes[1]
        setup_ax(ax2, "CBA Support Burden Relative to Starting HQLA", "Support / starting HQLA", "")
        ax2.barh(out2[bank], out2["support_to_hqla"], color=GOLD, alpha=0.95)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    savefig(fig, "13_normalized_cba_support_burden.png")


def chart_14_cba_shortfall_coverage(monthly: Optional[pd.DataFrame]) -> None:
    if monthly is None or "stress_month" not in monthly.columns:
        log("SKIP 14: no monthly data")
        return
    df = monthly.copy()
    cash = first_existing(df, ["aggregate_cash_gap_before_support", "aggregate_cash_shortfall_before_support", "total_cash_shortfall_before_support"])
    cb = first_existing(df, ["aggregate_cb_support_granted", "total_cb_support_granted", "cb_support_granted"])
    sales = first_existing(df, ["aggregate_asset_sale_proceeds", "total_asset_sale_proceeds", "asset_sale_proceeds"])
    residual = first_existing(df, ["aggregate_final_unresolved_shortfall", "residual_shortfall", "final_unresolved_shortfall_amount"])
    if cash is None or cb is None:
        log("SKIP 14: missing cash/support columns")
        return
    if sales is None:
        df["__sales"] = 0.0
        sales = "__sales"
    if residual is None:
        df["__residual"] = np.maximum(df[cash] - df[cb] - df[sales], 0)
        residual = "__residual"
    denom = df[cash].replace(0, np.nan)
    tmp = pd.DataFrame({
        "stress_month": df["stress_month"],
        "cba": (df[cb] / denom).fillna(0),
        "sales": (df[sales] / denom).fillna(0),
        "residual": (df[residual] / denom).fillna(0),
    })
    # Only plot months with a real cash gap; otherwise the empty months confuse the story.
    tmp = tmp[df[cash].fillna(0) > 0]
    if tmp.empty:
        log("SKIP 14: no positive cash gaps")
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "CBA Shortfall Coverage Ratio by Month", "Stress month", "Share of original cash shortfall")
    x = tmp["stress_month"].astype(int)
    ax.bar(x, tmp["cba"], color=BLUE, width=0.72, label="Covered by CBA refinancing")
    ax.bar(x, tmp["sales"], bottom=tmp["cba"], color=GOLD, width=0.72, label="Covered by asset sales")
    ax.bar(x, tmp["residual"], bottom=tmp["cba"] + tmp["sales"], color=BURGUNDY, width=0.72, label="Residual shortfall")
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=1.6)
    ax.set_ylim(0, max(1.05, float((tmp["cba"] + tmp["sales"] + tmp["residual"]).max()) * 1.15))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    ax.set_xticks(x)
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
    savefig(fig, "14_cba_shortfall_coverage_ratio_by_month.png")


# =============================================================================
# Policy scenario charts
# =============================================================================


def chart_policy_scenario_comparison(policy: Optional[pd.DataFrame]) -> None:
    df = tidy_policy_summary(policy)
    if df is None:
        log("SKIP policy comparison: no policy scenario summary")
        return
    needed_any = ["total_cb_support_granted", "total_fire_sale_mtm_loss", "minimum_LCR"]
    if not any(c in df.columns for c in needed_any):
        log("SKIP policy comparison: no usable policy metrics")
        return
    metrics = [c for c in needed_any if c in df.columns]
    titles = {
        "total_cb_support_granted": "CBA support granted",
        "total_fire_sale_mtm_loss": "Fire-sale MTM loss",
        "minimum_LCR": "Minimum LCR",
    }
    colors = {
        "total_cb_support_granted": BLUE,
        "total_fire_sale_mtm_loss": BURGUNDY,
        "minimum_LCR": GOLD,
    }
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 7.2), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    labels = [short_label(s, 30) for s in df["scenario"]]
    y = np.arange(len(df))
    for ax, metric in zip(axes, metrics):
        setup_ax(ax, titles[metric], "Value", "")
        ax.barh(y, df[metric], color=colors[metric], alpha=0.95)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        if metric != "minimum_LCR":
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
        else:
            ax.axvline(1.0, color=BURGUNDY, linestyle="--", linewidth=1.5)
    fig.suptitle("Policy Scenario Comparison: Key Regimes", fontsize=20, fontweight="bold", y=1.01, color=NAVY)
    savefig(fig, "16_policy_scenario_comparison_key_regimes.png")


def chart_policy_frontier(policy: Optional[pd.DataFrame]) -> None:
    df = tidy_policy_summary(policy)
    if df is None:
        log("SKIP policy frontier: no policy scenario summary")
        return
    if not require_cols(df, ["scenario", "total_cb_support_granted", "total_fire_sale_mtm_loss"], "policy frontier"):
        return
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Policy Frontier: Liquidity Burden vs Fire-sale Losses", "Total CBA support granted", "Total fire-sale MTM loss")
    size_col = first_existing(df, ["unresolved_banks", "final_terminal_distress_after_interbank_banks"])
    if size_col:
        sizes = 280 + 80 * pd.to_numeric(df[size_col], errors="coerce").fillna(0)
    else:
        sizes = 380
    ax.scatter(df["total_cb_support_granted"], df["total_fire_sale_mtm_loss"], s=sizes, color=BLUE, edgecolor=NAVY, alpha=0.85)
    for _, r in df.iterrows():
        ax.annotate(short_label(r["scenario"], 26), (r["total_cb_support_granted"], r["total_fire_sale_mtm_loss"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=10, color=NAVY)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.text(0.02, 0.03, "Lower-left is preferable: less central-bank liquidity and lower fire-sale damage.", transform=ax.transAxes,
            fontsize=11.5, color=NAVY, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
    savefig(fig, "17_policy_frontier_key_regimes.png")


# =============================================================================
# Monte Carlo charts
# =============================================================================


def ensure_stress_level_order(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "stress_level" not in out.columns:
        return out
    out["stress_level"] = out["stress_level"].astype(str).str.lower()
    out["stress_level"] = pd.Categorical(out["stress_level"], categories=STRESS_ORDER, ordered=True)
    return out.sort_values("stress_level")


def chart_mc_liquidity_probability(mc_runs: Optional[pd.DataFrame]) -> None:
    if mc_runs is None or "stress_level" not in mc_runs.columns:
        log("SKIP MC probability curve: no run summary")
        return
    df = mc_runs.copy()
    df = ensure_stress_level_order(df)

    support_col = first_existing(df, ["total_cb_support_granted", "aggregate_cb_support_granted"])
    min_lcr_col = first_existing(df, ["min_post_interbank_LCR", "minimum_final_post_interbank_LCR", "minimum_LCR"])
    unresolved_col = first_existing(df, ["ever_final_unresolved_shortfall_banks", "final_terminal_distress_after_interbank_banks"])
    terminal_col = first_existing(df, ["ever_terminal_distress_after_interbank_banks", "final_terminal_distress_after_interbank_banks", "ever_post_interbank_insolvent_banks"])

    tmp = pd.DataFrame({"stress_level": df["stress_level"]})
    if support_col:
        tmp["Any CBA support"] = pd.to_numeric(df[support_col], errors="coerce").fillna(0) > 0
    if min_lcr_col:
        min_lcr = pd.to_numeric(df[min_lcr_col], errors="coerce")
        tmp["Minimum LCR < 1"] = min_lcr < 1.0
        tmp["Minimum LCR < 0.75"] = min_lcr < 0.75
    if unresolved_col:
        tmp["Any unresolved shortfall"] = pd.to_numeric(df[unresolved_col], errors="coerce").fillna(0) > 0
    if terminal_col:
        tmp["Any terminal distress"] = pd.to_numeric(df[terminal_col], errors="coerce").fillna(0) > 0

    cols = [c for c in tmp.columns if c != "stress_level"]
    if not cols:
        log("SKIP MC probability curve: no event columns")
        return
    prob = tmp.groupby("stress_level", observed=False)[cols].mean().reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Monte Carlo Liquidity Stress Probability Curve", "Stress level", "Probability")
    palette = [BLUE, BURGUNDY, GOLD, NAVY, GREY]
    for col, color in zip(cols, palette):
        ax.plot(prob["stress_level"].astype(str), prob[col], marker="o", linewidth=3, color=color, label=col)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
    savefig(fig, "18_monte_carlo_liquidity_stress_probability_curve.png")


def chart_mc_min_lcr_quantiles(mc_runs: Optional[pd.DataFrame]) -> None:
    if mc_runs is None or "stress_level" not in mc_runs.columns:
        log("SKIP MC min LCR quantiles: no run summary")
        return
    df = mc_runs.copy()
    c = first_existing(df, ["min_post_interbank_LCR", "minimum_final_post_interbank_LCR", "minimum_LCR"])
    if c is None:
        log("SKIP MC min LCR quantiles: no min LCR column")
        return
    df = ensure_stress_level_order(df)
    q = df.groupby("stress_level", observed=False)[c].quantile([0.05, 0.25, 0.50, 0.75, 0.95]).unstack().reset_index()
    q = q.rename(columns={0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95"})
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Monte Carlo Distribution of Weakest-bank LCR", "Stress level", "Minimum post-interbank LCR")
    ax.axhline(1.0, color=BURGUNDY, linestyle="--", linewidth=1.8, label="LCR threshold = 1")
    ax.fill_between(x, q["p05"], q["p95"], color=LIGHT_BLUE, alpha=0.25, label="p05 to p95 range")
    ax.fill_between(x, q["p25"], q["p75"], color=LIGHT_BLUE, alpha=0.45, label="p25 to p75 range")
    ax.plot(x, q["p50"], marker="o", linewidth=3, color=BLUE, label="Median weakest-bank LCR")
    ax.plot(x, q["p05"], marker="o", linewidth=2.5, color=BURGUNDY, label="5th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(q["stress_level"].astype(str))
    ax.legend(frameon=False, fontsize=11, loc="best")
    savefig(fig, "19_monte_carlo_min_lcr_quantiles.png")


def chart_mc_bank_support_heatmap(mc_bank: Optional[pd.DataFrame]) -> None:
    if mc_bank is None or "stress_level" not in mc_bank.columns or "bank_name" not in mc_bank.columns:
        log("SKIP MC bank support heatmap: no bank summary")
        return
    support = first_existing(mc_bank, ["total_cb_support_granted", "cb_support_granted", "aggregate_cb_support_granted"])
    if support is None:
        log("SKIP MC bank support heatmap: no support column")
        return
    df = mc_bank.copy()
    df["stress_level"] = df["stress_level"].astype(str).str.lower()
    piv = df.pivot_table(index="bank_name", columns="stress_level", values=support, aggfunc="mean").reindex(columns=STRESS_ORDER)
    piv = piv.fillna(0)
    # Sort by severe expected support, then total support.
    sort_col = "severe" if "severe" in piv.columns else piv.sum(axis=1).idxmax()
    if isinstance(sort_col, str) and sort_col in piv.columns:
        piv = piv.loc[piv[sort_col].sort_values(ascending=False).index]
    else:
        piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    piv = piv.head(14)

    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)
    ax.set_facecolor(BG)
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues")
    ax.set_title("Monte Carlo Bank-level CBA Support Burden", fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Stress level", fontsize=13)
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=11)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    vmax = np.nanmax(piv.values) if np.isfinite(piv.values).any() else 0
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            color = "white" if vmax > 0 and val > 0.65 * vmax else "#111111"
            ax.text(j, i, format_k(val), ha="center", va="center", fontsize=9, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("Expected CBA support granted", fontsize=11)
    savefig(fig, "20_monte_carlo_bank_support_heatmap.png")


def chart_15_mc_cba_support_tail_risk(mc_runs: Optional[pd.DataFrame]) -> None:
    if mc_runs is None or "stress_level" not in mc_runs.columns:
        log("SKIP 15: no MC run summary")
        return
    df = mc_runs.copy()
    support = first_existing(df, ["total_cb_support_granted", "aggregate_cb_support_granted"])
    if support is None:
        log("SKIP 15: no support column")
        return
    df = ensure_stress_level_order(df)
    q = df.groupby("stress_level", observed=False)[support].quantile([0.50, 0.95, 0.99]).unstack().reset_index()
    mean = df.groupby("stress_level", observed=False)[support].mean().reset_index(name="mean")
    q = q.merge(mean, on="stress_level", how="left").rename(columns={0.50: "p50", 0.95: "p95", 0.99: "p99"})
    x = np.arange(len(q))
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Monte Carlo Tail Risk of CBA Liquidity Support", "Stress level", "Total CBA support granted")
    ax.fill_between(x, q["p95"], q["p99"], color=BURGUNDY, alpha=0.15, label="Tail liquidity buffer")
    ax.plot(x, q["mean"], marker="o", linewidth=3, color=BLUE, label="Expected support")
    ax.plot(x, q["p95"], marker="o", linewidth=3, color=GOLD, label="95th percentile")
    ax.plot(x, q["p99"], marker="o", linewidth=3, color=BURGUNDY, label="99th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(q["stress_level"].astype(str))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    savefig(fig, "15_monte_carlo_cba_support_tail_risk.png")


# =============================================================================
# Currency and selected-bank mechanism charts
# =============================================================================


def chart_currency_hqla_assets(csvs: list[Path]) -> None:
    # Try to find any currency sheet-like output.
    candidates = []
    for p in csvs:
        name = p.name.lower()
        if "currency" in name or "amd" in name or "fx" in name:
            df = read_csv_safe(p)
            if df is None:
                continue
            cols = set(df.columns)
            score = 0
            if "stress_month" in cols:
                score += 5
            if any(c in cols for c in ["currency", "currency_bucket", "currency_name"]):
                score += 5
            if any(c in cols for c in ["HQLA_assets_ratio", "HQLA", "total_assets"]):
                score += 8
            candidates.append((score, p, df))
    if not candidates or max(c[0] for c in candidates) < 10:
        log("SKIP currency HQLA/assets: no currency dataset")
        return
    score, path, df = sorted(candidates, key=lambda x: x[0], reverse=True)[0]
    log(f"using currency chart data: {path.relative_to(PROJECT_ROOT)} score={score}")
    month = get_month_col(df)
    cur_col = first_existing(df, ["currency", "currency_bucket", "currency_name"])
    ratio = get_hqla_assets_col(df)
    if month is None or cur_col is None or ratio is None:
        log("SKIP currency HQLA/assets: missing columns")
        return
    piv = df.pivot_table(index=month, columns=cur_col, values=ratio, aggfunc="mean").sort_index()
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    setup_ax(ax, "Change in System Liquidity Position by Currency", "Stress month", "HQLA / total assets")
    colors = [LIGHT_BLUE, BURGUNDY, OLIVE, BLUE]
    for col, color in zip(piv.columns, colors):
        ax.plot(piv.index, piv[col], marker="o", linewidth=3, color=color, label=str(col))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    ax.set_xticks(piv.index)
    ax.legend(frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=min(4, len(piv.columns)))
    savefig(fig, "21_liquidity_position_by_currency.png")


def chart_selected_bank_mechanism(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP selected bank mechanism: no bank-month data")
        return

    df = one_row_per_bank_month(bank_monthly.copy())

    month = get_month_col(df)
    bank = get_bank_col(df)
    final_lcr = get_lcr_col(df)

    if month is None or bank is None or final_lcr is None:
        log("SKIP selected bank mechanism: missing columns")
        return

    pre_lcr = get_pre_lcr_col(df)
    post_sale = first_existing(df, ["post_asset_sale_LCR", "post_reaction_LCR"])
    post_interbank = final_lcr

    support = first_existing(df, ["cb_support_granted", "support_granted"])
    assets = first_existing(df, ["total_assets", "assets", "final_total_assets"])

    agg_spec = {
        "trough": (post_interbank, "min"),
    }

    if support:
        agg_spec["total_support"] = (support, "sum")

    if assets:
        agg_spec["assets"] = (assets, "first")

    metrics = df.groupby(bank).agg(**agg_spec).reset_index()

    selected_candidates = []

    if not metrics.empty:
        selected_candidates.append(metrics.sort_values("trough").iloc[0][bank])

        if support and "total_support" in metrics.columns:
            selected_candidates.append(
                metrics.sort_values("total_support", ascending=False).iloc[0][bank]
            )

        if assets and "assets" in metrics.columns:
            selected_candidates.append(
                metrics.sort_values("assets", ascending=False).iloc[0][bank]
            )

    selected: list[str] = []
    for b in selected_candidates:
        if b not in selected:
            selected.append(b)

    selected = selected[:3]

    if not selected:
        log("SKIP selected bank mechanism: no banks selected")
        return

    for b in selected:
        sub = df[df[bank] == b].copy()
        sub = add_baseline_bank_rows(
            sub,
            value_col=post_interbank,
            baseline_value_col=baseline_lcr_col(sub),
            path_col="__selected_path_LCR",
        )
        sub = _collapse_duplicate_keys(sub, [bank, month])
        sub = sub.sort_values(month)

        if sub[month].duplicated().any():
            raise ValueError(f"Duplicate stress_month values remain for {b}")

        fig, ax1 = plt.subplots(figsize=FIGSIZE_WIDE)
        setup_ax(ax1, f"Selected-bank Mechanism: {b}", "Stress month", "LCR")

        threshold = get_active_lcr_threshold(bank_monthly=df)
        ax1.axhline(
            threshold,
            color=BURGUNDY,
            linestyle="--",
            linewidth=1.6,
            label=f"LCR threshold = {threshold:.1f}",
        )

        if pre_lcr and pre_lcr in sub.columns:
            ax1.plot(
                sub[month],
                sub[pre_lcr],
                marker="o",
                linewidth=2.5,
                color=GREY,
                label="Pre-reaction LCR",
            )

        if post_sale and post_sale in sub.columns and post_sale != post_interbank:
            ax1.plot(
                sub[month],
                sub[post_sale],
                marker="o",
                linewidth=2.5,
                color=GOLD,
                label="Post asset-sale LCR",
            )

        ax1.plot(
            sub[month],
            sub["__selected_path_LCR"],
            marker="o",
            linewidth=3.0,
            color=BLUE,
            label="Final post-transmission LCR",
        )

        ax1.set_xticks(sub[month])

        if support and support in sub.columns:
            ax2 = ax1.twinx()
            ax2.bar(
                sub[month],
                sub[support],
                color=NAVY,
                alpha=0.18,
                width=0.65,
                label="CBA support granted",
            )
            ax2.set_ylabel("CBA support", fontsize=12)
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                frameon=False,
                fontsize=10,
                loc="best",
            )
        else:
            ax1.legend(frameon=False, fontsize=10, loc="best")

        fname = re.sub(r"[^A-Za-z0-9]+", "_", str(b)).strip("_").lower()
        savefig(fig, f"22_selected_bank_mechanism_{fname}.png")

    sec_cols = [
        c
        for c in [
            "HQLA",
            "government_securities",
            "bank_securities",
            "corporate_securities",
            "cash",
            "cash_and_reserves",
        ]
        if c in df.columns
    ]

    if len(sec_cols) >= 2:
        for b in selected:
            sub = df[df[bank] == b].copy()
            sub = _collapse_duplicate_keys(sub, [bank, month])
            sub = sub.sort_values(month)

            if sub[month].duplicated().any():
                raise ValueError(f"Duplicate stress_month values remain for {b}")

            fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
            setup_ax(ax, f"Liquidity-buffer Composition: {b}", "Stress month", "Amount")

            for c, color in zip(
                sec_cols,
                [BLUE, GOLD, BURGUNDY, OLIVE, LIGHT_BLUE, GREY],
            ):
                ax.plot(
                    sub[month],
                    sub[c],
                    marker="o",
                    linewidth=2.5,
                    color=color,
                    label=c.replace("_", " "),
                )

            ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
            ax.set_xticks(sub[month])
            ax.legend(frameon=False, fontsize=10, loc="best", ncol=2)

            fname = re.sub(r"[^A-Za-z0-9]+", "_", str(b)).strip("_").lower()
            savefig(fig, f"23_liquidity_buffer_composition_{fname}.png")


def chart_bank_deterioration_ranking(bank_monthly: Optional[pd.DataFrame]) -> None:
    if bank_monthly is None:
        log("SKIP deterioration ranking: no bank-month data")
        return
    df = bank_monthly.copy()
    bank = get_bank_col(df)
    month = get_month_col(df)
    lcr = get_lcr_col(df)
    ratio = get_hqla_assets_col(df)
    if bank is None or month is None or lcr is None:
        log("SKIP deterioration ranking: missing columns")
        return
    first = df.sort_values(month).groupby(bank).first()
    trough = df.groupby(bank)[lcr].min()
    out = pd.DataFrame({
        "start_lcr": first[lcr],
        "trough_lcr": trough,
    })
    out["lcr_drop"] = out["start_lcr"] - out["trough_lcr"]
    if ratio:
        out["start_hqla_assets"] = first[ratio]
        out["trough_hqla_assets"] = df.groupby(bank)[ratio].min()
        out["hqla_assets_drop"] = out["start_hqla_assets"] - out["trough_hqla_assets"]
    out = out.sort_values("lcr_drop", ascending=False).head(10).iloc[::-1]

    fig, axes = plt.subplots(1, 2 if ratio else 1, figsize=(15, 7))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    ax = axes[0]
    setup_ax(ax, "Banks Ranked by LCR Deterioration", "Start LCR minus trough LCR", "Bank")
    ax.barh(out.index, out["lcr_drop"], color=BURGUNDY, alpha=0.92)
    if ratio:
        out2 = out.sort_values("hqla_assets_drop", ascending=False).iloc[::-1]
        ax2 = axes[1]
        setup_ax(ax2, "Banks Ranked by HQLA / Assets Deterioration", "Start ratio minus trough ratio", "")
        ax2.barh(out2.index, out2["hqla_assets_drop"], color=BLUE, alpha=0.92)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    savefig(fig, "24_bank_liquidity_deterioration_ranking.png")

# ===================================================
# LCR under different LCR rules
# ====================================================

def chart_policy_hqla_assets_paths(csvs: list[Path]) -> None:
    """
    Van den End-style policy-regime chart:
    HQLA/assets ratio through the stress horizon under alternative LCR rules.

    Uses outputs/tables/sensitivity_monthly_summary.csv produced by
    scripts/30_run_sensitivity.py.
    """
    target = None

    for p in csvs:
        if p.name.lower() == "sensitivity_monthly_summary.csv":
            target = p
            break

    if target is None:
        log("SKIP policy HQLA/assets paths: sensitivity_monthly_summary.csv missing")
        return

    df = read_csv_safe(target)

    if df is None or df.empty:
        log("SKIP policy HQLA/assets paths: sensitivity monthly data empty")
        return

    required = [
        "scenario",
        "policy_family",
        "stress_month",
        "average_post_interbank_HQLA_assets_ratio",
        "minimum_post_interbank_HQLA_assets_ratio",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        log(f"SKIP policy HQLA/assets paths: missing columns {missing}")
        return

    # Keep only the policy regimes comparable to the Van den End figures.
    scenario_map = {
        "lcr_req_120_no_lolr": "LCR_req 120%",
        "lcr_low_100_no_lolr": "LCR_low 100%",
        "lcr_star_wide_buffer_no_lolr": "LCR* wider buffer",
        "lcr_cb_collateral_capped_refinancing": "LCR_CB",
        "asset_purchase_state_bond_floor": "Asset purchases",
        "lcr_cb_plus_asset_purchases": "LCR_CB + AP",
        "base_full_engine": "Base full engine",

        # Backward-compatible aliases, in case older outputs are present.
        "lcr_req_100_no_lolr": "LCR_req 100%",
        "lcr_low_90_no_lolr": "LCR_low 90%",
    }

    df = df[df["scenario"].isin(scenario_map.keys())].copy()

    if df.empty:
        log("SKIP policy HQLA/assets paths: no Van den End-style scenarios found")
        return

    df["scenario_label"] = df["scenario"].map(scenario_map)

    order = [
        "LCR_req 120%",
        "LCR_low 100%",
        "LCR* wider buffer",
        "LCR_CB",
        "Asset purchases",
        "LCR_CB + AP",
        "Base full engine",
        "LCR_req 100%",
        "LCR_low 90%",
    ]

    df["scenario_label"] = pd.Categorical(
        df["scenario_label"],
        categories=order,
        ordered=True,
    )

    df = df.sort_values(["scenario_label", "stress_month"])

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), sharex=True)

    panels = [
        (
            axes[0],
            "average_post_interbank_HQLA_assets_ratio",
            "System average HQLA / assets",
            "Average HQLA / assets",
        ),
        (
            axes[1],
            "minimum_post_interbank_HQLA_assets_ratio",
            "Weakest-bank HQLA / assets",
            "Minimum HQLA / assets",
        ),
    ]

    color_cycle = {
        "LCR_req 120%": NAVY,
        "LCR_low 100%": GOLD,
        "LCR* wider buffer": DARK_GREY,
        "LCR_CB": BLUE,
        "Asset purchases": LIGHT_BLUE,
        "LCR_CB + AP": OLIVE,
        "Base full engine": BURGUNDY,
        "LCR_req 100%": NAVY,
        "LCR_low 90%": GOLD,
    }

    linestyle_cycle = {
        "LCR_req 120%": "-",
        "LCR_low 100%": "-",
        "LCR* wider buffer": "--",
        "LCR_CB": "-",
        "Asset purchases": "-",
        "LCR_CB + AP": "-.",
        "Base full engine": ":",
        "LCR_req 100%": "-",
        "LCR_low 90%": "-",
    }

    for ax, value_col, title, ylabel in panels:
        setup_ax(ax, title, "Stress month", ylabel)

        for scenario_label, group in df.groupby("scenario_label", observed=True):
            if group.empty:
                continue

            label = str(scenario_label)

            ax.plot(
                group["stress_month"],
                group[value_col],
                marker="o",
                linewidth=2.8,
                linestyle=linestyle_cycle.get(label, "-"),
                color=color_cycle.get(label, GREY),
                label=label,
            )

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
        ax.set_xticks(sorted(df["stress_month"].dropna().astype(int).unique()))

        # Diagnostic threshold only, not a regulatory minimum.
        ax.axhline(
            0.10,
            color=BURGUNDY,
            linestyle=":",
            linewidth=1.6,
            alpha=0.85,
            label="10% monitoring line" if ax is axes[0] else None,
        )

    axes[0].legend(
        frameon=False,
        fontsize=10.5,
        loc="lower center",
        bbox_to_anchor=(1.05, -0.28),
        ncol=4,
    )

    fig.suptitle(
        "HQLA / Assets Under Alternative LCR Policy Regimes",
        fontsize=20,
        fontweight="bold",
        y=1.02,
        color=NAVY,
    )

    savefig(fig, "16b_policy_hqla_assets_paths_by_lcr_regime.png")

# =============================================================================
# Main
# =============================================================================

def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    store = load_data()

    # Clean deterministic baseline datasets once globally.
    if store.base_bank_monthly is not None:
        store.base_bank_monthly = one_row_per_bank_month(store.base_bank_monthly)

    if store.base_monthly is not None:
        store.base_monthly = one_row_per_month(store.base_monthly)

    monthly = merged_monthly(store)

    if monthly is not None:
        monthly = one_row_per_month(monthly)
        log(f"monthly merged columns: {list(monthly.columns)}")

    if store.base_bank_monthly is not None:
        log(f"bank-month columns: {list(store.base_bank_monthly.columns)}")

    # -------------------------------------------------------------------------
    # Baseline deterministic storytelling charts.
    # -------------------------------------------------------------------------
    chart_01_base_lcr_path(monthly, store.base_bank_monthly)
    chart_01b_base_hqla_assets_path(monthly, store.base_bank_monthly)
    chart_01c_all_bank_lcr_lines(store.base_bank_monthly)
    chart_01d_all_bank_hqla_assets_lines(store.base_bank_monthly)
    chart_02_lcr_heatmap(store.base_bank_monthly)
    chart_03_hqla_assets_heatmap(store.base_bank_monthly)
    chart_04_bank_lcr_small_multiples(store.base_bank_monthly)
    chart_05_bank_hqla_assets_small_multiples(store.base_bank_monthly)
    chart_06_cash_gap_absorption(monthly)
    chart_07_lolr_mechanism(monthly)
    chart_08_loss_decomposition(monthly)
    chart_09_breach_counts(store.base_bank_monthly)
    chart_10_lcr_distribution(store.base_bank_monthly)
    chart_11_hqla_assets_distribution(store.base_bank_monthly)
    chart_12_support_concentration(store.base_bank_monthly)
    chart_13_normalized_support_burden(store.base_bank_monthly)
    chart_14_cba_shortfall_coverage(monthly)

    # -------------------------------------------------------------------------
    # Monte Carlo charts.
    # These will be skipped cleanly if Monte Carlo outputs are absent.
    # -------------------------------------------------------------------------
    chart_15_mc_cba_support_tail_risk(store.mc_runs)
    chart_mc_liquidity_probability(store.mc_runs)
    chart_mc_min_lcr_quantiles(store.mc_runs)
    chart_mc_bank_support_heatmap(store.mc_bank)

    # -------------------------------------------------------------------------
    # Policy scenario charts.
    # -------------------------------------------------------------------------
    chart_policy_scenario_comparison(store.policy_summary)
    chart_policy_frontier(store.policy_summary)
    chart_policy_hqla_assets_paths(store.csvs)

    # -------------------------------------------------------------------------
    # Additional monitoring / bank-mechanism charts.
    # -------------------------------------------------------------------------
    chart_currency_hqla_assets(store.csvs)
    chart_selected_bank_mechanism(store.base_bank_monthly)
    chart_bank_deterioration_ranking(store.base_bank_monthly)

    log(f"Done. Charts written to: {CHART_DIR}")


if __name__ == "__main__":
    main()