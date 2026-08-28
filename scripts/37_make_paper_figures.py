#!/usr/bin/env python3
"""
Generate paper-grade figures from the stage-aware scenario tables.

This script intentionally uses only the curated paper outputs, not the older
diagnostic chart pipeline. It is meant for the LaTeX paper.

Run from the project root:

    PYTHONPATH=src python scripts/37_make_paper_figures.py
"""

from __future__ import annotations

from pathlib import Path
import textwrap

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
# Paths
# =============================================================================

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIO_SUMMARY = TABLE_DIR / "paper_stage_aware_scenario_summary.csv"
MONTHLY_SUMMARY = TABLE_DIR / "paper_stage_aware_monthly_summary.csv"
BANK_RESULTS = TABLE_DIR / "paper_scenario_bank_results.csv"
MC_RUN_SUMMARY = ROOT / "outputs" / "monte_carlo_shock_run_summary.csv"

# Fallback if Monte Carlo outputs are saved under outputs/tables.
MC_RUN_SUMMARY_ALT = TABLE_DIR / "monte_carlo_shock_run_summary.csv"


# =============================================================================
# Styling
# =============================================================================

DPI = 300
NAVY = "#06183D"
BLUE = "#1F5A82"
LIGHT_BLUE = "#74A6BF"
BURGUNDY = "#8C0030"
GOLD = "#D8A719"
OLIVE = "#4A5D23"
GREY = "#8A8A8A"
DARK_GREY = "#404040"
BG = "#F2F2F2"
GRID = "#D7D7D7"

CHANNEL_ORDER = [
    "first_round_only",
    "plus_bank_reaction",
    "plus_lolr_support",
    "plus_fire_sales",
    "full_engine",
]

CHANNEL_LABELS = {
    "first_round_only": "First-round\nonly",
    "plus_bank_reaction": "+ Bank\nreaction",
    "plus_lolr_support": "+ LoLR\nsupport",
    "plus_fire_sales": "+ Fire-sale\nfeedback",
    "full_engine": "Full\nengine",
}

POLICY_ORDER = [
    "no_cb_support",
    "collateral_capped_lolr",
    "full_coverage_lolr",
    "asset_purchases_only",
    "lolr_plus_asset_purchases",
]

POLICY_LABELS = {
    "no_cb_support": "No CB\nsupport",
    "collateral_capped_lolr": "Collateral-\ncapped LoLR",
    "full_coverage_lolr": "Full-coverage\nLoLR",
    "asset_purchases_only": "Asset\npurchases",
    "lolr_plus_asset_purchases": "LoLR + asset\npurchases",
}

STRESS_ORDER = ["mild", "moderate", "adverse", "severe"]


def log(msg: str) -> None:
    print(f"[37_make_paper_figures] {msg}")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def format_k(x: float, _pos=None) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax >= 1_000_000:
        return f"{x/1_000_000:.1f}m"
    if ax >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.0f}"


def format_pct(x: float, _pos=None) -> str:
    return f"{100*x:.0f}%"


def setup_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linestyle="--", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.set_title(title, fontsize=14.5, fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(axis="both", labelsize=10)


def savefig(fig: plt.Figure, filename: str) -> None:
    path = FIG_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log(f"saved {path.relative_to(ROOT)}")


def ordered(df: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    out = df[df["scenario"].isin(scenarios)].copy()
    out["__order"] = out["scenario"].map({s: i for i, s in enumerate(scenarios)})
    return out.sort_values("__order").drop(columns="__order")


def add_value_labels(ax: plt.Axes, values, fmt="{:.2f}", dy=0.02, fontsize=9) -> None:
    for i, v in enumerate(values):
        if pd.notna(v):
            ax.text(i, v + dy, fmt.format(v), ha="center", va="bottom", fontsize=fontsize, color=DARK_GREY)


# =============================================================================
# Figure 1: Benchmark / channel attribution LCR
# =============================================================================

def figure_channel_lcr(summary: pd.DataFrame) -> None:
    df = ordered(summary, CHANNEL_ORDER)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    setup_ax(
        ax,
        "Channel attribution: active-stage LCR",
        "",
        "LCR",
    )

    width = 0.34
    ax.axhspan(0, 1.20, color=BURGUNDY, alpha=0.07)
    ax.axhline(1.20, color=BURGUNDY, linestyle="--", linewidth=1.6, label="LCR threshold = 1.20")
    ax.axhline(1.00, color=DARK_GREY, linestyle=":", linewidth=1.3, label="LCR = 1.00")

    ax.bar(
        x - width / 2,
        df["min_system_LCR"],
        width,
        color=BLUE,
        alpha=0.92,
        label="Minimum system LCR",
    )
    ax.bar(
        x + width / 2,
        df["min_weakest_bank_LCR"],
        width,
        color=GOLD,
        alpha=0.92,
        label="Minimum weakest-bank LCR",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[s] for s in df["scenario"]])
    ax.set_ylim(0, max(1.35, df[["min_system_LCR", "min_weakest_bank_LCR"]].max().max() + 0.12))
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=2)

    savefig(fig, "01_channel_attribution_active_stage_lcr.png")


# =============================================================================
# Figure 2: Channel attribution HQLA/assets and breach counts
# =============================================================================

def figure_channel_hqla(summary: pd.DataFrame) -> None:
    df = ordered(summary, CHANNEL_ORDER)
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    setup_ax(
        ax,
        "Channel attribution: liquidity-stock depletion",
        "",
        "HQLA / total assets",
    )

    width = 0.34
    ax.axhspan(0, 0.10, color=BURGUNDY, alpha=0.07)
    ax.axhline(0.10, color=BURGUNDY, linestyle="--", linewidth=1.6, label="10% monitoring line")

    ax.bar(
        x - width / 2,
        df["min_system_HQLA_assets"],
        width,
        color=BLUE,
        alpha=0.92,
        label="Minimum system HQLA/assets",
    )
    ax.bar(
        x + width / 2,
        df["min_weakest_bank_HQLA_assets"],
        width,
        color=GOLD,
        alpha=0.92,
        label="Minimum weakest-bank HQLA/assets",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[s] for s in df["scenario"]])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_pct))
    ax.set_ylim(0, max(0.34, df[["min_system_HQLA_assets", "min_weakest_bank_HQLA_assets"]].max().max() + 0.04))
    ax.legend(frameon=False, fontsize=10, loc="upper right")

    savefig(fig, "02_channel_attribution_hqla_assets.png")


# =============================================================================
# Figure 3: Channel attribution losses and support
# =============================================================================

def figure_channel_support_losses(summary: pd.DataFrame) -> None:
    df = ordered(summary, CHANNEL_ORDER)
    x = np.arange(len(df))

    cols = [
        ("total_cb_support_granted", "CB support granted", BLUE),
        ("total_asset_sale_loss", "Asset-sale liquidation loss", GOLD),
        ("total_fire_sale_mtm_loss", "Fire-sale MTM loss", BURGUNDY),
        ("total_interbank_contagion_loss", "Interbank contagion loss", OLIVE),
    ]

    fig, ax = plt.subplots(figsize=(11.2, 6.5))
    setup_ax(
        ax,
        "Channel attribution: support needs and loss amplification",
        "",
        "Amount",
    )

    bottom = np.zeros(len(df))
    for col, label, color in cols:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        ax.bar(x, values, bottom=bottom, color=color, alpha=0.90, label=label)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels([CHANNEL_LABELS[s] for s in df["scenario"]])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2)

    savefig(fig, "03_channel_attribution_support_and_losses.png")


# =============================================================================
# Figure 4: Policy counterfactuals
# =============================================================================

def figure_policy_lcr_distress(summary: pd.DataFrame) -> None:
    df = ordered(summary, POLICY_ORDER)
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(11.3, 6.4))
    setup_ax(
        ax1,
        "Policy counterfactuals: final liquidity and distress",
        "",
        "Final system LCR",
    )

    ax1.axhline(1.20, color=BURGUNDY, linestyle="--", linewidth=1.5, label="LCR threshold = 1.20")
    ax1.bar(
        x,
        df["final_system_LCR"],
        color=BLUE,
        alpha=0.90,
        label="Final system LCR",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([POLICY_LABELS[s] for s in df["scenario"]])
    ax1.set_ylim(0, max(1.65, df["final_system_LCR"].max() + 0.12))

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        df["final_terminal_distress_after_interbank_banks"],
        color=BURGUNDY,
        marker="o",
        linewidth=2.7,
        label="Terminal distress banks",
    )
    ax2.set_ylabel("Number of terminal-distress banks", fontsize=11.5)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=10, loc="upper left")

    savefig(fig, "04_policy_counterfactual_lcr_and_distress.png")


def figure_policy_support_losses(summary: pd.DataFrame) -> None:
    df = ordered(summary, POLICY_ORDER)
    labels = [POLICY_LABELS[s].replace("\n", " ") for s in df["scenario"]]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    setup_ax(
        ax,
        "Policy counterfactuals: support burden and loss channels",
        "Amount",
        "",
    )

    cols = [
        ("total_cb_support_granted", "CB support granted", BLUE),
        ("total_fire_sale_mtm_loss", "Fire-sale MTM loss", BURGUNDY),
        ("total_interbank_contagion_loss", "Interbank contagion loss", OLIVE),
    ]

    left = np.zeros(len(df))
    for col, label, color in cols:
        values = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        ax.barh(y, values, left=left, color=color, alpha=0.90, label=label)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.legend(frameon=False, fontsize=10, loc="lower right")

    savefig(fig, "05_policy_counterfactual_support_and_losses.png")


# =============================================================================
# Figure 5: Active-stage monthly benchmark paths
# =============================================================================

def figure_benchmark_monthly_paths(monthly: pd.DataFrame, bank_results: pd.DataFrame) -> None:
    df = monthly[monthly["scenario"] == "full_engine"].copy()
    if df.empty:
        log("SKIP benchmark monthly paths: full_engine missing")
        return

    full = bank_results[bank_results["scenario"] == "full_engine"].copy()
    if full.empty:
        log("SKIP benchmark baseline insertion: full_engine bank results missing")
        return

    # Keep one pre-shock row per bank. The pre-shock columns are repeated across
    # stress months, so the earliest month is enough.
    full = full.sort_values(["bank_name", "stress_month"]).drop_duplicates("bank_name", keep="first")

    def pick(candidates):
        for c in candidates:
            if c in full.columns and pd.to_numeric(full[c], errors="coerce").notna().any():
                return c
        return None

    hqla_col = pick(["baseline_HQLA", "pre_shock_HQLA", "HQLA"])
    ncof_col = pick(["baseline_NCOF", "pre_shock_NCOF", "NCOF"])
    assets_col = pick(["baseline_total_assets", "pre_shock_total_assets", "total_assets"])
    lcr_col = pick(["baseline_LCR", "pre_shock_LCR", "LCR"])
    hqla_assets_col = pick(["baseline_HQLA_assets_ratio", "pre_shock_HQLA_assets_ratio", "HQLA_assets_ratio"])

    if hqla_col and ncof_col and pd.to_numeric(full[ncof_col], errors="coerce").sum() > 0:
        baseline_system_lcr = (
            pd.to_numeric(full[hqla_col], errors="coerce").sum()
            / pd.to_numeric(full[ncof_col], errors="coerce").sum()
        )
    elif lcr_col:
        baseline_system_lcr = pd.to_numeric(full[lcr_col], errors="coerce").mean()
    else:
        log("SKIP benchmark baseline insertion: no usable baseline LCR columns")
        return

    if hqla_col and assets_col and pd.to_numeric(full[assets_col], errors="coerce").sum() > 0:
        baseline_system_hqla_assets = (
            pd.to_numeric(full[hqla_col], errors="coerce").sum()
            / pd.to_numeric(full[assets_col], errors="coerce").sum()
        )
    elif hqla_assets_col:
        baseline_system_hqla_assets = pd.to_numeric(full[hqla_assets_col], errors="coerce").mean()
    else:
        log("SKIP benchmark baseline insertion: no usable baseline HQLA/assets columns")
        return

    baseline_row = pd.DataFrame(
        {
            "scenario": ["full_engine"],
            "stress_month": [0],
            "system_LCR_active_stage": [baseline_system_lcr],
            "system_HQLA_assets_active_stage": [baseline_system_hqla_assets],
        }
    )

    plot_df = pd.concat(
        [
            baseline_row,
            df[["scenario", "stress_month", "system_LCR_active_stage", "system_HQLA_assets_active_stage"]],
        ],
        ignore_index=True,
    ).sort_values("stress_month")

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    setup_ax(
        ax,
        "Benchmark full-engine path",
        "Stress month",
        "Ratio",
    )

    ax.axhline(1.20, color=BURGUNDY, linestyle="--", linewidth=1.4, label="LCR threshold = 1.20")
    ax.axhline(0.10, color=DARK_GREY, linestyle=":", linewidth=1.3, label="HQLA/assets monitoring line = 10%")

    ax.plot(
        plot_df["stress_month"],
        plot_df["system_LCR_active_stage"],
        marker="o",
        linewidth=3.0,
        color=BLUE,
        label="System LCR, active stage",
    )
    ax.plot(
        plot_df["stress_month"],
        plot_df["system_HQLA_assets_active_stage"],
        marker="o",
        linewidth=3.0,
        color=GOLD,
        label="System HQLA/assets, active stage",
    )

    ax.set_xticks(plot_df["stress_month"].astype(int))
    ax.set_xlim(-0.1, plot_df["stress_month"].max() + 0.1)
    ax.legend(frameon=False, fontsize=10, loc="best")

    savefig(fig, "06_benchmark_full_engine_monthly_paths.png")


# =============================================================================
# Figure 6: Monte Carlo
# =============================================================================

def load_mc() -> pd.DataFrame | None:
    path = MC_RUN_SUMMARY if MC_RUN_SUMMARY.exists() else MC_RUN_SUMMARY_ALT
    if not path.exists():
        log("SKIP Monte Carlo figures: monte_carlo_shock_run_summary.csv missing")
        return None
    return pd.read_csv(path)


def figure_mc_support_tail(mc: pd.DataFrame) -> None:
    if "stress_level" not in mc.columns or "total_cb_support_granted" not in mc.columns:
        log("SKIP MC support tail: required columns missing")
        return

    df = mc.copy()
    df["stress_level"] = pd.Categorical(
        df["stress_level"].astype(str).str.lower(),
        categories=STRESS_ORDER,
        ordered=True,
    )
    q = (
        df.groupby("stress_level", observed=False)["total_cb_support_granted"]
        .quantile([0.50, 0.95, 0.99])
        .unstack()
        .reset_index()
        .rename(columns={0.50: "p50", 0.95: "p95", 0.99: "p99"})
    )
    mean = df.groupby("stress_level", observed=False)["total_cb_support_granted"].mean().reset_index(name="mean")
    q = q.merge(mean, on="stress_level", how="left")
    q = q.dropna(subset=["stress_level"])
    x = np.arange(len(q))

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    setup_ax(
        ax,
        "Monte Carlo tail risk of CB liquidity support",
        "Stress level",
        "Total CB support granted",
    )

    ax.fill_between(x, q["p95"], q["p99"], color=BURGUNDY, alpha=0.16, label="p95 to p99 range")
    ax.plot(x, q["mean"], marker="o", linewidth=2.8, color=BLUE, label="Mean")
    ax.plot(x, q["p95"], marker="o", linewidth=2.8, color=GOLD, label="95th percentile")
    ax.plot(x, q["p99"], marker="o", linewidth=2.8, color=BURGUNDY, label="99th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(q["stress_level"].astype(str))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_k))
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    savefig(fig, "07_monte_carlo_cb_support_tail_risk.png")


def figure_mc_min_lcr(mc: pd.DataFrame) -> None:
    lcr_col = "minimum_final_post_interbank_LCR"
    if "stress_level" not in mc.columns or lcr_col not in mc.columns:
        log("SKIP MC minimum LCR: required columns missing")
        return

    df = mc.copy()
    df["stress_level"] = pd.Categorical(
        df["stress_level"].astype(str).str.lower(),
        categories=STRESS_ORDER,
        ordered=True,
    )
    q = (
        df.groupby("stress_level", observed=False)[lcr_col]
        .quantile([0.05, 0.25, 0.50, 0.75, 0.95])
        .unstack()
        .reset_index()
        .rename(columns={0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.95: "p95"})
    )
    q = q.dropna(subset=["stress_level"])
    x = np.arange(len(q))

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    setup_ax(
        ax,
        "Monte Carlo distribution of weakest-bank LCR",
        "Stress level",
        "Minimum final post-interbank LCR",
    )

    ax.axhline(1.20, color=BURGUNDY, linestyle="--", linewidth=1.5, label="LCR threshold = 1.20")
    ax.axhline(1.00, color=DARK_GREY, linestyle=":", linewidth=1.2, label="LCR = 1.00")
    ax.fill_between(x, q["p05"], q["p95"], color=LIGHT_BLUE, alpha=0.26, label="p05 to p95")
    ax.fill_between(x, q["p25"], q["p75"], color=LIGHT_BLUE, alpha=0.46, label="p25 to p75")
    ax.plot(x, q["p50"], marker="o", linewidth=2.8, color=BLUE, label="Median")
    ax.plot(x, q["p05"], marker="o", linewidth=2.4, color=BURGUNDY, label="5th percentile")
    ax.set_xticks(x)
    ax.set_xticklabels(q["stress_level"].astype(str))
    ax.legend(frameon=False, fontsize=10, loc="best")

    savefig(fig, "08_monte_carlo_weakest_bank_lcr_quantiles.png")


# =============================================================================
# LaTeX helper
# =============================================================================

def write_latex_snippets(summary: pd.DataFrame) -> None:
    channel = ordered(summary, CHANNEL_ORDER).copy()
    policy = ordered(summary, POLICY_ORDER).copy()

    display_cols = [
        "scenario",
        "active_stage_label",
        "min_system_LCR",
        "min_weakest_bank_LCR",
        "min_system_HQLA_assets",
        "total_cb_support_granted",
        "total_fire_sale_mtm_loss",
        "total_interbank_contagion_loss",
        "final_terminal_distress_after_interbank_banks",
    ]

    for df, name in [
        (channel, "paper_table_channel_attribution.tex"),
        (policy, "paper_table_policy_counterfactuals.tex"),
    ]:
        out = df[display_cols].copy()
        out["scenario"] = out["scenario"].replace({**CHANNEL_LABELS, **POLICY_LABELS}).str.replace("\n", " ", regex=False)
        out["min_system_LCR"] = out["min_system_LCR"].map(lambda x: f"{x:.3f}")
        out["min_weakest_bank_LCR"] = out["min_weakest_bank_LCR"].map(lambda x: f"{x:.3f}")
        out["min_system_HQLA_assets"] = out["min_system_HQLA_assets"].map(lambda x: f"{100*x:.1f}\\%")
        for c in ["total_cb_support_granted", "total_fire_sale_mtm_loss", "total_interbank_contagion_loss"]:
            out[c] = out[c].map(lambda x: f"{x:,.0f}")
        out["final_terminal_distress_after_interbank_banks"] = out["final_terminal_distress_after_interbank_banks"].astype(int)

        tex = out.to_latex(
            index=False,
            escape=False,
            column_format="llrrrrrrr",
        )
        path = TABLE_DIR / name
        path.write_text(tex)
        log(f"saved {path.relative_to(ROOT)}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    require(SCENARIO_SUMMARY)
    require(MONTHLY_SUMMARY)
    require(BANK_RESULTS)

    plt.rcParams.update({
        "font.family": "serif",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
    })

    summary = pd.read_csv(SCENARIO_SUMMARY)
    monthly = pd.read_csv(MONTHLY_SUMMARY)
    bank_results = pd.read_csv(BANK_RESULTS)

    figure_channel_lcr(summary)
    figure_channel_hqla(summary)
    figure_channel_support_losses(summary)
    figure_policy_lcr_distress(summary)
    figure_policy_support_losses(summary)
    figure_benchmark_monthly_paths(monthly, bank_results)

    mc = load_mc()
    if mc is not None:
        figure_mc_support_tail(mc)
        figure_mc_min_lcr(mc)

    write_latex_snippets(summary)

    log(f"Done. Paper figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
