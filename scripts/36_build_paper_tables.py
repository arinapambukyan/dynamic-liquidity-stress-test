#!/usr/bin/env python3
"""
Build paper-grade, stage-aware scenario summary tables.

This script does not rerun the stress engine. It reads the existing outputs from
scripts/35_run_paper_scenarios.py and rebuilds the summary tables so that each
scenario is evaluated at its economically active final stage.

Why this is needed:
- first_round_only must be evaluated using the stressed_* columns;
- plus_bank_reaction must be evaluated using post_asset_sale_*;
- plus_lolr_support must be evaluated using post_cb_support_*;
- plus_fire_sales must be evaluated using post_fire_sale_*;
- full_engine and policy counterfactuals are evaluated using post_interbank_*.

Run from repo root:
    PYTHONPATH=src python scripts/36_build_paper_tables.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# Resolve paths from the repository root. This script lives in <repo>/scripts/.
SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1]
TABLE_DIR = ROOT / "outputs" / "tables"


BANK_RESULTS = TABLE_DIR / "paper_scenario_bank_results.csv"
MONTHLY_RESULTS = TABLE_DIR / "paper_scenario_monthly_summary.csv"

OUT_SCENARIO = TABLE_DIR / "paper_stage_aware_scenario_summary.csv"
OUT_MONTHLY = TABLE_DIR / "paper_stage_aware_monthly_summary.csv"
OUT_CHANNEL = TABLE_DIR / "paper_channel_attribution_stage_aware.csv"
OUT_POLICY = TABLE_DIR / "paper_policy_counterfactual_stage_aware.csv"


ACTIVE_STAGE = {
    # Channel attribution
    "first_round_only": "stressed",
    "plus_bank_reaction": "post_asset_sale",
    "plus_lolr_support": "post_cb_support",
    "plus_fire_sales": "post_fire_sale",
    "full_engine": "post_interbank",

    # Policy counterfactuals
    "no_cb_support": "post_interbank",
    "collateral_capped_lolr": "post_interbank",
    "full_coverage_lolr": "post_interbank",
    "asset_purchases_only": "post_interbank",
    "lolr_plus_asset_purchases": "post_interbank",
    "relaxed_lcr_100_no_lolr": "post_interbank",
    "widened_hqla_no_lolr": "post_interbank",
}

STAGE_LABEL = {
    "stressed": "after first-round stress",
    "post_asset_sale": "after bank reaction",
    "post_cb_support": "after LoLR support",
    "post_fire_sale": "after fire-sale feedback",
    "post_interbank": "after interbank contagion",
}


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def stage_cols(stage: str) -> dict[str, str]:
    return {
        "hqla": f"{stage}_HQLA",
        "ncof": f"{stage}_NCOF",
        "lcr": f"{stage}_LCR",
        "hqla_assets_ratio": f"{stage}_HQLA_assets_ratio",
    }


def system_ratio(numer: pd.Series, denom: pd.Series) -> float:
    denom_sum = denom.sum()
    if denom_sum == 0:
        return np.nan
    return numer.sum() / denom_sum


def summarize_one_month(g: pd.DataFrame, scenario: str, stage: str) -> dict:
    c = stage_cols(stage)

    system_lcr = system_ratio(g[c["hqla"]], g[c["ncof"]])

    # The model output does not store a stage-specific total-assets column.
    # For cross-scenario comparison we use the bank-level ratio distribution,
    # plus an aggregate stock ratio using the current total_assets field.
    system_hqla_assets = system_ratio(g[c["hqla"]], g["total_assets"])

    final_unresolved_shortfall_banks = int((g["final_unresolved_shortfall"].fillna(0) > 1e-9).sum())
    terminal_after_interbank_banks = int(g["terminal_distress_after_interbank"].fillna(False).astype(bool).sum())

    return {
        "scenario": scenario,
        "scenario_family": g["scenario_family"].iloc[0],
        "paper_order": int(g["paper_order"].iloc[0]),
        "active_stage": stage,
        "active_stage_label": STAGE_LABEL[stage],
        "stress_month": int(g["stress_month"].iloc[0]),

        "system_LCR_active_stage": system_lcr,
        "minimum_bank_LCR_active_stage": g[c["lcr"]].min(),
        "median_bank_LCR_active_stage": g[c["lcr"]].median(),
        "p05_bank_LCR_active_stage": g[c["lcr"]].quantile(0.05),
        "p95_bank_LCR_active_stage": g[c["lcr"]].quantile(0.95),

        "system_HQLA_assets_active_stage": system_hqla_assets,
        "minimum_bank_HQLA_assets_active_stage": g[c["hqla_assets_ratio"]].min(),
        "median_bank_HQLA_assets_active_stage": g[c["hqla_assets_ratio"]].median(),
        "p05_bank_HQLA_assets_active_stage": g[c["hqla_assets_ratio"]].quantile(0.05),
        "p95_bank_HQLA_assets_active_stage": g[c["hqla_assets_ratio"]].quantile(0.95),

        "banks_LCR_below_120_active_stage": int((g[c["lcr"]] < 1.20).sum()),
        "banks_LCR_below_100_active_stage": int((g[c["lcr"]] < 1.00).sum()),
        "banks_HQLA_assets_below_10pct_active_stage": int((g[c["hqla_assets_ratio"]] < 0.10).sum()),

        "aggregate_cash_shortfall_before_support": g["cash_shortfall_before_support"].fillna(0).sum(),
        "aggregate_asset_sale_loss": g["asset_sale_liquidation_loss"].fillna(0).sum(),
        "aggregate_cb_support_granted": g["cb_support_granted"].fillna(0).sum(),
        "aggregate_support_used_for_outflows": g["cb_support_used_for_outflows"].fillna(0).sum(),
        "aggregate_support_retained_as_cash_for_lcr": g["cb_support_retained_as_cash_for_lcr"].fillna(0).sum(),
        "aggregate_fire_sale_mtm_loss": g["total_fire_sale_mtm_loss"].fillna(0).sum(),
        "aggregate_interbank_contagion_loss": g["interbank_contagion_loss"].fillna(0).sum(),

        "final_unresolved_shortfall_banks": final_unresolved_shortfall_banks,
        "terminal_distress_after_interbank_banks": terminal_after_interbank_banks,
    }


def build_monthly_summary(bank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, stage in ACTIVE_STAGE.items():
        if scenario not in set(bank_df["scenario"]):
            continue
        scenario_df = bank_df.loc[bank_df["scenario"] == scenario].copy()
        c = stage_cols(stage)
        require_columns(
            scenario_df,
            [
                c["hqla"], c["ncof"], c["lcr"], c["hqla_assets_ratio"],
                "total_assets", "stress_month", "scenario_family", "paper_order",
                "cash_shortfall_before_support", "asset_sale_liquidation_loss",
                "cb_support_granted", "cb_support_used_for_outflows",
                "cb_support_retained_as_cash_for_lcr", "total_fire_sale_mtm_loss",
                "interbank_contagion_loss", "final_unresolved_shortfall",
                "terminal_distress_after_interbank",
            ],
        )

        for month, g in scenario_df.groupby("stress_month", sort=True):
            rows.append(summarize_one_month(g, scenario, stage))

    return pd.DataFrame(rows).sort_values(["paper_order", "stress_month"])


def build_scenario_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, g in monthly.groupby("scenario", sort=False):
        g = g.sort_values("stress_month")
        final = g.iloc[-1]
        rows.append({
            "paper_order": int(final["paper_order"]),
            "scenario": scenario,
            "scenario_family": final["scenario_family"],
            "active_stage": final["active_stage"],
            "active_stage_label": final["active_stage_label"],

            "min_system_LCR": g["system_LCR_active_stage"].min(),
            "final_system_LCR": final["system_LCR_active_stage"],
            "min_weakest_bank_LCR": g["minimum_bank_LCR_active_stage"].min(),
            "final_weakest_bank_LCR": final["minimum_bank_LCR_active_stage"],

            "min_system_HQLA_assets": g["system_HQLA_assets_active_stage"].min(),
            "final_system_HQLA_assets": final["system_HQLA_assets_active_stage"],
            "min_weakest_bank_HQLA_assets": g["minimum_bank_HQLA_assets_active_stage"].min(),
            "final_weakest_bank_HQLA_assets": final["minimum_bank_HQLA_assets_active_stage"],

            "max_banks_LCR_below_120": int(g["banks_LCR_below_120_active_stage"].max()),
            "max_banks_LCR_below_100": int(g["banks_LCR_below_100_active_stage"].max()),
            "max_banks_HQLA_assets_below_10pct": int(g["banks_HQLA_assets_below_10pct_active_stage"].max()),

            "total_cash_shortfall_before_support": g["aggregate_cash_shortfall_before_support"].sum(),
            "total_asset_sale_loss": g["aggregate_asset_sale_loss"].sum(),
            "total_cb_support_granted": g["aggregate_cb_support_granted"].sum(),
            "total_support_used_for_outflows": g["aggregate_support_used_for_outflows"].sum(),
            "total_support_retained_as_cash_for_lcr": g["aggregate_support_retained_as_cash_for_lcr"].sum(),
            "total_fire_sale_mtm_loss": g["aggregate_fire_sale_mtm_loss"].sum(),
            "total_interbank_contagion_loss": g["aggregate_interbank_contagion_loss"].sum(),

            "max_final_unresolved_shortfall_banks": int(g["final_unresolved_shortfall_banks"].max()),
            # Meaningful only for scenarios with interbank contagion active.
            "final_terminal_distress_after_interbank_banks": int(final["terminal_distress_after_interbank_banks"]),
        })

    return pd.DataFrame(rows).sort_values("paper_order")


def main() -> None:
    if not BANK_RESULTS.exists():
        raise FileNotFoundError(f"Cannot find {BANK_RESULTS}. Run scripts/35_run_paper_scenarios.py first.")

    bank_df = pd.read_csv(BANK_RESULTS)

    monthly = build_monthly_summary(bank_df)
    summary = build_scenario_summary(monthly)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    monthly.to_csv(OUT_MONTHLY, index=False)
    summary.to_csv(OUT_SCENARIO, index=False)

    channel = summary.loc[summary["scenario_family"] == "channel_attribution"].copy()
    policy = summary.loc[summary["scenario_family"] == "policy_counterfactual"].copy()

    channel.to_csv(OUT_CHANNEL, index=False)
    policy.to_csv(OUT_POLICY, index=False)

    print("\nSTAGE-AWARE PAPER SCENARIO SUMMARY")
    print("=" * 72)
    display_cols = [
        "paper_order", "scenario", "active_stage",
        "min_system_LCR", "min_weakest_bank_LCR",
        "final_system_LCR", "min_system_HQLA_assets",
        "total_cb_support_granted", "total_fire_sale_mtm_loss",
        "total_interbank_contagion_loss",
        "final_terminal_distress_after_interbank_banks",
    ]
    print(summary[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {OUT_SCENARIO}")
    print(f"  {OUT_MONTHLY}")
    print(f"  {OUT_CHANNEL}")
    print(f"  {OUT_POLICY}")


if __name__ == "__main__":
    main()
