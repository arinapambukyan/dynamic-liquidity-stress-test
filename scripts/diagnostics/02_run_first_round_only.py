"""
Diagnostic script: run only the dynamic first-round balance-sheet shock module.

This script is not the canonical full-engine run. It is useful for checking
how the first-round LCR shock affects bank balance sheets before reaction,
policy support, fire sales, or interbank contagion.

Run from project root:

    PYTHONPATH=src python scripts/diagnostics/02_run_first_round_only.py
"""

from pathlib import Path

import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    FIRST_ROUND_STRESS_VECTOR,
    FIRST_ROUND_SCENARIO_CONFIG,
    LCR_ASSUMPTIONS,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.first_round import (
    run_first_round_alpha_path,
    summarize_first_round_shock,
)
from liquidity_stress.scenario_config import load_and_validate_scenario_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenario_base.yaml"


def build_worst_case_by_bank(monthly_results: pd.DataFrame) -> pd.DataFrame:
    """
    Build one horizon-level vulnerability summary per bank.
    """
    worst_lcr_idx = monthly_results.groupby("bank_id")["stressed_LCR"].idxmin()
    worst_hqla_idx = monthly_results.groupby("bank_id")["stressed_HQLA_assets_ratio"].idxmin()

    worst_lcr = monthly_results.loc[
        worst_lcr_idx,
        [
            "bank_id",
            "bank_name",
            "stress_month",
            "first_round_alpha",
            "stressed_LCR",
        ],
    ].rename(columns={
        "stress_month": "worst_LCR_month",
        "first_round_alpha": "alpha_at_worst_LCR",
        "stressed_LCR": "min_stressed_LCR",
    })

    worst_hqla = monthly_results.loc[
        worst_hqla_idx,
        [
            "bank_id",
            "stress_month",
            "first_round_alpha",
            "stressed_HQLA_assets_ratio",
        ],
    ].rename(columns={
        "stress_month": "worst_HQLA_assets_month",
        "first_round_alpha": "alpha_at_worst_HQLA_assets",
        "stressed_HQLA_assets_ratio": "min_HQLA_assets_ratio",
    })

    aggregated = monthly_results.groupby(["bank_id", "bank_name"]).agg(
        ever_LCR_breach=("LCR_breach", "max"),
        ever_cash_shortfall=("cash_shortfall_flag", "max"),
        ever_stress_event=("ever_stress_event", "max"),
        max_liquidity_shortfall_to_LCR_100=("liquidity_shortfall_to_LCR_100", "max"),
        max_cash_shortfall_before_support=("cash_shortfall_before_support", "max"),
        cumulative_cash_shortfall_before_support=(
            "cumulative_cash_shortfall_before_support",
            "max",
        ),
        max_unpaid_liability_outflow=("unpaid_total_liability_outflow", "max"),
        max_security_valuation_loss=("total_security_valuation_loss", "max"),
        final_LCR=("stressed_LCR", "last"),
        final_HQLA_assets_ratio=("stressed_HQLA_assets_ratio", "last"),
        final_total_assets=("total_assets", "last"),
        final_equity=("equity", "last"),
        unresolved_shortfall_no_reaction=("unresolved_shortfall_no_reaction", "max"),
        first_unresolved_shortfall_month=("first_unresolved_shortfall_month", "min"),
        continues_without_reaction_final=("continues_without_reaction", "last"),
    ).reset_index()

    out = (
        aggregated
        .merge(worst_lcr, on=["bank_id", "bank_name"], how="left")
        .merge(worst_hqla, on="bank_id", how="left")
        .sort_values("min_stressed_LCR")
        .reset_index(drop=True)
    )

    return out


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    banks = BankingDataLoader(BANKING_INPUT_PATH).load_banks()
    scenario = load_and_validate_scenario_config(SCENARIO_CONFIG_PATH)

    alpha_path = scenario.get("severity", {}).get("alpha_path")

    if alpha_path is None:
        alpha_path = [FIRST_ROUND_SCENARIO_CONFIG["alpha"]]

    scenario_name = scenario.get(
        "scenario_name",
        FIRST_ROUND_SCENARIO_CONFIG["scenario_name"],
    )

    monthly_results = run_first_round_alpha_path(
        banks=banks,
        stress_vector=FIRST_ROUND_STRESS_VECTOR,
        alpha_path=alpha_path,
        lcr_assumptions=LCR_ASSUMPTIONS,
        scenario_name=scenario_name,
    )

    summary_rows = []

    for month, group in monthly_results.groupby("stress_month"):
        summary = summarize_first_round_shock(group)
        row = summary.set_index("metric")["value"].to_dict()
        row["stress_month"] = month
        summary_rows.append(row)

    summary_by_month = pd.DataFrame(summary_rows).sort_values("stress_month")

    worst_case_by_bank = build_worst_case_by_bank(monthly_results)

    monthly_results.to_csv(
        TABLES_DIR / "first_round_monthly_bank_results.csv",
        index=False,
    )

    summary_by_month.to_csv(
        TABLES_DIR / "first_round_monthly_summary.csv",
        index=False,
    )

    worst_case_by_bank.to_csv(
        TABLES_DIR / "first_round_worst_case_by_bank.csv",
        index=False,
    )

    print("First-round monthly bank results saved.")
    print("First-round monthly summary saved.")
    print("First-round worst-case bank summary saved.")
    print(summary_by_month)
    print(worst_case_by_bank[[
        "bank_name",
        "min_stressed_LCR",
        "worst_LCR_month",
        "ever_LCR_breach",
        "unresolved_shortfall_no_reaction",
        "first_unresolved_shortfall_month",
        "min_HQLA_assets_ratio",
        "cumulative_cash_shortfall_before_support",
    ]])


if __name__ == "__main__":
    main()