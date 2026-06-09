"""
Diagnostic script: run first-round stress separately for combined, AMD, and FX sheets.

This script is not the canonical currency-aware full-engine run. It only checks
the first-round balance-sheet shock by currency bucket.

Run from project root:

    PYTHONPATH=src python scripts/diagnostics/03_run_first_round_currency_only.py
"""

from pathlib import Path

import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    FIRST_ROUND_STRESS_VECTOR,
    LCR_ASSUMPTIONS,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.first_round import run_first_round_alpha_path
from liquidity_stress.scenario_config import (
    load_and_validate_scenario_config,
    get_alpha_path_for_currency,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenario_base.yaml"


def build_monthly_summary(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(["currency", "stress_month"]).agg(
        alpha=("first_round_alpha", "first"),
        number_of_banks=("bank_id", "count"),
        average_LCR=("stressed_LCR", "mean"),
        minimum_LCR=("stressed_LCR", "min"),
        average_HQLA_assets_ratio=("stressed_HQLA_assets_ratio", "mean"),
        minimum_HQLA_assets_ratio=("stressed_HQLA_assets_ratio", "min"),
        banks_with_LCR_breach=("LCR_breach", "sum"),
        banks_with_cash_shortfall=("cash_shortfall_flag", "sum"),
        aggregate_cash_shortfall_before_support=(
            "cash_shortfall_before_support",
            "sum",
        ),
        aggregate_unpaid_liability_outflow=(
            "unpaid_total_liability_outflow",
            "sum",
        ),
        aggregate_security_valuation_loss=(
            "total_security_valuation_loss",
            "sum",
        ),
        aggregate_total_assets=("total_assets", "sum"),
        aggregate_equity=("equity", "sum"),
        aggregate_HQLA=("HQLA", "sum"),
        aggregate_NCOF=("NCOF", "sum"),
    ).reset_index()


def aggregate_amd_fx_to_combined(
    amd_results: pd.DataFrame,
    fx_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate AMD and FX first-round results into a truthful combined view.

    Important:
    Ratios are recomputed from summed numerator/denominator whenever possible.
    """
    stacked = pd.concat([amd_results, fx_results], ignore_index=True)

    additive_columns = [
        "cash_reserves",
        "loans",
        "government_securities",
        "bank_securities",
        "corporate_securities",
        "total_securities",
        "interbank_assets",
        "total_assets",
        "retail_deposits",
        "corporate_deposits",
        "other_deposits",
        "deposits",
        "wholesale_funding",
        "interbank_liabilities",
        "total_liabilities",
        "equity",
        "HQLA",
        "NCOF",
        "gross_30d_outflows",
        "recognized_30d_inflows",
        "capped_30d_inflows",
        "cash_shortfall_before_support",
        "unpaid_total_liability_outflow",
        "total_security_valuation_loss",
        "planned_total_liability_outflow",
        "paid_total_liability_outflow",
    ]

    available_additive = [
        column for column in additive_columns
        if column in stacked.columns
    ]

    group_keys = [
        "bank_id",
        "bank_name",
        "stress_month",
    ]

    aggregated = (
        stacked.groupby(group_keys)[available_additive]
        .sum()
        .reset_index()
    )

    metadata = (
        stacked.groupby(group_keys)
        .agg(
            first_round_scenario=("first_round_scenario", "first"),
            first_round_alpha=("first_round_alpha", "mean"),
        )
        .reset_index()
    )

    out = aggregated.merge(metadata, on=group_keys, how="left")

    out["currency"] = "combined_from_amd_fx"

    out["stressed_HQLA_assets_ratio"] = out["HQLA"] / out["total_assets"]
    out["stressed_LCR"] = out["HQLA"] / out["NCOF"]

    if "active_lcr_requirement" in out.columns:
        active_lcr_requirement = pd.to_numeric(
            out["active_lcr_requirement"],
            errors="coerce",
        ).fillna(1.20)
    else:
        active_lcr_requirement = pd.Series(1.20, index=out.index)

    out["active_lcr_requirement"] = active_lcr_requirement
    out["LCR_breach"] = out["stressed_LCR"] < active_lcr_requirement
    out["cash_shortfall_flag"] = out["cash_shortfall_before_support"] > 1e-9

    out["equity_assets_ratio"] = out["equity"] / out["total_assets"]

    return out


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    loader = BankingDataLoader(BANKING_INPUT_PATH)
    scenario = load_and_validate_scenario_config(SCENARIO_CONFIG_PATH)

    scenario_name = scenario.get("scenario_name", "currency_first_round")

    results_by_currency = {}

    for currency in ["combined", "AMD", "FX"]:
        banks = loader.load_banks(currency)
        alpha_path = get_alpha_path_for_currency(scenario, currency)

        results = run_first_round_alpha_path(
            banks=banks,
            stress_vector=FIRST_ROUND_STRESS_VECTOR,
            alpha_path=alpha_path,
            lcr_assumptions=LCR_ASSUMPTIONS,
            scenario_name=f"{scenario_name}_{currency}",
        )

        results["currency"] = currency
        results_by_currency[currency] = results

    combined_from_currency = aggregate_amd_fx_to_combined(
        amd_results=results_by_currency["AMD"],
        fx_results=results_by_currency["FX"],
    )

    all_results = pd.concat(
        [
            results_by_currency["combined"],
            results_by_currency["AMD"],
            results_by_currency["FX"],
            combined_from_currency,
        ],
        ignore_index=True,
    )

    monthly_summary = build_monthly_summary(all_results)

    all_results.to_csv(
        TABLES_DIR / "first_round_currency_bank_results.csv",
        index=False,
    )

    monthly_summary.to_csv(
        TABLES_DIR / "first_round_currency_monthly_summary.csv",
        index=False,
    )

    print("Currency-aware first-round bank results saved.")
    print("Currency-aware first-round monthly summary saved.")
    print(monthly_summary.to_string(index=False))


if __name__ == "__main__":
    main()