"""
Diagnostic script: run first-round stress with bank-level asset-sale reaction.

This script is not the canonical full-engine run. It is useful for isolating
the bank reaction module before central-bank support, fire-sale feedback, and
interbank contagion.

Run from project root:

    PYTHONPATH=src python scripts/diagnostics/04_run_bank_reaction_only.py
"""

import numpy as np
import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    FIRST_ROUND_STRESS_VECTOR,
    FIRST_ROUND_SCENARIO_CONFIG,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.first_round import (
    apply_one_month_balance_sheet_shock,
    summarize_first_round_shock,
)
from liquidity_stress.reaction import apply_asset_sale_reaction
from liquidity_stress.scenario_config import load_and_validate_scenario_config


SCENARIO_CONFIG_PATH = "configs/scenario_base.yaml"


CURRENT_PERIOD_FLOW_COLUMNS = [
    "planned_retail_deposit_outflow",
    "planned_corporate_deposit_outflow",
    "planned_other_deposit_outflow",
    "planned_wholesale_funding_outflow",
    "planned_interbank_liability_outflow",
    "planned_total_liability_outflow",
    "paid_retail_deposit_outflow",
    "paid_corporate_deposit_outflow",
    "paid_other_deposit_outflow",
    "paid_wholesale_funding_outflow",
    "paid_interbank_liability_outflow",
    "paid_total_liability_outflow",
    "unpaid_total_liability_outflow",
    "cash_shortfall_before_support",
    "realized_loan_inflows",
    "realized_interbank_asset_inflows",
    "total_realized_asset_inflows",
    "government_security_loss",
    "bank_security_loss",
    "corporate_security_loss",
    "total_security_valuation_loss",
    "additional_FX_liability_pressure",
    "asset_sale_proceeds_used_for_outflows",
    "asset_sale_liquidation_loss",
    "total_securities_sold_for_liquidity",
    "cash_shortfall_after_asset_sales",
]


def zero_current_period_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For banks carried forward after unresolved post-reaction shortfall,
    zero out current-period flow variables to avoid repeated counting.
    """
    out = df.copy()

    for column in CURRENT_PERIOD_FLOW_COLUMNS:
        if column in out.columns:
            out[column] = 0.0

    return out


def run_dynamic_reaction_path(
    banks: pd.DataFrame,
    alpha_path: list[float],
    scenario_name: str,
) -> pd.DataFrame:
    """
    Run monthly first-round shocks with asset-sale reaction inside the loop.

    If asset sales fully cover the cash shortfall, the bank continues into the
    next month from the post-reaction balance sheet.

    If a shortfall remains after asset sales, the bank is carried forward as
    unresolved until a later CB-support module is added.
    """
    current_state = banks.copy()

    current_state["unresolved_shortfall_after_asset_sales"] = False
    current_state["first_unresolved_after_asset_sales_month"] = np.nan
    current_state["continues_after_asset_sale_reaction"] = True
    current_state["carried_forward_after_unresolved_asset_sale_shortfall"] = False

    monthly_results = []

    for month, alpha in enumerate(alpha_path, start=1):
        unresolved_mask = (
            current_state["unresolved_shortfall_after_asset_sales"]
            .fillna(False)
            .astype(bool)
        )

        active_banks = current_state.loc[~unresolved_mask].copy()
        unresolved_banks = current_state.loc[unresolved_mask].copy()

        month_parts = []

        if not active_banks.empty:
            shocked = apply_one_month_balance_sheet_shock(
                banks=active_banks,
                stress_vector=FIRST_ROUND_STRESS_VECTOR,
                alpha=alpha,
                lcr_assumptions=LCR_ASSUMPTIONS,
                scenario_name=scenario_name,
                month=month,
            )

            reacted = apply_asset_sale_reaction(
                banks=shocked,
                liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
                lcr_assumptions=LCR_ASSUMPTIONS,
            )

            reacted["asset_sale_reaction_success_if_required"] = (
                reacted["requires_asset_sale_reaction"]
                & reacted["asset_sale_reaction_success"]
            )

            reacted["no_asset_sale_reaction_needed"] = (
                ~reacted["requires_asset_sale_reaction"]
            )

            newly_unresolved = (
                reacted["cash_shortfall_after_asset_sales"] > 1e-9
            )

            reacted["unresolved_shortfall_after_asset_sales"] = (
                reacted["unresolved_shortfall_after_asset_sales"]
                .fillna(False)
                .astype(bool)
                | newly_unresolved
            )

            reacted["first_unresolved_after_asset_sales_month"] = np.where(
                newly_unresolved
                & reacted["first_unresolved_after_asset_sales_month"].isna(),
                month,
                reacted["first_unresolved_after_asset_sales_month"],
            )

            reacted["continues_after_asset_sale_reaction"] = (
                ~reacted["unresolved_shortfall_after_asset_sales"].astype(bool)
            )

            reacted["carried_forward_after_unresolved_asset_sale_shortfall"] = False

            month_parts.append(reacted)

        if not unresolved_banks.empty:
            carried = unresolved_banks.copy()

            carried["stress_month"] = month
            carried["first_round_alpha"] = float(alpha)
            carried["first_round_scenario"] = scenario_name

            carried["unresolved_shortfall_after_asset_sales"] = True
            carried["continues_after_asset_sale_reaction"] = False
            carried["carried_forward_after_unresolved_asset_sale_shortfall"] = True

            carried = zero_current_period_flows(carried)

            carried["requires_asset_sale_reaction"] = False
            carried["asset_sale_reaction_success"] = False
            carried["asset_sale_reaction_success_if_required"] = False
            carried["no_asset_sale_reaction_needed"] = False

            month_parts.append(carried)

        if not month_parts:
            raise ValueError(f"No month results produced at month {month}.")

        month_result = pd.concat(month_parts, ignore_index=True)

        monthly_results.append(month_result.copy())

        # The next month starts from the post-reaction balance sheet.
        current_state = month_result.copy()

    return pd.concat(monthly_results, ignore_index=True)


def build_reaction_monthly_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly system-level reaction summary.
    """
    return results.groupby("stress_month").agg(
        alpha=("first_round_alpha", "first"),
        number_of_banks=("bank_name", "count"),

        banks_requiring_asset_sale_reaction=(
            "requires_asset_sale_reaction",
            "sum",
        ),
        banks_not_needing_asset_sale_reaction=(
            "no_asset_sale_reaction_needed",
            "sum",
        ),
        successful_asset_sale_reactions=(
            "asset_sale_reaction_success_if_required",
            "sum",
        ),
        unresolved_after_asset_sales=(
            "unresolved_shortfall_after_asset_sales",
            "sum",
        ),

        aggregate_cash_shortfall_before_support=(
            "cash_shortfall_before_support",
            "sum",
        ),
        aggregate_cash_shortfall_after_asset_sales=(
            "cash_shortfall_after_asset_sales",
            "sum",
        ),
        aggregate_asset_sale_proceeds=(
            "asset_sale_proceeds_used_for_outflows",
            "sum",
        ),
        aggregate_asset_sale_loss=(
            "asset_sale_liquidation_loss",
            "sum",
        ),
        aggregate_securities_sold_for_liquidity=(
            "total_securities_sold_for_liquidity",
            "sum",
        ),

        average_stressed_LCR=("stressed_LCR", "mean"),
        minimum_stressed_LCR=("stressed_LCR", "min"),
        average_post_asset_sale_LCR=("post_asset_sale_LCR", "mean"),
        minimum_post_asset_sale_LCR=("post_asset_sale_LCR", "min"),
        average_post_asset_sale_HQLA_assets_ratio=(
            "post_asset_sale_HQLA_assets_ratio",
            "mean",
        ),
        minimum_post_asset_sale_HQLA_assets_ratio=(
            "post_asset_sale_HQLA_assets_ratio",
            "min",
        ),
    ).reset_index()


def build_reaction_worst_case_by_bank(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build one bank-level reaction summary over the full horizon.
    """
    worst_lcr_idx = results.groupby("bank_id")["post_asset_sale_LCR"].idxmin()

    worst_lcr = results.loc[
        worst_lcr_idx,
        [
            "bank_id",
            "bank_name",
            "stress_month",
            "first_round_alpha",
            "post_asset_sale_LCR",
        ],
    ].rename(columns={
        "stress_month": "worst_post_asset_sale_LCR_month",
        "first_round_alpha": "alpha_at_worst_post_asset_sale_LCR",
        "post_asset_sale_LCR": "min_post_asset_sale_LCR",
    })

    aggregated = results.groupby(["bank_id", "bank_name"]).agg(
        ever_LCR_breach_before_reaction=("LCR_breach", "max"),
        ever_required_asset_sale_reaction=(
            "requires_asset_sale_reaction",
            "max",
        ),
        ever_unresolved_after_asset_sales=(
            "unresolved_shortfall_after_asset_sales",
            "max",
        ),
        max_cash_shortfall_before_support=(
            "cash_shortfall_before_support",
            "max",
        ),
        max_cash_shortfall_after_asset_sales=(
            "cash_shortfall_after_asset_sales",
            "max",
        ),
        cumulative_cash_shortfall_before_support=(
            "cumulative_cash_shortfall_before_support",
            "max",
        ),
        total_asset_sale_proceeds=(
            "asset_sale_proceeds_used_for_outflows",
            "sum",
        ),
        total_asset_sale_loss=(
            "asset_sale_liquidation_loss",
            "sum",
        ),
        total_securities_sold_for_liquidity=(
            "total_securities_sold_for_liquidity",
            "sum",
        ),
        min_post_asset_sale_HQLA_assets_ratio=(
            "post_asset_sale_HQLA_assets_ratio",
            "min",
        ),
        final_post_asset_sale_LCR=(
            "post_asset_sale_LCR",
            "last",
        ),
        final_total_assets=("total_assets", "last"),
        final_equity=("equity", "last"),
    ).reset_index()

    reaction_months = (
        results.loc[
            results["requires_asset_sale_reaction"],
            ["bank_id", "stress_month"],
        ]
        .groupby("bank_id")["stress_month"]
        .min()
        .rename("first_asset_sale_reaction_month")
        .reset_index()
    )

    out = (
        aggregated
        .merge(worst_lcr, on=["bank_id", "bank_name"], how="left")
        .merge(reaction_months, on="bank_id", how="left")
        .sort_values("min_post_asset_sale_LCR")
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

    results = run_dynamic_reaction_path(
        banks=banks,
        alpha_path=alpha_path,
        scenario_name=scenario_name,
    )

    monthly_summary = build_reaction_monthly_summary(results)
    worst_case_by_bank = build_reaction_worst_case_by_bank(results)

    results.to_csv(
        TABLES_DIR / "asset_sale_reaction_bank_results.csv",
        index=False,
    )

    monthly_summary.to_csv(
        TABLES_DIR / "asset_sale_reaction_monthly_summary.csv",
        index=False,
    )

    worst_case_by_bank.to_csv(
        TABLES_DIR / "asset_sale_reaction_worst_case_by_bank.csv",
        index=False,
    )

    print("Dynamic asset-sale reaction bank results saved.")
    print("Dynamic asset-sale reaction monthly summary saved.")
    print("Dynamic asset-sale reaction worst-case bank summary saved.")

    print(monthly_summary.to_string(index=False))

    print(worst_case_by_bank[[
        "bank_name",
        "min_post_asset_sale_LCR",
        "worst_post_asset_sale_LCR_month",
        "ever_required_asset_sale_reaction",
        "ever_unresolved_after_asset_sales",
        "first_asset_sale_reaction_month",
        "max_cash_shortfall_before_support",
        "max_cash_shortfall_after_asset_sales",
        "total_asset_sale_proceeds",
        "total_asset_sale_loss",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()