"""
Diagnostic script: run first-round stress, bank reaction, and CBA support.

This script uses an older intermediate policy-support sequence and should not
be treated as the canonical final engine. The canonical full-engine run is:

    PYTHONPATH=src python scripts/10_run_base_engine.py

Run this script only for module-level diagnostics.

Run from project root:

    PYTHONPATH=src python scripts/diagnostics/05_run_policy_support_old_sequence.py
"""
import numpy as np
import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    FIRST_ROUND_STRESS_VECTOR,
    FIRST_ROUND_SCENARIO_CONFIG,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    CB_SUPPORT_CONFIG,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.first_round import apply_one_month_balance_sheet_shock
from liquidity_stress.policy_support import (
    apply_cb_refinancing_before_asset_sales,
    finalize_cb_policy_metrics_after_asset_sales,
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
    "cash_shortfall_after_asset_sales",
    "remaining_shortfall_after_cb_support",
    "realized_loan_inflows",
    "realized_interbank_asset_inflows",
    "total_realized_asset_inflows",
    "government_security_loss",
    "bank_security_loss",
    "corporate_security_loss",
    "total_security_valuation_loss",
    "asset_sale_proceeds_used_for_outflows",
    "asset_sale_liquidation_loss",
    "total_securities_sold_for_liquidity",
    "cb_support_required",
    "cb_support_granted",
    "cash_shortfall_before_cba_refinancing",
    "cash_shortfall_after_cb_refinancing",
    "cb_support_target_before_asset_sales",
    "newly_pledged_collateral",
    "newly_pledged_collateral_support_value",
]


def zero_current_period_flows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for column in CURRENT_PERIOD_FLOW_COLUMNS:
        if column in out.columns:
            out[column] = 0.0

    for column in out.columns:
        if column.startswith("newly_pledged_"):
            out[column] = 0.0

    return out


def run_dynamic_policy_support_path(
    banks: pd.DataFrame,
    alpha_path: list[float],
    scenario_name: str,
) -> pd.DataFrame:
    """
    Run dynamic monthly path:

        first-round shock
        -> asset-sale reaction
        -> CBA liquidity support
        -> next month balance sheet
    """
    current_state = banks.copy()

    current_state["final_unresolved_shortfall"] = False
    current_state["first_final_unresolved_shortfall_month"] = np.nan
    current_state["continues_after_policy_support"] = True
    current_state["carried_forward_after_final_unresolved_shortfall"] = False

    monthly_results = []

    for month, alpha in enumerate(alpha_path, start=1):
        unresolved_mask = (
            current_state["final_unresolved_shortfall"]
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

            refinanced = apply_cb_refinancing_before_asset_sales(
                banks=shocked,
                policy_config=CB_SUPPORT_CONFIG,
                lcr_assumptions=LCR_ASSUMPTIONS,
            )

            reacted = apply_asset_sale_reaction(
                banks=refinanced,
                liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
                lcr_assumptions=LCR_ASSUMPTIONS,
            )

            supported = finalize_cb_policy_metrics_after_asset_sales(
                banks=reacted,
                lcr_assumptions=LCR_ASSUMPTIONS,
            )

            supported["asset_sale_reaction_success_if_required"] = (
                supported["requires_asset_sale_reaction"]
                & supported["asset_sale_reaction_success"]
            )

            supported["no_asset_sale_reaction_needed"] = (
                ~supported["requires_asset_sale_reaction"]
            )

            newly_final_unresolved = (
                supported["remaining_shortfall_after_cb_support"] > 1e-9
            )

            supported["final_unresolved_shortfall"] = (
                supported["final_unresolved_shortfall"]
                .fillna(False)
                .astype(bool)
                | newly_final_unresolved
            )

            supported["first_final_unresolved_shortfall_month"] = np.where(
                newly_final_unresolved
                & supported["first_final_unresolved_shortfall_month"].isna(),
                month,
                supported["first_final_unresolved_shortfall_month"],
            )

            supported["continues_after_policy_support"] = (
                ~supported["final_unresolved_shortfall"].astype(bool)
            )

            supported["carried_forward_after_final_unresolved_shortfall"] = False

            month_parts.append(supported)

        if not unresolved_banks.empty:
            carried = unresolved_banks.copy()

            carried["stress_month"] = month
            carried["first_round_alpha"] = float(alpha)
            carried["first_round_scenario"] = scenario_name

            carried["final_unresolved_shortfall"] = True
            carried["continues_after_policy_support"] = False
            carried["carried_forward_after_final_unresolved_shortfall"] = True

            carried = zero_current_period_flows(carried)

            carried["requires_asset_sale_reaction"] = False
            carried["asset_sale_reaction_success"] = False
            carried["asset_sale_reaction_success_if_required"] = False
            carried["no_asset_sale_reaction_needed"] = False
            carried["policy_support_used"] = False

            month_parts.append(carried)

        if not month_parts:
            raise ValueError(f"No month results produced at month {month}.")

        month_result = pd.concat(month_parts, ignore_index=True)
        monthly_results.append(month_result.copy())

        current_state = month_result.copy()

    return pd.concat(monthly_results, ignore_index=True)


def build_policy_support_monthly_summary(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("stress_month").agg(
        alpha=("first_round_alpha", "first"),
        number_of_banks=("bank_name", "count"),

        banks_requiring_asset_sale_reaction=(
            "requires_asset_sale_reaction",
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

        banks_using_cb_support=("policy_support_used", "sum"),
        final_unresolved_shortfalls=("final_unresolved_shortfall", "sum"),

        aggregate_cash_shortfall_before_support=(
            "cash_shortfall_before_support",
            "sum",
        ),
        aggregate_cash_shortfall_after_asset_sales=(
            "cash_shortfall_after_asset_sales",
            "sum",
        ),
        aggregate_cb_support_required=(
            "cb_support_required",
            "sum",
        ),
        aggregate_cb_support_granted=(
            "cb_support_granted",
            "sum",
        ),
        aggregate_remaining_shortfall_after_cb=(
            "remaining_shortfall_after_cb_support",
            "sum",
        ),
        aggregate_asset_sale_loss=(
            "asset_sale_liquidation_loss",
            "sum",
        ),

        average_post_cb_support_LCR=("post_cb_support_LCR", "mean"),
        minimum_post_cb_support_LCR=("post_cb_support_LCR", "min"),
        average_post_cb_support_HQLA_assets_ratio=(
            "post_cb_support_HQLA_assets_ratio",
            "mean",
        ),
        minimum_post_cb_support_HQLA_assets_ratio=(
            "post_cb_support_HQLA_assets_ratio",
            "min",
        ),
    ).reset_index()


def build_policy_support_worst_case_by_bank(results: pd.DataFrame) -> pd.DataFrame:
    worst_lcr_idx = results.groupby("bank_id")["post_cb_support_LCR"].idxmin()

    worst_lcr = results.loc[
        worst_lcr_idx,
        [
            "bank_id",
            "bank_name",
            "stress_month",
            "first_round_alpha",
            "post_cb_support_LCR",
        ],
    ].rename(columns={
        "stress_month": "worst_post_cb_support_LCR_month",
        "first_round_alpha": "alpha_at_worst_post_cb_support_LCR",
        "post_cb_support_LCR": "min_post_cb_support_LCR",
    })

    cb_months = (
        results.loc[
            results["policy_support_used"],
            ["bank_id", "stress_month"],
        ]
        .groupby("bank_id")["stress_month"]
        .min()
        .rename("first_cb_support_month")
        .reset_index()
    )

    aggregated = results.groupby(["bank_id", "bank_name"]).agg(
        ever_LCR_breach_before_reaction=("LCR_breach", "max"),
        ever_required_asset_sale_reaction=(
            "requires_asset_sale_reaction",
            "max",
        ),
        ever_used_cb_support=("policy_support_used", "max"),
        ever_final_unresolved_shortfall=(
            "final_unresolved_shortfall",
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
        max_remaining_shortfall_after_cb=(
            "remaining_shortfall_after_cb_support",
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
        total_cb_support_granted=("cb_support_granted", "sum"),
        max_cb_support_assets_ratio=("cb_support_assets_ratio", "max"),
        min_post_cb_support_HQLA_assets_ratio=(
            "post_cb_support_HQLA_assets_ratio",
            "min",
        ),
        final_post_cb_support_LCR=("post_cb_support_LCR", "last"),
        final_total_assets=("total_assets", "last"),
        final_equity=("equity", "last"),
        final_cb_liquidity_funding=("cb_liquidity_funding", "last"),
    ).reset_index()

    return (
        aggregated
        .merge(worst_lcr, on=["bank_id", "bank_name"], how="left")
        .merge(cb_months, on="bank_id", how="left")
        .sort_values("min_post_cb_support_LCR")
        .reset_index(drop=True)
    )


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

    results = run_dynamic_policy_support_path(
        banks=banks,
        alpha_path=alpha_path,
        scenario_name=scenario_name,
    )

    monthly_summary = build_policy_support_monthly_summary(results)
    worst_case_by_bank = build_policy_support_worst_case_by_bank(results)

    results.to_csv(
        TABLES_DIR / "policy_support_bank_results.csv",
        index=False,
    )

    monthly_summary.to_csv(
        TABLES_DIR / "policy_support_monthly_summary.csv",
        index=False,
    )

    worst_case_by_bank.to_csv(
        TABLES_DIR / "policy_support_worst_case_by_bank.csv",
        index=False,
    )

    print("Policy support bank results saved.")
    print("Policy support monthly summary saved.")
    print("Policy support worst-case bank summary saved.")

    print(monthly_summary.to_string(index=False))

    print(worst_case_by_bank[[
        "bank_name",
        "min_post_cb_support_LCR",
        "worst_post_cb_support_LCR_month",
        "ever_used_cb_support",
        "ever_final_unresolved_shortfall",
        "first_cb_support_month",
        "total_cb_support_granted",
        "max_remaining_shortfall_after_cb",
        "final_cb_liquidity_funding",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
