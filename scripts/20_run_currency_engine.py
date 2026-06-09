"""
Run the canonical full liquidity stress engine separately for AMD and FX.

The script runs the full engine on AMD and FX balance-sheet sheets, then
aggregates the currency-bucket results into a combined currency-aware view.
Solvency is treated as a whole-bank concept, while liquidity stress can be
currency-specific.

Run from project root:

    PYTHONPATH=src python scripts/20_run_currency_engine.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import argparse
import copy

import numpy as np
import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    INTERBANK_EXPOSURE_CONFIG,
    TABLES_DIR,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    CB_SUPPORT_CONFIG,
    FIRE_SALE_CONFIG,
    INTERBANK_CONTAGION_CONFIG,
)

from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.network_interbank import get_or_create_interbank_exposure_matrix
from liquidity_stress.scenario_config import (
    load_and_validate_scenario_config,
    get_alpha_path_for_currency,
    get_stress_mode,
    build_runtime_configs_from_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenario_base.yaml"
ENGINE_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "10_run_base_engine.py"


def load_engine_function():
    """
    Import run_dynamic_interbank_contagion_path from scripts/10_run_base_engine.py.

    The engine script starts with a number, so normal Python imports are awkward.
    This safely loads the function without running the script's main().
    """
    spec = importlib.util.spec_from_file_location(
        "run_interbank_engine_module",
        ENGINE_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load engine script: {ENGINE_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.run_dynamic_interbank_contagion_path


def exposure_matrix_path_for_currency(currency: str) -> Path:
    base_path = Path(INTERBANK_EXPOSURE_CONFIG["path"])

    if currency.upper() == "AMD":
        return base_path.with_name("interbank_exposure_matrix_amd.xlsx")

    if currency.upper() == "FX":
        return base_path.with_name("interbank_exposure_matrix_fx.xlsx")

    raise ValueError("currency must be AMD or FX")

def attach_bank_level_reference(
    currency_banks: pd.DataFrame,
    combined_banks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach whole-bank solvency reference variables to AMD/FX sheets.

    Currency sheets measure currency liquidity buckets.
    Solvency, however, is a whole-bank concept.

    initial_currency_equity is used to measure the change in the currency
    bucket relative to its starting value.
    """
    out = currency_banks.copy()
    out["initial_currency_equity"] = out["equity"].astype(float)

    reference = combined_banks[
        [
            "bank_id",
            "equity",
            "total_assets",
            "HQLA",
            "NCOF",
            "LCR",
            "HQLA_assets_ratio",
        ]
    ].copy()

    reference = reference.rename(
        columns={
            "equity": "bank_level_equity",
            "total_assets": "bank_level_total_assets",
            "HQLA": "bank_level_HQLA",
            "NCOF": "bank_level_NCOF",
            "LCR": "bank_level_LCR",
            "HQLA_assets_ratio": "bank_level_HQLA_assets_ratio",
        }
    )

    return out.merge(reference, on="bank_id", how="left")

def run_currency_engine(
        currency: str,
        banks: pd.DataFrame,
        alpha_path: list[float],
        scenario_name: str,
        stress_mode: str,
        run_dynamic_interbank_contagion_path,
        runtime_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full engine for one currency sheet.
    """
    matrix_path = exposure_matrix_path_for_currency(currency)

    exposure_matrix = get_or_create_interbank_exposure_matrix(
        banks=banks,
        path=matrix_path,
        source=INTERBANK_EXPOSURE_CONFIG["source"],
        sheet_name=INTERBANK_EXPOSURE_CONFIG["sheet_name"],
        exposure_scale=INTERBANK_EXPOSURE_CONFIG["network_exposure_scale"],
    )

    results, market_summary, contagion_summary = run_dynamic_interbank_contagion_path(
        banks=banks,
        alpha_path=alpha_path,
        scenario_name=f"{scenario_name}_{currency}",
        exposure_matrix=exposure_matrix,
        stress_mode=stress_mode,

        lcr_assumptions=copy.deepcopy(runtime_config["lcr"]),
        liquidation_config=copy.deepcopy(runtime_config["liquidation"]),
        cb_support_config=copy.deepcopy(runtime_config["cb"]),
        fire_sale_config=copy.deepcopy(runtime_config["fire"]),
        interbank_contagion_config=copy.deepcopy(runtime_config["interbank"]),
        module_config=copy.deepcopy(runtime_config["modules"]),
    )

    results["currency"] = currency

    if not market_summary.empty:
        market_summary["currency"] = currency

    if not contagion_summary.empty:
        contagion_summary["currency"] = currency

    return results, market_summary, contagion_summary


def recompute_ratio(
    df: pd.DataFrame,
    numerator: str,
    denominator: str,
    output: str,
) -> None:
    if numerator in df.columns and denominator in df.columns:
        df[output] = np.where(
            df[denominator].astype(float) > 0,
            df[numerator].astype(float) / df[denominator].astype(float),
            np.nan,
        )


def aggregate_amd_fx_results(
    amd: pd.DataFrame,
    fx: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate AMD and FX bank-level full-engine results.

    Additive columns are summed.
    Boolean distress/status columns are OR-ed.
    Ratios are recomputed from summed numerators and denominators when possible.
    """
    stacked = pd.concat([amd, fx], ignore_index=True)

    group_keys = ["bank_id", "bank_name", "stress_month"]

    additive_candidates = [
        # balance sheet stocks
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

        # liquidity variables
        "HQLA",
        "NCOF",
        "gross_30d_outflows",
        "recognized_30d_inflows",
        "capped_30d_inflows",

        # first-round and reaction flows
        "planned_total_liability_outflow",
        "paid_total_liability_outflow",
        "unpaid_total_liability_outflow",
        "cash_shortfall_before_support",
        "cash_shortfall_before_cba_refinancing",
        "cash_shortfall_after_cb_refinancing",
        "cash_shortfall_after_asset_sales",
        "remaining_shortfall_after_cb_support",
        "asset_sale_proceeds_used_for_outflows",
        "asset_sale_liquidation_loss",
        "total_securities_sold_for_liquidity",

        # policy support
        "cb_support_required",
        "cb_support_granted",
        "cb_liquidity_funding",
        "newly_pledged_collateral",
        "newly_pledged_collateral_support_value",
        "total_cb_pledged_collateral",
        "pre_cb_unencumbered_collateral_capacity",
        "post_cb_unencumbered_collateral_capacity",
        "unencumbered_collateral_capacity",

        # losses
        "government_security_loss",
        "bank_security_loss",
        "corporate_security_loss",
        "total_security_valuation_loss",
        "total_fire_sale_mtm_loss",
        "government_securities_fire_sale_mtm_loss",
        "pledged_government_securities_fire_sale_mtm_loss",
        "interbank_contagion_loss",
    ]

    additive_columns = [
        column for column in additive_candidates
        if column in stacked.columns
    ]

    aggregated = (
        stacked.groupby(group_keys)[additive_columns]
        .sum()
        .reset_index()
    )

    metadata = (
        stacked.groupby(group_keys)
        .agg(
            first_round_alpha=("first_round_alpha", "mean"),
            first_round_scenario=("first_round_scenario", "first"),
            stress_mode=("stress_mode", "first"),
        )
        .reset_index()
    )

    out = aggregated.merge(metadata, on=group_keys, how="left")
    out["currency"] = "combined_from_amd_fx"

    boolean_candidates = [
        "LCR_breach",
        "cash_shortfall_flag",
        "requires_asset_sale_reaction",
        "asset_sale_reaction_success",
        "asset_sale_reaction_success_if_required",
        "no_asset_sale_reaction_needed",
        "policy_support_used",
        "solvent_before_lolr",
        "illiquid_before_lolr",
        "eligible_for_lolr",
        "final_unresolved_shortfall",
    ]

    for column in boolean_candidates:
        if column in stacked.columns:
            temp = (
                stacked.groupby(group_keys)[column]
                .max()
                .reset_index()
            )
            out = out.merge(temp, on=group_keys, how="left")

    # Recompute core ratios.
    recompute_ratio(out, "HQLA", "NCOF", "LCR")
    recompute_ratio(out, "HQLA", "NCOF", "stressed_LCR")
    recompute_ratio(out, "HQLA", "NCOF", "post_cb_refinancing_LCR")
    recompute_ratio(out, "HQLA", "NCOF", "post_asset_sale_LCR")
    recompute_ratio(out, "HQLA", "NCOF", "post_fire_sale_LCR")
    recompute_ratio(out, "HQLA", "NCOF", "post_interbank_LCR")

    recompute_ratio(out, "HQLA", "total_assets", "HQLA_assets_ratio")
    recompute_ratio(out, "HQLA", "total_assets", "stressed_HQLA_assets_ratio")
    recompute_ratio(out, "HQLA", "total_assets", "post_cb_refinancing_HQLA_assets_ratio")
    recompute_ratio(out, "HQLA", "total_assets", "post_asset_sale_HQLA_assets_ratio")
    recompute_ratio(out, "HQLA", "total_assets", "post_fire_sale_HQLA_assets_ratio")
    recompute_ratio(out, "HQLA", "total_assets", "post_interbank_HQLA_assets_ratio")
    recompute_ratio(out, "equity", "total_assets", "equity_assets_ratio")
    recompute_ratio(out, "total_fire_sale_mtm_loss", "total_assets", "fire_sale_mtm_loss_assets_ratio")

    # Re-impose terminal definition at the aggregated level.
    if "remaining_shortfall_after_cb_support" in out.columns:
        unresolved = out["remaining_shortfall_after_cb_support"].astype(float) > 1e-9
    else:
        unresolved = pd.Series(False, index=out.index)

    insolvent = out["equity"].astype(float) <= 0 if "equity" in out.columns else pd.Series(False, index=out.index)

    out["final_unresolved_shortfall"] = (
        out.get("final_unresolved_shortfall", False).astype(bool)
        | unresolved
    )

    # In the aggregated combined view, insolvency is a whole-bank condition.
    # Do not inherit AMD-only or FX-only insolvency flags, because currency-bucket
    # equity is not bank-level solvency.
    out["post_interbank_insolvent"] = insolvent

    # Liquidity shortfall can still be currency-specific.
    # If either AMD or FX has an unresolved liquidity shortfall, the bank has
    # unresolved currency liquidity stress.
    if "final_unresolved_shortfall" in stacked.columns:
        unresolved_from_currency = (
            stacked.groupby(group_keys)["final_unresolved_shortfall"]
            .max()
            .reset_index()["final_unresolved_shortfall"]
            .astype(bool)
        )
    else:
        unresolved_from_currency = pd.Series(False, index=out.index)

    out["final_unresolved_shortfall"] = unresolved | unresolved_from_currency.to_numpy()

    out["terminal_distress_after_interbank"] = (
            out["final_unresolved_shortfall"].astype(bool)
            | out["post_interbank_insolvent"].astype(bool)
    )

    out["continues_after_interbank_contagion"] = ~out["terminal_distress_after_interbank"]

    out["continues_after_interbank_contagion"] = ~out["terminal_distress_after_interbank"]

    return out


def build_currency_monthly_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly summary by currency.
    """
    df = results.copy()

    required_numeric_defaults = [
        "cb_support_granted",
        "asset_sale_proceeds_used_for_outflows",
        "asset_sale_liquidation_loss",
        "total_fire_sale_mtm_loss",
        "interbank_contagion_loss",
        "cash_shortfall_before_support",
        "remaining_shortfall_after_cb_support",
        "total_cb_pledged_collateral",
        "pre_cb_unencumbered_collateral_capacity",
        "post_cb_unencumbered_collateral_capacity",
        "post_interbank_LCR",
        "post_interbank_HQLA_assets_ratio",
    ]

    required_bool_defaults = [
        "policy_support_used",
        "solvent_before_lolr",
        "illiquid_before_lolr",
        "eligible_for_lolr",
        "final_unresolved_shortfall",
        "post_fire_sale_insolvent",
        "post_interbank_insolvent",
        "terminal_distress_after_interbank",
        "new_terminal_from_interbank",
        "requires_asset_sale_reaction",
    ]

    for column in required_numeric_defaults:
        if column not in df.columns:
            df[column] = 0.0

    for column in required_bool_defaults:
        if column not in df.columns:
            df[column] = False

    out = df.groupby(["currency", "stress_month"]).agg(
        alpha=("first_round_alpha", "first"),
        number_of_banks=("bank_id", "count"),
        banks_requiring_asset_sale_reaction=("requires_asset_sale_reaction", "sum"),
        banks_using_cb_support=("policy_support_used", "sum"),
        solvent_before_lolr_banks=("solvent_before_lolr", "sum"),
        illiquid_before_lolr_banks=("illiquid_before_lolr", "sum"),
        eligible_for_lolr_banks=("eligible_for_lolr", "sum"),
        final_unresolved_shortfalls=("final_unresolved_shortfall", "sum"),
        post_fire_sale_insolvent_banks=("post_fire_sale_insolvent", "sum"),
        post_interbank_insolvent_banks=("post_interbank_insolvent", "sum"),
        terminal_distress_after_interbank=("terminal_distress_after_interbank", "sum"),
        new_terminal_from_interbank=("new_terminal_from_interbank", "sum"),
        aggregate_cash_shortfall_before_support=("cash_shortfall_before_support", "sum"),
        aggregate_cb_support_granted=("cb_support_granted", "sum"),
        aggregate_asset_sale_proceeds=("asset_sale_proceeds_used_for_outflows", "sum"),
        aggregate_asset_sale_loss=("asset_sale_liquidation_loss", "sum"),
        aggregate_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
        aggregate_interbank_contagion_loss=("interbank_contagion_loss", "sum"),
        average_post_interbank_LCR=("post_interbank_LCR", "mean"),
        minimum_post_interbank_LCR=("post_interbank_LCR", "min"),
        average_post_interbank_HQLA_assets_ratio=("post_interbank_HQLA_assets_ratio", "mean"),
        minimum_post_interbank_HQLA_assets_ratio=("post_interbank_HQLA_assets_ratio", "min"),
        aggregate_cb_pledged_collateral=("total_cb_pledged_collateral", "sum"),
        aggregate_pre_cb_unencumbered_collateral_capacity=(
            "pre_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        aggregate_post_cb_unencumbered_collateral_capacity=(
            "post_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        aggregate_system_assets=("total_assets", "sum"),
        aggregate_system_equity=("equity", "sum"),
        aggregate_HQLA=("HQLA", "sum"),
        aggregate_NCOF=("NCOF", "sum"),
    ).reset_index()

    out["system_HQLA_assets_ratio"] = np.where(
        out["aggregate_system_assets"] > 0,
        out["aggregate_HQLA"] / out["aggregate_system_assets"],
        np.nan,
    )

    out["system_LCR"] = np.where(
        out["aggregate_NCOF"] > 0,
        out["aggregate_HQLA"] / out["aggregate_NCOF"],
        np.nan,
    )

    return out


def build_currency_worst_case_bank_summary(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()

    for column in [
        "post_interbank_LCR",
        "post_interbank_HQLA_assets_ratio",
        "cb_support_granted",
        "total_fire_sale_mtm_loss",
        "interbank_contagion_loss",
        "equity",
    ]:
        if column not in df.columns:
            df[column] = 0.0

    for column in [
        "policy_support_used",
        "final_unresolved_shortfall",
        "post_fire_sale_insolvent",
        "post_interbank_insolvent",
        "terminal_distress_after_interbank",
        "eligible_for_lolr",
    ]:
        if column not in df.columns:
            df[column] = False

    worst_idx = (
        df.groupby(["currency", "bank_id"])["post_interbank_LCR"]
        .idxmin()
        .dropna()
        .astype(int)
    )

    worst_rows = df.loc[
        worst_idx,
        [
            "currency",
            "bank_id",
            "bank_name",
            "stress_month",
            "post_interbank_LCR",
            "post_interbank_HQLA_assets_ratio",
        ],
    ].rename(
        columns={
            "stress_month": "worst_post_interbank_LCR_month",
            "post_interbank_LCR": "min_post_interbank_LCR",
            "post_interbank_HQLA_assets_ratio": "min_post_interbank_HQLA_assets_ratio",
        }
    )

    agg = df.groupby(["currency", "bank_id", "bank_name"]).agg(
        ever_used_cb_support=("policy_support_used", "max"),
        ever_eligible_for_lolr=("eligible_for_lolr", "max"),
        ever_final_unresolved_shortfall=("final_unresolved_shortfall", "max"),
        ever_post_fire_sale_insolvent=("post_fire_sale_insolvent", "max"),
        ever_post_interbank_insolvent=("post_interbank_insolvent", "max"),
        ever_terminal_distress_after_interbank=("terminal_distress_after_interbank", "max"),
        total_cb_support_granted=("cb_support_granted", "sum"),
        total_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
        total_interbank_contagion_loss=("interbank_contagion_loss", "sum"),
        final_equity=("equity", "last"),
    ).reset_index()

    out = worst_rows.merge(
        agg,
        on=["currency", "bank_id", "bank_name"],
        how="left",
    )

    return out.sort_values(["currency", "min_post_interbank_LCR"])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run currency-aware dynamic liquidity stress engine.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(SCENARIO_CONFIG_PATH),
        help="Path to scenario YAML config.",
    )

    return parser.parse_args()

def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    run_dynamic_interbank_contagion_path = load_engine_function()

    loader = BankingDataLoader(BANKING_INPUT_PATH)
    args = parse_args()
    scenario = load_and_validate_scenario_config(args.config)

    scenario_name = scenario.get("scenario_name", "currency_interbank_contagion")
    stress_mode = get_stress_mode(scenario)

    runtime_config = build_runtime_configs_from_scenario(
        scenario_config=scenario,
        base_lcr_assumptions=LCR_ASSUMPTIONS,
        base_liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
        base_cb_support_config=CB_SUPPORT_CONFIG,
        base_fire_sale_config=FIRE_SALE_CONFIG,
        base_interbank_contagion_config=INTERBANK_CONTAGION_CONFIG,
    )

    banks_combined = loader.load_banks("combined")
    banks_amd = attach_bank_level_reference(
        currency_banks=loader.load_banks("AMD"),
        combined_banks=banks_combined,
    )
    banks_fx = attach_bank_level_reference(
        currency_banks=loader.load_banks("FX"),
        combined_banks=banks_combined,
    )
    alpha_path_amd = get_alpha_path_for_currency(scenario, "AMD")
    alpha_path_fx = get_alpha_path_for_currency(scenario, "FX")

    amd_results, amd_market, amd_contagion = run_currency_engine(
        currency="AMD",
        banks=banks_amd,
        alpha_path=alpha_path_amd,
        scenario_name=scenario_name,
        stress_mode=stress_mode,
        run_dynamic_interbank_contagion_path=run_dynamic_interbank_contagion_path,
        runtime_config=runtime_config,
    )

    fx_results, fx_market, fx_contagion = run_currency_engine(
        currency="FX",
        banks=banks_fx,
        alpha_path=alpha_path_fx,
        scenario_name=scenario_name,
        stress_mode=stress_mode,
        run_dynamic_interbank_contagion_path=run_dynamic_interbank_contagion_path,
        runtime_config=runtime_config,
    )


    combined_from_currency = aggregate_amd_fx_results(
        amd=amd_results,
        fx=fx_results,
    )

    all_results = pd.concat(
        [amd_results, fx_results, combined_from_currency],
        ignore_index=True,
    )

    monthly_summary = build_currency_monthly_summary(all_results)
    worst_case = build_currency_worst_case_bank_summary(all_results)

    market_summary = pd.concat(
        [amd_market, fx_market],
        ignore_index=True,
    ) if not amd_market.empty or not fx_market.empty else pd.DataFrame()

    contagion_summary = pd.concat(
        [amd_contagion, fx_contagion],
        ignore_index=True,
    ) if not amd_contagion.empty or not fx_contagion.empty else pd.DataFrame()

    all_results.to_csv(
        TABLES_DIR / "currency_interbank_contagion_bank_results.csv",
        index=False,
    )

    monthly_summary.to_csv(
        TABLES_DIR / "currency_interbank_contagion_monthly_summary.csv",
        index=False,
    )

    worst_case.to_csv(
        TABLES_DIR / "currency_interbank_contagion_worst_case_bank_summary.csv",
        index=False,
    )

    if not market_summary.empty:
        market_summary.to_csv(
            TABLES_DIR / "currency_interbank_contagion_market_summary.csv",
            index=False,
        )

    if not contagion_summary.empty:
        contagion_summary.to_csv(
            TABLES_DIR / "currency_interbank_contagion_network_summary.csv",
            index=False,
        )

    print("Currency-aware full-engine bank results saved.")
    print("Currency-aware full-engine monthly summary saved.")
    print("Currency-aware full-engine worst-case bank summary saved.")

    if not market_summary.empty:
        print("Currency-aware market summary saved.")

    if not contagion_summary.empty:
        print("Currency-aware interbank network summary saved.")

    print(monthly_summary.to_string(index=False))


if __name__ == "__main__":
    main()