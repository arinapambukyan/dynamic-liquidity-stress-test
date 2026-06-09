"""
Run deterministic sensitivity and policy-scenario analysis.

The sensitivity engine reuses the canonical full liquidity stress engine and
varies policy settings, fire-sale intensity, interbank LGD, LCR requirements,
and stress paths.

Full engine:

    first-round shock
        -> bank asset-sale reaction
        -> central-bank liquidity support
        -> fire-sale feedback
        -> interbank contagion

Run from project root:

    PYTHONPATH=src python scripts/30_run_sensitivity.py
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    CB_SUPPORT_CONFIG,
    FIRE_SALE_CONFIG,
    INTERBANK_CONTAGION_CONFIG,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.scenario_config import (
    load_and_validate_scenario_config,
    get_stress_mode,
    build_runtime_configs_from_scenario,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = PROJECT_ROOT / "scripts" / "10_run_base_engine.py"
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenario_base.yaml"


def load_interbank_engine():
    """
    Load the existing full-engine script as a module.

    We do this because the file name starts with a number, so it cannot be
    imported with a normal Python import statement.
    """
    spec = importlib.util.spec_from_file_location(
        "interbank_engine",
        ENGINE_PATH,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load engine from {ENGINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def scale_price_impact(
    fire_sale_config: dict,
    multiplier: float,
) -> dict:
    """
    Scale fire-sale price impact parameters.
    """
    out = copy.deepcopy(fire_sale_config)

    out["price_impact_at_reference_sale_fraction"] = {
        asset: min(max(value * multiplier, 0.0), 0.95)
        for asset, value in out[
            "price_impact_at_reference_sale_fraction"
        ].items()
    }

    return out

def scale_lcr_buffer_weights(
    lcr_config: dict,
    government_weight: float | None = None,
    bank_weight: float | None = None,
    corporate_weight: float | None = None,
) -> dict:
    """
    Adjust HQLA recognition weights.

    Used for LCR* style wider-buffer scenarios.
    """
    out = copy.deepcopy(lcr_config)

    if government_weight is not None:
        out["government_securities_hqla_weight"] = government_weight

    if bank_weight is not None:
        out["bank_securities_hqla_weight"] = bank_weight

    if corporate_weight is not None:
        out["corporate_securities_hqla_weight"] = corporate_weight

    return out


def set_reaction_lcr_requirement(
    liquidation_config: dict,
    requirement: float,
) -> dict:
    """
    Set the active LCR requirement for bank reaction.

    This changes both:
        1. the trigger: when the bank reacts,
        2. the target: where the bank tries to restore the active LCR measure.

    This is essential for policy consistency:
        LCR_req  -> ordinary LCR target = 1.2
        LCR_low  -> ordinary LCR target = 1.0
        LCR_star -> LCR* target = 1.2, provided the scenario LCR assumptions
                    are passed into the engine.
    """
    out = copy.deepcopy(liquidation_config)

    requirement = float(requirement)

    out["reaction_lcr_trigger"] = requirement
    out["reaction_target_mode"] = "trigger"
    out["reaction_target_cap"] = requirement

    # Informational field for diagnostics.
    out["active_lcr_requirement"] = requirement

    return out

def set_cb_lcr_requirement(
    cb_config: dict,
    requirement: float,
) -> dict:
    """
    Set the active LCR requirement for CBA liquidity support.

    The CBA support gap should be computed against the same active LCR
    requirement as the policy scenario.

    Example:
        LCR_req  -> support target = 1.2
        LCR_low  -> support target = 1.0
        LCR_star -> support target = 1.2 under widened HQLA definition
    """
    out = copy.deepcopy(cb_config)

    requirement = float(requirement)

    out["illiquidity_lcr_threshold"] = requirement
    out["support_lcr_gap"] = True
    out["lcr_support_target"] = requirement

    return out



def set_asset_purchase_share(
    fire_sale_config: dict,
    purchase_share: float,
) -> dict:
    """
    Change CBA asset-purchase intensity in the fire-sale module.
    """
    out = copy.deepcopy(fire_sale_config)

    out.setdefault("central_bank_asset_purchases", {})
    out["central_bank_asset_purchases"]["enabled"] = purchase_share > 0

    out["central_bank_asset_purchases"]["purchase_share_of_bank_sales"] = {
        asset: float(purchase_share)
        for asset in out.get("asset_columns", [])
    }

    return out

def make_scenarios(
    base_alpha_path: list[float],
    base_runtime_config: dict,
) -> list[dict]:

    """
    Define sensitivity and policy-regime scenarios.

    Policy families:
        LCR_req:
            Current Armenian regulatory requirement.
            Ordinary LCR must remain at or above 1.20.

        LCR_low:
            Relaxed requirement during stress.
            Ordinary LCR must remain at or above 1.00.

        LCR_star:
            Widened HQLA buffer definition.
            LCR* must remain at or above 1.20.

        LCR_cb:
            Central-bank liquidity support / refinancing channel.
            Active LCR support target remains 1.20.

        Asset purchase:
            Central-bank asset purchases dampen state-bond fire-sale pressure.
            Ordinary LCR requirement remains 1.20.
    """
    base_cb = copy.deepcopy(base_runtime_config["cb"])
    base_fire = copy.deepcopy(base_runtime_config["fire"])
    base_interbank = copy.deepcopy(base_runtime_config["interbank"])
    base_lcr = copy.deepcopy(base_runtime_config["lcr"])
    base_liquidation = copy.deepcopy(base_runtime_config["liquidation"])
    base_modules = copy.deepcopy(base_runtime_config["modules"])

    mild_alpha = [0.25, 0.50, 0.75, 0.75, 0.50, 0.25]
    prolonged_stress_alpha = [0.75, 1.00, 1.00, 1.00, 1.00, 0.75]

    def scenario(
        name: str,
        policy_family: str,
        alpha_path: list[float] | None = None,
        cb: dict | None = None,
        fire: dict | None = None,
        interbank: dict | None = None,
        lcr: dict | None = None,
        liquidation: dict | None = None,
    ) -> dict:
        return {
            "scenario": name,
            "policy_family": policy_family,
            "alpha_path": alpha_path if alpha_path is not None else base_alpha_path,
            "cb": copy.deepcopy(cb if cb is not None else base_cb),
            "fire": copy.deepcopy(fire if fire is not None else base_fire),
            "interbank": copy.deepcopy(interbank if interbank is not None else base_interbank),
            "lcr": copy.deepcopy(lcr if lcr is not None else base_lcr),
            "liquidation": copy.deepcopy(
                liquidation if liquidation is not None else base_liquidation
            ),
            "modules": copy.deepcopy(base_modules),
        }

    # ----------------------------------------------------------
    # Central-bank liquidity support variants
    # ----------------------------------------------------------
    no_cb = copy.deepcopy(base_cb)
    no_cb["enabled"] = False

    cba_first_collateral_capped = copy.deepcopy(base_cb)
    cba_first_collateral_capped["enabled"] = True
    cba_first_collateral_capped["support_mode"] = "collateral_capped"
    cba_first_collateral_capped["refinancing_strategy"] = "cba_first"
    cba_first_collateral_capped["pre_liquidation_refinancing_share"] = 1.0
    cba_first_collateral_capped["support_only_solvent_illiquid"] = True
    cba_first_collateral_capped["outside_30d_lcr_window"] = True

    full_coverage_cb = copy.deepcopy(base_cb)
    full_coverage_cb["enabled"] = True
    full_coverage_cb["support_mode"] = "full_coverage"
    full_coverage_cb["refinancing_strategy"] = "cba_first"
    full_coverage_cb["pre_liquidation_refinancing_share"] = 1.0
    full_coverage_cb["support_only_solvent_illiquid"] = True
    full_coverage_cb["outside_30d_lcr_window"] = True

    # ----------------------------------------------------------
    # LCR switching rules
    # ----------------------------------------------------------
    # Armenia current requirement in this model:
    #     LCR_req = 1.20
    #
    # Relaxed policy:
    #     LCR_low = 1.00
    #
    # Widened-buffer policy:
    #     LCR_star keeps the 1.20 requirement, but applies it to the widened
    #     HQLA definition.

    LCR_REQ = 1.20
    LCR_LOW = 1.00

    lcr_req_120_liquidation = set_reaction_lcr_requirement(
        base_liquidation,
        requirement=LCR_REQ,
    )

    lcr_low_100_liquidation = set_reaction_lcr_requirement(
        base_liquidation,
        requirement=LCR_LOW,
    )

    lcr_star = scale_lcr_buffer_weights(
        base_lcr,
        government_weight=0.95,
        bank_weight=0.60,
        corporate_weight=0.35,
    )

    lcr_star_liquidation = set_reaction_lcr_requirement(
        base_liquidation,
        requirement=LCR_REQ,
    )

    lcr_cb = copy.deepcopy(base_lcr)
    lcr_cb["cb_liquidity_funding_runoff"] = 0.0

    cba_collateral_lcr_req = set_cb_lcr_requirement(
        cba_first_collateral_capped,
        requirement=LCR_REQ,
    )

    full_coverage_cb_lcr_req = set_cb_lcr_requirement(
        full_coverage_cb,
        requirement=LCR_REQ,
    )

    # ----------------------------------------------------------
    # Asset-purchase variants
    # ----------------------------------------------------------
    no_asset_purchases = set_asset_purchase_share(base_fire, purchase_share=0.0)
    moderate_asset_purchases = set_asset_purchase_share(base_fire, purchase_share=0.50)
    strong_asset_purchases = set_asset_purchase_share(base_fire, purchase_share=0.80)

    no_fire_sale = copy.deepcopy(base_fire)
    no_fire_sale["enabled"] = False

    high_fire_sale = scale_price_impact(base_fire, multiplier=2.0)
    low_fire_sale = scale_price_impact(base_fire, multiplier=0.5)

    high_lgd = copy.deepcopy(base_interbank)
    high_lgd["loss_given_default"] = 0.90

    low_lgd = copy.deepcopy(base_interbank)
    low_lgd["loss_given_default"] = 0.30

    no_interbank = copy.deepcopy(base_interbank)
    no_interbank["enabled"] = False

    severe_combined_fire = scale_price_impact(base_fire, multiplier=2.0)
    severe_combined_interbank = copy.deepcopy(base_interbank)
    severe_combined_interbank["loss_given_default"] = 0.90

    return [
        # ------------------------------------------------------
        # Main current model
        # ------------------------------------------------------
        scenario(
            name="base_full_engine",
            policy_family="baseline",
        ),

        # ------------------------------------------------------
        # Van den End-style LCR switching rules
        # ------------------------------------------------------
        scenario(
            name="lcr_req_120_no_lolr",
            policy_family="LCR_req",
            cb=no_cb,
            fire=no_asset_purchases,
            liquidation=lcr_req_120_liquidation,
        ),
        scenario(
            name="lcr_low_100_no_lolr",
            policy_family="LCR_low",
            cb=no_cb,
            fire=no_asset_purchases,
            liquidation=lcr_low_100_liquidation,
        ),
        scenario(
            name="lcr_star_wide_buffer_no_lolr",
            policy_family="LCR_star",
            cb=no_cb,
            fire=no_asset_purchases,
            lcr=lcr_star,
            liquidation=lcr_star_liquidation,
        ),
        scenario(
            name="lcr_cb_collateral_capped_refinancing",
            policy_family="LCR_cb",
            cb=cba_collateral_lcr_req,
            lcr=lcr_cb,
            liquidation=lcr_req_120_liquidation,
        ),
        scenario(
            name="lcr_cb_full_coverage_refinancing",
            policy_family="LCR_cb",
            cb=full_coverage_cb_lcr_req,
            lcr=lcr_cb,
            liquidation=lcr_req_120_liquidation,
        ),
        scenario(
            name="asset_purchase_state_bond_floor",
            policy_family="asset_purchase",
            cb=no_cb,
            fire=strong_asset_purchases,
            liquidation=lcr_req_120_liquidation,
        ),
        scenario(
            name="lcr_cb_plus_asset_purchases",
            policy_family="combined_policy",
            cb=cba_collateral_lcr_req,
            fire=strong_asset_purchases,
            lcr=lcr_cb,
            liquidation=lcr_req_120_liquidation,
        ),

        # ------------------------------------------------------
        # Existing sensitivity scenarios
        # ------------------------------------------------------
        scenario(
            name="cba_first_collateral_capped_refinancing",
            policy_family="support_mode",
            cb=cba_first_collateral_capped,
        ),
        scenario(
            name="full_coverage_cb_refinancing",
            policy_family="support_mode",
            cb=full_coverage_cb,
        ),
        scenario(
            name="mild_alpha_path",
            policy_family="stress_path",
            alpha_path=mild_alpha,
        ),
        scenario(
            name="prolonged_high_stress_alpha_path",
            policy_family="stress_path",
            alpha_path=prolonged_stress_alpha,
        ),
        scenario(
            name="no_cb_support",
            policy_family="support_mode",
            cb=no_cb,
        ),
        scenario(
            name="no_fire_sale_feedback",
            policy_family="fire_sale",
            fire=no_fire_sale,
        ),
        scenario(
            name="low_fire_sale_impact",
            policy_family="fire_sale",
            fire=low_fire_sale,
        ),
        scenario(
            name="high_fire_sale_impact",
            policy_family="fire_sale",
            fire=high_fire_sale,
        ),
        scenario(
            name="low_interbank_lgd",
            policy_family="interbank",
            interbank=low_lgd,
        ),
        scenario(
            name="high_interbank_lgd",
            policy_family="interbank",
            interbank=high_lgd,
        ),
        scenario(
            name="no_interbank_contagion",
            policy_family="interbank",
            interbank=no_interbank,
        ),
        scenario(
            name="severe_combined_fire_and_network",
            policy_family="combined_stress",
            alpha_path=prolonged_stress_alpha,
            fire=severe_combined_fire,
            interbank=severe_combined_interbank,
        ),
    ]

def ensure_sensitivity_columns(results: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure optional engine columns exist even when a module is disabled.
    """
    out = results.copy()

    bool_defaults = [
        "final_unresolved_shortfall",
        "post_fire_sale_insolvent",
        "post_interbank_insolvent",
        "new_terminal_from_interbank",
        "terminal_distress_after_interbank",
    ]

    numeric_defaults = [
        "cb_support_granted",
        "total_fire_sale_mtm_loss",
        "interbank_contagion_loss",
        "asset_sale_proceeds_used_for_outflows",
        "asset_sale_liquidation_loss",
        "post_interbank_LCR",
        "post_interbank_HQLA_assets_ratio",
        "cb_liquidity_funding",
    ]

    for column in bool_defaults:
        if column not in out.columns:
            out[column] = False

    for column in numeric_defaults:
        if column not in out.columns:
            out[column] = 0.0

    return out

def summarize_scenario(
    scenario_name: str,
    policy_family: str,
    results: pd.DataFrame,
    market_summary: pd.DataFrame,
    network_summary: pd.DataFrame,
) -> dict:
    """

    Produce one row of scenario-level metrics.
    """
    results = ensure_sensitivity_columns(results)
    last_month = results["stress_month"].max()
    final_rows = results.loc[results["stress_month"] == last_month].copy()

    monthly_cb_support = (
        results.groupby("stress_month")["cb_support_granted"].sum()
        if "cb_support_granted" in results.columns
        else pd.Series(dtype=float)
    )

    monthly_fire_loss = (
        results.groupby("stress_month")["total_fire_sale_mtm_loss"].sum()
        if "total_fire_sale_mtm_loss" in results.columns
        else pd.Series(dtype=float)
    )

    monthly_interbank_loss = (
        results.groupby("stress_month")["interbank_contagion_loss"].sum()
        if "interbank_contagion_loss" in results.columns
        else pd.Series(dtype=float)
    )

    return {
        "scenario": scenario_name,
        "policy_family": policy_family,
        "number_of_months": int(results["stress_month"].nunique()),

        "min_post_interbank_LCR": results["post_interbank_LCR"].min(),
        "average_final_post_interbank_LCR": final_rows[
            "post_interbank_LCR"
        ].mean(),
        "minimum_final_post_interbank_LCR": final_rows[
            "post_interbank_LCR"
        ].min(),

        "total_cash_shortfall_before_support": results[
            "cash_shortfall_before_support"
        ].sum(),
        "total_cb_support_granted": results["cb_support_granted"].sum(),
        "max_monthly_cb_support_granted": (
            monthly_cb_support.max() if not monthly_cb_support.empty else 0.0
        ),

        "total_asset_sale_proceeds": results[
            "asset_sale_proceeds_used_for_outflows"
        ].sum(),
        "total_asset_sale_loss": results["asset_sale_liquidation_loss"].sum(),

        "total_fire_sale_mtm_loss": results[
            "total_fire_sale_mtm_loss"
        ].sum(),
        "max_monthly_fire_sale_mtm_loss": (
            monthly_fire_loss.max() if not monthly_fire_loss.empty else 0.0
        ),

        "total_interbank_contagion_loss": results[
            "interbank_contagion_loss"
        ].sum(),
        "max_monthly_interbank_contagion_loss": (
            monthly_interbank_loss.max()
            if not monthly_interbank_loss.empty
            else 0.0
        ),

        "ever_final_unresolved_shortfall_banks": int(
            results.groupby("bank_id")["final_unresolved_shortfall"].max().sum()
        ),
        "ever_post_fire_sale_insolvent_banks": int(
            results.groupby("bank_id")["post_fire_sale_insolvent"].max().sum()
        ),
        "ever_post_interbank_insolvent_banks": int(
            results.groupby("bank_id")["post_interbank_insolvent"].max().sum()
        ),
        "ever_new_terminal_from_interbank_banks": int(
            results.groupby("bank_id")["new_terminal_from_interbank"].max().sum()
        ),
        "ever_terminal_distress_after_interbank_banks": int(
            results.groupby("bank_id")[
                "terminal_distress_after_interbank"
            ].max().sum()
        ),
        "final_terminal_distress_after_interbank_banks": int(
            final_rows["terminal_distress_after_interbank"].sum()
        ),

        "final_system_assets": final_rows["total_assets"].sum(),
        "final_system_equity": final_rows["equity"].sum(),
        "final_system_cb_liquidity_funding": final_rows[
            "cb_liquidity_funding"
        ].sum(),

        "market_summary_rows": len(market_summary),
        "network_summary_rows": len(network_summary),
    }


def build_sensitivity_bank_summary(
    scenario_name: str,
    policy_family: str,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bank-level summary for one scenario.
    """
    results = ensure_sensitivity_columns(results)

    out = results.groupby(["bank_id", "bank_name"]).agg(
        min_post_interbank_LCR=("post_interbank_LCR", "min"),
        final_post_interbank_LCR=("post_interbank_LCR", "last"),
        min_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "min",
        ),
        total_cb_support_granted=("cb_support_granted", "sum"),
        total_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
        total_interbank_contagion_loss=("interbank_contagion_loss", "sum"),
        ever_final_unresolved_shortfall=(
            "final_unresolved_shortfall",
            "max",
        ),
        ever_post_fire_sale_insolvent=(
            "post_fire_sale_insolvent",
            "max",
        ),
        ever_post_interbank_insolvent=(
            "post_interbank_insolvent",
            "max",
        ),
        ever_new_terminal_from_interbank=(
            "new_terminal_from_interbank",
            "max",
        ),
        ever_terminal_distress_after_interbank=(
            "terminal_distress_after_interbank",
            "max",
        ),
        final_total_assets=("total_assets", "last"),
        final_equity=("equity", "last"),
        final_cb_liquidity_funding=("cb_liquidity_funding", "last"),
    ).reset_index()

    out.insert(0, "scenario", scenario_name)
    out.insert(1, "policy_family", policy_family)

    return out

def build_lolr_diagnostic_summary(
    scenario_name: str,
    policy_family: str,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Monthly LoLR diagnostics for one sensitivity scenario.

    Purpose:
        make the CBA support mechanism transparent:
            cash-shortfall need
            LCR support gap
            required support
            support granted
            support used for outflows
            support retained as cash/HQLA
    """
    if "stress_month" not in results.columns:
        return pd.DataFrame()

    out = results.copy()

    numeric_defaults = [
        "lolr_cash_shortfall_need",
        "lolr_lcr_support_gap",
        "lolr_required_support",
        "cb_support_required",
        "cb_support_target_before_asset_sales",
        "cb_support_granted",
        "cb_support_used_for_outflows",
        "cb_support_retained_as_cash_for_lcr",
        "cash_shortfall_before_cba_refinancing",
        "cash_shortfall_after_cb_refinancing",
        "pre_lolr_HQLA",
        "pre_lolr_NCOF",
        "pre_lolr_LCR",
        "post_cb_support_LCR",
        "post_cb_support_HQLA_assets_ratio",
        "pre_cb_unencumbered_collateral_capacity",
        "post_cb_unencumbered_collateral_capacity",
    ]

    bool_defaults = [
        "solvent_before_lolr",
        "illiquid_before_lolr",
        "cash_shortfall_illiquid",
        "lcr_illiquid",
        "hqla_assets_illiquid",
        "eligible_for_lolr",
        "policy_support_used",
    ]

    for column in numeric_defaults:
        if column not in out.columns:
            out[column] = 0.0

    for column in bool_defaults:
        if column not in out.columns:
            out[column] = False

    diagnostic = out.groupby("stress_month").agg(
        number_of_banks=("bank_id", "count"),

        solvent_before_lolr_banks=("solvent_before_lolr", "sum"),
        illiquid_before_lolr_banks=("illiquid_before_lolr", "sum"),
        cash_shortfall_illiquid_banks=("cash_shortfall_illiquid", "sum"),
        lcr_illiquid_banks=("lcr_illiquid", "sum"),
        hqla_assets_illiquid_banks=("hqla_assets_illiquid", "sum"),
        eligible_for_lolr_banks=("eligible_for_lolr", "sum"),
        banks_using_cb_support=("policy_support_used", "sum"),

        aggregate_cash_shortfall_need=("lolr_cash_shortfall_need", "sum"),
        aggregate_lcr_support_gap=("lolr_lcr_support_gap", "sum"),
        aggregate_lolr_required_support=("lolr_required_support", "sum"),
        aggregate_cb_support_required=("cb_support_required", "sum"),
        aggregate_cb_support_target=("cb_support_target_before_asset_sales", "sum"),
        aggregate_cb_support_granted=("cb_support_granted", "sum"),

        aggregate_support_used_for_outflows=("cb_support_used_for_outflows", "sum"),
        aggregate_support_retained_as_cash_for_lcr=(
            "cb_support_retained_as_cash_for_lcr",
            "sum",
        ),

        aggregate_cash_shortfall_before_cba=(
            "cash_shortfall_before_cba_refinancing",
            "sum",
        ),
        aggregate_cash_shortfall_after_cba=(
            "cash_shortfall_after_cb_refinancing",
            "sum",
        ),

        average_pre_lolr_LCR=("pre_lolr_LCR", "mean"),
        minimum_pre_lolr_LCR=("pre_lolr_LCR", "min"),
        average_post_cb_support_LCR=("post_cb_support_LCR", "mean"),
        minimum_post_cb_support_LCR=("post_cb_support_LCR", "min"),

        aggregate_pre_cb_unencumbered_collateral_capacity=(
            "pre_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        aggregate_post_cb_unencumbered_collateral_capacity=(
            "post_cb_unencumbered_collateral_capacity",
            "sum",
        ),
    ).reset_index()

    diagnostic.insert(0, "scenario", scenario_name)
    diagnostic.insert(1, "policy_family", policy_family)

    diagnostic["support_retained_share"] = (
        diagnostic["aggregate_support_retained_as_cash_for_lcr"]
        / diagnostic["aggregate_cb_support_granted"].replace(0.0, pd.NA)
    ).fillna(0.0)

    diagnostic["lcr_gap_share_of_required_support"] = (
        diagnostic["aggregate_lcr_support_gap"]
        / diagnostic["aggregate_lolr_required_support"].replace(0.0, pd.NA)
    ).fillna(0.0)

    diagnostic["support_gap_after_targeting"] = (
        diagnostic["aggregate_lolr_required_support"]
        - diagnostic["aggregate_cb_support_granted"]
    ).clip(lower=0.0)

    return diagnostic

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sensitivity analysis for the liquidity stress engine.",
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

    args = parse_args()

    engine = load_interbank_engine()

    banks = BankingDataLoader(BANKING_INPUT_PATH).load_banks()

    base_scenario = load_and_validate_scenario_config(args.config)
    stress_mode = get_stress_mode(base_scenario)

    base_alpha_path = base_scenario.get("severity", {}).get("alpha_path")

    if base_alpha_path is None:
        raise ValueError("scenario_base.yaml must define severity.alpha_path")

    scenario_name_from_yaml = base_scenario.get(
        "scenario_name",
        "sensitivity_base",
    )

    base_runtime_config = build_runtime_configs_from_scenario(
        scenario_config=base_scenario,
        base_lcr_assumptions=LCR_ASSUMPTIONS,
        base_liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
        base_cb_support_config=CB_SUPPORT_CONFIG,
        base_fire_sale_config=FIRE_SALE_CONFIG,
        base_interbank_contagion_config=INTERBANK_CONTAGION_CONFIG,
    )

    sensitivity_scenarios = make_scenarios(
        base_alpha_path=base_alpha_path,
        base_runtime_config=base_runtime_config,
    )

    scenario_rows = []
    monthly_rows = []
    bank_rows = []
    network_rows = []
    market_rows = []
    lolr_diagnostic_rows = []

    for scenario_config in sensitivity_scenarios:
        scenario_name = scenario_config["scenario"]
        policy_family = scenario_config["policy_family"]

        print(f"\nRunning sensitivity scenario: {scenario_name}")

        results, market_summary, network_summary = (
            engine.run_dynamic_interbank_contagion_path(
                banks=banks,
                alpha_path=scenario_config["alpha_path"],
                scenario_name=f"{scenario_name_from_yaml}_{scenario_name}",
                stress_mode=stress_mode,

                # Pass scenario-specific mechanics explicitly.
                # This is what makes LCR_req, LCR_low, LCR_star, LCR_cb,
                # asset-purchase, fire-sale, and interbank scenarios genuinely different.
                lcr_assumptions=scenario_config["lcr"],
                liquidation_config=scenario_config["liquidation"],
                cb_support_config=scenario_config["cb"],
                fire_sale_config=scenario_config["fire"],
                interbank_contagion_config=scenario_config["interbank"],
                module_config=scenario_config["modules"],
            )
        )

        scenario_rows.append(
            summarize_scenario(
                scenario_name=scenario_name,
                policy_family=policy_family,
                results=results,
                market_summary=market_summary,
                network_summary=network_summary,
            )
        )

        monthly_summary = engine.build_interbank_monthly_summary(results)
        monthly_summary.insert(0, "scenario", scenario_name)
        monthly_summary.insert(1, "policy_family", policy_family)
        monthly_rows.append(monthly_summary)

        bank_rows.append(
            build_sensitivity_bank_summary(
                scenario_name=scenario_name,
                policy_family=policy_family,
                results=results,
            )
        )

        lolr_diagnostic = build_lolr_diagnostic_summary(
            scenario_name=scenario_name,
            policy_family=policy_family,
            results=results,
        )

        if not lolr_diagnostic.empty:
            lolr_diagnostic_rows.append(lolr_diagnostic)

        if not network_summary.empty:
            network_summary = network_summary.copy()
            network_summary.insert(0, "scenario", scenario_name)
            network_summary.insert(1, "policy_family", policy_family)
            network_rows.append(network_summary)

        if not market_summary.empty:
            market_summary = market_summary.copy()
            market_summary.insert(0, "scenario", scenario_name)
            market_summary.insert(1, "policy_family", policy_family)
            market_rows.append(market_summary)

    sensitivity_summary = pd.DataFrame(scenario_rows).sort_values(
        [
            "ever_terminal_distress_after_interbank_banks",
            "total_cb_support_granted",
            "total_fire_sale_mtm_loss",
        ],
        ascending=[False, False, False],
    )

    sensitivity_monthly_summary = pd.concat(
        monthly_rows,
        ignore_index=True,
    )

    sensitivity_bank_summary = pd.concat(
        bank_rows,
        ignore_index=True,
    )

    sensitivity_network_summary = (
        pd.concat(network_rows, ignore_index=True)
        if network_rows
        else pd.DataFrame()
    )

    sensitivity_market_summary = (
        pd.concat(market_rows, ignore_index=True)
        if market_rows
        else pd.DataFrame()
    )

    sensitivity_lolr_diagnostics = (
        pd.concat(lolr_diagnostic_rows, ignore_index=True)
        if lolr_diagnostic_rows
        else pd.DataFrame()
    )

    sensitivity_summary.to_csv(
        TABLES_DIR / "sensitivity_summary.csv",
        index=False,
    )

    sensitivity_monthly_summary.to_csv(
        TABLES_DIR / "sensitivity_monthly_summary.csv",
        index=False,
    )

    sensitivity_bank_summary.to_csv(
        TABLES_DIR / "sensitivity_bank_summary.csv",
        index=False,
    )

    sensitivity_network_summary.to_csv(
        TABLES_DIR / "sensitivity_network_summary.csv",
        index=False,
    )

    sensitivity_market_summary.to_csv(
        TABLES_DIR / "sensitivity_market_summary.csv",
        index=False,
    )

    sensitivity_lolr_diagnostics.to_csv(
        TABLES_DIR / "sensitivity_lolr_diagnostics.csv",
        index=False,
    )

    print("\nSensitivity summary saved.")
    print("Sensitivity monthly summary saved.")
    print("Sensitivity bank summary saved.")
    print("Sensitivity network summary saved.")
    print("Sensitivity market summary saved.")
    print("Sensitivity LoLR diagnostics saved.")

    display_columns = [
        "scenario",
        "policy_family",
        "min_post_interbank_LCR",
        "total_cb_support_granted",
        "total_fire_sale_mtm_loss",
        "total_interbank_contagion_loss",
        "ever_final_unresolved_shortfall_banks",
        "ever_post_fire_sale_insolvent_banks",
        "ever_new_terminal_from_interbank_banks",
        "ever_terminal_distress_after_interbank_banks",
        "final_system_equity",
    ]

    print("\nSCENARIO-LEVEL SENSITIVITY SUMMARY")
    print(
        sensitivity_summary[display_columns]
        .sort_values("total_cb_support_granted", ascending=False)
        .to_string(index=False)
    )

    lolr_display_columns = [
        "scenario",
        "policy_family",
        "stress_month",
        "eligible_for_lolr_banks",
        "banks_using_cb_support",
        "aggregate_cash_shortfall_need",
        "aggregate_lcr_support_gap",
        "aggregate_lolr_required_support",
        "aggregate_cb_support_granted",
        "aggregate_support_used_for_outflows",
        "aggregate_support_retained_as_cash_for_lcr",
        "minimum_pre_lolr_LCR",
        "minimum_post_cb_support_LCR",
    ]

    if not sensitivity_lolr_diagnostics.empty:
        print("\nLoLR DIAGNOSTIC SUMMARY: POLICY SCENARIOS")
        policy_mask = sensitivity_lolr_diagnostics["scenario"].str.contains(
            "lcr_cb|cba_first|full_coverage|base_full_engine",
            regex=True,
        )

        print(
            sensitivity_lolr_diagnostics.loc[
                policy_mask,
                [c for c in lolr_display_columns if c in sensitivity_lolr_diagnostics.columns],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()