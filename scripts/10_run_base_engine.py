"""
Run the canonical full liquidity stress engine.

Pipeline:
    first-round shock
    -> bank asset-sale reaction
    -> CBA liquidity support
    -> fire-sale price impact
    -> interbank contagion

Run from project root:

    PYTHONPATH=src python scripts/10_run_base_engine.py
"""
import argparse
import copy

import numpy as np
import pandas as pd


from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    FIRST_ROUND_STRESS_VECTOR,
    FIRST_ROUND_SCENARIO_CONFIG,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    CB_SUPPORT_CONFIG,
    FIRE_SALE_CONFIG,
    INTERBANK_CONTAGION_CONFIG,
    TABLES_DIR,
    INTERBANK_EXPOSURE_CONFIG,
)

from liquidity_stress.network_interbank import (
    get_or_create_interbank_exposure_matrix,
    build_synthetic_interbank_exposure_matrix,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.first_round import (
    apply_one_month_balance_sheet_shock,
    compute_lcr_from_balance_sheet,
)
from liquidity_stress.fire_sales import apply_fire_sale_price_impact
from liquidity_stress.interbank_contagion import apply_interbank_contagion
from liquidity_stress.policy_support import apply_cb_liquidity_support
from liquidity_stress.reaction import apply_asset_sale_reaction
from liquidity_stress.scenario_config import (
    load_and_validate_scenario_config,
    get_stress_mode,
    get_alpha_path_for_currency,
    build_runtime_configs_from_scenario,
)

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
    "total_fire_sale_mtm_loss",
    "interbank_contagion_loss",
    "cash_shortfall_before_cba_refinancing",
    "cash_shortfall_after_cb_refinancing",
    "cb_support_target_before_asset_sales",
    "newly_pledged_collateral",
    "newly_pledged_collateral_support_value",
]


def zero_current_period_flows(
    df: pd.DataFrame,
    fire_sale_config: dict | None = None,
) -> pd.DataFrame:
    out = df.copy()

    if fire_sale_config is None:
        fire_sale_config = FIRE_SALE_CONFIG

    for column in CURRENT_PERIOD_FLOW_COLUMNS:
        if column in out.columns:
            out[column] = 0.0

    for asset_column in fire_sale_config["asset_columns"]:
        sold_column = f"{asset_column}_sold_for_liquidity"
        loss_column = f"{asset_column}_fire_sale_mtm_loss"

        if sold_column in out.columns:
            out[sold_column] = 0.0

        if loss_column in out.columns:
            out[loss_column] = 0.0

    round_loss_columns = [
        column
        for column in out.columns
        if column.startswith("interbank_contagion_loss_round_")
    ]

    for column in round_loss_columns:
        out[column] = 0.0

    for column in out.columns:
        if column.startswith("newly_pledged_"):
            out[column] = 0.0

    return out

def refresh_liquidity_metrics(
    banks: pd.DataFrame,
    lcr_assumptions: dict,
    prefix: str,
) -> pd.DataFrame:
    """
    Recompute LCR and HQLA/assets after a model stage and store
    both current values and stage-specific diagnostic columns.
    """
    out = banks.copy()

    refreshed = compute_lcr_from_balance_sheet(
        banks=out,
        lcr_assumptions=lcr_assumptions,
    )

    out["HQLA"] = refreshed["HQLA"]
    out["HQLA_assets_ratio"] = refreshed["HQLA_assets_ratio"]
    out["NCOF"] = refreshed["NCOF"]
    out["LCR"] = refreshed["LCR"]

    out[f"{prefix}_HQLA"] = out["HQLA"]
    out[f"{prefix}_HQLA_assets_ratio"] = out["HQLA_assets_ratio"]
    out[f"{prefix}_NCOF"] = out["NCOF"]
    out[f"{prefix}_LCR"] = out["LCR"]

    return out

def _active_requirement_series(out: pd.DataFrame) -> pd.Series:
    if "active_lcr_requirement" in out.columns:
        return pd.to_numeric(
            out["active_lcr_requirement"],
            errors="coerce",
        ).fillna(1.0)

    return pd.Series(1.0, index=out.index)


def bypass_bank_reaction(
    banks: pd.DataFrame,
    lcr_assumptions: dict,
    liquidation_config: dict,
) -> pd.DataFrame:
    """
    Skip bank asset-sale reaction while preserving downstream columns.

    This is used when modules.bank_reaction = false.
    """
    out = banks.copy()

    active_lcr_requirement = float(
        liquidation_config.get(
            "active_lcr_requirement",
            liquidation_config.get("reaction_lcr_trigger", 1.0),
        )
    )

    out["active_lcr_requirement"] = active_lcr_requirement
    out["reaction_lcr_trigger"] = float(
        liquidation_config.get("reaction_lcr_trigger", active_lcr_requirement)
    )

    if "cash_shortfall_before_support" not in out.columns:
        out["cash_shortfall_before_support"] = 0.0

    out["cash_shortfall_after_asset_sales"] = out[
        "cash_shortfall_before_support"
    ].fillna(0.0)

    out["requires_asset_sale_reaction"] = False
    out["asset_sale_reaction_success"] = (
        out["cash_shortfall_after_asset_sales"] <= 1e-9
    )
    out["regulatory_lcr_reaction_triggered"] = False
    out["regulatory_lcr_target"] = out["reaction_lcr_trigger"]
    out["regulatory_hqla_gap_before_reaction"] = 0.0
    out["regulatory_hqla_gap_after_reaction"] = 0.0
    out["regulatory_lcr_reaction_success"] = True

    out["unresolved_shortfall_after_asset_sales"] = (
        out["cash_shortfall_after_asset_sales"] > 1e-9
    )

    zero_columns = [
        "asset_sale_proceeds_used_for_outflows",
        "asset_sale_cash_retained_for_lcr",
        "asset_sale_liquidation_loss",
        "total_securities_sold_for_liquidity",
    ]

    for column in zero_columns:
        out[column] = 0.0

    reaction_assets = set(liquidation_config.get("liquidation_order", []))
    reaction_assets.update(liquidation_config.get("regulatory_reaction_order", []))

    for asset_column in reaction_assets:
        out[f"{asset_column}_sold_for_liquidity"] = 0.0
        out[f"{asset_column}_sale_proceeds"] = 0.0
        out[f"{asset_column}_sale_loss"] = 0.0

    out = refresh_liquidity_metrics(
        banks=out,
        lcr_assumptions=lcr_assumptions,
        prefix="post_asset_sale",
    )

    out["post_asset_sale_LCR_breach"] = (
        out["post_asset_sale_LCR"] < out["active_lcr_requirement"]
    )

    return out


def bypass_policy_support(
    banks: pd.DataFrame,
    cb_support_config: dict,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Skip CBA support while preserving downstream columns.

    This is used when modules.policy_intervention = false.
    """
    out = banks.copy()

    if "cash_shortfall_after_asset_sales" not in out.columns:
        out["cash_shortfall_after_asset_sales"] = 0.0

    remaining_shortfall = out["cash_shortfall_after_asset_sales"].fillna(0.0)

    out["cb_liquidity_funding"] = out.get("cb_liquidity_funding", 0.0)
    out["cb_support_required"] = 0.0
    out["cb_support_collateral_capacity"] = 0.0
    out["cb_support_granted"] = 0.0
    out["cb_support_used_for_outflows"] = 0.0
    out["cb_support_retained_as_cash_for_lcr"] = 0.0
    out["aggregate_support_used_for_outflows"] = 0.0
    out["aggregate_support_retained_as_cash_for_lcr"] = 0.0
    out["remaining_shortfall_after_cb_support"] = remaining_shortfall
    out["final_unresolved_shortfall"] = remaining_shortfall > 1e-9
    out["policy_support_used"] = False

    out["eligible_for_lolr"] = False
    out["lolr_denial_reason"] = "policy_disabled"

    out["pre_lolr_HQLA"] = out.get("post_asset_sale_HQLA", out.get("HQLA", 0.0))
    out["pre_lolr_NCOF"] = out.get("post_asset_sale_NCOF", out.get("NCOF", 1.0))
    out["pre_lolr_LCR"] = out.get("post_asset_sale_LCR", out.get("LCR", np.inf))
    out["pre_lolr_lcr"] = out["pre_lolr_LCR"]
    out["pre_lolr_hqla_assets_ratio"] = out.get(
        "post_asset_sale_HQLA_assets_ratio",
        out.get("HQLA_assets_ratio", np.inf),
    )

    out["solvent_before_lolr"] = (
        out["equity"].astype(float)
        > float(cb_support_config.get("solvency_equity_threshold", 0.0))
    )

    active_requirement = _active_requirement_series(out)

    out["cash_shortfall_illiquid"] = remaining_shortfall > 1e-9
    out["lcr_illiquid"] = out["pre_lolr_LCR"].astype(float) < active_requirement
    out["hqla_assets_illiquid"] = False
    out["illiquid_before_lolr"] = (
        out["cash_shortfall_illiquid"] | out["lcr_illiquid"]
    )

    out["lolr_cash_shortfall_need"] = remaining_shortfall
    out["lolr_lcr_support_gap"] = 0.0
    out["lolr_hqla_assets_support_gap"] = 0.0
    out["lolr_buffer_support_need"] = 0.0
    out["lolr_required_support"] = 0.0

    out["pre_cb_unencumbered_collateral_capacity"] = 0.0
    out["post_cb_unencumbered_collateral_capacity"] = 0.0
    out["unencumbered_collateral_capacity"] = 0.0
    out["total_cb_pledged_collateral"] = 0.0
    out["cb_support_assets_ratio"] = 0.0

    out = refresh_liquidity_metrics(
        banks=out,
        lcr_assumptions=lcr_assumptions,
        prefix="post_cb_support",
    )

    out["post_cb_support_LCR_breach"] = (
        out["post_cb_support_LCR"] < _active_requirement_series(out)
    )

    return out


def bypass_fire_sales(
    banks: pd.DataFrame,
    fire_sale_config: dict,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Skip common-asset fire-sale feedback while preserving downstream columns.

    This is used when modules.fire_sales = false.
    """
    out = banks.copy()

    out["total_fire_sale_mtm_loss"] = 0.0
    out["total_fire_sale_mtm_loss_assets_ratio"] = 0.0

    for asset_column in fire_sale_config.get("asset_columns", []):
        out[f"{asset_column}_fire_sale_mtm_loss"] = 0.0

    out = refresh_liquidity_metrics(
        banks=out,
        lcr_assumptions=lcr_assumptions,
        prefix="post_fire_sale",
    )

    out["post_fire_sale_LCR_breach"] = (
        out["post_fire_sale_LCR"] < _active_requirement_series(out)
    )

    out["post_fire_sale_insolvent"] = out["equity"].astype(float) < 0.0

    if "final_unresolved_shortfall" not in out.columns:
        out["final_unresolved_shortfall"] = False

    out["terminal_distress_after_fire_sale"] = (
        out["final_unresolved_shortfall"].fillna(False).astype(bool)
        | out["post_fire_sale_insolvent"].fillna(False).astype(bool)
    )

    return out

def run_dynamic_interbank_contagion_path(
    banks: pd.DataFrame,
    alpha_path: list[float],
    scenario_name: str,
    exposure_matrix: pd.DataFrame | None = None,
    stress_mode: str = "full_systemic",
    lcr_assumptions: dict | None = None,
    liquidation_config: dict | None = None,
    cb_support_config: dict | None = None,
    fire_sale_config: dict | None = None,
    interbank_contagion_config: dict | None = None,
    module_config: dict | None = None,
    monthly_stress_vectors: list[pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run dynamic monthly path:

        first-round shock
        -> asset-sale reaction
        -> CBA liquidity support
        -> fire-sale price impact
        -> interbank contagion
        -> next month balance sheet

    Important:
        Scenario-specific configs must be passed through the whole engine.
        Otherwise policy scenarios such as LCR_low, LCR_star, and LCR_cb
        silently fall back to default global configs.
    """
    if lcr_assumptions is None:
        lcr_assumptions = copy.deepcopy(LCR_ASSUMPTIONS)

    if liquidation_config is None:
        liquidation_config = copy.deepcopy(LIQUIDATION_WATERFALL_CONFIG)

    if cb_support_config is None:
        cb_support_config = copy.deepcopy(CB_SUPPORT_CONFIG)

    if fire_sale_config is None:
        fire_sale_config = copy.deepcopy(FIRE_SALE_CONFIG)

    if interbank_contagion_config is None:
        interbank_contagion_config = copy.deepcopy(INTERBANK_CONTAGION_CONFIG)

    if module_config is None:
        module_config = {
            "first_round": True,
            "bank_reaction": True,
            "policy_intervention": True,
            "fire_sales": True,
            "interbank_contagion": True,
            "reputation_feedback": False,
        }
    else:
        module_config = copy.deepcopy(module_config)

    if monthly_stress_vectors is not None:
        if len(monthly_stress_vectors) != len(alpha_path):
            raise ValueError(
                "monthly_stress_vectors length must equal alpha_path length."
            )

    first_round_enabled = bool(module_config.get("first_round", True))
    bank_reaction_enabled = bool(module_config.get("bank_reaction", True))
    policy_enabled = bool(module_config.get("policy_intervention", True))
    fire_sales_enabled = bool(module_config.get("fire_sales", True))
    interbank_enabled = bool(module_config.get("interbank_contagion", True))

    cb_support_config = copy.deepcopy(cb_support_config)
    cb_support_config["enabled"] = (
        bool(cb_support_config.get("enabled", True)) and policy_enabled
    )

    fire_sale_config = copy.deepcopy(fire_sale_config)
    fire_sale_config["enabled"] = (
        bool(fire_sale_config.get("enabled", True)) and fire_sales_enabled
    )

    interbank_contagion_config = copy.deepcopy(interbank_contagion_config)
    interbank_contagion_config["enabled"] = (
        bool(interbank_contagion_config.get("enabled", True))
        and interbank_enabled
    )

    current_state = banks.copy()

    current_state["initial_interbank_assets"] = (
        current_state["interbank_assets"].astype(float)
    )

    if exposure_matrix is None:
        exposure_matrix = get_or_create_interbank_exposure_matrix(
            banks=banks,
            path=INTERBANK_EXPOSURE_CONFIG["path"],
            source=INTERBANK_EXPOSURE_CONFIG["source"],
            sheet_name=INTERBANK_EXPOSURE_CONFIG["sheet_name"],
            exposure_scale=INTERBANK_EXPOSURE_CONFIG["network_exposure_scale"],
        )

    current_state["final_unresolved_shortfall"] = False
    current_state["first_final_unresolved_shortfall_month"] = np.nan

    current_state["post_fire_sale_insolvent"] = False
    current_state["terminal_distress_after_fire_sale"] = False

    current_state["post_interbank_insolvent"] = False
    current_state["terminal_distress_after_interbank"] = False
    current_state["first_terminal_distress_after_interbank_month"] = np.nan
    current_state["continues_after_interbank_contagion"] = True
    current_state["carried_forward_after_interbank_terminal_distress"] = False

    market_prices = {
        asset_column: 1.0
        for asset_column in fire_sale_config["asset_columns"]
    }

    monthly_results: list[pd.DataFrame] = []
    market_summaries: list[pd.DataFrame] = []
    contagion_summaries: list[pd.DataFrame] = []

    for month, alpha in enumerate(alpha_path, start=1):

        if monthly_stress_vectors is None:
            stress_vector_for_month = FIRST_ROUND_STRESS_VECTOR
        else:
            stress_vector_for_month = monthly_stress_vectors[month - 1]

        terminal_mask = (
            current_state["terminal_distress_after_interbank"]
            .fillna(False)
            .astype(bool)
        )

        active_banks = current_state.loc[~terminal_mask].copy()
        terminal_banks = current_state.loc[terminal_mask].copy()

        month_parts: list[pd.DataFrame] = []

        if not active_banks.empty:
            if first_round_enabled:
                shocked = apply_one_month_balance_sheet_shock(
                    banks=active_banks,
                    stress_vector=stress_vector_for_month,
                    alpha=alpha,
                    lcr_assumptions=lcr_assumptions,
                    scenario_name=scenario_name,
                    month=month,
                    stress_mode=stress_mode,
                )
            else:
                shocked = active_banks.copy()
                shocked["stress_month"] = month
                shocked["first_round_alpha"] = float(alpha)
                shocked["first_round_scenario"] = scenario_name

                if "cash_shortfall_before_support" not in shocked.columns:
                    shocked["cash_shortfall_before_support"] = 0.0

            shocked = refresh_liquidity_metrics(
                shocked,
                lcr_assumptions,
                "post_first_round",
            )

            if bank_reaction_enabled:
                reacted = apply_asset_sale_reaction(
                    banks=shocked,
                    liquidation_config=liquidation_config,
                    lcr_assumptions=lcr_assumptions,
                )

                reacted = refresh_liquidity_metrics(
                    reacted,
                    lcr_assumptions,
                    "post_asset_sale",
                )
            else:
                reacted = bypass_bank_reaction(
                    banks=shocked,
                    lcr_assumptions=lcr_assumptions,
                    liquidation_config=liquidation_config,
                )

            if policy_enabled:
                supported = apply_cb_liquidity_support(
                    banks=reacted,
                    policy_config=cb_support_config,
                    lcr_assumptions=lcr_assumptions,
                )

                supported = refresh_liquidity_metrics(
                    supported,
                    lcr_assumptions,
                    "post_cb_support",
                )
            else:
                supported = bypass_policy_support(
                    banks=reacted,
                    cb_support_config=cb_support_config,
                    lcr_assumptions=lcr_assumptions,
                )

            if fire_sales_enabled:
                with_fire_sales, market_prices, market_summary = (
                    apply_fire_sale_price_impact(
                        banks=supported,
                        fire_sale_config=fire_sale_config,
                        lcr_assumptions=lcr_assumptions,
                        current_market_prices=market_prices,
                    )
                )

                market_summaries.append(market_summary)
            else:
                with_fire_sales = bypass_fire_sales(
                    banks=supported,
                    fire_sale_config=fire_sale_config,
                    lcr_assumptions=lcr_assumptions,
                )

            newly_final_unresolved = (
                    with_fire_sales["remaining_shortfall_after_cb_support"] > 1e-9
            )

            newly_insolvent_after_fire_sale = (
                with_fire_sales["post_fire_sale_insolvent"]
                .fillna(False)
                .astype(bool)
            )

            newly_terminal_fire_sale = (
                    newly_final_unresolved
                    | newly_insolvent_after_fire_sale
            )

            with_fire_sales["final_unresolved_shortfall"] = (
                    with_fire_sales["final_unresolved_shortfall"]
                    .fillna(False)
                    .astype(bool)
                    | newly_final_unresolved
            )

            with_fire_sales["first_final_unresolved_shortfall_month"] = np.where(
                newly_final_unresolved
                & with_fire_sales["first_final_unresolved_shortfall_month"].isna(),
                month,
                with_fire_sales["first_final_unresolved_shortfall_month"],
            )

            with_fire_sales["terminal_distress_after_fire_sale"] = (
                    with_fire_sales["terminal_distress_after_fire_sale"]
                    .fillna(False)
                    .astype(bool)
                    | newly_terminal_fire_sale
            )

            # Live network: row/col sums track current interbank_assets/liabilities.
            live_exposure_matrix = build_synthetic_interbank_exposure_matrix(
                banks=with_fire_sales[
                    ["bank_id", "interbank_assets", "interbank_liabilities"]
                ],
            )
            with_interbank, contagion_summary = apply_interbank_contagion(
                banks=with_fire_sales,
                contagion_config=interbank_contagion_config,
                lcr_assumptions=lcr_assumptions,
                exposure_matrix=live_exposure_matrix,
            )

            with_interbank = refresh_liquidity_metrics(
                with_interbank,
                lcr_assumptions,
                "post_interbank",
            )

            newly_terminal_after_interbank = (
                with_interbank["terminal_distress_after_interbank"]
                .fillna(False)
                .astype(bool)
            )

            with_interbank["first_terminal_distress_after_interbank_month"] = (
                np.where(
                    newly_terminal_after_interbank
                    & with_interbank[
                        "first_terminal_distress_after_interbank_month"
                    ].isna(),
                    month,
                    with_interbank["first_terminal_distress_after_interbank_month"],
                )
            )

            with_interbank["continues_after_interbank_contagion"] = (
                ~with_interbank["terminal_distress_after_interbank"]
                .fillna(False)
                .astype(bool)
            )

            with_interbank["carried_forward_after_interbank_terminal_distress"] = False

            with_interbank["asset_sale_reaction_success_if_required"] = (
                    with_interbank["requires_asset_sale_reaction"]
                    & with_interbank["asset_sale_reaction_success"]
            )

            with_interbank["no_asset_sale_reaction_needed"] = (
                ~with_interbank["requires_asset_sale_reaction"]
            )

            month_parts.append(with_interbank)
            contagion_summaries.append(contagion_summary)

        if not terminal_banks.empty:
            carried = terminal_banks.copy()

            carried["stress_month"] = month
            carried["first_round_alpha"] = float(alpha)
            carried["first_round_scenario"] = scenario_name

            carried["terminal_distress_after_interbank"] = True
            carried["continues_after_interbank_contagion"] = False
            carried["carried_forward_after_interbank_terminal_distress"] = True

            carried = zero_current_period_flows(
                carried,
                fire_sale_config=fire_sale_config,
            )

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

    results = pd.concat(monthly_results, ignore_index=True)

    market_summary = (
        pd.concat(market_summaries, ignore_index=True)
        if market_summaries
        else pd.DataFrame()
    )

    contagion_summary = (
        pd.concat(contagion_summaries, ignore_index=True)
        if contagion_summaries
        else pd.DataFrame()
    )

    return results, market_summary, contagion_summary


def build_interbank_monthly_summary(results: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly system-level summary for the full engine.

    Important:
        - stressed_LCR is the forward-looking stress metric.
        - post_interbank_LCR is the recovered/post-transmission diagnostic.
        - first-round security valuation losses are included so the loss
          decomposition does not hide the main damage channel.
    """
    summary = results.groupby("stress_month").agg(
        alpha=("first_round_alpha", "first"),
        number_of_banks=("bank_name", "count"),

        banks_requiring_asset_sale_reaction=(
            "requires_asset_sale_reaction",
            "sum",
        ),
        banks_using_cb_support=("policy_support_used", "sum"),
        solvent_before_lolr_banks=("solvent_before_lolr", "sum"),
        illiquid_before_lolr_banks=("illiquid_before_lolr", "sum"),
        eligible_for_lolr_banks=("eligible_for_lolr", "sum"),

        final_unresolved_shortfalls=("final_unresolved_shortfall", "sum"),
        post_fire_sale_insolvent_banks=("post_fire_sale_insolvent", "sum"),
        post_interbank_insolvent_banks=("post_interbank_insolvent", "sum"),
        terminal_distress_after_interbank=(
            "terminal_distress_after_interbank",
            "sum",
        ),
        new_terminal_from_interbank=("new_terminal_from_interbank", "sum"),

        aggregate_cash_shortfall_before_support=(
            "cash_shortfall_before_support",
            "sum",
        ),
        aggregate_cb_support_granted=("cb_support_granted", "sum"),
        aggregate_lcr_support_gap=("lolr_lcr_support_gap", "sum"),
        aggregate_support_used_for_outflows=(
            "cb_support_used_for_outflows",
            "sum",
        ),
        aggregate_support_retained_as_cash_for_lcr=(
            "cb_support_retained_as_cash_for_lcr",
            "sum",
        ),

        aggregate_security_valuation_loss=(
            "total_security_valuation_loss",
            "sum",
        ),
        aggregate_asset_sale_loss=("asset_sale_liquidation_loss", "sum"),
        aggregate_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
        aggregate_interbank_contagion_loss=(
            "interbank_contagion_loss",
            "sum",
        ),

        aggregate_HQLA=("HQLA", "sum"),
        aggregate_NCOF=("NCOF", "sum"),
        aggregate_system_assets=("total_assets", "sum"),
        aggregate_system_equity=("equity", "sum"),

        aggregate_stressed_HQLA=("stressed_HQLA", "sum"),
        aggregate_stressed_NCOF=("stressed_NCOF", "sum"),
        aggregate_stressed_assets=("pre_shock_total_assets", "sum"),

        average_stressed_LCR=("stressed_LCR", "mean"),
        minimum_stressed_LCR=("stressed_LCR", "min"),
        average_post_interbank_LCR=("post_interbank_LCR", "mean"),
        minimum_post_interbank_LCR=("post_interbank_LCR", "min"),
        average_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "mean",
        ),
        minimum_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "min",
        ),

        aggregate_cb_pledged_collateral=("total_cb_pledged_collateral", "sum"),
        aggregate_pre_cb_unencumbered_collateral_capacity=(
            "pre_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        aggregate_post_cb_unencumbered_collateral_capacity=(
            "post_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        aggregate_unencumbered_collateral_capacity=(
            "unencumbered_collateral_capacity",
            "sum",
        ),
    ).reset_index()

    summary["system_stressed_LCR"] = np.where(
        summary["aggregate_stressed_NCOF"] > 0,
        summary["aggregate_stressed_HQLA"] / summary["aggregate_stressed_NCOF"],
        np.nan,
    )

    summary["system_stressed_HQLA_assets_ratio"] = np.where(
        summary["aggregate_stressed_assets"] > 0,
        summary["aggregate_stressed_HQLA"] / summary["aggregate_stressed_assets"],
        np.nan,
    )

    summary["system_post_interbank_LCR"] = np.where(
        summary["aggregate_NCOF"] > 0,
        summary["aggregate_HQLA"] / summary["aggregate_NCOF"],
        np.nan,
    )

    summary["system_post_interbank_HQLA_assets_ratio"] = np.where(
        summary["aggregate_system_assets"] > 0,
        summary["aggregate_HQLA"] / summary["aggregate_system_assets"],
        np.nan,
    )

    # Backward-compatible name used by charts and old notes.
    summary["asset_weighted_stressed_LCR"] = summary["system_stressed_LCR"]

    summary["aggregate_total_modeled_loss"] = (
        summary["aggregate_security_valuation_loss"].fillna(0.0)
        + summary["aggregate_asset_sale_loss"].fillna(0.0)
        + summary["aggregate_fire_sale_mtm_loss"].fillna(0.0)
        + summary["aggregate_interbank_contagion_loss"].fillna(0.0)
    )

    return summary

def build_interbank_worst_case_by_bank(results: pd.DataFrame) -> pd.DataFrame:
    worst_lcr_idx = results.groupby("bank_id")["post_interbank_LCR"].idxmin()

    worst_lcr = results.loc[
        worst_lcr_idx,
        [
            "bank_id",
            "bank_name",
            "stress_month",
            "first_round_alpha",
            "post_interbank_LCR",
        ],
    ].rename(columns={
        "stress_month": "worst_post_interbank_LCR_month",
        "first_round_alpha": "alpha_at_worst_post_interbank_LCR",
        "post_interbank_LCR": "min_post_interbank_LCR",
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

    return (
        results.groupby(["bank_id", "bank_name"]).agg(
            ever_LCR_breach_before_reaction=("LCR_breach", "max"),
            ever_required_asset_sale_reaction=(
                "requires_asset_sale_reaction",
                "max",
            ),
            ever_used_cb_support=("policy_support_used", "max"),
            ever_eligible_for_lolr=("eligible_for_lolr", "max"),
            ever_illiquid_before_lolr=("illiquid_before_lolr", "max"),
            ever_solvent_before_lolr=("solvent_before_lolr", "max"),
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
            ever_terminal_distress_after_interbank=(
                "terminal_distress_after_interbank",
                "max",
            ),
            ever_new_terminal_from_interbank=(
                "new_terminal_from_interbank",
                "max",
            ),
            first_terminal_distress_after_interbank_month=(
                "first_terminal_distress_after_interbank_month",
                "min",
            ),
            max_cash_shortfall_before_support=(
                "cash_shortfall_before_support",
                "max",
            ),
            total_cb_support_granted=("cb_support_granted", "sum"),
            total_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
            total_interbank_contagion_loss=("interbank_contagion_loss", "sum"),
            min_post_interbank_HQLA_assets_ratio=(
                "post_interbank_HQLA_assets_ratio",
                "min",
            ),
            final_post_interbank_LCR=("post_interbank_LCR", "last"),
            final_total_assets=("total_assets", "last"),
            final_equity=("equity", "last"),
            final_cb_liquidity_funding=("cb_liquidity_funding", "last"),
        )
        .reset_index()
        .merge(worst_lcr, on=["bank_id", "bank_name"], how="left")
        .merge(cb_months, on="bank_id", how="left")
        .sort_values("min_post_interbank_LCR")
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic dynamic liquidity stress engine.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=SCENARIO_CONFIG_PATH,
        help="Path to scenario YAML config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    banks = BankingDataLoader(BANKING_INPUT_PATH).load_banks()

    scenario = load_and_validate_scenario_config(args.config)
    stress_mode = get_stress_mode(scenario)

    alpha_path = get_alpha_path_for_currency(
        config=scenario,
        currency="combined",
    )

    if alpha_path is None:
        alpha_path = [FIRST_ROUND_SCENARIO_CONFIG["alpha"]]

    scenario_name = scenario.get(
        "scenario_name",
        FIRST_ROUND_SCENARIO_CONFIG["scenario_name"],
    )

    runtime_config = build_runtime_configs_from_scenario(
        scenario_config=scenario,
        base_lcr_assumptions=LCR_ASSUMPTIONS,
        base_liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
        base_cb_support_config=CB_SUPPORT_CONFIG,
        base_fire_sale_config=FIRE_SALE_CONFIG,
        base_interbank_contagion_config=INTERBANK_CONTAGION_CONFIG,
    )

    results, market_summary, contagion_summary = (
        run_dynamic_interbank_contagion_path(
            banks=banks,
            alpha_path=alpha_path,
            scenario_name=scenario_name,
            stress_mode=stress_mode,
            lcr_assumptions=runtime_config["lcr"],
            liquidation_config=runtime_config["liquidation"],
            cb_support_config=runtime_config["cb"],
            fire_sale_config=runtime_config["fire"],
            interbank_contagion_config=runtime_config["interbank"],
            module_config=runtime_config["modules"],
        )
    )

    monthly_summary = build_interbank_monthly_summary(results)
    worst_case_by_bank = build_interbank_worst_case_by_bank(results)

    results.to_csv(
        TABLES_DIR / "interbank_contagion_bank_results.csv",
        index=False,
    )

    monthly_summary.to_csv(
        TABLES_DIR / "interbank_contagion_monthly_summary.csv",
        index=False,
    )

    worst_case_by_bank.to_csv(
        TABLES_DIR / "interbank_contagion_worst_case_by_bank.csv",
        index=False,
    )

    market_summary.to_csv(
        TABLES_DIR / "interbank_contagion_market_summary.csv",
        index=False,
    )

    contagion_summary.to_csv(
        TABLES_DIR / "interbank_contagion_network_summary.csv",
        index=False,
    )

    print("Interbank contagion bank results saved.")
    print("Interbank contagion monthly summary saved.")
    print("Interbank contagion worst-case bank summary saved.")
    print("Interbank contagion market summary saved.")
    print("Interbank contagion network summary saved.")

    print(monthly_summary.to_string(index=False))

    print(worst_case_by_bank[[
        "bank_name",
        "min_post_interbank_LCR",
        "worst_post_interbank_LCR_month",
        "ever_post_fire_sale_insolvent",
        "ever_post_interbank_insolvent",
        "ever_new_terminal_from_interbank",
        "ever_terminal_distress_after_interbank",
        "total_interbank_contagion_loss",
        "final_equity",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()