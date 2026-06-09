"""
Dynamic first-round balance-sheet shock module.

Correct interpretation
----------------------
For each month t:

    balance sheet at t
        -> apply alpha_t shock to balance-sheet components
        -> obtain balance sheet at t+1
        -> recompute HQLA, NCOF, LCR, and HQLA/assets

Important
---------
This module does not add emergency funding to the balance sheet. If cash is
insufficient to meet planned outflows, the unmet amount is recorded as a
liquidity shortfall, but it is not double-counted as future unstable funding.

Bank reaction, asset liquidation, fire sales, and policy support come later.
"""

import numpy as np
import pandas as pd

from liquidity_stress.stress_vectors import (
    apply_alpha_to_stress_vector,
    get_effective_weight,
)

VALID_STRESS_MODES = {
    "full_systemic",
    "outflow_only",
}


def filter_stress_vector_for_mode(
    stress_vector: pd.DataFrame,
    stress_mode: str,
) -> pd.DataFrame:
    """
    Filter the stress vector according to the scenario stress mode.

    full_systemic:
        keep all stress components.

    outflow_only:
        keep only liability runoff shocks.
        This means no exogenous security haircuts, no inflow-reduction shock,
        no FX pressure shock, and no FX liquid-resource haircut.

        Asset sales and fire-sale feedback may still occur endogenously if
        banks face cash shortfalls after outflows.
    """
    if stress_mode not in VALID_STRESS_MODES:
        raise ValueError(
            f"Unknown stress_mode={stress_mode}. "
            f"Allowed values: {sorted(VALID_STRESS_MODES)}"
        )

    out = stress_vector.copy()

    if stress_mode == "full_systemic":
        return out

    if "shock_type" not in out.columns:
        raise ValueError("stress_vector must contain a 'shock_type' column.")

    keep_mask = out["shock_type"].eq("runoff")

    out.loc[~keep_mask, "base_weight"] = 0.0

    if "cap" in out.columns:
        out.loc[~keep_mask, "cap"] = 0.0

    return out

def compute_lcr_from_balance_sheet(
    banks: pd.DataFrame,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Recompute HQLA, NCOF, LCR, and HQLA/assets from current balance-sheet items.
    """
    out = banks.copy()

    out["total_securities"] = (
        out["government_securities"]
        + out["bank_securities"]
        + out["corporate_securities"]
    )

    out["deposits"] = (
        out["retail_deposits"]
        + out["corporate_deposits"]
        + out["other_deposits"]
    )

    cash_hqla_weight = float(
        lcr_assumptions.get("cash_hqla_weight", 1.00)
    )
    government_hqla_weight = float(
        lcr_assumptions.get("government_securities_hqla_weight", 0.90)
    )
    bank_hqla_weight = float(
        lcr_assumptions.get("bank_securities_hqla_weight", 0.35)
    )
    corporate_hqla_weight = float(
        lcr_assumptions.get("corporate_securities_hqla_weight", 0.10)
    )

    out["HQLA"] = (
            cash_hqla_weight * out["cash_reserves"]
            + government_hqla_weight * out["government_securities"]
            + bank_hqla_weight * out["bank_securities"]
            + corporate_hqla_weight * out["corporate_securities"]
    )

    out["HQLA"] = np.maximum(out["HQLA"], 0.0)

    out["HQLA_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["HQLA"] / out["total_assets"],
        0.0,
    )

    if "wholesale_funding_30d_share" in lcr_assumptions:
        wholesale_base = (
            lcr_assumptions["wholesale_funding_30d_share"]
            * out["wholesale_funding"]
        )
    else:
        wholesale_base = out["wholesale_funding"]

    if "interbank_liability_30d_share" in lcr_assumptions:
        interbank_liability_base = (
            lcr_assumptions["interbank_liability_30d_share"]
            * out["interbank_liabilities"]
        )
    else:
        interbank_liability_base = out["interbank_liabilities"]


    if "cb_liquidity_funding" in out.columns:
        cb_liquidity_funding_base = out["cb_liquidity_funding"]
    else:
        cb_liquidity_funding_base = 0.0


    out["wholesale_funding_30d"] = wholesale_base
    out["interbank_liabilities_30d"] = interbank_liability_base
    out["cb_liquidity_funding_30d"] = cb_liquidity_funding_base

    out["gross_30d_outflows"] = (
            lcr_assumptions["retail_deposit_runoff"] * out["retail_deposits"]
            + lcr_assumptions["corporate_deposit_runoff"] * out["corporate_deposits"]
            + lcr_assumptions["other_deposit_runoff"] * out["other_deposits"]
            + lcr_assumptions["wholesale_funding_runoff"] * wholesale_base
            + lcr_assumptions["interbank_liability_runoff"] * interbank_liability_base
            + lcr_assumptions.get("cb_liquidity_funding_runoff", 0.0)
            * cb_liquidity_funding_base
    )

    out["recognized_30d_inflows"] = (
        lcr_assumptions["loan_maturing_30d_share"]
        * lcr_assumptions["loan_inflow_recognition_rate"]
        * out["loans"]
        + lcr_assumptions["interbank_asset_inflow_rate"]
        * out["interbank_assets"]
    )

    out["capped_30d_inflows"] = np.minimum(
        out["recognized_30d_inflows"],
        lcr_assumptions["inflow_cap_share_of_outflows"]
        * out["gross_30d_outflows"],
    )

    # ----------------------------------------------------------
    # Synthetic NCOF calibration
    # ----------------------------------------------------------
    # For synthetic/demo data, bank-level NCOF is calibrated at generation time
    # so that baseline LCRs are plausible. Preserve that bank-specific NCOF scale
    # whenever LCR is recomputed from the evolving balance sheet.
    #
    # Without this, the first stress month recomputes NCOF from raw LCR
    # assumptions and mechanically restores unrealistically high LCRs.
    if "synthetic_ncof_calibration_scale" in out.columns:
        calibration_scale = pd.to_numeric(
            out["synthetic_ncof_calibration_scale"],
            errors="coerce",
        ).fillna(1.0)

        calibration_scale = np.maximum(calibration_scale, 1e-9)

        out["gross_30d_outflows"] = (
                out["gross_30d_outflows"] * calibration_scale
        )

        out["recognized_30d_inflows"] = (
                out["recognized_30d_inflows"] * calibration_scale
        )

        out["capped_30d_inflows"] = np.minimum(
            out["recognized_30d_inflows"],
            lcr_assumptions["inflow_cap_share_of_outflows"]
            * out["gross_30d_outflows"],
        )

    out["NCOF"] = out["gross_30d_outflows"] - out["capped_30d_inflows"]

    out["NCOF"] = np.maximum(
        out["NCOF"],
        lcr_assumptions["minimum_ncof"],
    )

    out["LCR"] = out["HQLA"] / out["NCOF"]

    return out


def apply_one_month_balance_sheet_shock(
    banks: pd.DataFrame,
    stress_vector: pd.DataFrame,
    alpha: float,
    lcr_assumptions: dict,
    scenario_name: str = "dynamic_first_round_stress",
    month: int | None = None,
    stress_mode: str = "full_systemic",
) -> pd.DataFrame:
    """
    Apply one month of balance-sheet stress and recompute liquidity metrics.

    The shock affects the balance sheet itself. LCR is then recomputed from
    the new state.
    """
    stress_vector_for_mode = filter_stress_vector_for_mode(
        stress_vector=stress_vector,
        stress_mode=stress_mode,
    )

    effective_vector = apply_alpha_to_stress_vector(
        stress_vector=stress_vector_for_mode,
        alpha=alpha,
    )

    pre = compute_lcr_from_balance_sheet(
        banks=banks,
        lcr_assumptions=lcr_assumptions,
    )

    out = pre.copy()
    out["stress_mode"] = stress_mode

    # Preserve pre-shock diagnostics.
    out["pre_shock_total_assets"] = out["total_assets"]
    out["pre_shock_total_liabilities"] = out["total_liabilities"]
    out["pre_shock_equity"] = out["equity"]
    out["pre_shock_HQLA"] = out["HQLA"]
    out["pre_shock_NCOF"] = out["NCOF"]
    out["pre_shock_LCR"] = out["LCR"]
    out["pre_shock_HQLA_assets_ratio"] = out["HQLA_assets_ratio"]

    if "baseline_LCR" not in out.columns:
        out["baseline_LCR"] = out["pre_shock_LCR"]

    if "baseline_HQLA_assets_ratio" not in out.columns:
        out["baseline_HQLA_assets_ratio"] = out["pre_shock_HQLA_assets_ratio"]

    if "cumulative_cash_shortfall_before_support" not in out.columns:
        out["cumulative_cash_shortfall_before_support"] = 0.0

    if "ever_LCR_breach" not in out.columns:
        out["ever_LCR_breach"] = False

    if "ever_cash_shortfall" not in out.columns:
        out["ever_cash_shortfall"] = False

    if "ever_stress_event" not in out.columns:
        out["ever_stress_event"] = False

    # ----------------------------------------------------------
    # Effective monthly shock rates
    # ----------------------------------------------------------
    retail_runoff = get_effective_weight(effective_vector, "retail_deposits")
    corporate_runoff = get_effective_weight(effective_vector, "corporate_deposits")
    other_runoff = get_effective_weight(effective_vector, "other_deposits")
    wholesale_runoff = get_effective_weight(effective_vector, "wholesale_funding")
    interbank_liability_runoff = get_effective_weight(
        effective_vector,
        "interbank_liabilities",
    )

    government_haircut = get_effective_weight(
        effective_vector,
        "government_securities",
    )
    bank_security_haircut = get_effective_weight(
        effective_vector,
        "bank_securities",
    )
    corporate_security_haircut = get_effective_weight(
        effective_vector,
        "corporate_securities",
    )

    loan_inflow_reduction = get_effective_weight(effective_vector, "loan_inflows")
    interbank_inflow_reduction = get_effective_weight(
        effective_vector,
        "interbank_asset_inflows",
    )

    fx_liability_pressure = get_effective_weight(
        effective_vector,
        "fx_liability_pressure",
    )
    fx_asset_haircut = get_effective_weight(effective_vector, "fx_asset_haircut")

    # ----------------------------------------------------------
    # Planned liability outflows
    # ----------------------------------------------------------
    if "wholesale_funding_30d_share" in lcr_assumptions:
        wholesale_base = (
            lcr_assumptions["wholesale_funding_30d_share"]
            * out["wholesale_funding"]
        )
    else:
        wholesale_base = out["wholesale_funding"]

    if "interbank_liability_30d_share" in lcr_assumptions:
        interbank_liability_base = (
            lcr_assumptions["interbank_liability_30d_share"]
            * out["interbank_liabilities"]
        )
    else:
        interbank_liability_base = out["interbank_liabilities"]



    # ----------------------------------------------------------
    # Stress-measure LCR for reporting
    # ----------------------------------------------------------
    # This is the actual stressed LCR for the current month.
    # It is computed from the pre-shock balance sheet under current-month
    # stressed assumptions, before mechanical balance-sheet shrinkage.

    stress_retail_runoff = min(
        lcr_assumptions["retail_deposit_runoff"] + retail_runoff,
        1.0,
    )

    stress_corporate_runoff = min(
        lcr_assumptions["corporate_deposit_runoff"] + corporate_runoff,
        1.0,
    )

    stress_other_runoff = min(
        lcr_assumptions["other_deposit_runoff"] + other_runoff,
        1.0,
    )

    stress_wholesale_runoff = min(
        lcr_assumptions["wholesale_funding_runoff"] + wholesale_runoff,
        1.0,
    )

    stress_interbank_liability_runoff = min(
        lcr_assumptions["interbank_liability_runoff"] + interbank_liability_runoff,
        1.0,
    )

    stress_cash_hqla_weight = float(
        lcr_assumptions.get("cash_hqla_weight", 1.00)
    )

    stress_government_hqla_weight = float(
        lcr_assumptions.get("government_securities_hqla_weight", 0.90)
    )

    stress_bank_hqla_weight = float(
        lcr_assumptions.get("bank_securities_hqla_weight", 0.35)
    )

    stress_corporate_hqla_weight = float(
        lcr_assumptions.get("corporate_securities_hqla_weight", 0.10)
    )

    stress_measure_HQLA = (
            stress_cash_hqla_weight * out["cash_reserves"]
            + stress_government_hqla_weight
            * (1.0 - government_haircut)
            * out["government_securities"]
            + stress_bank_hqla_weight
            * (1.0 - bank_security_haircut)
            * out["bank_securities"]
            + stress_corporate_hqla_weight
            * (1.0 - corporate_security_haircut)
            * out["corporate_securities"]
    )

    stress_measure_HQLA = np.maximum(stress_measure_HQLA, 0.0)

    stress_measure_gross_outflows = (
        stress_retail_runoff * out["retail_deposits"]
        + stress_corporate_runoff * out["corporate_deposits"]
        + stress_other_runoff * out["other_deposits"]
        + stress_wholesale_runoff * wholesale_base
        + stress_interbank_liability_runoff * interbank_liability_base
    )

    stress_measure_loan_inflow_rate = (
        lcr_assumptions["loan_maturing_30d_share"]
        * lcr_assumptions["loan_inflow_recognition_rate"]
        * (1.0 - loan_inflow_reduction)
    )

    stress_measure_interbank_inflow_rate = (
        lcr_assumptions["interbank_asset_inflow_rate"]
        * (1.0 - interbank_inflow_reduction)
    )

    stress_measure_recognized_inflows = (
        stress_measure_loan_inflow_rate * out["loans"]
        + stress_measure_interbank_inflow_rate * out["interbank_assets"]
    )

    stress_measure_capped_inflows = np.minimum(
        stress_measure_recognized_inflows,
        lcr_assumptions["inflow_cap_share_of_outflows"]
        * stress_measure_gross_outflows,
    )

    # Preserve synthetic NCOF calibration in the stressed-measure denominator too.
    # This keeps stressed_LCR comparable to the calibrated baseline_LCR.
    if "synthetic_ncof_calibration_scale" in out.columns:
        calibration_scale = pd.to_numeric(
            out["synthetic_ncof_calibration_scale"],
            errors="coerce",
        ).fillna(1.0)

        calibration_scale = np.maximum(calibration_scale, 1e-9)

        stress_measure_gross_outflows = (
                stress_measure_gross_outflows * calibration_scale
        )

        stress_measure_recognized_inflows = (
                stress_measure_recognized_inflows * calibration_scale
        )

        stress_measure_capped_inflows = np.minimum(
            stress_measure_recognized_inflows,
            lcr_assumptions["inflow_cap_share_of_outflows"]
            * stress_measure_gross_outflows,
        )

    stress_measure_NCOF = (
            stress_measure_gross_outflows
            - stress_measure_capped_inflows
    )

    stress_measure_NCOF = np.maximum(
        stress_measure_NCOF,
        lcr_assumptions["minimum_ncof"],
    )

    stress_measure_LCR = stress_measure_HQLA / stress_measure_NCOF

    out["planned_retail_deposit_outflow"] = retail_runoff * out["retail_deposits"]
    out["planned_corporate_deposit_outflow"] = corporate_runoff * out["corporate_deposits"]
    out["planned_other_deposit_outflow"] = other_runoff * out["other_deposits"]
    out["planned_wholesale_funding_outflow"] = wholesale_runoff * wholesale_base
    out["planned_interbank_liability_outflow"] = (
        interbank_liability_runoff * interbank_liability_base
    )

    out["planned_total_liability_outflow"] = (
        out["planned_retail_deposit_outflow"]
        + out["planned_corporate_deposit_outflow"]
        + out["planned_other_deposit_outflow"]
        + out["planned_wholesale_funding_outflow"]
        + out["planned_interbank_liability_outflow"]
    )

    # ----------------------------------------------------------
    # Realized asset inflows
    # ----------------------------------------------------------
    out["realized_loan_inflows"] = (
        lcr_assumptions["loan_maturing_30d_share"]
        * out["loans"]
        * (1.0 - loan_inflow_reduction)
    )

    # The 0.75 LCR figure must NOT be re-realized as cash each month, or the
    # interbank network disappears within 2-3 months and contagion cannot fire.
    interbank_monthly_maturing_rate = lcr_assumptions.get(
        "interbank_asset_monthly_maturing_rate", 0.15
    )
    out["realized_interbank_asset_inflows"] = (
            interbank_monthly_maturing_rate
            * out["interbank_assets"]
            * (1.0 - interbank_inflow_reduction)
    )

    out["realized_loan_inflows"] = np.minimum(
        out["realized_loan_inflows"],
        out["loans"],
    )

    # Preserve a stable network core: interbank assets cannot be converted below it.
    network_core_share = lcr_assumptions.get("interbank_network_core_share", 0.60)
    if "initial_interbank_assets" in out.columns:
        interbank_core_floor = (
                network_core_share * out["initial_interbank_assets"].astype(float)
        )
    else:
        interbank_core_floor = network_core_share * out["interbank_assets"].astype(float)

    max_convertible_interbank = np.maximum(
        out["interbank_assets"].astype(float) - interbank_core_floor, 0.0
    )
    out["realized_interbank_asset_inflows"] = np.minimum(
        out["realized_interbank_asset_inflows"], max_convertible_interbank
    )

    out["total_realized_asset_inflows"] = (
        out["realized_loan_inflows"]
        + out["realized_interbank_asset_inflows"]
    )

    # Asset inflows convert loans/interbank assets into cash.
    out["loans"] = out["loans"] - out["realized_loan_inflows"]
    out["interbank_assets"] = (
        out["interbank_assets"] - out["realized_interbank_asset_inflows"]
    )

    cash_available = out["cash_reserves"] + out["total_realized_asset_inflows"]

    # ----------------------------------------------------------
    # Pay only outflows that can actually be paid from cash/inflows
    # ----------------------------------------------------------
    out["paid_total_liability_outflow"] = np.minimum(
        out["planned_total_liability_outflow"],
        cash_available,
    )

    out["cash_shortfall_before_support"] = np.maximum(
        out["planned_total_liability_outflow"] - cash_available,
        0.0,
    )

    payment_ratio = np.where(
        out["planned_total_liability_outflow"] > 0,
        out["paid_total_liability_outflow"]
        / out["planned_total_liability_outflow"],
        0.0,
    )

    out["paid_retail_deposit_outflow"] = (
        out["planned_retail_deposit_outflow"] * payment_ratio
    )
    out["paid_corporate_deposit_outflow"] = (
        out["planned_corporate_deposit_outflow"] * payment_ratio
    )
    out["paid_other_deposit_outflow"] = (
        out["planned_other_deposit_outflow"] * payment_ratio
    )
    out["paid_wholesale_funding_outflow"] = (
        out["planned_wholesale_funding_outflow"] * payment_ratio
    )
    out["paid_interbank_liability_outflow"] = (
        out["planned_interbank_liability_outflow"] * payment_ratio
    )

    out["unpaid_total_liability_outflow"] = (
        out["planned_total_liability_outflow"]
        - out["paid_total_liability_outflow"]
    )

    # Reduce only the liabilities that were actually paid.
    out["retail_deposits"] = np.maximum(
        out["retail_deposits"] - out["paid_retail_deposit_outflow"],
        0.0,
    )

    out["corporate_deposits"] = np.maximum(
        out["corporate_deposits"] - out["paid_corporate_deposit_outflow"],
        0.0,
    )

    out["other_deposits"] = np.maximum(
        out["other_deposits"] - out["paid_other_deposit_outflow"],
        0.0,
    )

    out["wholesale_funding"] = np.maximum(
        out["wholesale_funding"] - out["paid_wholesale_funding_outflow"],
        0.0,
    )

    out["interbank_liabilities"] = np.maximum(
        out["interbank_liabilities"] - out["paid_interbank_liability_outflow"],
        0.0,
    )

    out["cash_reserves"] = np.maximum(
        cash_available - out["paid_total_liability_outflow"],
        0.0,
    )

    # ----------------------------------------------------------
    # Securities valuation haircuts
    # ----------------------------------------------------------
    out["government_security_loss"] = government_haircut * out["government_securities"]
    out["bank_security_loss"] = bank_security_haircut * out["bank_securities"]
    out["corporate_security_loss"] = corporate_security_haircut * out["corporate_securities"]

    out["total_security_valuation_loss"] = (
        out["government_security_loss"]
        + out["bank_security_loss"]
        + out["corporate_security_loss"]
    )

    out["government_securities"] = np.maximum(
        out["government_securities"] - out["government_security_loss"],
        0.0,
    )

    out["bank_securities"] = np.maximum(
        out["bank_securities"] - out["bank_security_loss"],
        0.0,
    )

    out["corporate_securities"] = np.maximum(
        out["corporate_securities"] - out["corporate_security_loss"],
        0.0,
    )

    # ----------------------------------------------------------
    # Rebuild balance-sheet aggregates
    # ----------------------------------------------------------
    out["total_securities"] = (
        out["government_securities"]
        + out["bank_securities"]
        + out["corporate_securities"]
    )

    out["deposits"] = (
        out["retail_deposits"]
        + out["corporate_deposits"]
        + out["other_deposits"]
    )

    out["total_assets"] = (
        out["cash_reserves"]
        + out["loans"]
        + out["total_securities"]
        + out["interbank_assets"]
    )

    cb_liquidity_funding = (
        out["cb_liquidity_funding"]
        if "cb_liquidity_funding" in out.columns
        else 0.0
    )

    out["total_liabilities"] = (
            out["deposits"]
            + out["wholesale_funding"]
            + out["interbank_liabilities"]
            + cb_liquidity_funding
    )

    out["equity"] = out["total_assets"] - out["total_liabilities"]

    out["equity_change"] = out["equity"] - out["pre_shock_equity"]

    out["equity_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["equity"] / out["total_assets"],
        0.0,
    )

    # Keep currency structure internally consistent, approximately.
    pre_fx_asset_share = np.where(
        out["pre_shock_total_assets"] > 0,
        out["FX_assets"] / out["pre_shock_total_assets"],
        0.0,
    )

    pre_fx_liability_share = np.where(
        out["pre_shock_total_liabilities"] > 0,
        out["FX_liabilities"] / out["pre_shock_total_liabilities"],
        0.0,
    )

    out["FX_assets"] = pre_fx_asset_share * out["total_assets"]
    out["AMD_assets"] = out["total_assets"] - out["FX_assets"]

    out["FX_liabilities"] = pre_fx_liability_share * out["total_liabilities"]
    out["AMD_liabilities"] = out["total_liabilities"] - out["FX_liabilities"]

    out["FX_asset_share"] = np.where(
        out["total_assets"] > 0,
        out["FX_assets"] / out["total_assets"],
        0.0,
    )

    out["FX_liability_share"] = np.where(
        out["total_liabilities"] > 0,
        out["FX_liabilities"] / out["total_liabilities"],
        0.0,
    )

    out["net_fx_gap"] = out["FX_assets"] - out["FX_liabilities"]

    out["net_fx_gap_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["net_fx_gap"] / out["total_assets"],
        0.0,
    )

    # ----------------------------------------------------------
    # Recompute LCR from post-shock balance sheet
    # ----------------------------------------------------------
    post = compute_lcr_from_balance_sheet(
        banks=out,
        lcr_assumptions=lcr_assumptions,
    )

    out["HQLA"] = post["HQLA"]
    out["HQLA_assets_ratio"] = post["HQLA_assets_ratio"]
    out["gross_30d_outflows"] = post["gross_30d_outflows"]
    out["recognized_30d_inflows"] = post["recognized_30d_inflows"]
    out["capped_30d_inflows"] = post["capped_30d_inflows"]
    out["NCOF"] = post["NCOF"]
    out["LCR"] = post["LCR"]

    # Post-transition balance-sheet LCR.
    # This is diagnostic only. It may rise if liabilities shrink faster than HQLA.
    out["post_state_HQLA"] = out["HQLA"]
    out["post_state_HQLA_assets_ratio"] = out["HQLA_assets_ratio"]
    out["post_state_NCOF"] = out["NCOF"]
    out["post_state_LCR"] = out["LCR"]

    # Main stress-result metrics for reporting.
    out["stressed_HQLA"] = stress_measure_HQLA
    out["stressed_HQLA_assets_ratio"] = np.where(
        out["pre_shock_total_assets"] > 0,
        stress_measure_HQLA / out["pre_shock_total_assets"],
        0.0,
    )
    out["stressed_NCOF"] = stress_measure_NCOF
    out["stressed_LCR"] = stress_measure_LCR

    out["LCR_change"] = out["stressed_LCR"] - out["pre_shock_LCR"]
    out["LCR_pct_change"] = np.where(
        out["pre_shock_LCR"] != 0,
        out["LCR_change"] / out["pre_shock_LCR"],
        0.0,
    )

    # Keep the Basel 100% shortfall for diagnostics. This is not the active
    # Armenian regulatory requirement.
    out["liquidity_shortfall_to_LCR_100"] = np.maximum(
        out["stressed_NCOF"] - out["stressed_HQLA"],
        0.0,
    )

    if "active_lcr_requirement" in out.columns:
        active_lcr_requirement = pd.to_numeric(
            out["active_lcr_requirement"],
            errors="coerce",
        ).fillna(1.20)
    else:
        active_lcr_requirement = pd.Series(1.20, index=out.index)

    out["active_lcr_requirement"] = active_lcr_requirement

    out["liquidity_shortfall_to_active_LCR"] = np.maximum(
        active_lcr_requirement * out["stressed_NCOF"] - out["stressed_HQLA"],
        0.0,
    )

    out["liquidity_shortfall_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["liquidity_shortfall_to_active_LCR"] / out["total_assets"],
        0.0,
    )

    out["LCR_breach"] = out["stressed_LCR"] < active_lcr_requirement

    out["cash_shortfall_flag"] = out["cash_shortfall_before_support"] > 0.0

    out["current_stress_event"] = (
        out["LCR_breach"]
        | out["cash_shortfall_flag"]
    )

    out["cumulative_cash_shortfall_before_support"] = (
        out["cumulative_cash_shortfall_before_support"]
        + out["cash_shortfall_before_support"]
    )

    out["ever_LCR_breach"] = (
        out["ever_LCR_breach"].astype(bool)
        | out["LCR_breach"].astype(bool)
    )

    out["ever_cash_shortfall"] = (
        out["ever_cash_shortfall"].astype(bool)
        | out["cash_shortfall_flag"].astype(bool)
    )

    out["ever_stress_event"] = (
        out["ever_stress_event"].astype(bool)
        | out["current_stress_event"].astype(bool)
    )

    # ----------------------------------------------------------
    # FX stress diagnostics only
    # ----------------------------------------------------------
    out["stressed_FX_assets_proxy"] = out["FX_assets"] * (1.0 - fx_asset_haircut)

    out["additional_FX_liability_pressure"] = (
        out["FX_liabilities"] * fx_liability_pressure
    )

    out["stressed_net_fx_gap"] = (
        out["stressed_FX_assets_proxy"]
        - out["FX_liabilities"]
        - out["additional_FX_liability_pressure"]
    )

    out["stressed_net_fx_gap_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["stressed_net_fx_gap"] / out["total_assets"],
        0.0,
    )

    # ----------------------------------------------------------
    # Metadata
    # ----------------------------------------------------------
    out["first_round_alpha"] = float(alpha)
    out["first_round_scenario"] = scenario_name

    if month is not None:
        out["stress_month"] = int(month)

    return out


def apply_first_round_lcr_shock(
    banks: pd.DataFrame,
    stress_vector: pd.DataFrame,
    alpha: float,
    lcr_assumptions: dict,
    scenario_name: str = "dynamic_first_round_stress",
    month: int | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for the one-month dynamic shock.
    """
    return apply_one_month_balance_sheet_shock(
        banks=banks,
        stress_vector=stress_vector,
        alpha=alpha,
        lcr_assumptions=lcr_assumptions,
        scenario_name=scenario_name,
        month=month,
    )


def summarize_first_round_shock(post_shock_banks: pd.DataFrame) -> pd.DataFrame:
    """
    Create a compact system-level summary for one stress month.
    """
    total_assets = post_shock_banks["total_assets"].sum()
    pre_assets = post_shock_banks["pre_shock_total_assets"].sum()

    return pd.DataFrame({
        "metric": [
            "alpha",
            "number_of_banks",
            "number_of_LCR_breaches",
            "number_of_cash_shortfalls",
            "number_of_current_stress_events",
            "number_of_ever_LCR_breaches",
            "number_of_ever_cash_shortfalls",
            "number_of_ever_stress_events",
            "share_of_assets_in_LCR_breach",
            "system_pre_shock_HQLA_assets",
            "system_post_shock_HQLA_assets",
            "system_pre_shock_LCR_weighted",
            "system_post_shock_LCR_weighted",
            "minimum_post_shock_LCR",
            "minimum_post_shock_HQLA_assets_ratio",
            "aggregate_liquidity_shortfall_to_LCR_100",
            "aggregate_cash_shortfall_before_support",
            "aggregate_cumulative_cash_shortfall",
            "aggregate_unpaid_liability_outflow",
            "aggregate_security_valuation_loss",
            "aggregate_additional_FX_pressure",
        ],
        "value": [
            post_shock_banks["first_round_alpha"].iloc[0],
            len(post_shock_banks),
            int(post_shock_banks["LCR_breach"].sum()),
            int(post_shock_banks["cash_shortfall_flag"].sum()),
            int(post_shock_banks["current_stress_event"].sum()),
            int(post_shock_banks["ever_LCR_breach"].sum()),
            int(post_shock_banks["ever_cash_shortfall"].sum()),
            int(post_shock_banks["ever_stress_event"].sum()),
            (
                post_shock_banks.loc[
                    post_shock_banks["LCR_breach"],
                    "total_assets",
                ].sum()
                / total_assets
                if total_assets > 0 else 0.0
            ),
            (
                post_shock_banks["pre_shock_HQLA"].sum()
                / pre_assets
                if pre_assets > 0 else 0.0
            ),
            (
                post_shock_banks["HQLA"].sum()
                / total_assets
                if total_assets > 0 else 0.0
            ),
            np.average(
                post_shock_banks["pre_shock_LCR"],
                weights=post_shock_banks["pre_shock_total_assets"],
            ),
            np.average(
                post_shock_banks["stressed_LCR"],
                weights=post_shock_banks["total_assets"],
            ),
            post_shock_banks["stressed_LCR"].min(),
            post_shock_banks["stressed_HQLA_assets_ratio"].min(),
            post_shock_banks["liquidity_shortfall_to_LCR_100"].sum(),
            post_shock_banks["cash_shortfall_before_support"].sum(),
            post_shock_banks["cumulative_cash_shortfall_before_support"].sum(),
            post_shock_banks["unpaid_total_liability_outflow"].sum(),
            post_shock_banks["total_security_valuation_loss"].sum(),
            post_shock_banks["additional_FX_liability_pressure"].sum(),
        ],
    })


def run_first_round_alpha_path(
    banks: pd.DataFrame,
    stress_vector: pd.DataFrame,
    alpha_path: list[float],
    lcr_assumptions: dict,
    scenario_name: str = "dynamic_first_round_stress",
) -> pd.DataFrame:
    """
    Run recursive monthly first-round shocks over an alpha path.

    This is a no-reaction dynamic path.

    If a bank experiences a cash shortfall, it is marked as having an
    unresolved shortfall under the no-reaction path. It is then carried forward
    for reporting until a separate reaction/support module is added.

    This does not mean the bank has failed. It means the model has not yet
    resolved the shortfall through asset sales, central bank funding, or other
    liquidity actions.
    """
    current_state = banks.copy()

    current_state["unresolved_shortfall_no_reaction"] = False
    current_state["first_unresolved_shortfall_month"] = np.nan
    current_state["continues_without_reaction"] = True

    monthly_results = []

    for month, alpha in enumerate(alpha_path, start=1):
        if current_state.empty:
            raise ValueError(
                f"No banks available at month {month}. "
                "Current state became empty unexpectedly."
            )

        unresolved_mask = (
            current_state["unresolved_shortfall_no_reaction"]
            .fillna(False)
            .astype(bool)
        )

        active_banks = current_state.loc[~unresolved_mask].copy()
        unresolved_banks = current_state.loc[unresolved_mask].copy()

        month_parts = []

        # ------------------------------------------------------
        # Shock banks that still continue without reaction
        # ------------------------------------------------------
        if not active_banks.empty:
            shocked_active = apply_one_month_balance_sheet_shock(
                banks=active_banks,
                stress_vector=stress_vector,
                alpha=alpha,
                lcr_assumptions=lcr_assumptions,
                scenario_name=scenario_name,
                month=month,
            )

            newly_unresolved = (
                shocked_active["cash_shortfall_before_support"] > 0.0
            )

            shocked_active["unresolved_shortfall_no_reaction"] = (
                shocked_active["unresolved_shortfall_no_reaction"]
                .fillna(False)
                .astype(bool)
                | newly_unresolved
            )

            shocked_active["first_unresolved_shortfall_month"] = np.where(
                newly_unresolved
                & shocked_active["first_unresolved_shortfall_month"].isna(),
                month,
                shocked_active["first_unresolved_shortfall_month"],
            )

            shocked_active["continues_without_reaction"] = (
                ~shocked_active["unresolved_shortfall_no_reaction"].astype(bool)
            )

            shocked_active["carried_forward_after_unresolved_shortfall"] = False

            month_parts.append(shocked_active)

        # ------------------------------------------------------
        # Carry already-unresolved banks forward without re-shocking
        # ------------------------------------------------------
        if not unresolved_banks.empty:
            carried_unresolved = unresolved_banks.copy()

            carried_unresolved["stress_month"] = month
            carried_unresolved["first_round_alpha"] = float(alpha)
            carried_unresolved["first_round_scenario"] = scenario_name

            carried_unresolved["unresolved_shortfall_no_reaction"] = True
            carried_unresolved["continues_without_reaction"] = False
            carried_unresolved["carried_forward_after_unresolved_shortfall"] = True

            current_period_flow_columns = [
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
            ]

            for column in current_period_flow_columns:
                if column in carried_unresolved.columns:
                    carried_unresolved[column] = 0.0

            # No new current-period shortfall is generated because the bank
            # is not re-shocked. The previous unresolved shortfall remains.
            carried_unresolved["cash_shortfall_flag"] = False
            carried_unresolved["current_stress_event"] = True
            carried_unresolved["ever_stress_event"] = True
            carried_unresolved["ever_cash_shortfall"] = True

            month_parts.append(carried_unresolved)

        if not month_parts:
            raise ValueError(
                f"No month results produced at month {month}. "
                "Check unresolved-shortfall state logic."
            )

        month_result = pd.concat(month_parts, ignore_index=True)

        monthly_results.append(month_result.copy())
        current_state = month_result.copy()

    return pd.concat(monthly_results, ignore_index=True)