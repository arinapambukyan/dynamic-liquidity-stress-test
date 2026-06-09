"""
Central bank liquidity support module.

This module models policy liquidity support after bank-level asset sales:

    cash_shortfall_after_asset_sales
        -> calculate eligible collateral capacity
        -> grant central bank liquidity support
        -> use support to pay remaining unpaid liability outflows
        -> recompute balance sheet, HQLA, NCOF, and LCR

This is not yet an optimization module. It is a transparent intervention
layer that quantifies how much central bank support would be needed after
banks use their own liquid securities.
"""

import numpy as np
import pandas as pd

from liquidity_stress.first_round import compute_lcr_from_balance_sheet


LIABILITY_PAYMENT_MAP = {
    "retail_deposits": (
        "planned_retail_deposit_outflow",
        "paid_retail_deposit_outflow",
    ),
    "corporate_deposits": (
        "planned_corporate_deposit_outflow",
        "paid_corporate_deposit_outflow",
    ),
    "other_deposits": (
        "planned_other_deposit_outflow",
        "paid_other_deposit_outflow",
    ),
    "wholesale_funding": (
        "planned_wholesale_funding_outflow",
        "paid_wholesale_funding_outflow",
    ),
    "interbank_liabilities": (
        "planned_interbank_liability_outflow",
        "paid_interbank_liability_outflow",
    ),
}


def _safe_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return float(value)

def _equity_for_solvency(row: pd.Series) -> float:
    """
    Equity used for solvency tests.

    Normal combined run:
        use row["equity"]

    Currency run:
        use bank-level equity adjusted by the change in the currency bucket:
        bank_level_equity + (current_currency_equity - initial_currency_equity)
    """
    if (
        "bank_level_equity" in row.index
        and "initial_currency_equity" in row.index
        and "equity" in row.index
        and not pd.isna(row["bank_level_equity"])
        and not pd.isna(row["initial_currency_equity"])
    ):
        return float(row["bank_level_equity"]) + (
            float(row["equity"]) - float(row["initial_currency_equity"])
        )

    if "bank_level_equity" in row.index and not pd.isna(row["bank_level_equity"]):
        return float(row["bank_level_equity"])

    return _safe_value(row, "equity")

def _rebuild_balance_sheet_aggregates(row: pd.Series) -> pd.Series:
    """
    Rebuild accounting aggregates after CBA support and extra liability payment.
    """
    row["total_securities"] = (
        row["government_securities"]
        + row["bank_securities"]
        + row["corporate_securities"]
    )

    row["deposits"] = (
        row["retail_deposits"]
        + row["corporate_deposits"]
        + row["other_deposits"]
    )

    row["total_assets"] = (
        row["cash_reserves"]
        + row["loans"]
        + row["total_securities"]
        + row["interbank_assets"]
    )

    row["total_liabilities"] = (
        row["deposits"]
        + row["wholesale_funding"]
        + row["interbank_liabilities"]
        + _safe_value(row, "cb_liquidity_funding")
    )

    row["equity"] = row["total_assets"] - row["total_liabilities"]

    row["equity_assets_ratio"] = (
        row["equity"] / row["total_assets"]
        if row["total_assets"] > 0
        else 0.0
    )

    fx_asset_share = _safe_value(row, "FX_asset_share")
    fx_liability_share = _safe_value(row, "FX_liability_share")

    row["FX_assets"] = fx_asset_share * row["total_assets"]
    row["AMD_assets"] = row["total_assets"] - row["FX_assets"]

    row["FX_liabilities"] = fx_liability_share * row["total_liabilities"]
    row["AMD_liabilities"] = row["total_liabilities"] - row["FX_liabilities"]

    row["net_fx_gap"] = row["FX_assets"] - row["FX_liabilities"]

    row["net_fx_gap_assets_ratio"] = (
        row["net_fx_gap"] / row["total_assets"]
        if row["total_assets"] > 0
        else 0.0
    )

    return row


def _apply_extra_liability_payment(
    row: pd.Series,
    payment_amount: float,
) -> pd.Series:
    """
    Use CBA liquidity to pay remaining unpaid liability outflows.
    """
    payment_amount = max(float(payment_amount), 0.0)

    unpaid_by_liability = {}

    for liability_column, (planned_col, paid_col) in LIABILITY_PAYMENT_MAP.items():
        planned = _safe_value(row, planned_col)
        paid = _safe_value(row, paid_col)
        unpaid = max(planned - paid, 0.0)
        unpaid_by_liability[liability_column] = unpaid

    total_unpaid = sum(unpaid_by_liability.values())

    if total_unpaid <= 0 or payment_amount <= 0:
        return row

    actual_payment = min(payment_amount, total_unpaid)
    payment_ratio = actual_payment / total_unpaid

    for liability_column, unpaid in unpaid_by_liability.items():
        _, paid_col = LIABILITY_PAYMENT_MAP[liability_column]

        extra_paid = unpaid * payment_ratio

        row[paid_col] = _safe_value(row, paid_col) + extra_paid
        row[liability_column] = max(
            _safe_value(row, liability_column) - extra_paid,
            0.0,
        )

    row["paid_total_liability_outflow"] = (
        _safe_value(row, "paid_total_liability_outflow") + actual_payment
    )

    row["unpaid_total_liability_outflow"] = max(
        _safe_value(row, "unpaid_total_liability_outflow") - actual_payment,
        0.0,
    )

    return row


def _pledged_column(asset_column: str) -> str:
    """
    Name of the stock column tracking collateral already pledged to CBA.
    """
    return f"pledged_{asset_column}"


def _compute_collateral_capacity(
    row: pd.Series,
    policy_config: dict,
) -> tuple[float, dict[str, float], dict[str, float]]:
    """
    Compute maximum CBA support capacity from unencumbered eligible collateral.

    Collateral already pledged to CBA cannot be pledged again.

    Capacity equals:
        sum((asset_value - already_pledged_value) * (1 - haircut))

    Guard:
        If cb_liquidity_funding is already positive, pledged collateral
        tracking columns must exist. Otherwise the model would silently treat
        previously supported collateral as fully unencumbered.
    """
    eligible_collateral = policy_config["eligible_collateral"]
    collateral_haircuts = policy_config["collateral_haircuts"]

    existing_cb_funding = max(
        _safe_value(row, "cb_liquidity_funding"),
        0.0,
    )

    missing_pledged_columns = [
        _pledged_column(asset_column)
        for asset_column in eligible_collateral
        if _pledged_column(asset_column) not in row.index
    ]

    if existing_cb_funding > 1e-9 and missing_pledged_columns:
        raise ValueError(
            "Existing cb_liquidity_funding is positive, but pledged collateral "
            f"tracking columns are missing: {missing_pledged_columns}. "
            "This would overstate unencumbered collateral capacity. "
            "Initialize pledged_* columns or reconstruct collateral allocation "
            "before computing additional CBA support."
        )

    collateral_capacities = {}
    unencumbered_values = {}
    total_capacity = 0.0

    for asset_column in eligible_collateral:
        if asset_column not in collateral_haircuts:
            raise KeyError(
                f"Missing collateral haircut for eligible collateral '{asset_column}'."
            )

        asset_value = max(
            _safe_value(row, asset_column),
            0.0,
        )

        pledged_col = _pledged_column(asset_column)

        if pledged_col not in row.index:
            pledged_value = 0.0
        else:
            pledged_value = max(
                _safe_value(row, pledged_col),
                0.0,
            )

        unencumbered_value = max(
            asset_value - pledged_value,
            0.0,
        )

        haircut = float(collateral_haircuts[asset_column])

        capacity = unencumbered_value * max(
            1.0 - haircut,
            0.0,
        )

        unencumbered_values[asset_column] = unencumbered_value
        collateral_capacities[asset_column] = capacity
        total_capacity += capacity

    return total_capacity, collateral_capacities, unencumbered_values

def _first_existing_ratio(
    row: pd.Series,
    candidates: list[str],
    default: float = np.inf,
) -> float:
    """
    Return the first available numeric ratio from candidate columns.
    """
    for column in candidates:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])

    return default

def _first_existing_value(
    row: pd.Series,
    candidates: list[str],
    default: float = 0.0,
) -> float:
    """
    Return the first available numeric value from candidate columns.
    """
    for column in candidates:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])

    return default

def _compute_lolr_eligibility(
    row: pd.Series,
    policy_config: dict,
    liquidity_need: float,
    tolerance: float = 1e-9,
) -> dict:
    """
    Compute LoLR eligibility and required support.

    Economic rule:
        CBA support is available only to solvent but illiquid banks.

    Illiquidity can come from:
        1. actual unpaid cash shortfall,
        2. LCR below policy threshold,
        3. HQLA/assets below policy threshold.

    Support need is decomposed into:
        cash gap support:
            support used immediately to cover unpaid outflows

        buffer support:
            support retained as cash/HQLA to close LCR or HQLA/assets gaps
    """
    equity = _equity_for_solvency(row)
    total_assets = _safe_value(row, "total_assets")

    solvency_threshold = float(
        policy_config.get("solvency_equity_threshold", 0.0)
    )
    solvent = equity > solvency_threshold

    # Prefer current / post-reaction fields before old intermediate fields.
    hqla = _first_existing_value(
        row,
        [
            "post_asset_sale_HQLA",
            "post_reaction_HQLA",
            "post_bank_reaction_HQLA",
            "post_cb_support_HQLA",
            "HQLA",
            "stressed_HQLA",
            "post_cb_refinancing_HQLA",
        ],
        default=0.0,
    )

    ncof = _first_existing_value(
        row,
        [
            "post_asset_sale_NCOF",
            "post_reaction_NCOF",
            "post_bank_reaction_NCOF",
            "post_cb_support_NCOF",
            "NCOF",
            "stressed_NCOF",
            "post_cb_refinancing_NCOF",
        ],
        default=1.0,
    )

    lcr = hqla / ncof if ncof > 0 else np.inf

    hqla_assets_ratio = (
        hqla / total_assets
        if total_assets > 0
        else np.inf
    )

    cash_shortfall_need = max(float(liquidity_need), 0.0)

    cash_shortfall_illiquid = (
        bool(policy_config.get("illiquidity_cash_shortfall_enabled", True))
        and cash_shortfall_need > tolerance
    )

    lcr_threshold = policy_config.get("illiquidity_lcr_threshold")
    hqla_threshold = policy_config.get("illiquidity_hqla_assets_threshold")

    lcr_illiquid = (
        lcr_threshold is not None
        and lcr < float(lcr_threshold)
    )

    hqla_assets_illiquid = (
        hqla_threshold is not None
        and total_assets > 0
        and hqla_assets_ratio < float(hqla_threshold)
    )

    illiquid = (
        cash_shortfall_illiquid
        or lcr_illiquid
        or hqla_assets_illiquid
    )

    # LCR gap: extra cash/HQLA needed to bring HQLA / NCOF to target.
    support_lcr_gap = bool(policy_config.get("support_lcr_gap", False))

    if support_lcr_gap and lcr_threshold is not None:
        lcr_target = float(policy_config.get("lcr_support_target", lcr_threshold))
        lolr_lcr_support_gap = max(lcr_target * ncof - hqla, 0.0)
    else:
        lcr_target = np.nan
        lolr_lcr_support_gap = 0.0

    # HQLA/assets gap: if support is retained as cash, both HQLA and assets rise.
    # Need s such that (HQLA + s) / (Assets + s) >= target.
    support_hqla_assets_gap = bool(
        policy_config.get("support_hqla_assets_gap", False)
    )

    if support_hqla_assets_gap and hqla_threshold is not None and total_assets > 0:
        hqla_assets_target = float(
            policy_config.get("hqla_assets_support_target", hqla_threshold)
        )

        if hqla_assets_target >= 1.0:
            lolr_hqla_assets_support_gap = np.inf
        else:
            raw_gap = hqla_assets_target * total_assets - hqla
            lolr_hqla_assets_support_gap = max(
                raw_gap / max(1.0 - hqla_assets_target, tolerance),
                0.0,
            )
    else:
        hqla_assets_target = np.nan
        lolr_hqla_assets_support_gap = 0.0

    buffer_support_need = max(
        lolr_lcr_support_gap,
        lolr_hqla_assets_support_gap,
    )

    # Important: cash-gap support is used up by outflows.
    # It cannot simultaneously count as retained HQLA.
    required_support = cash_shortfall_need + buffer_support_need

    policy_enabled = bool(policy_config.get("enabled", True))
    support_only_solvent_illiquid = bool(
        policy_config.get("support_only_solvent_illiquid", True)
    )

    if not policy_enabled:
        eligible = False
        denial_reason = "policy_disabled"
    elif support_only_solvent_illiquid and not solvent:
        eligible = False
        denial_reason = "insolvent"
    elif support_only_solvent_illiquid and not illiquid:
        eligible = False
        denial_reason = "not_illiquid"
    elif required_support <= tolerance:
        eligible = False
        denial_reason = "no_support_need"
    else:
        eligible = True
        denial_reason = ""

    return {
        "solvency_equity_for_lolr": equity,
        "solvent_before_lolr": solvent,
        "illiquid_before_lolr": illiquid,
        "cash_shortfall_illiquid": cash_shortfall_illiquid,
        "hqla_assets_illiquid": hqla_assets_illiquid,
        "lcr_illiquid": lcr_illiquid,
        "eligible_for_lolr": eligible,
        "lolr_denial_reason": denial_reason,

        "pre_lolr_HQLA": hqla,
        "pre_lolr_NCOF": ncof,
        "pre_lolr_LCR": lcr,
        "pre_lolr_lcr": lcr,
        "pre_lolr_hqla_assets_ratio": hqla_assets_ratio,

        "lolr_cash_shortfall_need": cash_shortfall_need,
        "lolr_lcr_support_gap": lolr_lcr_support_gap,
        "lolr_lcr_target": lcr_target,
        "lolr_hqla_assets_support_gap": lolr_hqla_assets_support_gap,
        "lolr_hqla_assets_target": hqla_assets_target,
        "lolr_buffer_support_need": buffer_support_need,
        "lolr_required_support": required_support,
    }

def _allocate_pledged_collateral(
    row: pd.Series,
    support_granted: float,
    policy_config: dict,
) -> pd.Series:
    """
    Allocate newly granted CBA support to pledged collateral.

    The pledged amount is the gross collateral value. The support value is the
    collateral value after haircuts.
    """
    remaining_support_to_collateralize = max(float(support_granted), 0.0)

    eligible_collateral = policy_config["eligible_collateral"]
    collateral_haircuts = policy_config["collateral_haircuts"]

    newly_pledged_total = 0.0
    newly_pledged_support_value_total = 0.0

    for asset_column in eligible_collateral:
        pledged_col = _pledged_column(asset_column)

        if pledged_col not in row.index:
            row[pledged_col] = 0.0

        row[f"newly_pledged_{asset_column}"] = 0.0
        row[f"newly_pledged_{asset_column}_support_value"] = 0.0

        if remaining_support_to_collateralize <= 0:
            continue

        asset_value = max(_safe_value(row, asset_column), 0.0)
        already_pledged = max(_safe_value(row, pledged_col), 0.0)
        unencumbered = max(asset_value - already_pledged, 0.0)

        if unencumbered <= 0:
            continue

        haircut = float(collateral_haircuts[asset_column])
        support_per_unit = max(1.0 - haircut, 0.0)

        if support_per_unit <= 0:
            continue

        gross_collateral_needed = (
            remaining_support_to_collateralize / support_per_unit
        )

        gross_collateral_to_pledge = min(
            unencumbered,
            gross_collateral_needed,
        )

        support_value = gross_collateral_to_pledge * support_per_unit

        row[pledged_col] = already_pledged + gross_collateral_to_pledge
        row[f"newly_pledged_{asset_column}"] = gross_collateral_to_pledge
        row[f"newly_pledged_{asset_column}_support_value"] = support_value

        newly_pledged_total += gross_collateral_to_pledge
        newly_pledged_support_value_total += support_value

        remaining_support_to_collateralize = max(
            remaining_support_to_collateralize - support_value,
            0.0,
        )

    row["newly_pledged_collateral"] = newly_pledged_total
    row["newly_pledged_collateral_support_value"] = (
        newly_pledged_support_value_total
    )

    return row

def apply_cb_refinancing_before_asset_sales(
    banks: pd.DataFrame,
    policy_config: dict,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Apply CBA refinancing before bank asset liquidation.

    In cba_first mode:
        CBA tries to refinance the full cash shortfall before asset sales.

    In hybrid mode:
        CBA refinances a share of the cash shortfall first, subject to collateral.
        The asset-sale module then handles the remaining gap.

    This keeps refinancing economically meaningful while preserving the bank
    reaction and fire-sale channels.
    """
    rows = []
    tolerance = 1e-9

    for _, original_row in banks.iterrows():
        row = original_row.copy()

        if "cb_liquidity_funding" not in row.index:
            row["cb_liquidity_funding"] = 0.0

        original_shortfall = max(
            _safe_value(row, "cash_shortfall_before_support"),
            0.0,
        )

        if original_shortfall <= tolerance:
            original_shortfall = 0.0

        pre_capacity, pre_collateral_values, pre_unencumbered_values = (
            _compute_collateral_capacity(
                row=row,
                policy_config=policy_config,
            )
        )

        eligibility = _compute_lolr_eligibility(
            row=row,
            policy_config=policy_config,
            liquidity_need=original_shortfall,
            tolerance=tolerance,
        )

        for key, value in eligibility.items():
            row[key] = value

        support_need = float(eligibility["lolr_required_support"])

        refinancing_strategy = policy_config.get(
            "refinancing_strategy",
            "cba_first",
        )

        if refinancing_strategy == "cba_first":
            target_support = support_need

        elif refinancing_strategy == "hybrid":
            refinancing_share = float(
                policy_config.get("pre_liquidation_refinancing_share", 0.70)
            )
            refinancing_share = min(max(refinancing_share, 0.0), 1.0)
            target_support = refinancing_share * support_need
        else:
            raise ValueError(
                f"Unknown refinancing_strategy: {refinancing_strategy}"
            )

        if not row["eligible_for_lolr"]:
            support_granted = 0.0

        elif policy_config["support_mode"] == "full_coverage":
            support_granted = support_need

        elif policy_config["support_mode"] == "collateral_capped":
            support_granted = min(target_support, pre_capacity)
        else:
            raise ValueError(
                f"Unknown support_mode: {policy_config['support_mode']}"
            )

        if support_granted <= tolerance:
            support_granted = 0.0

        row = _allocate_pledged_collateral(
            row=row,
            support_granted=support_granted,
            policy_config=policy_config,
        )

        post_capacity, post_collateral_values, post_unencumbered_values = (
            _compute_collateral_capacity(
                row=row,
                policy_config=policy_config,
            )
        )

        support_used_for_outflows = min(support_granted, original_shortfall)
        support_retained_as_cash = max(
            support_granted - support_used_for_outflows,
            0.0,
        )

        shortfall_after_refinancing = max(
            original_shortfall - support_used_for_outflows,
            0.0,
        )

        if shortfall_after_refinancing <= tolerance:
            shortfall_after_refinancing = 0.0

        row["cash_shortfall_before_cba_refinancing"] = original_shortfall
        row["cb_support_required"] = support_need
        row["cb_support_target_before_asset_sales"] = target_support
        row["cb_support_granted"] = support_granted
        row["cb_support_used_for_outflows"] = support_used_for_outflows
        row["cb_support_retained_as_cash_for_lcr"] = support_retained_as_cash
        row["cash_shortfall_after_cb_refinancing"] = shortfall_after_refinancing

        row["pre_cb_unencumbered_collateral_capacity"] = pre_capacity
        row["post_cb_unencumbered_collateral_capacity"] = post_capacity

        # Backward-compatible name now means post-CBA remaining capacity.
        row["unencumbered_collateral_capacity"] = post_capacity

        for asset_column in policy_config["eligible_collateral"]:
            row[f"{asset_column}_pre_cb_collateral_capacity"] = (
                pre_collateral_values[asset_column]
            )
            row[f"{asset_column}_post_cb_collateral_capacity"] = (
                post_collateral_values[asset_column]
            )
            row[f"{asset_column}_pre_cb_unencumbered_collateral"] = (
                pre_unencumbered_values[asset_column]
            )
            row[f"{asset_column}_post_cb_unencumbered_collateral"] = (
                post_unencumbered_values[asset_column]
            )

            # Backward-compatible names.
            row[f"{asset_column}_cb_collateral_capacity"] = (
                post_collateral_values[asset_column]
            )
            row[f"{asset_column}_unencumbered_collateral"] = (
                post_unencumbered_values[asset_column]
            )

        total_pledged_collateral = 0.0

        for asset_column in policy_config["eligible_collateral"]:
            pledged_col = _pledged_column(asset_column)

            if pledged_col not in row.index:
                row[pledged_col] = 0.0

            total_pledged_collateral += max(_safe_value(row, pledged_col), 0.0)

        row["total_cb_pledged_collateral"] = total_pledged_collateral
        row["policy_support_used"] = support_granted > tolerance

        # CBA support enters as funding and is used immediately to pay outflows.
        row["cb_liquidity_funding"] = (
            _safe_value(row, "cb_liquidity_funding") + support_granted
        )

        row["cash_reserves"] = _safe_value(row, "cash_reserves") + support_granted

        row = _apply_extra_liability_payment(
            row=row,
            payment_amount=support_used_for_outflows,
        )

        # Only the part used to pay outflows leaves cash.
        # Any excess CBA support is retained as cash/HQLA to close the LCR gap.
        row["cash_reserves"] = max(
            _safe_value(row, "cash_reserves") - support_used_for_outflows,
            0.0,
        )

        row = _rebuild_balance_sheet_aggregates(row)

        recomputed = compute_lcr_from_balance_sheet(
            banks=pd.DataFrame([row]),
            lcr_assumptions=lcr_assumptions,
        ).iloc[0]

        row["post_cb_refinancing_HQLA"] = recomputed["HQLA"]
        row["post_cb_refinancing_HQLA_assets_ratio"] = recomputed[
            "HQLA_assets_ratio"
        ]
        row["post_cb_refinancing_NCOF"] = recomputed["NCOF"]
        row["post_cb_refinancing_LCR"] = recomputed["LCR"]
        row["post_cb_refinancing_LCR_breach"] = (
            row["post_cb_refinancing_LCR"] < 1.0
        )

        row["cb_support_assets_ratio"] = (
            support_granted / row["total_assets"]
            if row["total_assets"] > 0
            else 0.0
        )

        # The asset-sale module reads cash_shortfall_before_support.
        # Therefore, pass only the residual gap after CBA refinancing.
        row["cash_shortfall_before_support"] = shortfall_after_refinancing

        rows.append(row)

    return pd.DataFrame(rows)


def finalize_cb_policy_metrics_after_asset_sales(
    banks: pd.DataFrame,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Finalize policy-support fields after asset-sale reaction.

    No additional CBA cash is injected here. This function only records the
    remaining shortfall after pre-liquidation CBA refinancing plus asset sales.
    """
    rows = []
    tolerance = 1e-9

    for _, original_row in banks.iterrows():
        row = original_row.copy()

        remaining_shortfall = max(
            _safe_value(row, "cash_shortfall_after_asset_sales"),
            0.0,
        )

        if remaining_shortfall <= tolerance:
            remaining_shortfall = 0.0

        row["remaining_shortfall_after_cb_support"] = remaining_shortfall
        row["final_unresolved_shortfall"] = remaining_shortfall > tolerance

        row = _rebuild_balance_sheet_aggregates(row)

        recomputed = compute_lcr_from_balance_sheet(
            banks=pd.DataFrame([row]),
            lcr_assumptions=lcr_assumptions,
        ).iloc[0]



        row["post_cb_support_HQLA"] = recomputed["HQLA"]
        row["post_cb_support_HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
        row["post_cb_support_NCOF"] = recomputed["NCOF"]
        row["post_cb_support_LCR"] = recomputed["LCR"]
        row["post_cb_support_LCR_breach"] = (
                row["post_cb_support_LCR"] < row["active_lcr_requirement"]
        )

        # Restore original first-round cash shortfall for reporting.
        if "cash_shortfall_before_cba_refinancing" in row.index:
            row["cash_shortfall_before_support"] = row[
                "cash_shortfall_before_cba_refinancing"
            ]

        rows.append(row)

    return pd.DataFrame(rows)

def apply_cb_liquidity_support(
    banks: pd.DataFrame,
    policy_config: dict,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Apply central bank liquidity support after asset-sale reaction.

    Corrected logic:
        1. compute cash gap after asset sales,
        2. compute LCR/HQLA-assets support gap,
        3. grant support subject to eligibility and collateral capacity,
        4. split support into:
            a. support used to pay unpaid outflows,
            b. support retained as cash/HQLA,
        5. recompute balance sheet, HQLA, NCOF, and LCR.
    """
    tolerance = 1e-9
    rows = []

    for _, original_row in banks.iterrows():
        row = original_row.copy()

        if "cb_liquidity_funding" not in row.index:
            row["cb_liquidity_funding"] = 0.0

        cash_gap_after_asset_sales = max(
            _safe_value(row, "cash_shortfall_after_asset_sales"),
            0.0,
        )

        if cash_gap_after_asset_sales <= tolerance:
            cash_gap_after_asset_sales = 0.0

        pre_capacity, pre_collateral_values, pre_unencumbered_values = (
            _compute_collateral_capacity(
                row=row,
                policy_config=policy_config,
            )
        )

        eligibility = _compute_lolr_eligibility(
            row=row,
            policy_config=policy_config,
            liquidity_need=cash_gap_after_asset_sales,
            tolerance=tolerance,
        )

        for key, value in eligibility.items():
            row[key] = value

        support_required = float(eligibility["lolr_required_support"])

        if not row["eligible_for_lolr"]:
            support_granted = 0.0

        elif policy_config["support_mode"] == "full_coverage":
            support_granted = support_required

        elif policy_config["support_mode"] == "collateral_capped":
            support_granted = min(support_required, pre_capacity)

        else:
            raise ValueError(
                f"Unknown support_mode: {policy_config['support_mode']}"
            )

        if support_granted <= tolerance:
            support_granted = 0.0

        row = _allocate_pledged_collateral(
            row=row,
            support_granted=support_granted,
            policy_config=policy_config,
        )

        post_capacity, post_collateral_values, post_unencumbered_values = (
            _compute_collateral_capacity(
                row=row,
                policy_config=policy_config,
            )
        )

        support_used_for_outflows = min(
            support_granted,
            cash_gap_after_asset_sales,
        )

        support_retained_as_cash = max(
            support_granted - support_used_for_outflows,
            0.0,
        )

        remaining_shortfall = max(
            cash_gap_after_asset_sales - support_used_for_outflows,
            0.0,
        )

        if remaining_shortfall <= tolerance:
            remaining_shortfall = 0.0

        row["cash_shortfall_after_asset_sales_before_cb"] = (
            cash_gap_after_asset_sales
        )

        row["cb_support_required"] = support_required
        row["cb_support_collateral_capacity"] = pre_capacity
        row["pre_cb_unencumbered_collateral_capacity"] = pre_capacity
        row["post_cb_unencumbered_collateral_capacity"] = post_capacity
        row["unencumbered_collateral_capacity"] = post_capacity

        row["cb_support_granted"] = support_granted
        row["cb_support_used_for_outflows"] = support_used_for_outflows
        row["cb_support_retained_as_cash_for_lcr"] = support_retained_as_cash

        row["aggregate_support_used_for_outflows"] = support_used_for_outflows
        row["aggregate_support_retained_as_cash_for_lcr"] = support_retained_as_cash

        row["remaining_shortfall_after_cb_support"] = remaining_shortfall
        row["final_unresolved_shortfall"] = remaining_shortfall > tolerance
        row["policy_support_used"] = support_granted > tolerance

        for asset_column in policy_config["eligible_collateral"]:
            row[f"{asset_column}_pre_cb_collateral_capacity"] = (
                pre_collateral_values[asset_column]
            )
            row[f"{asset_column}_post_cb_collateral_capacity"] = (
                post_collateral_values[asset_column]
            )
            row[f"{asset_column}_pre_cb_unencumbered_collateral"] = (
                pre_unencumbered_values[asset_column]
            )
            row[f"{asset_column}_post_cb_unencumbered_collateral"] = (
                post_unencumbered_values[asset_column]
            )

            # Backward-compatible names.
            row[f"{asset_column}_cb_collateral_capacity"] = (
                post_collateral_values[asset_column]
            )
            row[f"{asset_column}_unencumbered_collateral"] = (
                post_unencumbered_values[asset_column]
            )

        total_pledged_collateral = 0.0

        for asset_column in policy_config["eligible_collateral"]:
            pledged_col = _pledged_column(asset_column)

            if pledged_col not in row.index:
                row[pledged_col] = 0.0

            total_pledged_collateral += max(
                _safe_value(row, pledged_col),
                0.0,
            )

        row["total_cb_pledged_collateral"] = total_pledged_collateral

        # CBA support enters as funding and cash.
        row["cb_liquidity_funding"] = (
            _safe_value(row, "cb_liquidity_funding") + support_granted
        )

        row["cash_reserves"] = (
            _safe_value(row, "cash_reserves") + support_granted
        )

        # Only the part used for unpaid outflows leaves cash.
        row = _apply_extra_liability_payment(
            row=row,
            payment_amount=support_used_for_outflows,
        )

        row["cash_reserves"] = max(
            _safe_value(row, "cash_reserves") - support_used_for_outflows,
            0.0,
        )

        row = _rebuild_balance_sheet_aggregates(row)

        recomputed = compute_lcr_from_balance_sheet(
            banks=pd.DataFrame([row]),
            lcr_assumptions=lcr_assumptions,
        ).iloc[0]

        row["post_cb_support_HQLA"] = recomputed["HQLA"]
        row["post_cb_support_HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
        row["post_cb_support_NCOF"] = recomputed["NCOF"]
        row["post_cb_support_LCR"] = recomputed["LCR"]

        if (
            "active_lcr_requirement" in row.index
            and not pd.isna(row["active_lcr_requirement"])
        ):
            active_lcr_requirement = float(row["active_lcr_requirement"])
        else:
            active_lcr_requirement = float(
                policy_config.get(
                    "lcr_support_target",
                    policy_config.get("illiquidity_lcr_threshold", 1.0),
                )
            )

        row["active_lcr_requirement"] = active_lcr_requirement
        row["post_cb_support_LCR_breach"] = (
            row["post_cb_support_LCR"] < row["active_lcr_requirement"]
        )

        row["cb_support_assets_ratio"] = (
            support_granted / row["total_assets"]
            if row["total_assets"] > 0
            else 0.0
        )

        row["cb_support_to_pre_lolr_hqla_ratio"] = (
            support_granted / row["pre_lolr_HQLA"]
            if row["pre_lolr_HQLA"] > 0
            else 0.0
        )

        rows.append(row)

    return pd.DataFrame(rows)