"""
Bank-level liquidity reaction module.

This module models the first reaction to a liquidity problem:

    cash_shortfall_before_support
        -> sell securities through a liquidation waterfall
        -> use sale proceeds to cover unpaid liability outflows
        -> recompute post-reaction balance sheet, HQLA, NCOF, and LCR

The module also supports a regulatory LCR-buffer reaction:

    LCR below reaction trigger
        -> sell selected securities only if conversion to cash creates
           a meaningful net HQLA gain
        -> retain proceeds as cash/HQLA
        -> recompute post-reaction balance sheet, HQLA, NCOF, and LCR

Fire-sale price impact is not applied here. Fire-sale feedback comes later,
after the asset-sale quantities are known.
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
    """
    Safely get a numeric value from a row.
    """
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return float(value)


def _rebuild_balance_sheet_aggregates(row: pd.Series) -> pd.Series:
    """
    Rebuild accounting aggregates after asset sales and extra liability payments.
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

    # Preserve current AMD/FX shares approximately after reaction.
    fx_asset_share = _safe_value(row, "FX_asset_share", 0.0)
    fx_liability_share = _safe_value(row, "FX_liability_share", 0.0)

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
    Use asset-sale proceeds to pay previously unpaid liability outflows.

    The payment is allocated proportionally across unpaid liability outflow
    components.
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


def _first_available_lcr(row: pd.Series) -> float:
    for column in [
        "post_cb_refinancing_LCR",
        "stressed_LCR",
        "LCR",
    ]:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])

    return np.inf


def _first_available_hqla(row: pd.Series) -> float:
    for column in [
        "post_cb_refinancing_HQLA",
        "stressed_HQLA",
        "HQLA",
    ]:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])

    return _safe_value(row, "HQLA")


def _first_available_ncof(row: pd.Series) -> float:
    for column in [
        "post_cb_refinancing_NCOF",
        "stressed_NCOF",
        "NCOF",
    ]:
        if column in row.index and not pd.isna(row[column]):
            return float(row[column])

    return _safe_value(row, "NCOF", default=1.0)


def _reaction_target_lcr(
    row: pd.Series,
    liquidation_config: dict,
) -> float:
    trigger = float(liquidation_config.get("reaction_lcr_trigger", 1.0))
    target_cap = float(liquidation_config.get("reaction_target_cap", 2.0))
    target_mode = liquidation_config.get("reaction_target_mode", "baseline_lcr")

    if target_mode == "baseline_lcr":
        baseline_lcr = _safe_value(row, "LCR", default=trigger)
        return min(max(baseline_lcr, trigger), target_cap)

    if target_mode == "trigger":
        return trigger

    return min(max(trigger, 0.0), target_cap)


def _asset_hqla_weight(
    asset_column: str,
    liquidation_config: dict,
    lcr_assumptions: dict,
) -> float:
    """
    Return the HQLA recognition weight of an asset.

    Prefer LCR assumptions because LCR* scenarios may modify these weights.
    """
    key = f"{asset_column}_hqla_weight"

    if key in lcr_assumptions:
        return float(lcr_assumptions[key])

    weights = liquidation_config.get("hqla_recognition_weights", {})
    return float(weights.get(asset_column, 0.0))


def _unique_asset_order(*orders: list[str]) -> list[str]:
    """
    Merge asset-order lists while preserving order and removing duplicates.
    """
    assets = []

    for order in orders:
        for asset in order:
            if asset not in assets:
                assets.append(asset)

    return assets


def apply_asset_sale_reaction(
    banks: pd.DataFrame,
    liquidation_config: dict,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Apply bank reaction after the first-round shock and possible CBA refinancing.

    A bank reacts if either:
        1. it has an operational cash shortfall, or
        2. its LCR is below the active reaction trigger.

    Cash-shortfall reaction:
        sell assets according to liquidation_order and use proceeds to pay
        unpaid liability outflows.

    Regulatory LCR reaction:
        sell assets according to regulatory_reaction_order and retain proceeds
        as cash, but only if conversion to cash creates a meaningful net HQLA
        gain. This prevents massive government-bond liquidation merely to gain
        a few percentage points of HQLA recognition.
    """
    rows = []

    liquidation_order = list(liquidation_config["liquidation_order"])
    sale_haircuts = liquidation_config["sale_haircuts"]
    reaction_lcr_trigger = float(liquidation_config.get("reaction_lcr_trigger", 1.0))

    active_lcr_requirement = float(
        liquidation_config.get(
            "active_lcr_requirement",
            liquidation_config.get("reaction_lcr_trigger", 1.0),
        )
    )

    regulatory_reaction_order = list(
        liquidation_config.get(
            "regulatory_reaction_order",
            liquidation_order,
        )
    )

    minimum_net_hqla_gain_per_unit = float(
        liquidation_config.get("minimum_net_hqla_gain_per_unit", 0.0)
    )

    all_reaction_assets = _unique_asset_order(
        liquidation_order,
        regulatory_reaction_order,
    )

    for _, original_row in banks.iterrows():
        row = original_row.copy()

        initial_shortfall = _safe_value(row, "cash_shortfall_before_support")
        pre_reaction_lcr = _first_available_lcr(row)
        pre_reaction_hqla = _first_available_hqla(row)
        pre_reaction_ncof = _first_available_ncof(row)

        regulatory_triggered = pre_reaction_lcr < reaction_lcr_trigger
        target_lcr = _reaction_target_lcr(row, liquidation_config)

        regulatory_hqla_gap = 0.0
        if regulatory_triggered and pre_reaction_ncof > 0:
            regulatory_hqla_gap = max(
                target_lcr * pre_reaction_ncof - pre_reaction_hqla,
                0.0,
            )

        row["pre_reaction_LCR"] = pre_reaction_lcr
        row["reaction_lcr_trigger"] = reaction_lcr_trigger
        row["active_lcr_requirement"] = active_lcr_requirement
        row["regulatory_lcr_reaction_triggered"] = regulatory_triggered
        row["regulatory_lcr_target"] = target_lcr
        row["regulatory_hqla_gap_before_reaction"] = regulatory_hqla_gap

        row["requires_asset_sale_reaction"] = (
            initial_shortfall > 1e-9 or regulatory_triggered
        )

        for asset_column in all_reaction_assets:
            row[f"{asset_column}_sold_for_liquidity"] = 0.0
            row[f"{asset_column}_sale_proceeds"] = 0.0
            row[f"{asset_column}_sale_loss"] = 0.0

        row["total_securities_sold_for_liquidity"] = 0.0
        row["asset_sale_proceeds_used_for_outflows"] = 0.0
        row["asset_sale_cash_retained_for_lcr"] = 0.0
        row["asset_sale_liquidation_loss"] = 0.0

        remaining_shortfall = initial_shortfall
        remaining_hqla_gap = regulatory_hqla_gap

        if row["requires_asset_sale_reaction"]:
            # --------------------------------------------------
            # 1. Operational liquidity reaction:
            # sell assets to cover unpaid outflows.
            # --------------------------------------------------
            for asset_column in liquidation_order:
                if remaining_shortfall <= 1e-9:
                    break

                if asset_column not in sale_haircuts:
                    raise KeyError(
                        f"Missing sale haircut for asset '{asset_column}'."
                    )

                pledged_col = f"pledged_{asset_column}"
                pledged_asset = max(_safe_value(row, pledged_col), 0.0)

                total_asset = max(_safe_value(row, asset_column), 0.0)
                available_asset = max(total_asset - pledged_asset, 0.0)

                haircut = float(sale_haircuts[asset_column])

                if available_asset <= 0:
                    continue

                net_proceeds_per_unit = max(1.0 - haircut, 0.0)

                if net_proceeds_per_unit <= 0:
                    continue

                amount_to_sell = min(
                    available_asset,
                    remaining_shortfall / net_proceeds_per_unit,
                )

                proceeds = amount_to_sell * net_proceeds_per_unit
                loss = amount_to_sell * haircut

                # Keep pledged assets on the balance sheet. Only the available
                # unpledged portion is sold.
                row[asset_column] = total_asset - amount_to_sell

                # Explicit gross-flow accounting:
                # sale proceeds enter cash first, then are used to pay unpaid outflows.
                row["cash_reserves"] = _safe_value(row, "cash_reserves") + proceeds

                row[f"{asset_column}_sold_for_liquidity"] += amount_to_sell
                row[f"{asset_column}_sale_proceeds"] += proceeds
                row[f"{asset_column}_sale_loss"] += loss

                row["total_securities_sold_for_liquidity"] += amount_to_sell
                row["asset_sale_proceeds_used_for_outflows"] += proceeds
                row["asset_sale_liquidation_loss"] += loss

                remaining_shortfall = max(remaining_shortfall - proceeds, 0.0)

            unpaid_before_asset_sale_payment = _safe_value(
                row,
                "unpaid_total_liability_outflow",
            )

            asset_sale_payment_amount = _safe_value(
                row,
                "asset_sale_proceeds_used_for_outflows",
            )

            row = _apply_extra_liability_payment(
                row=row,
                payment_amount=asset_sale_payment_amount,
            )

            unpaid_after_asset_sale_payment = _safe_value(
                row,
                "unpaid_total_liability_outflow",
            )

            actual_asset_sale_payment = max(
                unpaid_before_asset_sale_payment - unpaid_after_asset_sale_payment,
                0.0,
            )

            # Cash must leave when the liability outflow is actually paid.
            row["cash_reserves"] = max(
                _safe_value(row, "cash_reserves") - actual_asset_sale_payment,
                0.0,
            )

            row["asset_sale_proceeds_used_for_outflows"] = actual_asset_sale_payment
            row["asset_sale_cash_remaining_after_outflows"] = max(
                asset_sale_payment_amount - actual_asset_sale_payment,
                0.0,
            )

            # --------------------------------------------------
            # 2. Regulatory LCR reaction:
            # sell assets only if converting them to cash improves HQLA
            # sufficiently. Proceeds are retained as cash/HQLA.
            # --------------------------------------------------
            if remaining_hqla_gap > 1e-9:
                for asset_column in regulatory_reaction_order:
                    if remaining_hqla_gap <= 1e-9:
                        break

                    if asset_column not in sale_haircuts:
                        raise KeyError(
                            f"Missing sale haircut for asset '{asset_column}'."
                        )

                    pledged_col = f"pledged_{asset_column}"
                    pledged_asset = max(_safe_value(row, pledged_col), 0.0)

                    total_asset = max(_safe_value(row, asset_column), 0.0)
                    available_asset = max(total_asset - pledged_asset, 0.0)

                    haircut = float(sale_haircuts[asset_column])
                    net_proceeds_per_unit = max(1.0 - haircut, 0.0)

                    if available_asset <= 0 or net_proceeds_per_unit <= 0:
                        continue

                    hqla_weight = _asset_hqla_weight(
                        asset_column=asset_column,
                        liquidation_config=liquidation_config,
                        lcr_assumptions=lcr_assumptions,
                    )

                    # Selling one unit and retaining proceeds as cash changes HQLA by:
                    #     net cash proceeds - lost recognized HQLA value.
                    net_hqla_gain_per_unit = net_proceeds_per_unit - hqla_weight

                    if net_hqla_gain_per_unit < minimum_net_hqla_gain_per_unit:
                        continue

                    if net_hqla_gain_per_unit <= 1e-9:
                        continue

                    amount_to_sell = min(
                        available_asset,
                        remaining_hqla_gap / net_hqla_gain_per_unit,
                    )

                    proceeds = amount_to_sell * net_proceeds_per_unit
                    loss = amount_to_sell * haircut
                    hqla_gain = amount_to_sell * net_hqla_gain_per_unit

                    row[asset_column] = total_asset - amount_to_sell
                    row["cash_reserves"] = (
                        _safe_value(row, "cash_reserves") + proceeds
                    )

                    row[f"{asset_column}_sold_for_liquidity"] += amount_to_sell
                    row[f"{asset_column}_sale_proceeds"] += proceeds
                    row[f"{asset_column}_sale_loss"] += loss

                    row["total_securities_sold_for_liquidity"] += amount_to_sell
                    row["asset_sale_cash_retained_for_lcr"] += proceeds
                    row["asset_sale_liquidation_loss"] += loss

                    remaining_hqla_gap = max(
                        remaining_hqla_gap - hqla_gain,
                        0.0,
                    )

        row["cash_shortfall_after_asset_sales"] = remaining_shortfall

        row["unresolved_shortfall_after_asset_sales"] = (
            row["cash_shortfall_after_asset_sales"] > 1e-9
        )

        row = _rebuild_balance_sheet_aggregates(row)

        recomputed = compute_lcr_from_balance_sheet(
            banks=pd.DataFrame([row]),
            lcr_assumptions=lcr_assumptions,
        ).iloc[0]

        row["post_asset_sale_HQLA"] = recomputed["HQLA"]
        row["post_asset_sale_HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
        row["post_asset_sale_NCOF"] = recomputed["NCOF"]
        row["post_asset_sale_LCR"] = recomputed["LCR"]

        active_lcr_requirement = float(
            liquidation_config.get(
                "active_lcr_requirement",
                liquidation_config.get("reaction_lcr_trigger", 1.0),
            )
        )

        row["active_lcr_requirement"] = active_lcr_requirement
        row["post_asset_sale_LCR_breach"] = (
                row["post_asset_sale_LCR"] < row["active_lcr_requirement"]
        )

        row["regulatory_hqla_gap_after_reaction"] = max(
            target_lcr * row["post_asset_sale_NCOF"] - row["post_asset_sale_HQLA"],
            0.0,
        )

        row["regulatory_lcr_reaction_success"] = (
            not regulatory_triggered
            or row["post_asset_sale_LCR"] >= reaction_lcr_trigger
        )

        row["asset_sale_reaction_success"] = (
            row["cash_shortfall_after_asset_sales"] <= 1e-9
            and row["regulatory_lcr_reaction_success"]
        )

        rows.append(row)

    return pd.DataFrame(rows)
