"""
Fire-sale price impact module.

This module models common-asset fire-sale feedback after bank-level reactions:

    asset sales by banks
        -> aggregate sale volume by asset class
        -> market price impact
        -> mark-to-market losses on remaining securities
        -> recompute HQLA, equity, and LCR

The price-impact function follows the common exponential form used in
fire-sale contagion models:

    p_new = p_old * exp(-beta * fraction_sold)

where beta is calibrated so that selling a reference fraction of the market
causes a chosen price drop.
"""

import numpy as np
import pandas as pd

from liquidity_stress.first_round import compute_lcr_from_balance_sheet


def _safe_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row.index:
        return default

    value = row[column]

    if pd.isna(value):
        return default

    return float(value)

def _solvency_equity_series(df: pd.DataFrame) -> pd.Series:
    """
    Equity used for insolvency tests.

    For normal combined runs:
        use equity.

    For currency runs:
        use bank-level equity adjusted by the change in the currency bucket.
    """
    if {
        "bank_level_equity",
        "initial_currency_equity",
        "equity",
    }.issubset(df.columns):
        return (
            df["bank_level_equity"].astype(float)
            + (
                df["equity"].astype(float)
                - df["initial_currency_equity"].astype(float)
            )
        )

    if "bank_level_equity" in df.columns:
        return df["bank_level_equity"].astype(float)

    return df["equity"].astype(float)

def _compute_new_price(
    current_price: float,
    fraction_sold: float,
    price_impact_at_reference_sale_fraction: float,
    reference_sale_fraction: float,
    minimum_market_price: float,
) -> float:
    """
    Compute new market price after aggregate fire-sale pressure.
    """
    current_price = float(current_price)
    fraction_sold = max(float(fraction_sold), 0.0)



    if fraction_sold <= 0:
        return current_price

    impact = float(price_impact_at_reference_sale_fraction)

    if impact <= 0:
        return current_price

    if impact >= 1:
        raise ValueError("Price impact must be below 1.")

    beta = -np.log(1.0 - impact) / reference_sale_fraction

    new_price = current_price * np.exp(-beta * fraction_sold)

    return max(float(new_price), float(minimum_market_price))


def _rebuild_balance_sheet_aggregates(row: pd.Series) -> pd.Series:
    """
    Rebuild balance-sheet aggregates after fire-sale mark-to-market losses.
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


def apply_fire_sale_price_impact(
    banks: pd.DataFrame,
    fire_sale_config: dict,
    lcr_assumptions: dict,
    current_market_prices: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    """
    Apply one round of fire-sale price impact to a monthly bank table.

    Parameters
    ----------
    banks:
        Bank table after first-round shock, asset sales, and policy support.
    fire_sale_config:
        Fire-sale configuration from config.py.
    lcr_assumptions:
        LCR assumptions used to recompute post-fire-sale LCR.
    current_market_prices:
        Current market price index for each securities bucket. If None,
        every asset starts at price 1.

    Returns
    -------
    tuple:
        updated bank table,
        updated market prices,
        one-row market summary table.
    """
    out = banks.copy()

    asset_columns = fire_sale_config["asset_columns"]
    reference_sale_fraction = fire_sale_config["reference_sale_fraction"]
    impact_params = fire_sale_config["price_impact_at_reference_sale_fraction"]
    minimum_market_price = fire_sale_config["minimum_market_price"]

    if current_market_prices is None:
        current_market_prices = {asset: 1.0 for asset in asset_columns}

    updated_market_prices = dict(current_market_prices)

    market_summary = {}

    out["total_fire_sale_mtm_loss"] = 0.0

    for asset_column in asset_columns:
        sold_column = f"{asset_column}_sold_for_liquidity"

        if sold_column not in out.columns:
            out[sold_column] = 0.0

        quantity_sold = float(out[sold_column].sum())

        # Approximate market size at the moment of sale:
        # remaining post-sale holdings + quantity sold.
        modeled_banking_market_quantity = float(
            (out[asset_column] + out[sold_column]).sum()
        )

        market_depth_by_asset = fire_sale_config.get("market_depth_by_asset", {})
        market_depth_multiplier_by_asset = fire_sale_config.get(
            "market_depth_multiplier_by_asset",
            {},
        )

        if asset_column in market_depth_by_asset:
            # External market-depth estimate, e.g. total outstanding state bonds.
            market_quantity_before_sale = max(
                float(market_depth_by_asset[asset_column]),
                modeled_banking_market_quantity,
            )
        else:
            # Conservative default: modeled banking-sector holdings only.
            # A multiplier above 1.0 can approximate non-bank market depth.
            multiplier = float(
                market_depth_multiplier_by_asset.get(asset_column, 1.0)
            )

            market_quantity_before_sale = (
                    modeled_banking_market_quantity * max(multiplier, 1.0)
            )

        if market_quantity_before_sale > 0:
            fraction_sold = quantity_sold / market_quantity_before_sale
        else:
            fraction_sold = 0.0

        fraction_sold = min(max(fraction_sold, 0.0), 1.0)

        cb_asset_purchase_config = fire_sale_config.get(
            "central_bank_asset_purchases",
            {},
        )

        cb_asset_purchases_enabled = cb_asset_purchase_config.get(
            "enabled",
            False,
        )

        purchase_share_by_asset = cb_asset_purchase_config.get(
            "purchase_share_of_bank_sales",
            {},
        )

        if cb_asset_purchases_enabled:
            cb_purchase_share = float(
                purchase_share_by_asset.get(asset_column, 0.0)
            )
        else:
            cb_purchase_share = 0.0

        cb_purchase_share = min(max(cb_purchase_share, 0.0), 1.0)

        cb_purchase_fraction = fraction_sold * cb_purchase_share
        effective_fraction_sold = max(fraction_sold - cb_purchase_fraction, 0.0)

        cb_purchase_quantity = quantity_sold * cb_purchase_share

        price_before = float(current_market_prices.get(asset_column, 1.0))

        if fire_sale_config.get("enabled", True):
            price_after = _compute_new_price(
                current_price=price_before,
                fraction_sold=effective_fraction_sold,
                price_impact_at_reference_sale_fraction=impact_params[asset_column],
                reference_sale_fraction=reference_sale_fraction,
                minimum_market_price=minimum_market_price,
            )
        else:
            price_after = price_before

        updated_market_prices[asset_column] = price_after

        if price_before > 0:
            incremental_price_drop = max(1.0 - price_after / price_before, 0.0)
        else:
            incremental_price_drop = 0.0

        loss_column = f"{asset_column}_fire_sale_mtm_loss"

        out[loss_column] = out[asset_column] * incremental_price_drop
        out[asset_column] = np.maximum(out[asset_column] - out[loss_column], 0.0)

        pledged_col = f"pledged_{asset_column}"
        pledged_loss_col = f"{pledged_col}_fire_sale_mtm_loss"

        if pledged_col in out.columns:
            out[pledged_loss_col] = out[pledged_col] * incremental_price_drop
            out[pledged_col] = np.maximum(
                out[pledged_col] - out[pledged_loss_col],
                0.0,
            )

            # Numerical safety: pledged collateral cannot exceed remaining holdings.
            out[pledged_col] = np.minimum(out[pledged_col], out[asset_column])

        out["total_fire_sale_mtm_loss"] += out[loss_column]

        market_summary[f"{asset_column}_quantity_sold"] = quantity_sold
        market_summary[f"{asset_column}_market_quantity_before_sale"] = (
            market_quantity_before_sale
        )
        market_summary[f"{asset_column}_fraction_sold"] = fraction_sold
        market_summary[f"{asset_column}_cb_purchase_share"] = cb_purchase_share
        market_summary[f"{asset_column}_cb_purchase_quantity"] = cb_purchase_quantity
        market_summary[f"{asset_column}_cb_purchase_fraction"] = cb_purchase_fraction
        market_summary[f"{asset_column}_effective_fraction_sold_after_cb"] = (
            effective_fraction_sold
        )
        market_summary[f"{asset_column}_price_before"] = price_before
        market_summary[f"{asset_column}_price_after"] = price_after
        market_summary[f"{asset_column}_incremental_price_drop"] = (
            incremental_price_drop
        )
        market_summary[f"{asset_column}_system_mtm_loss"] = float(
            out[loss_column].sum()
        )

    rows = []

    for _, row in out.iterrows():
        rebuilt = _rebuild_balance_sheet_aggregates(row)
        rows.append(rebuilt)

    out = pd.DataFrame(rows)

    recomputed = compute_lcr_from_balance_sheet(
        banks=out,
        lcr_assumptions=lcr_assumptions,
    )

    out["post_fire_sale_HQLA"] = recomputed["HQLA"]
    out["post_fire_sale_HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
    out["post_fire_sale_NCOF"] = recomputed["NCOF"]
    out["post_fire_sale_LCR"] = recomputed["LCR"]
    if "active_lcr_requirement" in out.columns:
        active_lcr_requirement = pd.to_numeric(
            out["active_lcr_requirement"],
            errors="coerce",
        ).fillna(1.0)
    else:
        active_lcr_requirement = 1.0

    out["post_fire_sale_LCR_breach"] = (
            out["post_fire_sale_LCR"] < active_lcr_requirement
    )

    out["HQLA"] = recomputed["HQLA"]
    out["HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
    out["NCOF"] = recomputed["NCOF"]
    out["LCR"] = recomputed["LCR"]

    out["solvency_equity"] = _solvency_equity_series(out)
    out["post_fire_sale_insolvent"] = out["solvency_equity"] < 0.0

    out["fire_sale_mtm_loss_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["total_fire_sale_mtm_loss"] / out["total_assets"],
        0.0,
    )

    market_summary["total_system_fire_sale_mtm_loss"] = float(
        out["total_fire_sale_mtm_loss"].sum()
    )

    if "stress_month" in out.columns and len(out) > 0:
        market_summary["stress_month"] = int(out["stress_month"].iloc[0])

    if "first_round_alpha" in out.columns and len(out) > 0:
        market_summary["alpha"] = float(out["first_round_alpha"].iloc[0])

    return out, updated_market_prices, pd.DataFrame([market_summary])