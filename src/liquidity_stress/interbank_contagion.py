"""
Interbank contagion module.

This module propagates losses from terminally distressed banks through a
synthetic interbank exposure matrix.

Logic
-----
A terminally distressed debtor bank j creates losses for creditor bank i:

    loss_i,j = LGD * exposure_i,j

The creditor's interbank assets fall by the loss amount, reducing total assets
and equity. LCR and solvency are then recomputed. If the creditor becomes
insolvent, it can trigger a further contagion round.

This module is synthetic-data compatible: if no exposure matrix is provided, it
constructs one from current interbank_assets and interbank_liabilities using a
gravity/IPF-style allocation with zero diagonal.
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

def _rebuild_balance_sheet_aggregates(row: pd.Series) -> pd.Series:
    """
    Rebuild accounting aggregates after interbank losses.
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


def _recompute_post_interbank_metrics(
    banks: pd.DataFrame,
    lcr_assumptions: dict,
) -> pd.DataFrame:
    """
    Recompute accounting and LCR metrics after interbank contagion losses.

    The LCR breach flag uses the active policy requirement if available.
    This keeps LCR_req, LCR_low, LCR_star, and LCR_cb internally consistent.
    """
    rows = []

    for _, row in banks.iterrows():
        rows.append(_rebuild_balance_sheet_aggregates(row.copy()))

    out = pd.DataFrame(rows)

    recomputed = compute_lcr_from_balance_sheet(
        banks=out,
        lcr_assumptions=lcr_assumptions,
    )

    out["post_interbank_HQLA"] = recomputed["HQLA"]
    out["post_interbank_HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
    out["post_interbank_NCOF"] = recomputed["NCOF"]
    out["post_interbank_LCR"] = recomputed["LCR"]

    if "active_lcr_requirement" in out.columns:
        out["active_lcr_requirement"] = pd.to_numeric(
            out["active_lcr_requirement"],
            errors="coerce",
        ).fillna(1.0)
    else:
        out["active_lcr_requirement"] = 1.0

    out["post_interbank_LCR_breach"] = (
        out["post_interbank_LCR"] < out["active_lcr_requirement"]
    )

    out["HQLA"] = recomputed["HQLA"]
    out["HQLA_assets_ratio"] = recomputed["HQLA_assets_ratio"]
    out["NCOF"] = recomputed["NCOF"]
    out["LCR"] = recomputed["LCR"]

    out["solvency_equity"] = _solvency_equity_series(out)
    out["post_interbank_insolvent"] = out["solvency_equity"] < 0.0

    return out

def build_synthetic_interbank_exposure_matrix(
    banks: pd.DataFrame,
    exposure_scale: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> pd.DataFrame:
    """
    Deprecated wrapper.

    The maintained implementation lives in network_interbank.py.
    """
    from liquidity_stress.network_interbank import (
        build_synthetic_interbank_exposure_matrix as build_matrix,
    )

    matrix = build_matrix(
        banks=banks,
        max_iter=max_iterations,
        tol=tolerance,
    )

    return matrix * float(exposure_scale)


def apply_interbank_contagion(
    banks: pd.DataFrame,
    contagion_config: dict,
    lcr_assumptions: dict,
    exposure_matrix: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply interbank contagion losses after fire-sale terminal distress.

    Terminal debtor banks transmit losses to creditor banks through the exposure
    matrix. Newly insolvent creditors can trigger additional contagion rounds.

    Returns
    -------
    tuple:
        updated bank table,
        one-row contagion summary table.
    """
    out = banks.copy().reset_index(drop=True)

    tolerance = float(contagion_config.get("tolerance", 1e-9))
    loss_given_default = float(contagion_config["loss_given_default"])
    max_rounds = int(contagion_config["max_contagion_rounds"])

    if not contagion_config.get("enabled", True):
        out["interbank_contagion_loss"] = 0.0
        out["new_terminal_from_interbank"] = False

        out["terminal_distress_after_interbank"] = (
            out["terminal_distress_after_fire_sale"].fillna(False).astype(bool)
            if "terminal_distress_after_fire_sale" in out.columns
            else False
        )

        out = _recompute_post_interbank_metrics(out, lcr_assumptions)

        summary = pd.DataFrame([{
            "stress_month": int(out["stress_month"].iloc[0]) if "stress_month" in out.columns else np.nan,
            "alpha": float(out["first_round_alpha"].iloc[0]) if "first_round_alpha" in out.columns else np.nan,
            "initial_terminal_debtors": int(out["terminal_distress_after_interbank"].sum()),
            "interbank_contagion_rounds": 0,
            "total_interbank_contagion_loss": 0.0,
            "new_terminal_banks_from_interbank": 0,
            "terminal_distress_after_interbank": int(out["terminal_distress_after_interbank"].sum()),
        }])

        return out, summary

    if exposure_matrix is None:
        raise ValueError(
            "exposure_matrix must be provided to apply_interbank_contagion. "
            "Use get_or_create_interbank_exposure_matrix(...) before calling the engine."
        )

    out["bank_id"] = out["bank_id"].astype(str)
    out = out.set_index("bank_id", drop=False)

    exposure_matrix = exposure_matrix.copy()
    exposure_matrix.index = exposure_matrix.index.astype(str)
    exposure_matrix.columns = exposure_matrix.columns.astype(str)

    if "terminal_distress_after_fire_sale" not in out.columns:
        out["terminal_distress_after_fire_sale"] = False

    if "final_unresolved_shortfall" not in out.columns:
        out["final_unresolved_shortfall"] = False

    if "post_fire_sale_insolvent" not in out.columns:
        out["post_fire_sale_insolvent"] = False

    # Liquidity-based terminal distress must be computed regardless of whether
    # post_fire_sale_insolvent already exists. In the full engine it usually
    # exists, so keeping this block inside the if above creates a KeyError.
    resolution_floor = float(
        contagion_config.get("liquidity_resolution_floor", 1.00)
    )
    collateral_tol = float(
        contagion_config.get("collateral_exhaustion_tol", 1e-6)
    )

    post_support_lcr = pd.to_numeric(
        out["post_cb_support_LCR"]
        if "post_cb_support_LCR" in out.columns
        else out.get("LCR"),
        errors="coerce",
    ).fillna(np.inf)

    remaining_collateral = pd.to_numeric(
        out["post_cb_unencumbered_collateral_capacity"]
        if "post_cb_unencumbered_collateral_capacity" in out.columns
        else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)

    out["terminally_illiquid_after_lolr"] = (
        (post_support_lcr < resolution_floor)
        & (remaining_collateral <= collateral_tol)
    )

    out["terminal_distress_after_interbank"] = (
        out["terminal_distress_after_fire_sale"].fillna(False).astype(bool)
        | out["final_unresolved_shortfall"].fillna(False).astype(bool)
        | out["post_fire_sale_insolvent"].fillna(False).astype(bool)
        | out["terminally_illiquid_after_lolr"].fillna(False).astype(bool)
    )

    out["interbank_contagion_loss"] = 0.0
    out["new_terminal_from_interbank"] = False

    initial_terminal_ids = set(
        out.index[out["terminal_distress_after_interbank"].astype(bool)]
    )

    terminal_ids = set(initial_terminal_ids)
    processed_debtors: set[str] = set()

    total_loss = 0.0
    rounds_used = 0

    for round_number in range(1, max_rounds + 1):
        new_debtors = sorted(terminal_ids - processed_debtors)

        if not new_debtors:
            break

        processed_debtors.update(new_debtors)

        losses = pd.Series(0.0, index=out.index)

        for debtor_id in new_debtors:
            if debtor_id not in exposure_matrix.columns:
                continue

            debtor_exposures = exposure_matrix[debtor_id].reindex(out.index).fillna(0.0)
            losses = losses + loss_given_default * debtor_exposures

        # Do not spend effort updating banks already terminal before this round.
        active_creditors = ~out["terminal_distress_after_interbank"].astype(bool)
        losses = losses.where(active_creditors, 0.0)

        # Cap loss by current interbank assets.
        current_interbank_assets = out["interbank_assets"].astype(float).clip(lower=0.0)
        losses = np.minimum(losses, current_interbank_assets)

        round_loss = float(losses.sum())

        if round_loss <= tolerance:
            continue

        rounds_used = round_number
        total_loss += round_loss

        loss_column = f"interbank_contagion_loss_round_{round_number}"
        out[loss_column] = losses

        out["interbank_assets"] = np.maximum(
            out["interbank_assets"].astype(float) - losses,
            0.0,
        )

        out["interbank_contagion_loss"] = (
            out["interbank_contagion_loss"].astype(float) + losses
        )

        out = _recompute_post_interbank_metrics(
            banks=out.reset_index(drop=True),
            lcr_assumptions=lcr_assumptions,
        ).set_index("bank_id", drop=False)

        newly_insolvent = (
            out["post_interbank_insolvent"].fillna(False).astype(bool)
            & ~out["terminal_distress_after_interbank"].fillna(False).astype(bool)
        )

        if newly_insolvent.any():
            new_terminal_ids = set(out.index[newly_insolvent])
            terminal_ids.update(new_terminal_ids)

            out.loc[list(new_terminal_ids), "new_terminal_from_interbank"] = True

        out["terminal_distress_after_interbank"] = (
            out["terminal_distress_after_interbank"].fillna(False).astype(bool)
            | out["post_interbank_insolvent"].fillna(False).astype(bool)
        )

    out = _recompute_post_interbank_metrics(
        banks=out.reset_index(drop=True),
        lcr_assumptions=lcr_assumptions,
    )

    out["terminal_distress_after_interbank"] = (
        out["terminal_distress_after_interbank"].fillna(False).astype(bool)
        | out["post_interbank_insolvent"].fillna(False).astype(bool)
    )

    out["interbank_contagion_loss_assets_ratio"] = np.where(
        out["total_assets"] > 0,
        out["interbank_contagion_loss"] / out["total_assets"],
        0.0,
    )

    new_terminal_from_interbank_count = int(out["new_terminal_from_interbank"].sum())

    summary = pd.DataFrame([{
        "stress_month": int(out["stress_month"].iloc[0]) if "stress_month" in out.columns else np.nan,
        "alpha": float(out["first_round_alpha"].iloc[0]) if "first_round_alpha" in out.columns else np.nan,
        "initial_terminal_debtors": len(initial_terminal_ids),
        "interbank_contagion_rounds": rounds_used,
        "total_interbank_contagion_loss": total_loss,
        "new_terminal_banks_from_interbank": new_terminal_from_interbank_count,
        "terminal_distress_after_interbank": int(out["terminal_distress_after_interbank"].sum()),
    }])

    return out, summary