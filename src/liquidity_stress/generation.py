"""
Synthetic bank-level data generation.

This module generates a synthetic banking input table from public Armenian
banking-sector reference data. It is used only for the synthetic/demo input
file. The core stress-testing model should later read the generated Excel file,
not depend directly on this module.
"""

import numpy as np
import pandas as pd

from liquidity_stress.config import (
    LCR_ASSUMPTIONS,
    FX_ASSUMPTIONS,
    BASELINE_LCR_TARGETS,
    BASELINE_LCR_HARD_FLOOR,
    BASELINE_LCR_HARD_CAP,
)

from liquidity_stress.reference_data import add_derived_bank_reference_fields
from liquidity_stress.utils import make_rng


def assign_bank_type_from_public_reference(row: pd.Series) -> str:
    """
    Assign an interpretable synthetic bank type from public balance-sheet ratios.

    These are modelling labels, not official classifications.
    """
    market_share = float(row["market_share"])
    loans_assets = float(row["loans_assets_ratio"])
    deposits_assets = float(row["deposits_assets_ratio"])

    if market_share >= 0.10:
        return "large_universal"

    if deposits_assets >= 0.65:
        return "retail_deposit_heavy"

    if loans_assets >= 0.68:
        return "loan_heavy"

    if loans_assets <= 0.35:
        return "securities_interbank_active"

    return "mid_universal"

def draw_target_baseline_lcr(
    bank_type: str,
    rng: np.random.Generator,
    baseline_lcr_targets: dict = BASELINE_LCR_TARGETS,
    hard_floor: float = BASELINE_LCR_HARD_FLOOR,
    hard_cap: float = BASELINE_LCR_HARD_CAP,
) -> float:
    """
    Draw a plausible synthetic baseline LCR target.

    The generated dataset is synthetic/demo data. Public balance sheets give
    headline assets, liabilities, loans and deposits, but not bank-level
    30-day NCOF. Therefore NCOF is calibrated so baseline LCRs are plausible
    relative to Armenia's 120% requirement.
    """
    low, high = baseline_lcr_targets.get(
        bank_type,
        baseline_lcr_targets.get("default", (1.35, 1.80)),
    )

    target = float(rng.uniform(low, high))
    return float(np.clip(target, hard_floor, hard_cap))


def recalibrate_ncof_to_target_lcr(
    liquidity_metrics: dict,
    target_lcr: float,
) -> dict:
    """
    Scale synthetic NCOF so that HQLA / NCOF equals target_lcr.

    HQLA is kept balance-sheet-driven. NCOF is synthetic and can be calibrated
    without changing public balance-sheet totals.
    """
    out = dict(liquidity_metrics)

    hqla = float(out["HQLA"])
    old_ncof = max(float(out["NCOF"]), 1.0)
    target_lcr = max(float(target_lcr), 1e-9)

    new_ncof = max(hqla / target_lcr, 1.0)
    scale = new_ncof / old_ncof

    out["NCOF"] = float(new_ncof)
    out["LCR"] = float(hqla / new_ncof)

    for column in [
        "gross_30d_outflows",
        "recognized_30d_inflows",
        "capped_30d_inflows",
        "wholesale_funding_30d",
        "interbank_liabilities_30d",
    ]:
        if column in out:
            out[column] = float(out[column]) * scale

    out["synthetic_target_LCR"] = float(target_lcr)
    out["synthetic_ncof_calibration_scale"] = float(scale)

    return out

def allocate_residual_assets(
    residual_assets: float,
    bank_type: str,
    rng: np.random.Generator,
) -> dict:
    """
    Allocate non-loan assets into cash/reserves, securities, and interbank assets.

    Public data provide total assets and loans. The residual is synthetically
    decomposed while preserving actual public headline values.
    """
    if residual_assets < 0:
        raise ValueError("Residual assets cannot be negative.")

    if bank_type == "large_universal":
        weights = np.array([0.34, 0.54, 0.12])
    elif bank_type == "retail_deposit_heavy":
        weights = np.array([0.42, 0.48, 0.10])
    elif bank_type == "loan_heavy":
        weights = np.array([0.50, 0.40, 0.10])
    elif bank_type == "securities_interbank_active":
        weights = np.array([0.24, 0.60, 0.16])
    else:
        weights = np.array([0.36, 0.52, 0.12])

    noise = rng.normal(loc=1.0, scale=0.08, size=3)
    weights = np.maximum(weights * noise, 0.01)
    weights = weights / weights.sum()

    return {
        "cash_reserves": residual_assets * weights[0],
        "total_securities": residual_assets * weights[1],
        "interbank_assets": residual_assets * weights[2],
    }


def split_securities(
    total_securities: float,
    bank_type: str,
    rng: np.random.Generator,
) -> dict:
    """
    Split total securities into government, bank, and corporate securities.
    """
    if bank_type == "securities_interbank_active":
        base = np.array([0.60, 0.25, 0.15])
    elif bank_type == "large_universal":
        base = np.array([0.68, 0.18, 0.14])
    else:
        base = np.array([0.72, 0.16, 0.12])

    noise = rng.normal(loc=1.0, scale=0.07, size=3)
    weights = np.maximum(base * noise, 0.01)
    weights = weights / weights.sum()

    return {
        "government_securities": total_securities * weights[0],
        "bank_securities": total_securities * weights[1],
        "corporate_securities": total_securities * weights[2],
    }


def decompose_deposits_and_other_liabilities(
    deposits: float,
    total_liabilities: float,
    bank_type: str,
    system_reference: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Split public deposits and non-deposit liabilities into stress-relevant buckets.
    """
    other_liabilities = max(total_liabilities - deposits, 0.0)

    retail_base = system_reference["household_resident_deposit_share"]
    corporate_base = system_reference["private_company_resident_deposit_share"]

    if bank_type == "retail_deposit_heavy":
        retail_weight = retail_base * 1.18
        corporate_weight = corporate_base * 0.88
    elif bank_type == "large_universal":
        retail_weight = retail_base * 0.98
        corporate_weight = corporate_base * 1.05
    elif bank_type == "securities_interbank_active":
        retail_weight = retail_base * 0.85
        corporate_weight = corporate_base * 1.10
    else:
        retail_weight = retail_base
        corporate_weight = corporate_base

    other_deposit_weight = max(1.0 - retail_weight - corporate_weight, 0.05)

    deposit_weights = np.array([
        retail_weight,
        corporate_weight,
        other_deposit_weight,
    ])

    deposit_weights = np.maximum(
        deposit_weights * rng.normal(1.0, 0.06, size=3),
        0.01,
    )
    deposit_weights = deposit_weights / deposit_weights.sum()

    retail_deposits = deposits * deposit_weights[0]
    corporate_deposits = deposits * deposit_weights[1]
    other_deposits = deposits * deposit_weights[2]

    if bank_type == "securities_interbank_active":
        interbank_share = 0.32
    elif bank_type == "large_universal":
        interbank_share = 0.24
    else:
        interbank_share = 0.18

    interbank_share = float(np.clip(interbank_share * rng.normal(1.0, 0.15), 0.05, 0.50))

    interbank_liabilities = other_liabilities * interbank_share
    wholesale_funding = other_liabilities - interbank_liabilities

    return {
        "retail_deposits": retail_deposits,
        "corporate_deposits": corporate_deposits,
        "other_deposits": other_deposits,
        "wholesale_funding": wholesale_funding,
        "interbank_liabilities": interbank_liabilities,
    }


def assign_currency_structure(
    total_assets: float,
    total_liabilities: float,
    bank_type: str,
    rng: np.random.Generator,
    fx_assumptions: dict = FX_ASSUMPTIONS,
) -> dict:
    """
    Assign AMD/FX asset and liability decomposition.

    FX liabilities are anchored to the system FX deposit share. FX assets are
    centered around FX liabilities in level terms to avoid mechanically forcing
    every bank into the same currency gap.
    """
    fx_liab_anchor = fx_assumptions["system_fx_liability_share_anchor"]

    if bank_type == "large_universal":
        fx_liab_multiplier = 1.00
    elif bank_type == "retail_deposit_heavy":
        fx_liab_multiplier = 0.90
    elif bank_type == "securities_interbank_active":
        fx_liab_multiplier = 1.10
    else:
        fx_liab_multiplier = 1.00

    fx_liability_share = fx_liab_anchor * fx_liab_multiplier * rng.normal(1.0, 0.10)
    fx_liability_share = float(np.clip(
        fx_liability_share,
        fx_assumptions["min_fx_liability_share"],
        fx_assumptions["max_fx_liability_share"],
    ))

    fx_liabilities = total_liabilities * fx_liability_share

    fx_asset_to_fx_liability = rng.normal(
        fx_assumptions["fx_asset_to_fx_liability_center"],
        fx_assumptions["fx_asset_to_fx_liability_sd"],
    )
    fx_asset_to_fx_liability = float(np.clip(fx_asset_to_fx_liability, 0.75, 1.25))

    fx_assets = fx_liabilities * fx_asset_to_fx_liability
    fx_assets = float(np.clip(
        fx_assets,
        fx_assumptions["min_fx_asset_share"] * total_assets,
        fx_assumptions["max_fx_asset_share"] * total_assets,
    ))

    amd_assets = total_assets - fx_assets
    amd_liabilities = total_liabilities - fx_liabilities

    return {
        "AMD_assets": amd_assets,
        "FX_assets": fx_assets,
        "AMD_liabilities": amd_liabilities,
        "FX_liabilities": fx_liabilities,
        "FX_asset_share": fx_assets / total_assets,
        "FX_liability_share": fx_liabilities / total_liabilities,
        "net_fx_gap": fx_assets - fx_liabilities,
        "net_fx_gap_assets_ratio": (fx_assets - fx_liabilities) / total_assets,
    }


def compute_accounting_hqla(
    cash_reserves: float,
    government_securities: float,
    bank_securities: float,
    corporate_securities: float,
    total_assets: float,
) -> float:
    """
    Compute synthetic HQLA from asset components.
    """
    hqla = (
        cash_reserves
        + 0.90 * government_securities
        + 0.35 * bank_securities
        + 0.10 * corporate_securities
    )

    return float(np.clip(hqla, 0.0, total_assets))


def compute_synthetic_ncof(
    retail_deposits: float,
    corporate_deposits: float,
    other_deposits: float,
    wholesale_funding: float,
    interbank_liabilities: float,
    loans: float,
    interbank_assets: float,
    lcr_assumptions: dict = LCR_ASSUMPTIONS,
) -> dict:
    """
    Compute synthetic 30-day net cash outflows.

    Purpose
    -------
    Constructs a mechanically consistent LCR denominator from the synthetic
    balance-sheet structure.

    Important
    ---------
    Retail, corporate, and other deposits enter through runoff rates applied
    to the deposit stock. For wholesale funding and interbank liabilities,
    only the assumed 30-day maturing share enters the baseline LCR denominator.

    This avoids treating all wholesale/interbank funding as immediately
    maturing inside the 30-day LCR horizon.

    Formula
    -------
    gross_outflows = deposit runoff
                   + wholesale runoff on 30-day maturing wholesale funding
                   + interbank runoff on 30-day maturing interbank liabilities

    recognized_inflows = recognized loan and interbank inflows
    capped_inflows = min(recognized_inflows, 75% of gross_outflows)
    NCOF = gross_outflows - capped_inflows

    Returns
    -------
    dict
        Gross outflows, recognized inflows, capped inflows, and NCOF.
    """
    wholesale_30d = (
        lcr_assumptions["wholesale_funding_30d_share"]
        * wholesale_funding
    )

    interbank_liabilities_30d = (
        lcr_assumptions["interbank_liability_30d_share"]
        * interbank_liabilities
    )

    gross_outflows = (
        lcr_assumptions["retail_deposit_runoff"] * retail_deposits
        + lcr_assumptions["corporate_deposit_runoff"] * corporate_deposits
        + lcr_assumptions["other_deposit_runoff"] * other_deposits
        + lcr_assumptions["wholesale_funding_runoff"] * wholesale_30d
        + lcr_assumptions["interbank_liability_runoff"] * interbank_liabilities_30d
    )

    loan_inflows = (
        lcr_assumptions["loan_maturing_30d_share"]
        * lcr_assumptions["loan_inflow_recognition_rate"]
        * loans
    )

    interbank_inflows = (
        lcr_assumptions["interbank_asset_inflow_rate"]
        * interbank_assets
    )

    recognized_inflows = loan_inflows + interbank_inflows

    capped_inflows = min(
        recognized_inflows,
        lcr_assumptions["inflow_cap_share_of_outflows"] * gross_outflows,
    )

    ncof = gross_outflows - capped_inflows
    ncof = max(ncof, lcr_assumptions["minimum_ncof"])

    return {
        "gross_30d_outflows": float(gross_outflows),
        "recognized_30d_inflows": float(recognized_inflows),
        "capped_30d_inflows": float(capped_inflows),
        "NCOF": float(ncof),
        "wholesale_funding_30d": float(wholesale_30d),
        "interbank_liabilities_30d": float(interbank_liabilities_30d),
    }



def assign_liquidity_metrics(
    total_assets: float,
    cash_reserves: float,
    government_securities: float,
    bank_securities: float,
    corporate_securities: float,
    retail_deposits: float,
    corporate_deposits: float,
    other_deposits: float,
    wholesale_funding: float,
    interbank_liabilities: float,
    loans: float,
    interbank_assets: float,
    lcr_assumptions: dict = LCR_ASSUMPTIONS,
) -> dict:
    """
    Assign HQLA, NCOF, and baseline LCR from balance-sheet components.

    Final identity:
        LCR = HQLA / NCOF
    """
    hqla = compute_accounting_hqla(
        cash_reserves=cash_reserves,
        government_securities=government_securities,
        bank_securities=bank_securities,
        corporate_securities=corporate_securities,
        total_assets=total_assets,
    )

    ncof_detail = compute_synthetic_ncof(
        retail_deposits=retail_deposits,
        corporate_deposits=corporate_deposits,
        other_deposits=other_deposits,
        wholesale_funding=wholesale_funding,
        interbank_liabilities=interbank_liabilities,
        loans=loans,
        interbank_assets=interbank_assets,
        lcr_assumptions=lcr_assumptions,
    )

    ncof = ncof_detail["NCOF"]

    return {
        "HQLA": float(hqla),
        "HQLA_assets_ratio": float(hqla / total_assets),
        "NCOF": float(ncof),
        "gross_30d_outflows": ncof_detail["gross_30d_outflows"],
        "recognized_30d_inflows": ncof_detail["recognized_30d_inflows"],
        "capped_30d_inflows": ncof_detail["capped_30d_inflows"],
        "wholesale_funding_30d": ncof_detail["wholesale_funding_30d"],
        "interbank_liabilities_30d": ncof_detail["interbank_liabilities_30d"],
        "LCR": float(hqla / ncof),
    }

def generate_armenian_public_reference_banks(
    bank_reference: pd.DataFrame,
    system_reference: dict,
    system_config: dict,
) -> pd.DataFrame:
    """
    Generate Armenia-mimicking bank-level synthetic data from public references.

    Public variables are preserved. Missing liquidity and stress-testing
    components are generated synthetically around the public reference values.
    """
    rng = make_rng(system_config["random_seed"])

    reference = add_derived_bank_reference_fields(bank_reference)
    rows = []

    for _, row in reference.iterrows():
        bank_type = assign_bank_type_from_public_reference(row)

        total_assets = float(row["total_assets"])
        loans = float(row["loans"])
        total_liabilities = float(row["total_liabilities"])
        deposits = float(row["deposits"])
        equity = float(row["equity"])

        residual_assets = total_assets - loans

        residual_decomposition = allocate_residual_assets(
            residual_assets=residual_assets,
            bank_type=bank_type,
            rng=rng,
        )

        securities_split = split_securities(
            total_securities=residual_decomposition["total_securities"],
            bank_type=bank_type,
            rng=rng,
        )

        liability_decomposition = decompose_deposits_and_other_liabilities(
            deposits=deposits,
            total_liabilities=total_liabilities,
            bank_type=bank_type,
            system_reference=system_reference,
            rng=rng,
        )

        currency_structure = assign_currency_structure(
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            bank_type=bank_type,
            rng=rng,
        )

        liquidity_metrics = assign_liquidity_metrics(
            total_assets=total_assets,
            cash_reserves=residual_decomposition["cash_reserves"],
            government_securities=securities_split["government_securities"],
            bank_securities=securities_split["bank_securities"],
            corporate_securities=securities_split["corporate_securities"],
            retail_deposits=liability_decomposition["retail_deposits"],
            corporate_deposits=liability_decomposition["corporate_deposits"],
            other_deposits=liability_decomposition["other_deposits"],
            wholesale_funding=liability_decomposition["wholesale_funding"],
            interbank_liabilities=liability_decomposition["interbank_liabilities"],
            loans=loans,
            interbank_assets=residual_decomposition["interbank_assets"],
        )

        if system_config.get("use_synthetic_ncof_calibration", True):
            target_lcr = draw_target_baseline_lcr(bank_type=bank_type, rng=rng)
            liquidity_metrics = recalibrate_ncof_to_target_lcr(
                liquidity_metrics=liquidity_metrics,
                target_lcr=target_lcr,
            )
        else:
            liquidity_metrics["synthetic_target_LCR"] = float("nan")
            liquidity_metrics["synthetic_ncof_calibration_scale"] = 1.0

        rows.append({
            "bank_id": row["bank_id"],
            "bank_name": row["bank_name"],
            "legal_form": row["legal_form"],
            "bank_type": bank_type,

            "total_assets": total_assets,
            "interest_earning_assets": float(row["interest_earning_assets"]),
            "gross_advances_to_customers": float(row["gross_advances_to_customers"]),
            "loans": loans,
            "total_liabilities": total_liabilities,
            "deposits": deposits,
            "equity": equity,
            "charter_capital": float(row["charter_capital"]),
            "market_share": float(row["market_share"]),

            "cash_reserves": residual_decomposition["cash_reserves"],
            "total_securities": residual_decomposition["total_securities"],
            "interbank_assets": residual_decomposition["interbank_assets"],

            "government_securities": securities_split["government_securities"],
            "bank_securities": securities_split["bank_securities"],
            "corporate_securities": securities_split["corporate_securities"],

            "retail_deposits": liability_decomposition["retail_deposits"],
            "corporate_deposits": liability_decomposition["corporate_deposits"],
            "other_deposits": liability_decomposition["other_deposits"],
            "wholesale_funding": liability_decomposition["wholesale_funding"],
            "interbank_liabilities": liability_decomposition["interbank_liabilities"],

            "AMD_assets": currency_structure["AMD_assets"],
            "FX_assets": currency_structure["FX_assets"],
            "AMD_liabilities": currency_structure["AMD_liabilities"],
            "FX_liabilities": currency_structure["FX_liabilities"],
            "FX_asset_share": currency_structure["FX_asset_share"],
            "FX_liability_share": currency_structure["FX_liability_share"],
            "net_fx_gap": currency_structure["net_fx_gap"],
            "net_fx_gap_assets_ratio": currency_structure["net_fx_gap_assets_ratio"],

            "HQLA": liquidity_metrics["HQLA"],
            "HQLA_assets_ratio": liquidity_metrics["HQLA_assets_ratio"],
            "NCOF": liquidity_metrics["NCOF"],
            "gross_30d_outflows": liquidity_metrics["gross_30d_outflows"],
            "recognized_30d_inflows": liquidity_metrics["recognized_30d_inflows"],
            "capped_30d_inflows": liquidity_metrics["capped_30d_inflows"],
            "wholesale_funding_30d": liquidity_metrics["wholesale_funding_30d"],
            "interbank_liabilities_30d": liquidity_metrics["interbank_liabilities_30d"],
            "LCR": liquidity_metrics["LCR"],

            "synthetic_target_LCR": liquidity_metrics["synthetic_target_LCR"],
            "synthetic_ncof_calibration_scale": liquidity_metrics[
                "synthetic_ncof_calibration_scale"
            ],

            "loans_assets_ratio": loans / total_assets,
            "deposits_assets_ratio": deposits / total_assets,
            "equity_assets_ratio": equity / total_assets,
            "deposits_liabilities_ratio": (
                deposits / total_liabilities if total_liabilities > 0 else np.nan
            ),
        })

    return pd.DataFrame(rows)