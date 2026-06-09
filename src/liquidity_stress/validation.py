"""
Validation checks for generated or loaded banking data.
"""

import numpy as np
import pandas as pd


def run_baseline_sanity_checks(
    banks: pd.DataFrame,
    tolerance: float = 1e-6,
    public_balance_tolerance: float = 1.0,
) -> pd.DataFrame:
    """
    Run concrete accounting and liquidity sanity checks.

    Checks
    ------
    1. Assets = liabilities + equity
    2. Asset components sum to total assets
    3. Deposit components sum to deposits
    4. Liability components sum to total liabilities
    5. AMD + FX assets sum to total assets
    6. AMD + FX liabilities sum to total liabilities
    7. LCR = HQLA / NCOF
    """
    checks = []

    asset_component_sum = (
        banks["cash_reserves"]
        + banks["loans"]
        + banks["total_securities"]
        + banks["interbank_assets"]
    )

    deposit_component_sum = (
        banks["retail_deposits"]
        + banks["corporate_deposits"]
        + banks["other_deposits"]
    )

    liability_component_sum = (
        deposit_component_sum
        + banks["wholesale_funding"]
        + banks["interbank_liabilities"]
    )

    check_specs = [
        (
            "assets_equal_liabilities_plus_equity",
            banks["total_assets"] - (banks["total_liabilities"] + banks["equity"]),
            public_balance_tolerance,
        ),
        (
            "asset_components_sum_to_total_assets",
            banks["total_assets"] - asset_component_sum,
            tolerance,
        ),
        (
            "deposit_components_sum_to_deposits",
            banks["deposits"] - deposit_component_sum,
            tolerance,
        ),
        (
            "liability_components_sum_to_total_liabilities",
            banks["total_liabilities"] - liability_component_sum,
            tolerance,
        ),
        (
            "AMD_FX_assets_sum_to_total_assets",
            banks["total_assets"] - (banks["AMD_assets"] + banks["FX_assets"]),
            tolerance,
        ),
        (
            "AMD_FX_liabilities_sum_to_total_liabilities",
            banks["total_liabilities"] - (banks["AMD_liabilities"] + banks["FX_liabilities"]),
            tolerance,
        ),
        (
            "LCR_equals_HQLA_over_NCOF",
            banks["LCR"] - (banks["HQLA"] / banks["NCOF"]),
            tolerance,
        ),
    ]

    for check_name, error_series, check_tolerance in check_specs:
        max_abs_error = float(np.max(np.abs(error_series)))

        checks.append({
            "check": check_name,
            "passed": bool(max_abs_error <= check_tolerance),
            "max_abs_error": max_abs_error,
            "tolerance": check_tolerance,
        })

    return pd.DataFrame(checks)


def run_currency_sheet_consistency_checks(
    banks: pd.DataFrame,
    banks_amd: pd.DataFrame,
    banks_fx: pd.DataFrame,
    tolerance: float = 1e-6,
    public_balance_tolerance: float = 1.0,
) -> pd.DataFrame:
    """
    Check that banks_amd + banks_fx approximately reconstructs banks.

    This is a level-variable check only. Ratios such as LCR, HQLA/assets,
    equity/assets, and market shares are intentionally excluded because they
    are not additive.
    """
    combined = banks.copy().set_index("bank_id")
    amd = banks_amd.copy().set_index("bank_id")
    fx = banks_fx.copy().set_index("bank_id")

    common_ids = combined.index.intersection(amd.index).intersection(fx.index)

    if len(common_ids) != len(combined.index):
        missing_from_amd = sorted(set(combined.index) - set(amd.index))
        missing_from_fx = sorted(set(combined.index) - set(fx.index))

        return pd.DataFrame([{
            "check": "currency_sheets_have_same_bank_ids",
            "passed": False,
            "max_abs_error": np.nan,
            "tolerance": np.nan,
            "detail": (
                f"missing_from_amd={missing_from_amd}; "
                f"missing_from_fx={missing_from_fx}"
            ),
        }])

    combined = combined.loc[common_ids]
    amd = amd.loc[common_ids]
    fx = fx.loc[common_ids]

    additive_columns = [
        "total_assets",
        "interest_earning_assets",
        "gross_advances_to_customers",
        "loans",
        "total_liabilities",
        "deposits",
        "equity",
        "charter_capital",
        "cash_reserves",
        "total_securities",
        "interbank_assets",
        "government_securities",
        "bank_securities",
        "corporate_securities",
        "retail_deposits",
        "corporate_deposits",
        "other_deposits",
        "wholesale_funding",
        "interbank_liabilities",
        "HQLA",
        "gross_30d_outflows",
        "recognized_30d_inflows",
        "capped_30d_inflows",
        "NCOF",
    ]

    rows = []

    for column in additive_columns:
        if column not in combined.columns:
            continue

        if column not in amd.columns or column not in fx.columns:
            rows.append({
                "check": f"{column}_available_in_all_currency_sheets",
                "passed": False,
                "max_abs_error": np.nan,
                "tolerance": np.nan,
                "detail": "Column missing from banks_amd or banks_fx.",
            })
            continue

        error = combined[column].astype(float) - (
            amd[column].astype(float) + fx[column].astype(float)
        )

        check_tolerance = (
            public_balance_tolerance
            if column in ["total_assets", "total_liabilities", "equity"]
            else tolerance
        )

        # Avoid false failures from machine precision around public rounding gaps.
        comparison_tolerance = check_tolerance + 1e-9

        max_abs_error = float(np.max(np.abs(error)))

        rows.append({
            "check": f"{column}_combined_equals_AMD_plus_FX",
            "passed": bool(max_abs_error <= comparison_tolerance),
            "max_abs_error": max_abs_error,
            "tolerance": check_tolerance,
            "detail": "",
        })

    return pd.DataFrame(rows)

def run_behavioral_invariants(
    results: pd.DataFrame,
    contagion_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    checks = []
    by_month = results.groupby("stress_month")["interbank_assets"].sum()
    core_share = float(by_month.min() / by_month.iloc[0]) if by_month.iloc[0] > 0 else 0.0
    checks.append({
        "check": "interbank_network_not_drained",
        "passed": bool(core_share >= 0.30),
        "value": core_share,
    })
    if "stressed_LCR" in results.columns:
        peak_month = results.loc[results["first_round_alpha"].idxmax(), "stress_month"]
        m1 = results[results["stress_month"] == results["stress_month"].min()]["stressed_LCR"].min()
        mp = results[results["stress_month"] == peak_month]["stressed_LCR"].min()
        checks.append({
            "check": "stressed_LCR_responds_to_alpha",
            "passed": bool(mp < m1),
            "value": float(mp - m1),
        })
    if contagion_summary is not None and "total_interbank_contagion_loss" in contagion_summary.columns:
        total = float(contagion_summary["total_interbank_contagion_loss"].sum())
        checks.append({
            "check": "interbank_channel_can_fire",
            "passed": bool(total > 0.0),  # assert this against the SEVERE scenario
            "value": total,
        })
    return pd.DataFrame(checks)