"""
Stress-vector utilities.

This module applies scenario severity parameters to the base first-round
stress vector.

The stress vector contains LCR-style runoff rates, asset haircut rates,
inflow reduction rates, and FX stress rates. Alpha scales these rates by
scenario or by month.
"""

import numpy as np
import pandas as pd


def apply_alpha_to_stress_vector(
    stress_vector: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """
    Apply scalar shock severity alpha to the base stress vector.

    Formula
    -------
    effective_weight = min(alpha * base_weight, cap)

    Parameters
    ----------
    stress_vector : pd.DataFrame
        Base stress vector with columns:
        component, shock_type, base_weight, cap.
    alpha : float
        Shock severity multiplier.

    Returns
    -------
    pd.DataFrame
        Stress vector with alpha and effective_weight columns.
    """
    out = stress_vector.copy()

    out["alpha"] = float(alpha)
    out["effective_weight"] = np.minimum(
        float(alpha) * out["base_weight"],
        out["cap"],
    )

    return out


def get_effective_weight(
    effective_vector: pd.DataFrame,
    component: str,
) -> float:
    """
    Return the effective stress weight for one component.
    """
    matched = effective_vector.loc[
        effective_vector["component"] == component,
        "effective_weight",
    ]

    if len(matched) != 1:
        raise ValueError(f"Component '{component}' not found exactly once.")

    return float(matched.iloc[0])


def build_monthly_stress_vectors(
    stress_vector: pd.DataFrame,
    alpha_path: list[float],
) -> dict[int, pd.DataFrame]:
    """
    Build one effective stress vector per stress month.

    Parameters
    ----------
    stress_vector : pd.DataFrame
        Base first-round stress vector.
    alpha_path : list[float]
        Monthly alpha values.

    Returns
    -------
    dict[int, pd.DataFrame]
        Mapping from month number to effective stress vector.
    """
    monthly_vectors = {}

    for month, alpha in enumerate(alpha_path, start=1):
        monthly_vectors[month] = apply_alpha_to_stress_vector(
            stress_vector=stress_vector,
            alpha=alpha,
        )

    return monthly_vectors