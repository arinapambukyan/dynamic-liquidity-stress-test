from __future__ import annotations

import numpy as np
import pandas as pd


# Indirect contagion is state-bond-only in the current data setup.
TRADABLE_ASSET_COLUMNS = [
    "government_securities",
]


def _default_market_depths(
    banks: pd.DataFrame,
    asset_columns: list[str],
    market_depth_multiplier: float = 1.0,
) -> dict[str, float]:
    """
    Build simple market-depth proxies when no external market-depth data exists.

    For the Armenian synthetic setup, we use:
        D_a = market_depth_multiplier * total modeled holdings of asset a

    This is not an empirical market-depth estimate. It is a transparent
    normalization that can later be replaced by real market-depth data.
    """
    depths = {}

    for asset in asset_columns:
        total_holding = float(banks[asset].sum())
        depths[asset] = max(float(market_depth_multiplier) * total_holding, 1e-9)

    return depths


def build_liquidity_weighted_holdings_matrix(
    banks: pd.DataFrame,
    asset_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a bank-by-asset holdings matrix for indirect contagion.

    Current model assumption:
        only government securities / state bonds are used for indirect contagion.
    """
    if asset_columns is None:
        asset_columns = TRADABLE_ASSET_COLUMNS.copy()

    missing = [col for col in asset_columns if col not in banks.columns]
    if missing:
        raise ValueError(f"Missing tradable asset columns: {missing}")

    holdings = banks[["bank_id"] + asset_columns].copy()

    for col in asset_columns:
        holdings[col] = holdings[col].astype(float)

    holdings = holdings.set_index("bank_id")

    return holdings


def compute_overlap_matrix(
    banks: pd.DataFrame,
    asset_columns: list[str] | None = None,
    market_depths: dict[str, float] | None = None,
    market_depth_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Compute the liquidity-weighted overlap matrix Omega.

    Paper-style logic:
        Omega_ij = sum_a holdings_i,a * holdings_j,a / D_a

    Current model simplification:
        asset_columns = ["government_securities"]

    Therefore:
        Omega_ij = government_securities_i * government_securities_j / D_state_bonds

    This is the indirect contagion network: two banks are linked if they are
    exposed to the same state-bond market.
    """
    if asset_columns is None:
        asset_columns = TRADABLE_ASSET_COLUMNS.copy()

    holdings = build_liquidity_weighted_holdings_matrix(
        banks=banks,
        asset_columns=asset_columns,
    )

    if market_depths is None:
        market_depths = _default_market_depths(
            banks=banks,
            asset_columns=asset_columns,
            market_depth_multiplier=market_depth_multiplier,
        )

    bank_ids = holdings.index.tolist()
    n = len(bank_ids)

    omega = np.zeros((n, n), dtype=float)

    for asset in asset_columns:
        values = holdings[asset].to_numpy(dtype=float)
        depth = max(float(market_depths.get(asset, 0.0)), 1e-9)

        omega += np.outer(values, values) / depth

    return pd.DataFrame(omega, index=bank_ids, columns=bank_ids)


def compute_overlap_systemic_importance(omega: pd.DataFrame) -> pd.Series:
    """
    Overlap-network systemic importance:
        SI_i = sum_{j != i} Omega_ij
    """
    diag = np.diag(omega.to_numpy(dtype=float))
    importance = omega.sum(axis=1) - diag

    return pd.Series(
        importance,
        index=omega.index,
        name="systemic_importance",
    )


def build_overlap_edge_table(
    omega: pd.DataFrame,
    min_overlap: float = 0.0,
) -> pd.DataFrame:
    """
    Convert overlap matrix to undirected edge table.
    """
    rows = []
    bank_ids = omega.index.tolist()

    for i, src in enumerate(bank_ids):
        for j, tgt in enumerate(bank_ids):
            if j <= i:
                continue

            overlap = float(omega.loc[src, tgt])

            if overlap > min_overlap:
                rows.append({
                    "source": src,
                    "target": tgt,
                    "overlap": overlap,
                    "log_overlap": np.log1p(overlap),
                })

    return pd.DataFrame(rows)


def build_overlap_node_table(
    banks: pd.DataFrame,
    omega: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build node attributes for overlap-network plotting.
    """
    importance = compute_overlap_systemic_importance(omega)

    nodes = banks[["bank_id", "bank_name", "total_assets", "government_securities"]].copy()
    nodes["systemic_importance"] = nodes["bank_id"].map(importance)

    max_val = nodes["systemic_importance"].max()

    if max_val > 0:
        nodes["systemic_importance_normalized"] = (
            nodes["systemic_importance"] / max_val
        )
    else:
        nodes["systemic_importance_normalized"] = 0.0

    return nodes