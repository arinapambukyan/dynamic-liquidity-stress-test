from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from pathlib import Path


INTERBANK_ASSET_COLUMN = "interbank_assets"
INTERBANK_LIABILITY_COLUMN = "interbank_liabilities"


def build_synthetic_interbank_exposure_matrix(
    banks: pd.DataFrame,
    asset_column: str = INTERBANK_ASSET_COLUMN,
    liability_column: str = INTERBANK_LIABILITY_COLUMN,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    Build a synthetic directed interbank exposure matrix E.

    E[i, j] = exposure of creditor bank i to debtor bank j.

    Row sums match interbank assets.
    Column sums match interbank liabilities.
    Diagonal is zero.
    """
    required = ["bank_id", asset_column, liability_column]
    missing = [col for col in required if col not in banks.columns]
    if missing:
        raise ValueError(f"Missing required columns for interbank matrix: {missing}")

    bank_ids = banks["bank_id"].tolist()

    row_targets = banks[asset_column].astype(float).clip(lower=0.0).to_numpy()
    col_targets = banks[liability_column].astype(float).clip(lower=0.0).to_numpy()

    # Make system totals consistent if minor mismatch exists
    row_total = row_targets.sum()
    col_total = col_targets.sum()

    if row_total > 0 and col_total > 0 and not np.isclose(row_total, col_total):
        col_targets = col_targets * (row_total / col_total)

    n = len(bank_ids)

    base = np.outer(row_targets + 1e-12, col_targets + 1e-12)
    np.fill_diagonal(base, 0.0)

    if base.sum() == 0:
        return pd.DataFrame(np.zeros((n, n)), index=bank_ids, columns=bank_ids)

    matrix = base.copy()

    converged = False
    row_error = np.inf
    col_error = np.inf

    for iteration in range(1, max_iter + 1):
        # Row scaling
        row_sums = matrix.sum(axis=1)
        row_factors = np.divide(
            row_targets,
            row_sums,
            out=np.ones_like(row_targets),
            where=row_sums > 0,
        )
        matrix = (matrix.T * row_factors).T
        np.fill_diagonal(matrix, 0.0)

        # Column scaling
        col_sums = matrix.sum(axis=0)
        col_factors = np.divide(
            col_targets,
            col_sums,
            out=np.ones_like(col_targets),
            where=col_sums > 0,
        )
        matrix = matrix * col_factors
        np.fill_diagonal(matrix, 0.0)

        row_error = float(np.max(np.abs(matrix.sum(axis=1) - row_targets)))
        col_error = float(np.max(np.abs(matrix.sum(axis=0) - col_targets)))

        if max(row_error, col_error) < tol:
            converged = True
            break

    final_row_error = float(np.max(np.abs(matrix.sum(axis=1) - row_targets)))
    final_col_error = float(np.max(np.abs(matrix.sum(axis=0) - col_targets)))
    final_error = max(final_row_error, final_col_error)

    if not converged or final_error > tol:
        warnings.warn(
            "Synthetic interbank exposure matrix IPF did not fully converge. "
            f"max_row_error={final_row_error:.6g}, "
            f"max_col_error={final_col_error:.6g}, "
            f"tolerance={tol:.6g}, "
            f"max_iter={max_iter}. "
            "The returned matrix is approximate.",
            RuntimeWarning,
        )

    return pd.DataFrame(matrix, index=bank_ids, columns=bank_ids)


def compute_interbank_systemic_importance(
    exposure_matrix: pd.DataFrame,
    banks: pd.DataFrame,
    lgd: float = 1.0,
) -> pd.Series:
    """
    Compute direct-network systemic importance.

    SI_j = sum_i (LGD * E_ij / Equity_i)

    Interpretation:
    if bank j defaults, how much normalized damage can it impose on
    creditor banks i.
    """
    equity = banks.set_index("bank_id")["equity"].astype(float).copy()
    equity = equity.replace(0.0, np.nan)

    impact_matrix = exposure_matrix.copy().astype(float)

    for creditor in impact_matrix.index:
        creditor_equity = equity.loc[creditor]
        if pd.notna(creditor_equity) and creditor_equity > 0:
            impact_matrix.loc[creditor, :] = lgd * impact_matrix.loc[creditor, :] / creditor_equity
        else:
            impact_matrix.loc[creditor, :] = 0.0

    importance = impact_matrix.sum(axis=0)
    importance.name = "systemic_importance"
    return importance


def build_interbank_edge_table(
    exposure_matrix: pd.DataFrame,
    min_exposure: float = 0.0,
) -> pd.DataFrame:
    """
    Convert directed interbank matrix to edge table.
    """
    rows = []

    for src in exposure_matrix.index:
        for tgt in exposure_matrix.columns:
            if src == tgt:
                continue

            exposure = float(exposure_matrix.loc[src, tgt])

            if exposure > min_exposure:
                rows.append(
                    {
                        "source": src,
                        "target": tgt,
                        "exposure": exposure,
                        "log_exposure": np.log1p(exposure),
                    }
                )

    return pd.DataFrame(rows)


def build_interbank_node_table(
    banks: pd.DataFrame,
    exposure_matrix: pd.DataFrame,
    lgd: float = 1.0,
) -> pd.DataFrame:
    """
    Build node attributes for direct interbank network plotting.
    """
    importance = compute_interbank_systemic_importance(
        exposure_matrix=exposure_matrix,
        banks=banks,
        lgd=lgd,
    )

    nodes = banks[["bank_id", "bank_name", "total_assets"]].copy()
    nodes["systemic_importance"] = nodes["bank_id"].map(importance)

    max_val = nodes["systemic_importance"].max()
    if max_val > 0:
        nodes["systemic_importance_normalized"] = nodes["systemic_importance"] / max_val
    else:
        nodes["systemic_importance_normalized"] = 0.0

    return nodes


def save_interbank_exposure_matrix_to_excel(
    exposure_matrix: pd.DataFrame,
    path: str | Path,
    sheet_name: str = "exposures",
) -> Path:
    """
    Save directed interbank exposure matrix to Excel.

    Matrix format:
        rows = creditor banks
        columns = debtor banks
        values = exposure of creditor i to debtor j
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = exposure_matrix.copy()
    out.index = out.index.astype(str)
    out.columns = out.columns.astype(str)
    out = out.reset_index().rename(columns={"index": "creditor_bank_id"})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    return path


def read_interbank_exposure_matrix_from_excel(
    path: str | Path,
    sheet_name: str = "exposures",
) -> pd.DataFrame:
    """
    Read directed interbank exposure matrix from Excel.

    Expected matrix format:
        first column = creditor_bank_id
        remaining columns = debtor bank IDs
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Interbank exposure matrix not found: {path}")

    raw = pd.read_excel(path, sheet_name=sheet_name)

    if raw.empty:
        raise ValueError(f"Interbank exposure matrix is empty: {path}")

    first_col = raw.columns[0]
    matrix = raw.set_index(first_col)

    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # No self-exposure.
    common = matrix.index.intersection(matrix.columns)
    for bank_id in common:
        matrix.loc[bank_id, bank_id] = 0.0

    return matrix


def get_or_create_interbank_exposure_matrix(
    banks: pd.DataFrame,
    path: str | Path,
    source: str = "generate_if_missing",
    sheet_name: str = "exposures",
    exposure_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Get interbank exposure matrix from Excel, generating it only if explicitly
    allowed and only if the file is missing.

    This makes the contagion engine input-driven:
        generate once -> save Excel -> read Excel
    """
    path = Path(path)

    if source not in ["generate_if_missing", "read_existing"]:
        raise ValueError(
            "source must be either 'generate_if_missing' or 'read_existing'."
        )

    if path.exists():
        return read_interbank_exposure_matrix_from_excel(
            path=path,
            sheet_name=sheet_name,
        )

    if source == "read_existing":
        raise FileNotFoundError(
            f"source='read_existing', but matrix does not exist: {path}"
        )

    exposure_matrix = build_synthetic_interbank_exposure_matrix(
        banks=banks,
    )

    if exposure_scale != 1.0:
        exposure_matrix = exposure_matrix * float(exposure_scale)

    save_interbank_exposure_matrix_to_excel(
        exposure_matrix=exposure_matrix,
        path=path,
        sheet_name=sheet_name,
    )

    print(f"Generated and saved interbank exposure matrix: {path}")

    return read_interbank_exposure_matrix_from_excel(
        path=path,
        sheet_name=sheet_name,
    )