"""
Run Monte Carlo simulations for stochastic first-round liquidity shocks.

This is a shock-distribution Monte Carlo, not random policy-parameter
sensitivity.

Fixed:

    central-bank refinancing policy
    central-bank asset-purchase rule
    fire-sale price-impact parameters
    interbank LGD

Randomized:

    first-round stress-vector weights by month and shock component

Run from project root:

    PYTHONPATH=src python scripts/40_run_monte_carlo.py
"""

from __future__ import annotations

import copy
from pathlib import Path
import argparse
import importlib.util

import numpy as np
import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    FIRST_ROUND_STRESS_VECTOR,
    FIRST_ROUND_SCENARIO_CONFIG,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    CB_SUPPORT_CONFIG,
    FIRE_SALE_CONFIG,
    INTERBANK_CONTAGION_CONFIG,
    TABLES_DIR,
    INTERBANK_EXPOSURE_CONFIG,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.network_interbank import get_or_create_interbank_exposure_matrix
from liquidity_stress.scenario_config import (
    load_and_validate_scenario_config,
    get_stress_mode,
    build_runtime_configs_from_scenario,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenario_base.yaml"

# Number of simulations per stress level.
# Total runs = len(STRESS_LEVELS) * N_SIMULATIONS_PER_STRESS_LEVEL.
N_SIMULATIONS_PER_STRESS_LEVEL = 50
RANDOM_SEED = 42

STRESS_LEVELS = {
    "mild": 0.35,
    "moderate": 0.50,
    "adverse": 0.75,
    "severe": 1.00,
}

STRESS_LEVEL_ORDER = ["mild", "moderate", "adverse", "severe"]

# Choose which fixed policy regime Monte Carlo simulates.
# Options:
#   "base_config"                = exactly use config.py
#   "lcr_cb_collateral_capped"   = LoLR covers cash gap and LCR gap, capped by collateral
#   "lcr_cb_full_coverage"       = LoLR covers cash gap and LCR gap without collateral cap
#   "lcr_cb_plus_asset_purchases"= collateral-capped LCR-CB + strong state-bond asset purchases
MONTE_CARLO_POLICY_MODE = "lcr_cb_collateral_capped"

# Randomness of first-round shock weights.
SHOCK_MULTIPLIER_SIGMA = 0.20
SHOCK_MULTIPLIER_PERSISTENCE = 0.50
SHOCK_MULTIPLIER_MIN = 0.50
SHOCK_MULTIPLIER_MAX = 1.75

LCR_BREACH_THRESHOLD = 1.00

CURRENT_PERIOD_FLOW_COLUMNS = [
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
    "cash_shortfall_after_asset_sales",
    "remaining_shortfall_after_cb_support",
    "cash_shortfall_before_cba_refinancing",
    "cash_shortfall_after_cb_refinancing",
    "cb_support_target_before_asset_sales",
    "realized_loan_inflows",
    "realized_interbank_asset_inflows",
    "total_realized_asset_inflows",
    "government_security_loss",
    "bank_security_loss",
    "corporate_security_loss",
    "total_security_valuation_loss",
    "asset_sale_proceeds_used_for_outflows",
    "asset_sale_liquidation_loss",
    "total_securities_sold_for_liquidity",
    "cb_support_required",
    "cb_support_granted",
    "cb_support_used_for_outflows",
    "cb_support_retained_as_cash_for_lcr",
    "lolr_cash_shortfall_need",
    "lolr_lcr_support_gap",
    "lolr_required_support",
    "newly_pledged_collateral",
    "newly_pledged_collateral_support_value",
    "total_fire_sale_mtm_loss",
    "interbank_contagion_loss",
]

def load_interbank_engine():
    """
    Load scripts/10_run_base_engine.py as a module.

    We use importlib because the filename starts with a number and therefore
    cannot be imported with a normal Python import statement.
    """
    engine_path = PROJECT_ROOT / "scripts" / "10_run_base_engine.py"

    spec = importlib.util.spec_from_file_location(
        "interbank_engine_07",
        engine_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load engine from {engine_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def make_monte_carlo_policy_configs(
    policy_mode: str,
    base_runtime_config: dict,
) -> dict[str, dict]:
    """
    Build the fixed policy configuration used by the Monte Carlo.

    Monte Carlo randomizes shocks, not policy. The selected policy mode is held
    fixed across all simulations.

    The starting point is the YAML-derived runtime config, not raw config.py,
    so Armenia's active LCR requirement and module switches are preserved.
    """
    cb = copy.deepcopy(base_runtime_config["cb"])
    fire = copy.deepcopy(base_runtime_config["fire"])
    interbank = copy.deepcopy(base_runtime_config["interbank"])
    lcr = copy.deepcopy(base_runtime_config["lcr"])
    liquidation = copy.deepcopy(base_runtime_config["liquidation"])
    modules = copy.deepcopy(base_runtime_config["modules"])

    active_requirement = float(
        liquidation.get(
            "active_lcr_requirement",
            liquidation.get("reaction_lcr_trigger", 1.0),
        )
    )

    if policy_mode == "base_config":
        return {
            "cb": cb,
            "fire": fire,
            "interbank": interbank,
            "lcr": lcr,
            "liquidation": liquidation,
            "modules": modules,
        }

    if policy_mode in {
        "lcr_cb_collateral_capped",
        "lcr_cb_full_coverage",
        "lcr_cb_plus_asset_purchases",
    }:
        cb["enabled"] = True
        cb["support_timing"] = "after_asset_sales"
        cb["support_only_solvent_illiquid"] = True

        cb["illiquidity_lcr_threshold"] = active_requirement
        cb["support_lcr_gap"] = True
        cb["lcr_support_target"] = active_requirement

        if policy_mode == "lcr_cb_full_coverage":
            cb["support_mode"] = "full_coverage"
        else:
            cb["support_mode"] = "collateral_capped"

        # CBA liquidity funding is assumed not to run off inside the 30-day LCR window.
        lcr["cb_liquidity_funding_runoff"] = 0.0

        if policy_mode == "lcr_cb_plus_asset_purchases":
            fire.setdefault("central_bank_asset_purchases", {})
            fire["central_bank_asset_purchases"]["enabled"] = True
            fire["central_bank_asset_purchases"]["purchase_share_of_bank_sales"] = {
                asset: 0.80 for asset in fire.get("asset_columns", [])
            }

        return {
            "cb": cb,
            "fire": fire,
            "interbank": interbank,
            "lcr": lcr,
            "liquidation": liquidation,
            "modules": modules,
        }

    raise ValueError(
        "Unknown MONTE_CARLO_POLICY_MODE. Use one of: "
        "base_config, lcr_cb_collateral_capped, lcr_cb_full_coverage, "
        "lcr_cb_plus_asset_purchases."
    )


def sort_by_stress_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stress_level"] = pd.Categorical(
        out["stress_level"],
        categories=STRESS_LEVEL_ORDER,
        ordered=True,
    )
    return out.sort_values("stress_level").reset_index(drop=True)


def zero_current_period_flows(
    df: pd.DataFrame,
    fire_sale_config: dict,
) -> pd.DataFrame:
    """
    Zero only current-period flow variables for carried-forward terminal banks.

    Stock/state variables such as pledged collateral, cumulative CBA funding,
    securities holdings, HQLA, NCOF, LCR, and HQLA/assets are intentionally kept.
    """
    out = df.copy()

    for column in CURRENT_PERIOD_FLOW_COLUMNS:
        if column in out.columns:
            out[column] = 0.0

    for asset_column in fire_sale_config.get("asset_columns", []):
        sold_column = f"{asset_column}_sold_for_liquidity"
        loss_column = f"{asset_column}_fire_sale_mtm_loss"

        if sold_column in out.columns:
            out[sold_column] = 0.0

        if loss_column in out.columns:
            out[loss_column] = 0.0

    for column in out.columns:
        if column.startswith("newly_pledged_"):
            out[column] = 0.0

    for column in [c for c in out.columns if c.startswith("interbank_contagion_loss_round_")]:
        out[column] = 0.0

    return out


def ensure_columns(results: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure optional columns exist and derive the key Monte Carlo risk flags.
    """
    out = results.copy()

    bool_defaults = [
        "policy_support_used",
        "final_unresolved_shortfall",
        "post_fire_sale_insolvent",
        "post_interbank_insolvent",
        "new_terminal_from_interbank",
        "terminal_distress_after_interbank",
    ]

    numeric_defaults = [
        "cb_support_granted",
        "cb_support_required",
        "cb_support_used_for_outflows",
        "cb_support_retained_as_cash_for_lcr",
        "lolr_cash_shortfall_need",
        "lolr_lcr_support_gap",
        "lolr_required_support",
        "asset_sale_proceeds_used_for_outflows",
        "asset_sale_liquidation_loss",
        "total_fire_sale_mtm_loss",
        "interbank_contagion_loss",
        "stressed_LCR",
        "stressed_HQLA_assets_ratio",
        "post_interbank_LCR",
        "post_interbank_HQLA_assets_ratio",
        "HQLA",
        "NCOF",
        "equity",
        "total_assets",
        "cb_liquidity_funding",
        "cash_shortfall_before_support",
        "cash_shortfall_after_asset_sales",
        "remaining_shortfall_after_cb_support",
        "cash_shortfall_before_cba_refinancing",
        "cash_shortfall_after_cb_refinancing",
        "total_cb_pledged_collateral",
        "pre_cb_unencumbered_collateral_capacity",
        "post_cb_unencumbered_collateral_capacity",
    ]

    for column in bool_defaults:
        if column not in out.columns:
            out[column] = False
        out[column] = out[column].fillna(False).astype(bool)

    for column in numeric_defaults:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    if "active_lcr_requirement" in out.columns:
        active_requirement = pd.to_numeric(
            out["active_lcr_requirement"],
            errors="coerce",
        ).fillna(LCR_BREACH_THRESHOLD)
    else:
        active_requirement = pd.Series(LCR_BREACH_THRESHOLD, index=out.index)

    if "LCR_breach" in out.columns:
        out["LCR_breach"] = out["LCR_breach"].fillna(False).astype(bool)
    else:
        out["LCR_breach"] = out["stressed_LCR"] < active_requirement

    if "post_interbank_LCR_breach" in out.columns:
        out["post_interbank_LCR_breach"] = (
            out["post_interbank_LCR_breach"].fillna(False).astype(bool)
        )
    else:
        out["post_interbank_LCR_breach"] = (
                out["post_interbank_LCR"] < active_requirement
        )


    if "ever_LCR_breach" in out.columns:
        out["ever_LCR_breach"] = (
            out["ever_LCR_breach"].fillna(False).astype(bool)
            | out["LCR_breach"].astype(bool)
        )
    else:
        out["ever_LCR_breach"] = out["LCR_breach"].astype(bool)

    out["policy_support_used"] = (
        out["policy_support_used"].astype(bool)
        | (out["cb_support_granted"] > 1e-9)
    )

    return out


def get_numeric_shock_columns(stress_vector: pd.DataFrame) -> list[str]:
    """
    Select numeric stress-weight columns to perturb.
    """
    excluded_terms = [
        "id",
        "name",
        "month",
        "year",
        "scenario",
        "level",
        "alpha",
    ]

    numeric_columns = stress_vector.select_dtypes(include=[np.number]).columns
    candidates = []

    for column in numeric_columns:
        lowered = column.lower()

        if any(term in lowered for term in excluded_terms):
            continue

        values = stress_vector[column].dropna()

        if values.empty:
            continue

        if values.min() >= 0 and values.max() <= 1.0 + 1e-9:
            candidates.append(column)

    if not candidates:
        candidates = [
            column
            for column in numeric_columns
            if not any(term in column.lower() for term in excluded_terms)
        ]

    if not candidates:
        raise ValueError("No numeric stress-vector columns found to perturb.")

    return candidates


def draw_persistent_log_multipliers(
    n_months: int,
    shape: tuple[int, int],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """
    Draw persistent log-normal multipliers for stress weights.
    """
    n_rows, n_cols = shape
    n_factors = n_rows * n_cols

    shocks = np.zeros((n_months, n_factors))
    shocks[0] = rng.normal(0.0, SHOCK_MULTIPLIER_SIGMA, size=n_factors)

    innovation_sigma = SHOCK_MULTIPLIER_SIGMA * np.sqrt(
        1.0 - SHOCK_MULTIPLIER_PERSISTENCE**2
    )

    for month in range(1, n_months):
        shocks[month] = (
            SHOCK_MULTIPLIER_PERSISTENCE * shocks[month - 1]
            + rng.normal(0.0, innovation_sigma, size=n_factors)
        )

    multipliers = np.exp(shocks - 0.5 * SHOCK_MULTIPLIER_SIGMA**2)
    multipliers = np.clip(
        multipliers,
        SHOCK_MULTIPLIER_MIN,
        SHOCK_MULTIPLIER_MAX,
    )

    return [multipliers[month].reshape(shape) for month in range(n_months)]


def sample_monthly_stress_vectors(
    base_stress_vector: pd.DataFrame,
    n_months: int,
    rng: np.random.Generator,
) -> list[pd.DataFrame]:
    """
    Generate one randomized stress vector per month.
    """
    if not isinstance(base_stress_vector, pd.DataFrame):
        raise TypeError(
            "FIRST_ROUND_STRESS_VECTOR must be a pandas DataFrame "
            "for this Monte Carlo script."
        )

    numeric_columns = get_numeric_shock_columns(base_stress_vector)
    base_values = base_stress_vector[numeric_columns].astype(float).to_numpy()

    multipliers_by_month = draw_persistent_log_multipliers(
        n_months=n_months,
        shape=base_values.shape,
        rng=rng,
    )

    monthly_vectors = []

    for month in range(n_months):
        sampled = base_stress_vector.copy(deep=True)
        sampled_values = base_values * multipliers_by_month[month]
        sampled.loc[:, numeric_columns] = np.clip(sampled_values, 0.0, 1.0)
        monthly_vectors.append(sampled)

    return monthly_vectors


def summarize_run(
    run_id: int,
    simulation_id: int,
    stress_level: str,
    alpha_scale: float,
    alpha_path: list[float],
    results: pd.DataFrame,
) -> dict:
    """
    Create one run-level summary row.
    """
    results = ensure_columns(results)

    last_month = results["stress_month"].max()
    final_rows = results.loc[results["stress_month"] == last_month].copy()
    grouped = results.groupby("bank_id")

    row = {
        "run_id": run_id,
        "simulation_id": simulation_id,
        "stress_level": stress_level,
        "alpha_scale": alpha_scale,
        "policy_mode": MONTE_CARLO_POLICY_MODE,
        "number_of_months": int(results["stress_month"].nunique()),
        "mean_alpha": float(np.mean(alpha_path)),
        "max_alpha": float(np.max(alpha_path)),
        "min_alpha": float(np.min(alpha_path)),

        "total_cash_shortfall_before_support": float(
            results["cash_shortfall_before_support"].sum()
        ),
        "total_lcr_support_gap": float(results["lolr_lcr_support_gap"].sum()),
        "total_lolr_required_support": float(results["lolr_required_support"].sum()),
        "total_cb_support_granted": float(results["cb_support_granted"].sum()),
        "total_support_used_for_outflows": float(
            results["cb_support_used_for_outflows"].sum()
        ),
        "total_support_retained_as_cash_for_lcr": float(
            results["cb_support_retained_as_cash_for_lcr"].sum()
        ),
        "total_asset_sale_proceeds": float(
            results["asset_sale_proceeds_used_for_outflows"].sum()
        ),
        "total_asset_sale_loss": float(results["asset_sale_liquidation_loss"].sum()),
        "total_fire_sale_mtm_loss": float(results["total_fire_sale_mtm_loss"].sum()),
        "total_interbank_contagion_loss": float(
            results["interbank_contagion_loss"].sum()
        ),

        "min_stressed_LCR": float(results["stressed_LCR"].min()),
        "min_post_interbank_LCR": float(results["post_interbank_LCR"].min()),
        "average_final_post_interbank_LCR": float(
            final_rows["post_interbank_LCR"].mean()
        ),
        "minimum_final_post_interbank_LCR": float(
            final_rows["post_interbank_LCR"].min()
        ),
        "min_post_interbank_HQLA_assets_ratio": float(
            results["post_interbank_HQLA_assets_ratio"].min()
        ),
        "average_final_post_interbank_HQLA_assets_ratio": float(
            final_rows["post_interbank_HQLA_assets_ratio"].mean()
        ),
        "minimum_final_post_interbank_HQLA_assets_ratio": float(
            final_rows["post_interbank_HQLA_assets_ratio"].min()
        ),

        "ever_stressed_LCR_breach_banks": int(
            grouped["LCR_breach"].max().sum()
        ),
        "ever_post_interbank_LCR_breach_banks": int(
            grouped["post_interbank_LCR_breach"].max().sum()
        ),
        "final_post_interbank_LCR_breach_banks": int(
            final_rows["post_interbank_LCR_breach"].sum()
        ),
        "ever_cb_support_used_banks": int(grouped["policy_support_used"].max().sum()),
        "ever_final_unresolved_shortfall_banks": int(
            grouped["final_unresolved_shortfall"].max().sum()
        ),
        "ever_post_fire_sale_insolvent_banks": int(
            grouped["post_fire_sale_insolvent"].max().sum()
        ),
        "ever_post_interbank_insolvent_banks": int(
            grouped["post_interbank_insolvent"].max().sum()
        ),
        "ever_new_terminal_from_interbank_banks": int(
            grouped["new_terminal_from_interbank"].max().sum()
        ),
        "ever_terminal_distress_after_interbank_banks": int(
            grouped["terminal_distress_after_interbank"].max().sum()
        ),
        "final_terminal_distress_after_interbank_banks": int(
            final_rows["terminal_distress_after_interbank"].sum()
        ),

        "final_system_assets": float(final_rows["total_assets"].sum()),
        "final_system_equity": float(final_rows["equity"].sum()),
        "final_system_HQLA": float(final_rows["HQLA"].sum()),
        "final_system_NCOF": float(final_rows["NCOF"].sum()),
        "final_system_cb_liquidity_funding": float(
            final_rows["cb_liquidity_funding"].sum()
        ),
    }

    row["final_system_HQLA_assets_ratio"] = (
        row["final_system_HQLA"] / row["final_system_assets"]
        if row["final_system_assets"] > 0
        else np.nan
    )
    row["final_system_LCR"] = (
        row["final_system_HQLA"] / row["final_system_NCOF"]
        if row["final_system_NCOF"] > 0
        else np.nan
    )

    for month, alpha in enumerate(alpha_path, start=1):
        row[f"alpha_month_{month}"] = float(alpha)

    return row


def build_monthly_summary(
    run_id: int,
    simulation_id: int,
    stress_level: str,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly summary for one Monte Carlo run.
    """
    results = ensure_columns(results)

    out = results.groupby("stress_month").agg(
        alpha=("first_round_alpha", "first"),
        number_of_banks=("bank_name", "count"),
        stressed_LCR_breach_banks=("LCR_breach", "sum"),
        post_interbank_LCR_breach_banks=("post_interbank_LCR_breach", "sum"),
        banks_using_cb_support=("policy_support_used", "sum"),
        final_unresolved_shortfalls=("final_unresolved_shortfall", "sum"),
        post_fire_sale_insolvent_banks=("post_fire_sale_insolvent", "sum"),
        post_interbank_insolvent_banks=("post_interbank_insolvent", "sum"),
        terminal_distress_after_interbank=("terminal_distress_after_interbank", "sum"),
        aggregate_cash_shortfall_before_support=("cash_shortfall_before_support", "sum"),
        aggregate_lcr_support_gap=("lolr_lcr_support_gap", "sum"),
        aggregate_cb_support_granted=("cb_support_granted", "sum"),
        aggregate_support_used_for_outflows=("cb_support_used_for_outflows", "sum"),
        aggregate_support_retained_as_cash_for_lcr=(
            "cb_support_retained_as_cash_for_lcr",
            "sum",
        ),
        aggregate_asset_sale_proceeds=("asset_sale_proceeds_used_for_outflows", "sum"),
        aggregate_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
        aggregate_interbank_contagion_loss=("interbank_contagion_loss", "sum"),
        aggregate_cb_pledged_collateral=("total_cb_pledged_collateral", "sum"),
        aggregate_pre_cb_unencumbered_collateral_capacity=(
            "pre_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        aggregate_post_cb_unencumbered_collateral_capacity=(
            "post_cb_unencumbered_collateral_capacity",
            "sum",
        ),
        average_stressed_LCR=("stressed_LCR", "mean"),
        minimum_stressed_LCR=("stressed_LCR", "min"),
        average_post_interbank_LCR=("post_interbank_LCR", "mean"),
        minimum_post_interbank_LCR=("post_interbank_LCR", "min"),
        average_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "mean",
        ),
        minimum_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "min",
        ),
        aggregate_HQLA=("HQLA", "sum"),
        aggregate_NCOF=("NCOF", "sum"),
        aggregate_system_assets=("total_assets", "sum"),
        aggregate_system_equity=("equity", "sum"),
    ).reset_index()

    out.insert(0, "stress_level", stress_level)
    out.insert(0, "simulation_id", simulation_id)
    out.insert(0, "run_id", run_id)
    out.insert(3, "policy_mode", MONTE_CARLO_POLICY_MODE)

    out["system_HQLA_assets_ratio"] = np.where(
        out["aggregate_system_assets"] > 0,
        out["aggregate_HQLA"] / out["aggregate_system_assets"],
        np.nan,
    )

    out["system_LCR"] = np.where(
        out["aggregate_NCOF"] > 0,
        out["aggregate_HQLA"] / out["aggregate_NCOF"],
        np.nan,
    )

    return out


def build_bank_summary(
    run_id: int,
    simulation_id: int,
    stress_level: str,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Bank-level summary for one Monte Carlo run.
    """
    results = ensure_columns(results)

    out = results.groupby(["bank_id", "bank_name"]).agg(
        min_stressed_LCR=("stressed_LCR", "min"),
        min_post_interbank_LCR=("post_interbank_LCR", "min"),
        final_post_interbank_LCR=("post_interbank_LCR", "last"),
        min_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "min",
        ),
        final_post_interbank_HQLA_assets_ratio=(
            "post_interbank_HQLA_assets_ratio",
            "last",
        ),
        total_lcr_support_gap=("lolr_lcr_support_gap", "sum"),
        total_cb_support_granted=("cb_support_granted", "sum"),
        total_asset_sale_proceeds=("asset_sale_proceeds_used_for_outflows", "sum"),
        total_fire_sale_mtm_loss=("total_fire_sale_mtm_loss", "sum"),
        total_interbank_contagion_loss=("interbank_contagion_loss", "sum"),
        ever_stressed_LCR_breach=("LCR_breach", "max"),
        ever_post_interbank_LCR_breach=("post_interbank_LCR_breach", "max"),
        ever_used_cb_support=("policy_support_used", "max"),
        ever_final_unresolved_shortfall=("final_unresolved_shortfall", "max"),
        ever_post_fire_sale_insolvent=("post_fire_sale_insolvent", "max"),
        ever_post_interbank_insolvent=("post_interbank_insolvent", "max"),
        ever_terminal_distress_after_interbank=(
            "terminal_distress_after_interbank",
            "max",
        ),
        final_total_assets=("total_assets", "last"),
        final_equity=("equity", "last"),
        final_cb_liquidity_funding=("cb_liquidity_funding", "last"),
    ).reset_index()

    out.insert(0, "stress_level", stress_level)
    out.insert(0, "simulation_id", simulation_id)
    out.insert(0, "run_id", run_id)
    out.insert(3, "policy_mode", MONTE_CARLO_POLICY_MODE)

    return out


def build_quantile_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Monte Carlo quantiles by stress level.
    """
    metrics = [
        "total_cash_shortfall_before_support",
        "total_lcr_support_gap",
        "total_lolr_required_support",
        "total_cb_support_granted",
        "total_support_used_for_outflows",
        "total_support_retained_as_cash_for_lcr",
        "total_asset_sale_proceeds",
        "total_fire_sale_mtm_loss",
        "total_interbank_contagion_loss",
        "min_stressed_LCR",
        "min_post_interbank_LCR",
        "min_post_interbank_HQLA_assets_ratio",
        "final_system_HQLA_assets_ratio",
        "ever_stressed_LCR_breach_banks",
        "ever_post_interbank_LCR_breach_banks",
        "ever_cb_support_used_banks",
        "ever_final_unresolved_shortfall_banks",
        "ever_post_fire_sale_insolvent_banks",
        "ever_terminal_distress_after_interbank_banks",
        "final_system_equity",
    ]

    rows = []

    for stress_level, group in run_summary.groupby("stress_level", observed=False):
        for metric in metrics:
            if metric not in group.columns:
                continue

            values = pd.to_numeric(group[metric], errors="coerce").dropna()

            if values.empty:
                continue

            rows.append({
                "stress_level": stress_level,
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
                "p01": values.quantile(0.01),
                "p05": values.quantile(0.05),
                "p25": values.quantile(0.25),
                "p50": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p95": values.quantile(0.95),
                "p99": values.quantile(0.99),
                "min": values.min(),
                "max": values.max(),
            })

    return pd.DataFrame(rows)


def build_bank_risk_ranking(bank_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Rank banks by Monte Carlo vulnerability within each stress level.
    """
    out = bank_summary.groupby(
        ["stress_level", "bank_id", "bank_name"],
        observed=False,
    ).agg(
        probability_stressed_LCR_breach=("ever_stressed_LCR_breach", "mean"),
        probability_post_policy_LCR_breach=(
            "ever_post_interbank_LCR_breach",
            "mean",
        ),
        probability_used_cb_support=("ever_used_cb_support", "mean"),
        probability_final_unresolved_shortfall=(
            "ever_final_unresolved_shortfall",
            "mean",
        ),
        probability_fire_sale_insolvent=("ever_post_fire_sale_insolvent", "mean"),
        probability_interbank_insolvent=("ever_post_interbank_insolvent", "mean"),
        probability_terminal_distress=(
            "ever_terminal_distress_after_interbank",
            "mean",
        ),
        average_min_post_interbank_LCR=("min_post_interbank_LCR", "mean"),
        p05_min_post_interbank_LCR=(
            "min_post_interbank_LCR",
            lambda x: x.quantile(0.05),
        ),
        average_min_post_interbank_HQLA_assets_ratio=(
            "min_post_interbank_HQLA_assets_ratio",
            "mean",
        ),
        p05_min_post_interbank_HQLA_assets_ratio=(
            "min_post_interbank_HQLA_assets_ratio",
            lambda x: x.quantile(0.05),
        ),
        average_total_cb_support=("total_cb_support_granted", "mean"),
        p95_total_cb_support=(
            "total_cb_support_granted",
            lambda x: x.quantile(0.95),
        ),
        average_final_equity=("final_equity", "mean"),
        p05_final_equity=("final_equity", lambda x: x.quantile(0.05)),
    ).reset_index()

    out = out.sort_values(
        [
            "stress_level",
            "probability_post_policy_LCR_breach",
            "probability_used_cb_support",
            "probability_terminal_distress",
            "p05_min_post_interbank_LCR",
        ],
        ascending=[True, False, False, False, True],
    )

    return out


def build_risk_probability_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build probability indicators by stress level.
    """
    rows = []

    for stress_level, group in run_summary.groupby("stress_level", observed=False):
        rows.append({
            "stress_level": stress_level,
            "number_of_simulations": len(group),
            "policy_mode": MONTE_CARLO_POLICY_MODE,

            "prob_any_stressed_LCR_breach": (
                group["ever_stressed_LCR_breach_banks"] > 0
            ).mean(),
            "prob_5plus_stressed_LCR_breach_banks": (
                group["ever_stressed_LCR_breach_banks"] >= 5
            ).mean(),
            "prob_any_post_policy_LCR_breach": (
                group["ever_post_interbank_LCR_breach_banks"] > 0
            ).mean(),
            "prob_5plus_post_policy_LCR_breach_banks": (
                group["ever_post_interbank_LCR_breach_banks"] >= 5
            ).mean(),
            "prob_any_cb_support": (
                group["ever_cb_support_used_banks"] > 0
            ).mean(),
            "prob_5plus_banks_using_cb_support": (
                group["ever_cb_support_used_banks"] >= 5
            ).mean(),

            "prob_any_unresolved_liquidity_shortfall": (
                group["ever_final_unresolved_shortfall_banks"] > 0
            ).mean(),
            "prob_3plus_unresolved_liquidity_shortfall_banks": (
                group["ever_final_unresolved_shortfall_banks"] >= 3
            ).mean(),
            "prob_5plus_unresolved_liquidity_shortfall_banks": (
                group["ever_final_unresolved_shortfall_banks"] >= 5
            ).mean(),
            "prob_10plus_unresolved_liquidity_shortfall_banks": (
                group["ever_final_unresolved_shortfall_banks"] >= 10
            ).mean(),

            "prob_any_fire_sale_insolvency": (
                group["ever_post_fire_sale_insolvent_banks"] > 0
            ).mean(),
            "prob_3plus_fire_sale_insolvencies": (
                group["ever_post_fire_sale_insolvent_banks"] >= 3
            ).mean(),

            "prob_any_terminal_distress": (
                group["ever_terminal_distress_after_interbank_banks"] > 0
            ).mean(),
            "prob_5plus_terminal_distress_banks": (
                group["ever_terminal_distress_after_interbank_banks"] >= 5
            ).mean(),
            "prob_10plus_terminal_distress_banks": (
                group["ever_terminal_distress_after_interbank_banks"] >= 10
            ).mean(),

            "expected_stressed_LCR_breach_banks": (
                group["ever_stressed_LCR_breach_banks"].mean()
            ),
            "expected_post_policy_LCR_breach_banks": (
                group["ever_post_interbank_LCR_breach_banks"].mean()
            ),
            "expected_banks_using_cb_support": (
                group["ever_cb_support_used_banks"].mean()
            ),
            "expected_unresolved_liquidity_shortfall_banks": (
                group["ever_final_unresolved_shortfall_banks"].mean()
            ),
            "expected_fire_sale_insolvent_banks": (
                group["ever_post_fire_sale_insolvent_banks"].mean()
            ),
            "expected_terminal_distress_banks": (
                group["ever_terminal_distress_after_interbank_banks"].mean()
            ),
        })

    return pd.DataFrame(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulations for stochastic liquidity shocks.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(SCENARIO_CONFIG_PATH),
        help="Path to scenario YAML config.",
    )

    return parser.parse_args()

def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    engine = load_interbank_engine()

    rng = np.random.default_rng(RANDOM_SEED)

    banks = BankingDataLoader(BANKING_INPUT_PATH).load_banks()

    scenario = load_and_validate_scenario_config(args.config)
    stress_mode = get_stress_mode(scenario)
    base_alpha_path = scenario.get("severity", {}).get("alpha_path")

    base_runtime_config = build_runtime_configs_from_scenario(
        scenario_config=scenario,
        base_lcr_assumptions=LCR_ASSUMPTIONS,
        base_liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
        base_cb_support_config=CB_SUPPORT_CONFIG,
        base_fire_sale_config=FIRE_SALE_CONFIG,
        base_interbank_contagion_config=INTERBANK_CONTAGION_CONFIG,
    )

    if base_alpha_path is None:
        base_alpha_path = [FIRST_ROUND_SCENARIO_CONFIG["alpha"]]

    policy_configs = make_monte_carlo_policy_configs(
        policy_mode=MONTE_CARLO_POLICY_MODE,
        base_runtime_config=base_runtime_config,
    )

    exposure_matrix = get_or_create_interbank_exposure_matrix(
        banks=banks,
        path=INTERBANK_EXPOSURE_CONFIG["path"],
        source=INTERBANK_EXPOSURE_CONFIG.get("source", "generate_if_missing"),
        sheet_name=INTERBANK_EXPOSURE_CONFIG.get("sheet_name", "exposures"),
        exposure_scale=INTERBANK_EXPOSURE_CONFIG.get("network_exposure_scale", 1.0),
    )

    run_rows = []
    monthly_rows = []
    bank_rows = []

    global_run_id = 0

    print(f"Monte Carlo policy mode: {MONTE_CARLO_POLICY_MODE}")
    print(f"Stress mode: {stress_mode}")

    for stress_level, alpha_scale in STRESS_LEVELS.items():
        alpha_path = [float(alpha) * alpha_scale for alpha in base_alpha_path]

        for simulation_id in range(1, N_SIMULATIONS_PER_STRESS_LEVEL + 1):
            global_run_id += 1

            if simulation_id == 1 or simulation_id % 10 == 0:
                print(
                    f"Running {stress_level} shock Monte Carlo "
                    f"{simulation_id}/{N_SIMULATIONS_PER_STRESS_LEVEL}"
                )

            monthly_stress_vectors = sample_monthly_stress_vectors(
                base_stress_vector=FIRST_ROUND_STRESS_VECTOR,
                n_months=len(alpha_path),
                rng=rng,
            )

            results, market_summary, contagion_summary = (
                engine.run_dynamic_interbank_contagion_path(
                    banks=banks,
                    alpha_path=alpha_path,
                    monthly_stress_vectors=monthly_stress_vectors,
                    scenario_name=(
                        f"shock_monte_carlo_{stress_level}_"
                        f"simulation_{simulation_id}"
                    ),
                    exposure_matrix=exposure_matrix,
                    stress_mode=stress_mode,
                    lcr_assumptions=copy.deepcopy(policy_configs["lcr"]),
                    liquidation_config=copy.deepcopy(policy_configs["liquidation"]),
                    cb_support_config=copy.deepcopy(policy_configs["cb"]),
                    fire_sale_config=copy.deepcopy(policy_configs["fire"]),
                    interbank_contagion_config=copy.deepcopy(policy_configs["interbank"]),
                    module_config=copy.deepcopy(policy_configs["modules"]),
                )
            )

            run_rows.append(
                summarize_run(
                    run_id=global_run_id,
                    simulation_id=simulation_id,
                    stress_level=stress_level,
                    alpha_scale=alpha_scale,
                    alpha_path=alpha_path,
                    results=results,
                )
            )

            monthly_rows.append(
                build_monthly_summary(
                    run_id=global_run_id,
                    simulation_id=simulation_id,
                    stress_level=stress_level,
                    results=results,
                )
            )

            bank_rows.append(
                build_bank_summary(
                    run_id=global_run_id,
                    simulation_id=simulation_id,
                    stress_level=stress_level,
                    results=results,
                )
            )

    run_summary = pd.DataFrame(run_rows)
    monthly_summary = pd.concat(monthly_rows, ignore_index=True)
    bank_summary = pd.concat(bank_rows, ignore_index=True)

    quantile_summary = build_quantile_summary(run_summary)
    bank_risk_ranking = build_bank_risk_ranking(bank_summary)
    risk_probability_summary = build_risk_probability_summary(run_summary)

    run_summary = sort_by_stress_level(run_summary)
    monthly_summary = sort_by_stress_level(monthly_summary)
    bank_summary = sort_by_stress_level(bank_summary)
    quantile_summary = sort_by_stress_level(quantile_summary)
    bank_risk_ranking = sort_by_stress_level(bank_risk_ranking)
    risk_probability_summary = sort_by_stress_level(risk_probability_summary)

    run_summary.to_csv(
        TABLES_DIR / "monte_carlo_shock_run_summary.csv",
        index=False,
    )

    monthly_summary.to_csv(
        TABLES_DIR / "monte_carlo_shock_monthly_summary.csv",
        index=False,
    )

    bank_summary.to_csv(
        TABLES_DIR / "monte_carlo_shock_bank_summary.csv",
        index=False,
    )

    quantile_summary.to_csv(
        TABLES_DIR / "monte_carlo_shock_quantile_summary.csv",
        index=False,
    )

    bank_risk_ranking.to_csv(
        TABLES_DIR / "monte_carlo_shock_bank_risk_ranking.csv",
        index=False,
    )

    risk_probability_summary.to_csv(
        TABLES_DIR / "monte_carlo_shock_risk_probability_summary.csv",
        index=False,
    )

    print("\nShock Monte Carlo outputs saved.")
    print("monte_carlo_shock_run_summary.csv")
    print("monte_carlo_shock_monthly_summary.csv")
    print("monte_carlo_shock_bank_summary.csv")
    print("monte_carlo_shock_quantile_summary.csv")
    print("monte_carlo_shock_bank_risk_ranking.csv")
    print("monte_carlo_shock_risk_probability_summary.csv")

    print("\nMONTE CARLO SHOCK RISK PROBABILITIES BY STRESS LEVEL")
    print(risk_probability_summary.to_string(index=False))

    print("\nMONTE CARLO SHOCK QUANTILE SUMMARY")
    print(quantile_summary.to_string(index=False))

    print("\nBANK RISK RANKING")
    print(bank_risk_ranking.to_string(index=False))


if __name__ == "__main__":
    main()
