#!/usr/bin/env python3
"""
Run paper-grade channel-attribution and policy-counterfactual scenarios.

Purpose
-------
This script creates a clean scenario set for the paper, separate from the
larger sensitivity/debug grid. It keeps the CBA-style benchmark stress path
fixed and varies only the mechanism or policy channel under examination.

Outputs
-------
outputs/tables/paper_scenario_summary.csv
outputs/tables/paper_scenario_monthly_summary.csv
outputs/tables/paper_scenario_bank_results.csv
outputs/tables/paper_channel_attribution_summary.csv
outputs/tables/paper_policy_counterfactual_summary.csv

Run from project root:
    PYTHONPATH=src python scripts/35_run_paper_scenarios.py --config configs/scenario_base.yaml
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    CB_SUPPORT_CONFIG,
    FIRE_SALE_CONFIG,
    INTERBANK_CONTAGION_CONFIG,
    LCR_ASSUMPTIONS,
    LIQUIDATION_WATERFALL_CONFIG,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.scenario_config import (
    build_runtime_configs_from_scenario,
    get_alpha_path_for_currency,
    get_stress_mode,
    load_and_validate_scenario_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = PROJECT_ROOT / "scripts" / "10_run_base_engine.py"
DEFAULT_SCENARIO_CONFIG_PATH = PROJECT_ROOT / "configs" / "scenario_base.yaml"


# -----------------------------------------------------------------------------
# Engine loading
# -----------------------------------------------------------------------------


def load_engine_module():
    """Load scripts/10_run_base_engine.py even though its filename starts with a number."""
    spec = importlib.util.spec_from_file_location("paper_interbank_engine", ENGINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load engine from {ENGINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def set_module_switches(base_modules: dict[str, bool], **overrides: bool) -> dict[str, bool]:
    out = copy.deepcopy(base_modules)
    for key, value in overrides.items():
        if key not in out:
            raise KeyError(f"Unknown module switch: {key}")
        out[key] = bool(value)
    return out


def disable_cb(cb_config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cb_config)
    out["enabled"] = False
    return out


def set_cb_full_coverage(cb_config: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cb_config)
    out["enabled"] = True
    out["support_mode"] = "full_coverage"
    out["support_only_solvent_illiquid"] = True
    # Keep support after private reaction for consistency with the benchmark.
    out["support_timing"] = "after_asset_sales"
    return out


def set_reaction_lcr_requirement(liquidation_config: dict[str, Any], requirement: float) -> dict[str, Any]:
    out = copy.deepcopy(liquidation_config)
    out["active_lcr_requirement"] = float(requirement)
    out["reaction_lcr_trigger"] = float(requirement)
    out["reaction_target_mode"] = "trigger"
    out["reaction_target_cap"] = float(requirement)
    return out


def set_cb_lcr_requirement(cb_config: dict[str, Any], requirement: float) -> dict[str, Any]:
    out = copy.deepcopy(cb_config)
    out["illiquidity_lcr_threshold"] = float(requirement)
    out["support_lcr_gap"] = True
    out["lcr_support_target"] = float(requirement)
    return out


def widen_hqla_definition(lcr_config: dict[str, Any]) -> dict[str, Any]:
    """LCR* style wider HQLA recognition rule used as a policy counterfactual."""
    out = copy.deepcopy(lcr_config)
    out["government_securities_hqla_weight"] = 0.95
    out["bank_securities_hqla_weight"] = 0.60
    out["corporate_securities_hqla_weight"] = 0.35
    return out


def set_asset_purchase_share(fire_config: dict[str, Any], share: float) -> dict[str, Any]:
    out = copy.deepcopy(fire_config)
    out.setdefault("central_bank_asset_purchases", {})
    out["central_bank_asset_purchases"]["enabled"] = float(share) > 0.0
    out["central_bank_asset_purchases"]["purchase_share_of_bank_sales"] = {
        asset: float(share) for asset in out.get("asset_columns", [])
    }
    return out


# -----------------------------------------------------------------------------
# Scenario construction
# -----------------------------------------------------------------------------


def build_paper_scenarios(base_runtime: dict[str, Any], alpha_path: list[float]) -> list[dict[str, Any]]:
    base_lcr = copy.deepcopy(base_runtime["lcr"])
    base_liquidation = copy.deepcopy(base_runtime["liquidation"])
    base_cb = copy.deepcopy(base_runtime["cb"])
    base_fire = copy.deepcopy(base_runtime["fire"])
    base_interbank = copy.deepcopy(base_runtime["interbank"])
    base_modules = copy.deepcopy(base_runtime["modules"])

    no_cb = disable_cb(base_cb)
    full_cb = set_cb_lcr_requirement(set_cb_full_coverage(base_cb), 1.20)
    relaxed_liquidation = set_reaction_lcr_requirement(base_liquidation, 1.00)
    relaxed_cb = set_cb_lcr_requirement(no_cb, 1.00)
    wide_lcr = widen_hqla_definition(base_lcr)
    ap_50 = set_asset_purchase_share(base_fire, 0.50)

    def scenario(
        name: str,
        family: str,
        modules: dict[str, bool],
        lcr: dict[str, Any] | None = None,
        liquidation: dict[str, Any] | None = None,
        cb: dict[str, Any] | None = None,
        fire: dict[str, Any] | None = None,
        interbank: dict[str, Any] | None = None,
        order: int = 999,
    ) -> dict[str, Any]:
        return {
            "scenario": name,
            "scenario_family": family,
            "paper_order": order,
            "alpha_path": list(alpha_path),
            "modules": copy.deepcopy(modules),
            "lcr": copy.deepcopy(lcr if lcr is not None else base_lcr),
            "liquidation": copy.deepcopy(liquidation if liquidation is not None else base_liquidation),
            "cb": copy.deepcopy(cb if cb is not None else base_cb),
            "fire": copy.deepcopy(fire if fire is not None else base_fire),
            "interbank": copy.deepcopy(interbank if interbank is not None else base_interbank),
        }

    first_round_only = set_module_switches(
        base_modules,
        bank_reaction=False,
        policy_intervention=False,
        fire_sales=False,
        interbank_contagion=False,
    )
    reaction_only = set_module_switches(
        base_modules,
        bank_reaction=True,
        policy_intervention=False,
        fire_sales=False,
        interbank_contagion=False,
    )
    reaction_lolr = set_module_switches(
        base_modules,
        bank_reaction=True,
        policy_intervention=True,
        fire_sales=False,
        interbank_contagion=False,
    )
    reaction_lolr_fire = set_module_switches(
        base_modules,
        bank_reaction=True,
        policy_intervention=True,
        fire_sales=True,
        interbank_contagion=False,
    )
    full_engine = set_module_switches(
        base_modules,
        bank_reaction=True,
        policy_intervention=True,
        fire_sales=True,
        interbank_contagion=True,
    )
    no_policy_full_channels = set_module_switches(
        base_modules,
        bank_reaction=True,
        policy_intervention=False,
        fire_sales=True,
        interbank_contagion=True,
    )

    return [
        # Channel attribution: same shock path, progressively activated mechanisms.
        scenario(
            "first_round_only",
            "channel_attribution",
            modules=first_round_only,
            cb=no_cb,
            order=1,
        ),
        scenario(
            "plus_bank_reaction",
            "channel_attribution",
            modules=reaction_only,
            cb=no_cb,
            order=2,
        ),
        scenario(
            "plus_lolr_support",
            "channel_attribution",
            modules=reaction_lolr,
            order=3,
        ),
        scenario(
            "plus_fire_sales",
            "channel_attribution",
            modules=reaction_lolr_fire,
            order=4,
        ),
        scenario(
            "full_engine",
            "channel_attribution",
            modules=full_engine,
            order=5,
        ),
        # Policy counterfactuals: same full transmission environment unless the policy itself is disabled.
        scenario(
            "no_cb_support",
            "policy_counterfactual",
            modules=no_policy_full_channels,
            cb=no_cb,
            order=10,
        ),
        scenario(
            "collateral_capped_lolr",
            "policy_counterfactual",
            modules=full_engine,
            order=11,
        ),
        scenario(
            "full_coverage_lolr",
            "policy_counterfactual",
            modules=full_engine,
            cb=full_cb,
            order=12,
        ),
        scenario(
            "asset_purchases_only",
            "policy_counterfactual",
            modules=no_policy_full_channels,
            cb=no_cb,
            fire=ap_50,
            order=13,
        ),
        scenario(
            "lolr_plus_asset_purchases",
            "policy_counterfactual",
            modules=full_engine,
            fire=ap_50,
            order=14,
        ),
        scenario(
            "relaxed_lcr_100_no_lolr",
            "policy_counterfactual",
            modules=no_policy_full_channels,
            liquidation=relaxed_liquidation,
            cb=relaxed_cb,
            order=15,
        ),
        scenario(
            "widened_hqla_no_lolr",
            "policy_counterfactual",
            modules=no_policy_full_channels,
            lcr=wide_lcr,
            cb=no_cb,
            order=16,
        ),
    ]


# -----------------------------------------------------------------------------
# Summary metrics
# -----------------------------------------------------------------------------


def _safe_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0) > 0
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "t"])


def _safe_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _safe_min(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(s.min()) if not s.empty else float("nan")


def _safe_last_month_value(monthly: pd.DataFrame, col: str) -> float:
    if col not in monthly.columns or monthly.empty:
        return float("nan")
    last_month = monthly["stress_month"].max()
    s = pd.to_numeric(monthly.loc[monthly["stress_month"] == last_month, col], errors="coerce").dropna()
    return float(s.iloc[-1]) if not s.empty else float("nan")


def summarize_for_paper(
    scenario_config: dict[str, Any],
    results: pd.DataFrame,
    monthly: pd.DataFrame,
) -> dict[str, Any]:
    df = results.copy()
    month = "stress_month"
    active_lcr = 1.20
    if "active_lcr_requirement" in df.columns:
        active = pd.to_numeric(df["active_lcr_requirement"], errors="coerce").dropna()
        if not active.empty:
            active_lcr = float(active.median())

    final_month = int(df[month].max())
    final_rows = df.loc[df[month] == final_month].copy()

    if "system_post_interbank_LCR" in monthly.columns:
        min_system_lcr = _safe_min(monthly, "system_post_interbank_LCR")
        final_system_lcr = _safe_last_month_value(monthly, "system_post_interbank_LCR")
    else:
        min_system_lcr = float("nan")
        final_system_lcr = float("nan")

    if "system_post_interbank_HQLA_assets_ratio" in monthly.columns:
        min_system_hqla_assets = _safe_min(monthly, "system_post_interbank_HQLA_assets_ratio")
        final_system_hqla_assets = _safe_last_month_value(monthly, "system_post_interbank_HQLA_assets_ratio")
    else:
        min_system_hqla_assets = float("nan")
        final_system_hqla_assets = float("nan")

    # First-round diagnostics are useful for anchoring every scenario to the same shock.
    min_first_round_system_lcr = _safe_min(monthly, "system_stressed_LCR")
    min_first_round_weakest_lcr = _safe_min(monthly, "minimum_stressed_LCR")

    lcr_col = "post_interbank_LCR" if "post_interbank_LCR" in df.columns else "LCR"
    hqla_assets_col = (
        "post_interbank_HQLA_assets_ratio"
        if "post_interbank_HQLA_assets_ratio" in df.columns
        else "HQLA_assets_ratio"
    )

    terminal = _safe_bool_series(df, "terminal_distress_after_interbank")
    final_terminal = _safe_bool_series(final_rows, "terminal_distress_after_interbank")
    final_unresolved = _safe_bool_series(final_rows, "final_unresolved_shortfall")

    bank_group = df.assign(_terminal=terminal).groupby("bank_id")
    ever_terminal_banks = int(bank_group["_terminal"].max().sum())

    lcr_breaches = df.assign(_breach=pd.to_numeric(df[lcr_col], errors="coerce") < active_lcr)
    hqla_breaches = df.assign(_breach=pd.to_numeric(df[hqla_assets_col], errors="coerce") < 0.10)

    row = {
        "paper_order": int(scenario_config["paper_order"]),
        "scenario": scenario_config["scenario"],
        "scenario_family": scenario_config["scenario_family"],
        "active_lcr_requirement": active_lcr,
        "number_of_months": int(df[month].nunique()),
        "min_first_round_system_LCR": min_first_round_system_lcr,
        "min_first_round_weakest_bank_LCR": min_first_round_weakest_lcr,
        "min_system_LCR": min_system_lcr,
        "min_weakest_bank_LCR": _safe_min(df, lcr_col),
        "final_system_LCR": final_system_lcr,
        "final_weakest_bank_LCR": _safe_min(final_rows, lcr_col),
        "min_system_HQLA_assets": min_system_hqla_assets,
        "min_weakest_bank_HQLA_assets": _safe_min(df, hqla_assets_col),
        "final_system_HQLA_assets": final_system_hqla_assets,
        "final_weakest_bank_HQLA_assets": _safe_min(final_rows, hqla_assets_col),
        "total_cash_shortfall_before_support": _safe_sum(df, "cash_shortfall_before_support"),
        "total_cb_support_granted": _safe_sum(df, "cb_support_granted"),
        "total_support_used_for_outflows": _safe_sum(df, "cb_support_used_for_outflows"),
        "total_support_retained_as_cash_for_lcr": _safe_sum(df, "cb_support_retained_as_cash_for_lcr"),
        "total_asset_sale_proceeds_for_outflows": _safe_sum(df, "asset_sale_proceeds_used_for_outflows"),
        "total_asset_sale_loss": _safe_sum(df, "asset_sale_liquidation_loss"),
        "total_security_valuation_loss": _safe_sum(df, "total_security_valuation_loss"),
        "total_fire_sale_mtm_loss": _safe_sum(df, "total_fire_sale_mtm_loss"),
        "total_interbank_contagion_loss": _safe_sum(df, "interbank_contagion_loss"),
        "total_modeled_loss": (
            _safe_sum(df, "total_security_valuation_loss")
            + _safe_sum(df, "asset_sale_liquidation_loss")
            + _safe_sum(df, "total_fire_sale_mtm_loss")
            + _safe_sum(df, "interbank_contagion_loss")
        ),
        "max_banks_LCR_below_active_requirement": int(lcr_breaches.groupby(month)["_breach"].sum().max()),
        "max_banks_LCR_below_100": int(
            df.assign(_breach=pd.to_numeric(df[lcr_col], errors="coerce") < 1.00)
            .groupby(month)["_breach"]
            .sum()
            .max()
        ),
        "max_banks_HQLA_assets_below_10pct": int(hqla_breaches.groupby(month)["_breach"].sum().max()),
        "ever_terminal_distress_banks": ever_terminal_banks,
        "final_terminal_distress_banks": int(final_terminal.sum()),
        "final_unresolved_shortfall_banks": int(final_unresolved.sum()),
        "final_system_assets": float(pd.to_numeric(final_rows.get("total_assets", pd.Series(dtype=float)), errors="coerce").sum()),
        "final_system_equity": float(pd.to_numeric(final_rows.get("equity", pd.Series(dtype=float)), errors="coerce").sum()),
    }

    return row


def add_scenario_columns(df: pd.DataFrame, scenario: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "scenario", scenario["scenario"])
    out.insert(1, "scenario_family", scenario["scenario_family"])
    out.insert(2, "paper_order", scenario["paper_order"])
    return out


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-grade liquidity stress scenarios.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SCENARIO_CONFIG_PATH),
        help="Path to scenario YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    engine = load_engine_module()
    banks = BankingDataLoader(BANKING_INPUT_PATH).load_banks()

    base_scenario = load_and_validate_scenario_config(args.config)
    stress_mode = get_stress_mode(base_scenario)
    alpha_path = get_alpha_path_for_currency(base_scenario, currency="combined")

    base_runtime = build_runtime_configs_from_scenario(
        scenario_config=base_scenario,
        base_lcr_assumptions=LCR_ASSUMPTIONS,
        base_liquidation_config=LIQUIDATION_WATERFALL_CONFIG,
        base_cb_support_config=CB_SUPPORT_CONFIG,
        base_fire_sale_config=FIRE_SALE_CONFIG,
        base_interbank_contagion_config=INTERBANK_CONTAGION_CONFIG,
    )

    scenarios = build_paper_scenarios(base_runtime=base_runtime, alpha_path=alpha_path)

    scenario_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    bank_rows: list[pd.DataFrame] = []

    print("\nRunning paper-grade scenario set")
    print("=" * 72)

    for sc in scenarios:
        print(f"Running {sc['paper_order']:02d}. {sc['scenario']} [{sc['scenario_family']}]")

        results, market_summary, network_summary = engine.run_dynamic_interbank_contagion_path(
            banks=banks,
            alpha_path=sc["alpha_path"],
            scenario_name=sc["scenario"],
            stress_mode=stress_mode,
            lcr_assumptions=sc["lcr"],
            liquidation_config=sc["liquidation"],
            cb_support_config=sc["cb"],
            fire_sale_config=sc["fire"],
            interbank_contagion_config=sc["interbank"],
            module_config=sc["modules"],
        )

        monthly = engine.build_interbank_monthly_summary(results)

        scenario_rows.append(summarize_for_paper(sc, results, monthly))
        monthly_rows.append(add_scenario_columns(monthly, sc))
        bank_rows.append(add_scenario_columns(results, sc))

    summary = pd.DataFrame(scenario_rows).sort_values("paper_order")
    monthly_summary = pd.concat(monthly_rows, ignore_index=True).sort_values(
        ["paper_order", "stress_month"]
    )
    bank_results = pd.concat(bank_rows, ignore_index=True).sort_values(
        ["paper_order", "stress_month", "bank_id"]
    )

    channel = summary.loc[summary["scenario_family"] == "channel_attribution"].copy()
    policy = summary.loc[summary["scenario_family"] == "policy_counterfactual"].copy()

    summary.to_csv(TABLES_DIR / "paper_scenario_summary.csv", index=False)
    monthly_summary.to_csv(TABLES_DIR / "paper_scenario_monthly_summary.csv", index=False)
    bank_results.to_csv(TABLES_DIR / "paper_scenario_bank_results.csv", index=False)
    channel.to_csv(TABLES_DIR / "paper_channel_attribution_summary.csv", index=False)
    policy.to_csv(TABLES_DIR / "paper_policy_counterfactual_summary.csv", index=False)

    display_cols = [
        "paper_order",
        "scenario",
        "scenario_family",
        "min_system_LCR",
        "min_weakest_bank_LCR",
        "final_system_LCR",
        "min_system_HQLA_assets",
        "total_cb_support_granted",
        "total_fire_sale_mtm_loss",
        "total_interbank_contagion_loss",
        "final_terminal_distress_banks",
    ]

    print("\nPAPER SCENARIO SUMMARY")
    print(summary[display_cols].to_string(index=False))

    print("\nSaved:")
    for filename in [
        "paper_scenario_summary.csv",
        "paper_scenario_monthly_summary.csv",
        "paper_scenario_bank_results.csv",
        "paper_channel_attribution_summary.csv",
        "paper_policy_counterfactual_summary.csv",
    ]:
        print(f"  outputs/tables/{filename}")


if __name__ == "__main__":
    main()
