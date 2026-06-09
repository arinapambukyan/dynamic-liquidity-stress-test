"""
Scenario configuration loader and runtime-config builder.

This file reads user-editable YAML scenario files and converts them into the
actual runtime dictionaries used by the stress-testing engine.

YAML controls:
    - stress mode
    - alpha paths
    - module switches
    - LCR thresholds / reaction targets
    - CBA policy settings
    - optional fire-sale and interbank overrides

The core idea:
    scenario_base.yaml is the user-facing configuration layer;
    src/liquidity_stress/config.py provides default model parameters.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODULE_SWITCHES = {
    "first_round": True,
    "bank_reaction": True,
    "policy_intervention": True,
    "fire_sales": True,
    "interbank_contagion": True,
    "reputation_feedback": False,
}


SUPPORTED_STRESS_MODES = {
    "full_systemic",
    "outflow_only",
}


def load_scenario_config(path: str | Path) -> dict[str, Any]:
    """
    Load a scenario YAML file.

    Parameters
    ----------
    path:
        Path to the scenario YAML file.

    Returns
    -------
    dict
        Scenario configuration dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scenario config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Scenario config file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError("Scenario config must parse to a dictionary.")

    return config


def get_stress_mode(config: dict[str, Any]) -> str:
    """
    Read stress mode from scenario config.

    Supported YAML forms:

        stress_mode: outflow_only

    or:

        stress:
          mode: outflow_only

    Default:
        full_systemic
    """
    if "stress_mode" in config:
        stress_mode = config["stress_mode"]
    else:
        stress_mode = config.get("stress", {}).get("mode", "full_systemic")

    if stress_mode not in SUPPORTED_STRESS_MODES:
        raise ValueError(
            f"Unknown stress mode '{stress_mode}'. "
            f"Allowed values: {sorted(SUPPORTED_STRESS_MODES)}"
        )

    return stress_mode


def get_module_switches(config: dict[str, Any]) -> dict[str, bool]:
    """
    Return module switches with defaults filled in.

    Missing module switches default to the canonical full-engine behavior.
    """
    modules = copy.deepcopy(DEFAULT_MODULE_SWITCHES)

    user_modules = config.get("modules", {})

    if user_modules is None:
        user_modules = {}

    if not isinstance(user_modules, dict):
        raise ValueError("modules must be a dictionary of true/false switches.")

    for module_name, value in user_modules.items():
        if module_name not in modules:
            raise ValueError(
                f"Unknown module switch '{module_name}'. "
                f"Allowed switches: {sorted(modules)}"
            )

        if not isinstance(value, bool):
            raise ValueError(
                f"Module switch '{module_name}' must be true or false."
            )

        modules[module_name] = value

    return modules


def get_alpha_path_for_currency(
    config: dict[str, Any],
    currency: str,
) -> list[float]:
    """
    Return the correct alpha path for a currency mode.

    currency:
        combined -> severity.alpha_path
        AMD      -> severity.alpha_path_amd if present, else alpha_path
        FX       -> severity.alpha_path_fx if present, else alpha_path
    """
    severity = config.get("severity", {})
    fallback = severity.get("alpha_path")

    if fallback is None:
        raise ValueError("scenario config must define severity.alpha_path")

    currency_upper = currency.upper()

    if currency_upper == "AMD":
        return list(severity.get("alpha_path_amd", fallback))

    if currency_upper == "FX":
        return list(severity.get("alpha_path_fx", fallback))

    if currency.lower() in ["combined", "all"]:
        return list(fallback)

    raise ValueError("currency must be one of: combined, AMD, FX")


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively update a dictionary.

    This lets YAML override nested fields such as:
        collateral_haircuts:
          government_securities: 0.05
    without wiping the rest of the base dictionary.
    """
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value

    return base


def _optional_dict(value: Any, field_name: str) -> dict[str, Any]:
    """
    Normalize optional YAML override blocks.
    """
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    raise ValueError(f"{field_name} must be a dictionary if provided.")


def build_runtime_configs_from_scenario(
    scenario_config: dict[str, Any],
    base_lcr_assumptions: dict[str, Any],
    base_liquidation_config: dict[str, Any],
    base_cb_support_config: dict[str, Any],
    base_fire_sale_config: dict[str, Any],
    base_interbank_contagion_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build scenario-specific runtime configuration dictionaries.

    This is the bridge between user-editable YAML and the actual model engine.

    Returns
    -------
    dict with keys:
        lcr
        liquidation
        cb
        fire
        interbank
        modules
    """
    modules = get_module_switches(scenario_config)

    lcr_config = copy.deepcopy(base_lcr_assumptions)
    liquidation_config = copy.deepcopy(base_liquidation_config)
    cb_config = copy.deepcopy(base_cb_support_config)
    fire_config = copy.deepcopy(base_fire_sale_config)
    interbank_config = copy.deepcopy(base_interbank_contagion_config)

    # ------------------------------------------------------------
    # Optional direct overrides.
    # ------------------------------------------------------------
    lcr_overrides = scenario_config.get(
        "lcr_assumptions",
        scenario_config.get("lcr", {}),
    )
    _deep_update(lcr_config, _optional_dict(lcr_overrides, "lcr_assumptions"))

    liquidation_overrides = scenario_config.get(
        "liquidation",
        scenario_config.get("bank_reaction", {}),
    )
    _deep_update(
        liquidation_config,
        _optional_dict(liquidation_overrides, "liquidation"),
    )

    fire_overrides = scenario_config.get("fire_sales", {})
    _deep_update(
        fire_config,
        _optional_dict(fire_overrides, "fire_sales"),
    )

    interbank_overrides = scenario_config.get("interbank_contagion", {})
    _deep_update(
        interbank_config,
        _optional_dict(interbank_overrides, "interbank_contagion"),
    )

    # ------------------------------------------------------------
    # Thresholds.
    # ------------------------------------------------------------
    thresholds = scenario_config.get("thresholds", {})

    if thresholds is None:
        thresholds = {}

    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dictionary if provided.")

    current_lcr_requirement = float(
        thresholds.get(
            "lcr_minimum",
            liquidation_config.get("reaction_lcr_trigger", 1.0),
        )
    )

    reaction_lcr_trigger = float(
        thresholds.get(
            "reaction_lcr_trigger",
            current_lcr_requirement,
        )
    )

    reaction_target_mode = str(
        thresholds.get(
            "reaction_target_mode",
            liquidation_config.get("reaction_target_mode", "trigger"),
        )
    )

    reaction_target_cap = float(
        thresholds.get(
            "reaction_target_cap",
            reaction_lcr_trigger,
        )
    )

    liquidation_config["active_lcr_requirement"] = reaction_lcr_trigger
    liquidation_config["reaction_lcr_trigger"] = reaction_lcr_trigger
    liquidation_config["reaction_target_mode"] = reaction_target_mode
    liquidation_config["reaction_target_cap"] = reaction_target_cap

    # ------------------------------------------------------------
    # CBA policy intervention.
    # ------------------------------------------------------------
    policy_overrides = scenario_config.get("policy_intervention", {})

    if isinstance(policy_overrides, bool):
        policy_enabled_from_block = policy_overrides
        policy_overrides = {}
    else:
        policy_enabled_from_block = None
        policy_overrides = _optional_dict(
            policy_overrides,
            "policy_intervention",
        )

    _deep_update(cb_config, policy_overrides)

    # Module switch has priority over the block default.
    policy_module_enabled = bool(modules.get("policy_intervention", True))

    if policy_enabled_from_block is not None:
        policy_block_enabled = bool(policy_enabled_from_block)
    else:
        policy_block_enabled = bool(cb_config.get("enabled", True))

    cb_config["enabled"] = policy_module_enabled and policy_block_enabled
    cb_config["support_timing"] = "after_asset_sales"

    # If the YAML changes the active LCR requirement but does not explicitly
    # restate CBA thresholds, make CBA support follow the active requirement.
    if "illiquidity_lcr_threshold" not in policy_overrides:
        cb_config["illiquidity_lcr_threshold"] = reaction_lcr_trigger

    if "lcr_support_target" not in policy_overrides:
        cb_config["lcr_support_target"] = reaction_lcr_trigger

    if "support_lcr_gap" not in policy_overrides:
        cb_config["support_lcr_gap"] = bool(
            cb_config.get("support_lcr_gap", True)
        )

    # ------------------------------------------------------------
    # Module switches for downstream engines.
    # ------------------------------------------------------------
    fire_config["enabled"] = (
        bool(modules.get("fire_sales", True))
        and bool(fire_config.get("enabled", True))
    )

    interbank_config["enabled"] = (
        bool(modules.get("interbank_contagion", True))
        and bool(interbank_config.get("enabled", True))
    )

    return {
        "lcr": lcr_config,
        "liquidation": liquidation_config,
        "cb": cb_config,
        "fire": fire_config,
        "interbank": interbank_config,
        "modules": modules,
    }


def validate_scenario_config(config: dict[str, Any]) -> None:
    """
    Validate basic scenario configuration rules.

    Current constraints
    -------------------
    - Stress horizon must be between 1 and 12 months.
    - If alpha_path is provided, its length must match horizon_months.
    - Alpha must be non-negative.
    - Module switches must be boolean.
    - Stress mode must be supported.
    """
    if "time" not in config or "horizon_months" not in config["time"]:
        raise ValueError("Scenario config must define time.horizon_months.")

    horizon = int(config["time"]["horizon_months"])

    if horizon < 1:
        raise ValueError("Stress horizon must be at least 1 month.")

    if horizon > 12:
        raise ValueError(
            "Stress horizon cannot exceed 12 months in the current version."
        )

    severity = config.get("severity", {})

    if "alpha" in severity and severity["alpha"] < 0:
        raise ValueError("Alpha must be non-negative.")

    if "alpha_grid" in severity:
        if any(alpha < 0 for alpha in severity["alpha_grid"]):
            raise ValueError("All alpha_grid values must be non-negative.")

    if "alpha_path" in severity:
        alpha_path = severity["alpha_path"]

        if len(alpha_path) != horizon:
            raise ValueError("Length of alpha_path must equal horizon_months.")

        if any(alpha < 0 for alpha in alpha_path):
            raise ValueError("All alpha_path values must be non-negative.")

    for currency_alpha_key in ["alpha_path_amd", "alpha_path_fx"]:
        if currency_alpha_key in severity:
            alpha_path_currency = severity[currency_alpha_key]

            if len(alpha_path_currency) != horizon:
                raise ValueError(
                    f"Length of {currency_alpha_key} must equal horizon_months."
                )

            if any(alpha < 0 for alpha in alpha_path_currency):
                raise ValueError(
                    f"All values in {currency_alpha_key} must be non-negative."
                )

    # Validate modules and stress mode.
    get_module_switches(config)
    get_stress_mode(config)


def load_and_validate_scenario_config(path: str | Path) -> dict[str, Any]:
    """
    Load and validate a scenario YAML file.
    """
    config = load_scenario_config(path)
    validate_scenario_config(config)
    return config