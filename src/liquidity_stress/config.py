"""
Configuration settings for the liquidity stress-testing model.

This file contains user-editable assumptions only.
No model logic should be placed here.
"""

from pathlib import Path

import pandas as pd


# ==============================================================
# File paths
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = OUTPUT_DIR

TABLES_DIR = OUTPUT_DIR / "tables"
CHARTS_DIR = OUTPUT_DIR / "charts"

BANKING_INPUT_PATH = INPUT_DIR / "banking_input.xlsx"
INTERBANK_EXPOSURE_MATRIX_PATH = INPUT_DIR / "interbank_exposure_matrix.xlsx"

# ==============================================================
# System configuration
# ==============================================================

SYSTEM_CONFIG = {
    "n_banks": 17,
    "system_total_assets": 12_830_323.0,
    "random_seed": 42,
    "currencies": ["AMD", "FX"],
    "balance_tolerance": 1e-6,
    "anchor_year": 2025,
    "use_synthetic_ncof_calibration": True,   # set False when loading REAL bank LCR/NCOF
}


# ==============================================================
# Calibration targets
# ==============================================================

CALIBRATION_TARGETS = {
    "system_liquidity_ratio_target": 0.290,
    "system_fx_deposit_share_target": 0.439,
    "top_5_asset_share_target": 0.620,
    "hhi_assets_target": 0.1087,
}

# ==============================================================
# Synthetic baseline LCR calibration
# ==============================================================

BASELINE_LCR_TARGETS = {
    # Armenia's regulatory LCR minimum is 120%.
    # Synthetic banks should start above the minimum, but not mechanically
    # at 300-450%.
    "large_universal": (1.35, 1.65),
    "mid_universal": (1.35, 1.80),
    "retail_deposit_heavy": (1.35, 1.75),
    "loan_heavy": (1.25, 1.55),
    "securities_interbank_active": (1.55, 2.20),
    "default": (1.35, 1.80),
}

BASELINE_LCR_HARD_FLOOR = 1.20
BASELINE_LCR_HARD_CAP = 2.30

# ==============================================================
# LCR assumptions
# ==============================================================

LCR_ASSUMPTIONS = {
    "retail_deposit_runoff": 0.07,
    "corporate_deposit_runoff": 0.25,
    "other_deposit_runoff": 0.20,
    "wholesale_funding_runoff": 0.75,
    "interbank_liability_runoff": 1.00,

    # Baseline maturity-window assumptions.
    # Only the part of wholesale/interbank funding assumed to mature inside
    # the 30-day LCR window enters baseline NCOF.
    "wholesale_funding_30d_share": 0.50,
    "interbank_liability_30d_share": 0.60,

    "loan_maturing_30d_share": 0.03,
    "loan_inflow_recognition_rate": 0.50,
    "interbank_asset_inflow_rate": 0.75,

    "inflow_cap_share_of_outflows": 0.75,
    "minimum_ncof": 1.0,
    "cb_liquidity_funding_runoff": 0.0,

    # HQLA recognition weights.
    # These are used by compute_lcr_from_balance_sheet.
    "cash_hqla_weight": 1.00,
    "government_securities_hqla_weight": 0.90,
    "bank_securities_hqla_weight": 0.35,
    "corporate_securities_hqla_weight": 0.10,
}

# ==============================================================
# FX assumptions
# ==============================================================

FX_ASSUMPTIONS = {
    "system_fx_liability_share_anchor": 0.439,

    "fx_asset_to_fx_liability_center": 1.00,
    "fx_asset_to_fx_liability_sd": 0.08,

    "min_fx_liability_share": 0.25,
    "max_fx_liability_share": 0.65,
    "min_fx_asset_share": 0.20,
    "max_fx_asset_share": 0.65,
}


# ==============================================================
# First-round shock scenario controls
# ==============================================================

FIRST_ROUND_SCENARIO_CONFIG = {
    "alpha": 1.00,
    "alpha_grid": [0.50, 0.75, 1.00, 1.25, 1.50],
    "scenario_name": "baseline_first_round_stress",
}


FIRST_ROUND_STRESS_VECTOR = pd.DataFrame({
    "component": [
        "retail_deposits",
        "corporate_deposits",
        "other_deposits",
        "wholesale_funding",
        "interbank_liabilities",

        "government_securities",
        "bank_securities",
        "corporate_securities",

        "loan_inflows",
        "interbank_asset_inflows",

        "fx_liability_pressure",
        "fx_asset_haircut",
    ],

    "shock_type": [
        "runoff",
        "runoff",
        "runoff",
        "runoff",
        "runoff",

        "asset_haircut",
        "asset_haircut",
        "asset_haircut",

        "inflow_reduction",
        "inflow_reduction",

        "fx_pressure",
        "fx_haircut",
    ],

    "base_weight": [
        LCR_ASSUMPTIONS["retail_deposit_runoff"],
        LCR_ASSUMPTIONS["corporate_deposit_runoff"],
        LCR_ASSUMPTIONS["other_deposit_runoff"],
        LCR_ASSUMPTIONS["wholesale_funding_runoff"],
        LCR_ASSUMPTIONS["interbank_liability_runoff"],

        0.03,
        0.08,
        0.12,

        0.10,
        0.20,

        0.05,
        0.03,
    ],

    "cap": [
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,

        0.50,
        0.75,
        0.90,

        0.90,
        0.90,

        0.75,
        0.50,
    ],

    "description": [
        "Retail deposit runoff rate",
        "Corporate deposit runoff rate",
        "Other deposit runoff rate",
        "Wholesale funding runoff rate",
        "Interbank liability runoff rate",

        "Additional haircut on government securities",
        "Additional haircut on bank securities",
        "Additional haircut on corporate securities",

        "Reduction in recognized loan inflows",
        "Reduction in recognized interbank asset inflows",

        "Additional FX liability pressure",
        "Haircut on FX liquid resources proxy",
    ],
})


# ==============================================================
# Bank reaction and liquidation waterfall controls
# ==============================================================

LIQUIDATION_WATERFALL_CONFIG = {
        "liquidation_order": [
        "government_securities",
        "bank_securities",
        "corporate_securities",
    ],

    "regulatory_reaction_order": [
        "corporate_securities",
        "bank_securities",
    ],

    "minimum_net_hqla_gain_per_unit": 0.10,

    "sale_haircuts": {
        "government_securities": 0.04,
        "bank_securities": 0.10,
        "corporate_securities": 0.18,
    },

    "hqla_recognition_weights": {
        "government_securities": 0.90,
        "bank_securities": 0.35,
        "corporate_securities": 0.10,
    },

    "reaction_lcr_trigger": 1.20,
    "reaction_target_mode": "trigger",
    "reaction_target_cap": 1.20,
}

# ==============================================================
# Central bank liquidity support controls
# ==============================================================
CB_SUPPORT_CONFIG = {
    "enabled": True,
    "support_timing": "after_asset_sales",
    "support_mode": "collateral_capped",

    "eligible_collateral": [
        "government_securities",
        "bank_securities",
        "corporate_securities",
    ],

    "collateral_haircuts": {
        "government_securities": 0.05,
        "bank_securities": 0.15,
        "corporate_securities": 0.25,
    },

    "outside_30d_lcr_window": True,
    "system_support_cap": 250_000.0,

    "support_only_solvent_illiquid": True,
    "solvency_equity_threshold": 0.0,

    "illiquidity_cash_shortfall_enabled": True,

    # CBA can support if post-reaction LCR is below the minimum.
    "illiquidity_lcr_threshold": 1.20,
    "support_lcr_gap": True,
    "lcr_support_target": 1.20,

    # Monitor but do not use as support trigger.
    "illiquidity_hqla_assets_threshold": None,
    "support_hqla_assets_gap": False,
    "hqla_assets_support_target": 0.10,
}

# ==============================================================
# Fire-sale price impact controls
# ==============================================================

FIRE_SALE_CONFIG = {
    "enabled": True,

    # Indirect contagion is modeled only through state/government bonds.
    #
    # Logic:
    #   bank sells government securities
    #   -> aggregate state-bond sales depress the state-bond price
    #   -> all banks holding government securities take mark-to-market losses
    #   -> HQLA and equity are reduced
    #
    # Bank and corporate securities may still be sold in the bank-level
    # liquidation waterfall, but they do not create system-wide price-mediated
    # contagion in the current data setup.
    "asset_columns": [
        "government_securities",
    ],

    # Fraction of the state-bond market sold used for calibration.
    # Example: if reference_sale_fraction = 0.05 and
    # price_impact_at_reference_sale_fraction = 0.02,
    # then selling 5% of the modeled state-bond market lowers the price by 2%.
    "reference_sale_fraction": 0.05,

    "price_impact_at_reference_sale_fraction": {
        "government_securities": 0.02,
    },

    # Avoid numerical collapse to exactly zero.
    # Keep this conservative for now; we can later make it 0.50 / 0.70 / 0.80
    # in a market-depth sensitivity setup.
    "minimum_market_price": 0.01,

    # Van den End-style CBA asset-purchase intervention.
    # This reduces the effective market sale fraction for state bonds.
    "central_bank_asset_purchases": {
        "enabled": False,

        "purchase_share_of_bank_sales": {
            "government_securities": 0.50,
        },
    },

    # Example:
    # A multiplier of 2.0 means total market depth is assumed to be twice the
    # modeled banking-sector holdings.
    "market_depth_multiplier_by_asset": {
        "government_securities": 1.0,
    },
}

# ==============================================================
# Interbank contagion controls
# ==============================================================

INTERBANK_CONTAGION_CONFIG = {
    "enabled": True,

    # Loss-given-default on interbank exposure to a terminally distressed bank.
    # 0.60 means creditors recover 40% and lose 60%.
    "loss_given_default": 0.60,

    # Maximum contagion rounds inside one stress month.
    "max_contagion_rounds": 5,

    # Numerical tolerance.
    "tolerance": 1e-9,

    # Fraction of interbank assets/liabilities assumed to be inside the
    # domestic network. 1.0 means all modeled interbank positions are networked.
    "network_exposure_scale": 1.00,

    "liquidity_resolution_floor": 1.00,
    "collateral_exhaustion_tol": 1e-6,
}

# ==============================================================
# Interbank exposure matrix input controls
# ==============================================================

INTERBANK_EXPOSURE_CONFIG = {
    # "generate_if_missing":
    #     if the Excel matrix exists, read it;
    #     if it does not exist, generate once, save to Excel, then read it.
    #
    # "read_existing":
    #     require the Excel matrix to already exist.
    "source": "generate_if_missing",

    "path": INTERBANK_EXPOSURE_MATRIX_PATH,
    "sheet_name": "exposures",
    "format": "matrix",

    # Used only if source is generate_if_missing and file does not exist.
    "network_exposure_scale": INTERBANK_CONTAGION_CONFIG["network_exposure_scale"],
}