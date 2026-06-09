"""
Public Armenian banking-sector reference data.

This file contains public headline reference values used to generate
the synthetic input workbook. The core model should not depend directly
on this file in the final real-data version.
"""

import pandas as pd


ARMENIA_BANK_REFERENCE_2025 = pd.DataFrame({
    "bank_name": [
        "ACBA Bank",
        "Ameriabank",
        "Amio Bank",
        "AraratBank",
        "Ardshinbank",
        "ArmEconomBank",
        "ArmSwissBank",
        "ArtsakhBank",
        "Byblos Bank Armenia",
        "Converse Bank",
        "Evocabank",
        "Fast Bank",
        "ID Bank",
        "Inecobank",
        "Mellat Bank",
        "Unibank",
        "VTB Bank Armenia",
    ],

    "legal_form": [
        "OJSC", "CJSC", "CJSC", "OJSC", "CJSC", "OJSC", "CJSC",
        "CJSC", "CJSC", "CJSC", "CJSC", "CJSC", "CJSC", "CJSC",
        "CJSC", "OJSC", "CJSC",
    ],

    "total_assets": [
        1_067_703, 2_334_766, 957_467, 425_071, 2_670_358,
        604_499, 474_466, 186_099, 135_253, 604_299,
        735_729, 422_890, 512_633, 918_938, 128_287,
        394_648, 257_217,
    ],

    "interest_earning_assets": [
        908_376, 2_146_452, 851_554, 365_367, 2_221_473,
        515_330, 429_686, 170_056, 127_194, 563_586,
        591_127, 382_135, 442_253, 769_097, 115_067,
        303_984, 213_427,
    ],

    "gross_advances_to_customers": [
        835_092, 1_836_389, 608_883, 304_470, 1_784_588,
        409_954, 238_999, 39_990, 94_039, 397_469,
        414_482, 348_033, 355_952, 574_917, 55_065,
        279_317, 190_096,
    ],

    "loans": [
        736_844, 1_672_456, 564_781, 234_157, 1_464_143,
        401_239, 221_280, 26_560, 72_779, 342_312,
        360_015, 337_489, 269_264, 517_269, 47_924,
        266_437, 164_805,
    ],

    "total_liabilities": [
        872_759, 1_997_818, 764_594, 346_141, 2_244_395,
        515_614, 384_064, 146_338, 99_340, 498_130,
        630_522, 339_151, 427_214, 795_731, 62_572,
        341_987, 189_173,
    ],

    "deposits": [
        622_062, 1_362_274, 519_869, 264_608, 1_677_998,
        209_353, 181_820, 67_064, 86_163, 392_622,
        434_049, 247_969, 259_763, 559_594, 8_111,
        271_528, 170_456,
    ],

    "equity": [
        194_944, 336_948, 192_874, 78_929, 425_964,
        88_885, 90_402, 39_761, 35_912, 106_170,
        105_206, 83_738, 85_419, 123_206, 65_715,
        52_661, 68_045,
    ],

    "charter_capital": [
        89_775, 54_466, 190_094, 9_367, 200_000,
        31_578, 10_000, 23_261, 26_249, 19_948,
        23_000, 46_100, 51_026, 54_546, 32_100,
        24_925, 63_776,
    ],
})


ARMENIA_SYSTEM_REFERENCE_2025 = {
    "n_banks": 17,
    "total_assets": 12_830_323,
    "interest_earning_assets": 11_116_162,
    "gross_advances_to_customers": 8_767_734,
    "loan_portfolio": 7_699_753,
    "total_liabilities": 10_655_544,
    "deposits": 7_335_303,
    "equity": 2_174_779,

    "loan_portfolio_to_assets": 0.600,
    "liquidity_ratio": 0.290,
    "equity_to_assets": 0.170,
    "capital_adequacy_ratio": 0.207,
    "loans_to_deposits": 1.05,

    "top_5_asset_share": 0.620,
    "hhi_assets": 0.1087,

    "fx_deposit_share": 0.439,
    "resident_deposit_share": 0.788,
    "resident_term_deposit_share": 0.572,

    "household_resident_deposit_share": 0.58,
    "private_company_resident_deposit_share": 0.28,
    "ofi_resident_deposit_share": 0.09,
    "non_profit_resident_deposit_share": 0.04,
    "state_owned_company_resident_deposit_share": 0.01,
}


def add_derived_bank_reference_fields(bank_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived ratios and identifiers to the public bank reference table.

    The public headline values are kept exactly as reported. Small accounting
    gaps may exist because the source values are reported in AMD million.
    """
    out = bank_reference.copy()

    out["bank_id"] = [f"B{i + 1:02d}" for i in range(len(out))]
    out["market_share"] = out["total_assets"] / out["total_assets"].sum()

    out["balance_sheet_rounding_gap"] = (
        out["total_assets"] - out["total_liabilities"] - out["equity"]
    )

    out["loans_assets_ratio"] = out["loans"] / out["total_assets"]
    out["deposits_assets_ratio"] = out["deposits"] / out["total_assets"]
    out["equity_assets_ratio"] = out["equity"] / out["total_assets"]
    out["deposits_liabilities_ratio"] = out["deposits"] / out["total_liabilities"]
    out["liabilities_assets_ratio"] = out["total_liabilities"] / out["total_assets"]

    return out