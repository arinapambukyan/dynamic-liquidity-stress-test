"""
Generate the synthetic banking input workbook.

This script creates the Excel input file used by the liquidity stress-testing
model. The workbook contains:

    banks      = combined bank-level balance sheet
    banks_amd  = AMD-only synthetic balance sheet
    banks_fx   = FX-only synthetic balance sheet

Run from project root:

    PYTHONPATH=src python scripts/00_generate_input.py
"""
from liquidity_stress.config import BANKING_INPUT_PATH, SYSTEM_CONFIG
from liquidity_stress.reference_data import (
    ARMENIA_BANK_REFERENCE_2025,
    ARMENIA_SYSTEM_REFERENCE_2025,
)
from liquidity_stress.synthetic_data_generator import SyntheticBankingDataGenerator


def main() -> None:
    generator = SyntheticBankingDataGenerator(
        bank_reference=ARMENIA_BANK_REFERENCE_2025,
        system_reference=ARMENIA_SYSTEM_REFERENCE_2025,
        system_config=SYSTEM_CONFIG,
        output_path=BANKING_INPUT_PATH,
    )

    generator.export_excel()


if __name__ == "__main__":
    main()