"""
Build and validate the baseline banking input.

This script loads the generated or provided banking input workbook, validates
accounting identities, checks AMD/FX consistency, and saves baseline tables.

Run from project root:

    PYTHONPATH=src python scripts/01_validate_baseline.py
"""
import pandas as pd

from liquidity_stress.config import (
    BANKING_INPUT_PATH,
    SYSTEM_CONFIG,
    TABLES_DIR,
)
from liquidity_stress.data_loader import BankingDataLoader
from liquidity_stress.validation import (
    run_baseline_sanity_checks,
    run_currency_sheet_consistency_checks,
)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    loader = BankingDataLoader(BANKING_INPUT_PATH)

    banks = loader.load_banks("combined")
    banks_amd = loader.load_banks("AMD")
    banks_fx = loader.load_banks("FX")

    combined_checks = run_baseline_sanity_checks(
        banks=banks,
        tolerance=SYSTEM_CONFIG["balance_tolerance"],
    )

    amd_checks = run_baseline_sanity_checks(
        banks=banks_amd,
        tolerance=SYSTEM_CONFIG["balance_tolerance"],
    )

    fx_checks = run_baseline_sanity_checks(
        banks=banks_fx,
        tolerance=SYSTEM_CONFIG["balance_tolerance"],
    )

    currency_checks = run_currency_sheet_consistency_checks(
        banks=banks,
        banks_amd=banks_amd,
        banks_fx=banks_fx,
        tolerance=SYSTEM_CONFIG["balance_tolerance"],
    )

    combined_checks.insert(0, "sheet", "banks")
    amd_checks.insert(0, "sheet", "banks_amd")
    fx_checks.insert(0, "sheet", "banks_fx")
    currency_checks.insert(0, "sheet", "currency_consistency")

    checks = pd.concat(
        [combined_checks, amd_checks, fx_checks, currency_checks],
        ignore_index=True,
    )

    banks.to_csv(TABLES_DIR / "baseline_banks.csv", index=False)
    banks_amd.to_csv(TABLES_DIR / "baseline_banks_amd.csv", index=False)
    banks_fx.to_csv(TABLES_DIR / "baseline_banks_fx.csv", index=False)
    checks.to_csv(TABLES_DIR / "baseline_sanity_checks.csv", index=False)

    print("Baseline combined banks saved.")
    print("Baseline AMD banks saved.")
    print("Baseline FX banks saved.")
    print("Baseline sanity checks saved.")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()