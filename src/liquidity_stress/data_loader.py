"""
Data loader for the liquidity stress-testing model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CURRENCY_SHEET_MAP = {
    "combined": "banks",
    "all": "banks",
    "AMD": "banks_amd",
    "amd": "banks_amd",
    "FX": "banks_fx",
    "fx": "banks_fx",
}


class BankingDataLoader:
    """
    Load banking-system input data from an Excel workbook.

    Expected workbook structure:
        banks
        banks_amd
        banks_fx
    """

    def __init__(self, input_path: str | Path):
        self.input_path = Path(input_path)

    def read_workbook(self) -> dict[str, pd.DataFrame]:
        return pd.read_excel(self.input_path, sheet_name=None)

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        workbook = self.read_workbook()

        if sheet_name not in workbook:
            available = sorted(workbook.keys())
            raise KeyError(
                f"Sheet '{sheet_name}' not found in {self.input_path}. "
                f"Available sheets: {available}"
            )

        return workbook[sheet_name]

    def load_banks(self, currency: str = "combined") -> pd.DataFrame:
        """
        Load bank-level table by currency mode.

        Parameters
        ----------
        currency:
            "combined", "AMD", or "FX".

        Backward-compatible default:
            load_banks() returns the combined banks sheet.
        """
        if currency not in CURRENCY_SHEET_MAP:
            raise ValueError(
                "currency must be one of: combined, AMD, FX"
            )

        sheet_name = CURRENCY_SHEET_MAP[currency]
        banks = self.load_sheet(sheet_name).copy()

        if "currency" not in banks.columns:
            if sheet_name == "banks":
                banks["currency"] = "combined"
            elif sheet_name == "banks_amd":
                banks["currency"] = "AMD"
            elif sheet_name == "banks_fx":
                banks["currency"] = "FX"

        return banks

    def load_banks_combined(self) -> pd.DataFrame:
        return self.load_banks("combined")

    def load_banks_amd(self) -> pd.DataFrame:
        return self.load_banks("AMD")

    def load_banks_fx(self) -> pd.DataFrame:
        return self.load_banks("FX")

    def load_currency_sheets(self) -> dict[str, pd.DataFrame]:
        """
        Load combined, AMD, and FX sheets together.
        """
        return {
            "combined": self.load_banks("combined"),
            "AMD": self.load_banks("AMD"),
            "FX": self.load_banks("FX"),
        }