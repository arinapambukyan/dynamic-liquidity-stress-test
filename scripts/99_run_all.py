"""
Run the full liquidity stress-testing pipeline.

Pipeline:

    1. Generate synthetic input workbook.
    2. Validate baseline accounting and currency consistency.
    3. Run the canonical combined full-engine stress test.
    4. Run the AMD/FX currency-aware full-engine stress test.
    5. Run deterministic sensitivity scenarios.
    6. Run Monte Carlo simulations.
    7. Plot direct and indirect contagion networks.
    8. Generate reporting charts.

Run from project root:

    PYTHONPATH=src python scripts/99_run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PIPELINE = [
    "scripts/00_generate_input.py",
    "scripts/01_validate_baseline.py",
    "scripts/10_run_base_engine.py",
    "scripts/20_run_currency_engine.py",
    "scripts/30_run_sensitivity.py",
    "scripts/40_run_monte_carlo.py",
    "scripts/50_plot_indirect_contagion_network.py",
    "scripts/51_plot_direct_interbank_network.py",
    "scripts/60_make_charts.py",
]


def run_script(script_path: str) -> None:
    print(f"\n=== Running {script_path} ===")

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"

    subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    for script in PIPELINE:
        run_script(script)

    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()