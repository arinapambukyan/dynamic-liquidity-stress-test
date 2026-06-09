# Liquidity Stress-Testing Framework for the Armenian Banking System

This repository implements a multi-period liquidity stress-testing framework for the Armenian banking system. The model combines LCR-based first-round liquidity shocks, bank-level asset-sale reactions, central-bank liquidity support, common-asset fire-sale feedback, and direct interbank contagion.

The current input data are synthetic and Armenia-calibrated using public bank-level balance-sheet references. The framework is designed so that the synthetic workbook can later be replaced by a real supervisory input workbook with the same sheet structure.

## Model overview

The canonical full-engine pipeline is:

1. Generate or load bank-level balance-sheet input data.
2. Validate baseline accounting, liquidity, and AMD/FX consistency.
3. Apply monthly first-round liquidity shocks.
4. Model bank-level asset-sale reactions.
5. Model central-bank liquidity support.
6. Apply common-asset fire-sale price-impact feedback.
7. Propagate direct interbank contagion.
8. Run sensitivity and Monte Carlo scenarios.
9. Produce tables, network figures, and reporting charts.

The main deterministic model is implemented in:

```bash
scripts/10_run_base_engine.py
```

## Repository structure

```text
liquidity-stress-armenia/
│
├── README.md
├── requirements.txt
├── configs/
│   └── scenario_base.yaml
│
├── data/
│   └── input/
│       ├── banking_input.xlsx
│       ├── interbank_exposure_matrix.xlsx
│       ├── interbank_exposure_matrix_amd.xlsx
│       └── interbank_exposure_matrix_fx.xlsx
│
├── docs/
│   ├── architecture.mmd
│   └── audits/
│       └── path_audit.txt
│
├── outputs/
│   ├── architecture/
│   ├── charts/
│   └── tables/
│
├── scripts/
│   ├── 00_generate_input.py
│   ├── 01_validate_baseline.py
│   ├── 10_run_base_engine.py
│   ├── 20_run_currency_engine.py
│   ├── 30_run_sensitivity.py
│   ├── 40_run_monte_carlo.py
│   ├── 50_plot_indirect_contagion_network.py
│   ├── 51_plot_direct_interbank_network.py
│   ├── 60_make_charts.py
│   ├── 99_run_all.py
│   └── diagnostics/
│       ├── 02_run_first_round_only.py
│       ├── 03_run_first_round_currency_only.py
│       ├── 04_run_bank_reaction_only.py
│       ├── 05_run_policy_support_old_sequence.py
│       └── 06_run_fire_sales_old_sequence.py
│
└── src/
    └── liquidity_stress/
        ├── __init__.py
        ├── config.py
        ├── data_loader.py
        ├── first_round.py
        ├── reaction.py
        ├── policy_support.py
        ├── fire_sales.py
        ├── interbank_contagion.py
        ├── network_interbank.py
        ├── network_overlap.py
        ├── scenario_config.py
        ├── synthetic_data_generator.py
        ├── generation.py
        ├── reference_data.py
        ├── stress_vectors.py
        ├── validation.py
        └── utils.py
```

## Canonical run

Run the full pipeline from the project root:

```bash
PYTHONPATH=src python scripts/99_run_all.py
```

This executes:

```text
00_generate_input.py
01_validate_baseline.py
10_run_base_engine.py
20_run_currency_engine.py
30_run_sensitivity.py
40_run_monte_carlo.py
50_plot_indirect_contagion_network.py
51_plot_direct_interbank_network.py
60_make_charts.py
```

## Manual run order

If scripts need to be run separately, use:

```bash
PYTHONPATH=src python scripts/00_generate_input.py
PYTHONPATH=src python scripts/01_validate_baseline.py
PYTHONPATH=src python scripts/10_run_base_engine.py
PYTHONPATH=src python scripts/20_run_currency_engine.py
PYTHONPATH=src python scripts/30_run_sensitivity.py
PYTHONPATH=src python scripts/40_run_monte_carlo.py
PYTHONPATH=src python scripts/50_plot_indirect_contagion_network.py
PYTHONPATH=src python scripts/51_plot_direct_interbank_network.py
PYTHONPATH=src python scripts/60_make_charts.py
```

## Main scripts

| Script                                  | Purpose                                                                    |
| --------------------------------------- | -------------------------------------------------------------------------- |
| `00_generate_input.py`                  | Generates the synthetic Armenia-calibrated banking input workbook.         |
| `01_validate_baseline.py`               | Validates accounting identities, LCR identity, and AMD/FX consistency.     |
| `10_run_base_engine.py`                 | Runs the canonical combined full-engine liquidity stress test.             |
| `20_run_currency_engine.py`             | Runs the full engine separately for AMD and FX and aggregates the results. |
| `30_run_sensitivity.py`                 | Runs deterministic policy and parameter sensitivity scenarios.             |
| `40_run_monte_carlo.py`                 | Runs stochastic first-round shock simulations.                             |
| `50_plot_indirect_contagion_network.py` | Plots the common-asset overlap network.                                    |
| `51_plot_direct_interbank_network.py`   | Plots the direct interbank exposure network.                               |
| `60_make_charts.py`                     | Generates reporting charts from available CSV outputs.                     |
| `99_run_all.py`                         | Runs the full end-to-end pipeline.                                         |

Diagnostic scripts are stored in `scripts/diagnostics/`. They isolate intermediate modules and are not the canonical final engine.

## Core model modules

| Module                        | Purpose                                                           |
| ----------------------------- | ----------------------------------------------------------------- |
| `first_round.py`              | Applies first-round LCR-style liquidity shocks.                   |
| `reaction.py`                 | Models bank asset-sale reactions.                                 |
| `policy_support.py`           | Models central-bank liquidity support and collateral constraints. |
| `fire_sales.py`               | Applies common-asset fire-sale feedback.                          |
| `interbank_contagion.py`      | Propagates direct interbank contagion losses.                     |
| `network_interbank.py`        | Builds and reads directed interbank exposure matrices.            |
| `network_overlap.py`          | Builds common-asset overlap networks.                             |
| `synthetic_data_generator.py` | Generates the synthetic input workbook.                           |
| `validation.py`               | Runs baseline accounting and consistency checks.                  |
| `scenario_config.py`          | Loads YAML scenario settings and builds runtime configs.          |

## Output files

The main outputs are written to:

```text
outputs/tables/
outputs/charts/
outputs/architecture/
```

Key deterministic full-engine outputs include:

```text
outputs/tables/interbank_contagion_bank_results.csv
outputs/tables/interbank_contagion_monthly_summary.csv
outputs/tables/interbank_contagion_worst_case_by_bank.csv
outputs/tables/interbank_contagion_market_summary.csv
outputs/tables/interbank_contagion_network_summary.csv
```

Key charts include:

```text
outputs/charts/01_base_lcr_path.png
outputs/charts/08_loss_decomposition_by_month.png
outputs/charts/09_bank_distress_breach_counts.png
```

## Final validation checks

After running the pipeline, check that baseline validation passes:

```bash
python - <<'PY'
import pandas as pd

checks = pd.read_csv("outputs/tables/baseline_sanity_checks.csv")
failed = checks.loc[~checks["passed"].astype(bool)]

print("Failed checks:", len(failed))
print(failed.to_string(index=False))
PY
```

Expected result:

```text
Failed checks: 0
```

To inspect the main full-engine stress summary:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv("outputs/tables/interbank_contagion_monthly_summary.csv")

cols = [
    "stress_month",
    "system_stressed_LCR",
    "asset_weighted_stressed_LCR",
    "minimum_stressed_LCR",
    "system_post_interbank_LCR",
    "minimum_post_interbank_LCR",
    "aggregate_security_valuation_loss",
    "aggregate_asset_sale_loss",
    "aggregate_fire_sale_mtm_loss",
    "aggregate_interbank_contagion_loss",
    "aggregate_total_modeled_loss",
]

print(df[[c for c in cols if c in df.columns]].round(4).to_string(index=False))
PY
```

The expected qualitative behavior is:

```text
minimum_stressed_LCR falls below the active LCR requirement in stress months
aggregate_security_valuation_loss is nonzero
aggregate_total_modeled_loss includes first-round, fire-sale, and interbank channels
interbank contagion loss is nonzero in at least one severe or full-engine setting
```
