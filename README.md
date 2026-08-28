# Liquidity Stress-Testing Framework for the Armenian Banking System

This repository implements a multi-period liquidity stress-testing framework for the Armenian banking system. The model combines LCR-based first-round liquidity shocks, bank-level asset-sale reactions, central-bank liquidity support, common-asset fire-sale feedback, and direct interbank contagion.

The current input data are synthetic and Armenia-calibrated using public bank-level balance-sheet references. The framework is designed so that the synthetic input files can later be replaced by real supervisory inputs with the same structure.

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
9. Produce tables, network figures, reporting charts, and paper-ready outputs.

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
├── outputs/
│   ├── architecture/
│   ├── charts/
│   ├── paper_figures/
│   └── tables/
│
├── scripts/
│   ├── 00_generate_input.py
│   ├── 01_validate_baseline.py
│   ├── 10_run_base_engine.py
│   ├── 20_run_currency_engine.py
│   ├── 30_run_sensitivity.py
│   ├── 35_run_paper_scenarios.py
│   ├── 36_build_paper_tables.py
│   ├── 37_make_paper_figures.py
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

## Input data and supervisory use

The files currently stored in:

```text
data/input/
```

are the synthetic, Armenia-calibrated inputs used for the current implementation.

For use with supervisory data, the corresponding synthetic files should be replaced by real bank-level inputs while preserving the structure expected by the model:

```text
data/input/banking_input.xlsx
data/input/interbank_exposure_matrix.xlsx
data/input/interbank_exposure_matrix_amd.xlsx
data/input/interbank_exposure_matrix_fx.xlsx
```

The model architecture itself does not depend on the synthetic calibration; it depends on the format and variables expected by the data-loading layer.

When replacing the synthetic files with supervisory inputs:

1. Keep the same workbook and matrix structure expected by the model.
2. Replace the synthetic values with the corresponding supervisory values.
3. Run baseline validation before running the stress engine.
4. Do not rerun `00_generate_input.py`, since this script regenerates the synthetic input files and may overwrite the supervisory replacements.

With supervisory data in place, the recommended starting point is therefore:

```bash
PYTHONPATH=src python scripts/01_validate_baseline.py
```

followed by the required downstream model scripts.

The synthetic input files currently included in the repository are intended for reproducibility and model inspection.

## Canonical run

Run the full synthetic pipeline from the project root:

```bash
PYTHONPATH=src python scripts/99_run_all.py
```

This executes the main pipeline, including:

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

The paper-specific scripts can be run separately when paper tables and figures are required.

Because `99_run_all.py` includes `00_generate_input.py`, it is intended primarily for the synthetic implementation. When using supervisory inputs, run the downstream scripts manually beginning from `01_validate_baseline.py`.

## Manual run order

If scripts need to be run separately, use:

```bash
PYTHONPATH=src python scripts/00_generate_input.py
PYTHONPATH=src python scripts/01_validate_baseline.py
PYTHONPATH=src python scripts/10_run_base_engine.py
PYTHONPATH=src python scripts/20_run_currency_engine.py
PYTHONPATH=src python scripts/30_run_sensitivity.py
PYTHONPATH=src python scripts/35_run_paper_scenarios.py
PYTHONPATH=src python scripts/36_build_paper_tables.py
PYTHONPATH=src python scripts/37_make_paper_figures.py
PYTHONPATH=src python scripts/40_run_monte_carlo.py
PYTHONPATH=src python scripts/50_plot_indirect_contagion_network.py
PYTHONPATH=src python scripts/51_plot_direct_interbank_network.py
PYTHONPATH=src python scripts/60_make_charts.py
```

## Main scripts

| Script | Purpose |
| --- | --- |
| `00_generate_input.py` | Generates the synthetic Armenia-calibrated banking input files. |
| `01_validate_baseline.py` | Validates accounting identities, LCR identity, and AMD/FX consistency. |
| `10_run_base_engine.py` | Runs the canonical combined full-engine liquidity stress test. |
| `20_run_currency_engine.py` | Runs the full engine separately for AMD and FX and aggregates the results. |
| `30_run_sensitivity.py` | Runs deterministic policy and parameter sensitivity scenarios. |
| `35_run_paper_scenarios.py` | Runs the scenarios used for the paper analysis. |
| `36_build_paper_tables.py` | Builds paper-ready summary tables. |
| `37_make_paper_figures.py` | Produces paper-ready figures. |
| `40_run_monte_carlo.py` | Runs stochastic first-round shock simulations. |
| `50_plot_indirect_contagion_network.py` | Plots the common-asset overlap network. |
| `51_plot_direct_interbank_network.py` | Plots the direct interbank exposure network. |
| `60_make_charts.py` | Generates reporting charts from available model outputs. |
| `99_run_all.py` | Runs the full end-to-end pipeline. |

Diagnostic scripts are stored in `scripts/diagnostics/`. They isolate intermediate modules and are not the canonical final engine.

## Core model modules

| Module | Purpose |
| --- | --- |
| `first_round.py` | Applies first-round LCR-style liquidity shocks. |
| `reaction.py` | Models bank asset-sale reactions. |
| `policy_support.py` | Models central-bank liquidity support and collateral constraints. |
| `fire_sales.py` | Applies common-asset fire-sale feedback. |
| `interbank_contagion.py` | Propagates direct interbank contagion losses. |
| `network_interbank.py` | Builds and reads directed interbank exposure matrices. |
| `network_overlap.py` | Builds common-asset overlap networks. |
| `synthetic_data_generator.py` | Generates the synthetic input workbook. |
| `validation.py` | Runs baseline accounting and consistency checks. |
| `scenario_config.py` | Loads YAML scenario settings and builds runtime configs. |

## LCR thresholds

The framework distinguishes between the regulatory LCR minimum and the level at which banks are assumed to begin precautionary reactions.

The regulatory minimum is:

```text
LCR = 100%
```

The behavioral reaction threshold is set separately and can be higher than the regulatory minimum. In the benchmark implementation, banks may therefore begin reacting before an actual regulatory breach occurs.

A bank crossing the behavioral reaction threshold should consequently not be interpreted as having breached the regulatory LCR requirement.

## Monte Carlo simulations

The stochastic simulations are implemented in:

```text
scripts/40_run_monte_carlo.py
```

The current implementation uses a relatively small number of Monte Carlo replications for convenience and to keep routine model runs, debugging, and validation reasonably fast.

This setting is suitable for checking the model and reproducing the repository quickly, but it is not intended to provide highly precise tail estimates.

For substantive analysis, the number of Monte Carlo replications should be increased, especially when using results such as:

- stress probabilities;
- lower-tail LCR quantiles;
- bank-level risk rankings;
- central-bank support distributions;
- or other tail-risk measures.

The simulation-count setting in the Monte Carlo script or configuration can therefore be increased depending on the required precision and available computation time.

For final empirical results, it is advisable to check that the main Monte Carlo statistics are stable when the number of simulations is increased.

## Output files

The model generates outputs under:

```text
outputs/tables/
outputs/charts/
outputs/paper_figures/
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

Paper-specific outputs include files generated by:

```text
scripts/35_run_paper_scenarios.py
scripts/36_build_paper_tables.py
scripts/37_make_paper_figures.py
```

with final paper tables and figures written primarily to:

```text
outputs/tables/paper_*
outputs/paper_figures/
```

The framework also produces a larger number of intermediate diagnostic, sensitivity, Monte Carlo, and reporting outputs.

Not all generated intermediate outputs are committed to the repository. This is intentional, since they can be reproduced directly from the supplied code, configuration, and input files and would otherwise create unnecessary repository size and version-control changes after every run.

The complete set of intermediate outputs can be regenerated by running the relevant scripts and can also be shared separately if needed.

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
banks may begin reacting before reaching the 100% regulatory LCR minimum
aggregate_security_valuation_loss is nonzero under relevant market shocks
aggregate_total_modeled_loss includes the activated first-round, fire-sale, and interbank channels
interbank contagion loss is nonzero in at least one sufficiently severe or full-engine setting
```

## Notes on reproducibility

The repository is intended to provide a reproducible implementation of the liquidity stress-testing architecture using synthetic Armenia-calibrated inputs.

The synthetic data allow the model mechanics, calibration structure, transmission channels, and outputs to be reviewed without using confidential supervisory information.

For operational application, the synthetic input layer should be replaced with appropriately structured supervisory data, the baseline validation rerun, and the relevant behavioral and stress parameters reviewed before interpreting the model results.