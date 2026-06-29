# In-Memory DataFrame API Guide

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

The In-Memory API is the fastest way to experiment with **DPSynth**. Built on
top of Pandas and NumPy, this interface is designed for researchers, rapid
prototypers, and software engineers operating on datasets that comfortably fit
within a single machine's RAM.

--------------------------------------------------------------------------------

## Python API: `dpsynth.TabularSynthesizer`

The primary entry point for in-memory synthesis is
`dpsynth.TabularSynthesizer`. It accepts a dictionary of attribute domains,
is calibrated with a privacy budget, and generates a fully synthetic,
differentially private DataFrame matching the exact schema and data types of
your input.

### Usage

```python
import dpsynth
from dpsynth import discrete_mechanisms
import numpy as np
import pandas as pd

synth = dpsynth.TabularSynthesizer(
    domains=domains,
    discrete_mechanism=discrete_mechanisms.MSTMechanism(),
)
result = synth.calibrate(
    epsilon=1.0,
    delta=1e-6,
)(np.random.default_rng(), sensitive_df)
synthetic_df = result.synthetic_data
```

### Key Arguments

*   `data`: The sensitive input `pd.DataFrame`.
*   `domains`: Mapping of column names to domain specifications
    ([`CategoricalAttribute`, `NumericalAttribute`, or `OpenSetCategoricalAttribute`](data_and_terminology.md)).
    Every key must exist in `data.columns`.
*   `epsilon`, `delta`: Total differential privacy budget parameters.
*   `discrete_config`: Configuration object specifying which DP synthesis
    mechanism to run (e.g., `MSTConfig()`, `AIMConfig()`,
    `IndependentConfig()`).
*   `numerical_bins`: Number of equal-frequency quantile buckets used to
    discretize continuous numerical columns (default: `32`).
*   `one_way_marginal_budget_fraction`: Fraction of total `(epsilon, delta)`
    allocated for one-way marginal measurements and domain compression (default:
    `0.1`).
*   `skip_compression`: If `True`, bypasses the rare-category merging phase.
    Note: Compression cannot currently be used simultaneously with
    cross-attribute constraints.

--------------------------------------------------------------------------------

## End-to-End Python Example

Here is a complete Python script demonstrating how to load data, parse a domain
YAML file, configure the AIM mechanism with a fixed random seed, and generate
synthetic records.

```python
import dpsynth
from dpsynth import discrete_mechanisms
from dpsynth import domain
import numpy as np
import pandas as pd

# 1. Load sensitive tabular data into Pandas
sensitive_df = pd.read_csv("sensitive_transactions.csv")

# 2. Load domain schema from YAML
attribute_domains = domain.from_yaml_file("transaction_domain.yaml")

# 3. Configure and calibrate the synthesizer (AIM)
synth = dpsynth.TabularSynthesizer(
    domains=attribute_domains,
    discrete_mechanism=discrete_mechanisms.AIMConfig(
        seed=42,
        rounds=50,
        pgm_iters=1000,
    ),
)
calibrated = synth.calibrate(
    epsilon=1.0,
    delta=1e-6,
    numerical_bins=16,  # Use 16 quantile buckets for numerical columns
)

# 4. Generate Differentially Private synthetic data
result = calibrated(np.random.default_rng(), sensitive_df)
synthetic_df = result.synthetic_data

# 5. Save the synthetic dataframe
synthetic_df.to_csv("synthetic_transactions.csv", index=False)
print("Synthetic data successfully generated!")
```

--------------------------------------------------------------------------------

## Command-Line Interface: `bin/main.py`

For immediate execution without writing custom Python scripts, use the
standalone
binary [`bin/main.py`](../bin/main.py).
It provides command-line flags for all standard configuration parameters.

### CLI Execution Syntax

```bash
python3 bin/main.py \
  --dataset=/path/to/dataset.csv \
  --domain=/path/to/domain.yaml \
  --epsilon=1.0 \
  --delta=1e-8 \
  --mechanism=mst \
  --seed=12345 \
  --output_path=/tmp/synthetic_output.csv
```

### Supported CLI Flags

*   `--dataset`: Path to the input CSV file. (Supports standard CSV parsing
    arguments via `--read_csv_args`).
*   `--domain`: Path to the YAML domain specification file.
*   `--epsilon`, `--delta`: Total DP privacy budget.
*   `--mechanism`: Supported options are `mst`, `aim`, `independent`, and
    `aim_gdp`.
*   `--seed`: Integer seed for reproducible randomness across DP sampling and
    PGM inference.
*   `--output_path`: Destination filepath where the synthetic CSV will be
    written.

--------------------------------------------------------------------------------

## Under the Hood: The In-Memory Lifecycle

When you invoke `TabularSynthesizer`, the library performs the following
single-machine pipeline:

1.  **Discretization**: Continuous numerical columns are bucketed into
    `numerical_bins` quantiles using `pipeline_dp.LocalBackend`. Open-set
    strings are evaluated via DP partition selection.
2.  **Integer Encoding**: All columns are mapped to dense integer indices `[0,
    K-1]`.
3.  **Domain Compression**: DPSynth measures 1-way marginals with Gaussian noise
    and merges rare categories into an `"Other"` bucket, producing an un-noised
    discrete dataset (`mbi.Dataset`).
4.  **Mechanism Execution**: Calls `discrete_mechanisms.run_mechanism()` to
    execute the selected algorithm (`AIM`, `MST`, etc.) on the discrete dataset.
    The mechanism fits a Markov Random Field (`mbi.MarkovRandomField`) via
    Private-PGM mirror descent.
5.  **Sampling & Inversion**: Samples synthetic integer records from the
    graphical model, unpacks `"Other"` categories, and inverts the integer
    encoding back to original Pandas dtypes (strings, integers, floating
    points).
