# In-Memory DataFrame API Guide

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

The In-Memory API is the fastest way to experiment with **DPSynth**. Built on
top of Pandas and NumPy, this interface is designed for researchers, rapid
prototypers, and software engineers operating on datasets that comfortably fit
within a single machine's RAM.

--------------------------------------------------------------------------------

## Python API: `dpsynth.TabularConfig`

The primary entry point for in-memory synthesis is
`dpsynth.TabularConfig`. It configures the algorithm hyperparameters (e.g.
discrete mechanism, numerical bin count, budget allocation), is calibrated
with a dataset `dpsynth.Schema` and privacy budget to produce a
`dpsynth.TabularMechanism`, and generates a fully synthetic, differentially
private DataFrame matching the exact schema and data types of your input.

### Usage

```python
import dpsynth
from dpsynth import discrete_mechanisms
from dpsynth import domain
import numpy as np
import pandas as pd

# Define schema (or load from schema.to_yaml_file / domain.from_yaml_file)
schema = dpsynth.Schema({
    "age": domain.NumericalAttribute(min_value=18, max_value=90),
    "workclass": domain.CategoricalAttribute(possible_values=["Private", "Gov", "Other"]),
})

# Reusable algorithm preset
config = dpsynth.TabularConfig(
    discrete_mechanism=discrete_mechanisms.MSTConfig(),
    numerical_bins=32,
)

# Calibrate with schema and privacy budget
mechanism = config.calibrate(schema=schema, epsilon=1.0, delta=1e-6)
result = mechanism(np.random.default_rng(), sensitive_df)
synthetic_df = result.synthetic_data
```

### Key Configuration Arguments

When initializing `dpsynth.TabularConfig`:

*   `discrete_mechanism`: Configuration object specifying which DP synthesis
    mechanism to run (e.g., `MSTConfig()`, `AIMConfig()`, `SWIFTConfig()`,
    `IndependentConfig()`).
*   `numerical_bins`: Number of equal-frequency quantile buckets used to
    discretize continuous numerical columns (default: `32`).
*   `init_budget_fraction`: Fraction of total `(epsilon, delta)` budget
    allocated for per-column initialization such as bounds computation and
    partition selection (default: `0.1`).
*   `cross_attribute_constraints`: Optional sequence of `Constraint` objects to
    enforce on generated data.
*   `schema`: Optional schema specification if binding the config to a specific
    dataset at construction time rather than calibration time.

When calling `config.calibrate(...)`:

*   `schema`: The `dpsynth.Schema` (or mapping of column names to attribute
    domains) defining the dataset columns and optional constraints.
*   `epsilon`, `delta`: Total differential privacy budget parameters. Returns a
    runnable `TabularMechanism`.

--------------------------------------------------------------------------------

## Standalone End-to-End Python Example

Here is a complete, self-contained Python script demonstrating how to specify a
schema, set up a `TabularConfig`, calibrate the mechanism with a privacy budget,
load sensitive data, synthesize records, and print the first few rows.

```python
import dpsynth
from dpsynth import discrete_mechanisms
from dpsynth import domain
import numpy as np
import pandas as pd

# 1. Schema Specification: Define the schema of the tabular dataset
schema = dpsynth.Schema({
    "age": domain.NumericalAttribute(min_value=18, max_value=90),
    "workclass": domain.CategoricalAttribute(
        possible_values=["Private", "Self-emp", "Gov", "Other"]
    ),
    "education": domain.CategoricalAttribute(
        possible_values=["HS-grad", "Bachelors", "Masters", "PhD"]
    ),
})

# 2. Setup Config: Configure synthesizer hyperparameter preset
config = dpsynth.TabularConfig(
    discrete_mechanism=discrete_mechanisms.MSTConfig(),
    numerical_bins=16,
)

# 3. Calibrate Mechanism: Allocate privacy budget with schema to get runnable mechanism
mechanism = config.calibrate(schema=schema, epsilon=1.0, delta=1e-5)

# 4. Load Data: Create sensitive input DataFrame matching the domain schema
sensitive_df = pd.DataFrame({
    "age": [25, 42, 30, 55, 62, 29, 38, 47, 51, 33],
    "workclass": [
        "Private",
        "Gov",
        "Private",
        "Self-emp",
        "Other",
        "Private",
        "Gov",
        "Private",
        "Self-emp",
        "Private",
    ],
    "education": [
        "Bachelors",
        "Masters",
        "HS-grad",
        "PhD",
        "HS-grad",
        "Bachelors",
        "HS-grad",
        "Masters",
        "Bachelors",
        "HS-grad",
    ],
})

# 5. Synthesize Data: Run the calibrated mechanism on the sensitive data
rng = np.random.default_rng(seed=42)
result = mechanism(rng, sensitive_df)
synthetic_df = result.synthetic_data

# 6. Print the first few rows of the generated synthetic dataset
print("Generated Synthetic Data:")
print(synthetic_df.head())
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

When you configure and run `TabularConfig`, the library performs the following
single-machine pipeline:

1.  **Discretization**: Continuous numerical columns are bucketed into
    `numerical_bins` quantiles using `pipeline_dp.LocalBackend`. Open-set
    strings are evaluated via DP partition selection.
2.  **Integer Encoding**: All columns are mapped to dense integer indices `[0,
    K-1]`.
3.  **Domain Compression**: DPSynth measures 1-way marginals with Gaussian noise
    and merges rare categories into an `"Other"` bucket, producing an un-noised
    discrete dataset (`mbi.Dataset`).
4.  **Mechanism Execution**: Calls the configured discrete mechanism (`AIM`,
    `MST`, etc.) on the discrete dataset. The mechanism fits a Markov Random
    Field (`mbi.MarkovRandomField`) via Private-PGM mirror descent.
5.  **Sampling & Inversion**: Samples synthetic integer records from the
    graphical model, unpacks `"Other"` categories, and inverts the integer
    encoding back to original Pandas dtypes (strings, integers, floating
    points).

