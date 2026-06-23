# DPSynth: Differentially Private Synthetic Tabular Data

DPSynth is a library for differentially private synthetic tabular data
generation. Given a sensitive dataset of records defined w.r.t. a single-table
schema, our library can generate a synthetic version of the dataset, preserving
the structure and statistical properties of the source data while satisfying
differential privacy.

> [!WARNING]
> **This library is under active development.** APIs may change without notice,
> and you may encounter bugs, rough edges, or incomplete features. For standard
> tabular data settings (categorical and numerical attributes, single-table
> schemas), DPSynth should work well out-of-the-box — but more advanced use
> cases may hit limitations we haven't smoothed out yet.
>
> We have a long roadmap of features we plan to add. In the meantime, we
> welcome early adopters to:
>
> *   **Try it out** on your own datasets and use cases.
> *   **Report issues** — bugs, confusing behavior, or sharp edges.
> *   **Benchmark it** against other DP synthetic data implementations.
> *   **Suggest features** that would be valuable for your workflows.
> *   **Contribute** — whether it's a GitHub issue, pull request, new mechanism,
>     bug fix, or added functionality, contributions are welcome!
>
> Your feedback directly shapes the library's direction. Thank you for your
> patience as we build toward a stable release!

## Two Code Paths

DPSynth contains **two independent implementations** of differentially private
synthetic data generation. While both produce synthetic data using the same
underlying mathematical principles (marginal measurement + Private-PGM
inference), they were developed independently and have different trade-offs:

### 1. In-Memory (Local) Mode

**Entry point:** [`dpsynth.generate()`](dpsynth/__init__.py) (backed by
[`data_generation_v2.py`](dpsynth/data_generation_v2.py))

Designed for **datasets that fit in memory** (e.g., Pandas DataFrames). We have
tested this on datasets up to ~100M rows, though performance will depend on the
number of attributes and domain sizes. This code path:

*   Operates directly on NumPy / Jax arrays and Pandas DataFrames via
    [`discrete_mechanisms/`](dpsynth/discrete_mechanisms/README.md).
*   Accepts domains and data as Python objects — no
    `DatasetDescriptor` required.
*   May be more feature-rich, including experimental mechanisms
    not yet ported to the pipeline mode.
*   Has limited scalability compared to the pipeline mode.

**CLI binary:** [`bin/main.py`](dpsynth/bin/README.md)

### 2. Scalable Pipeline Mode

**Entry point:**
[`data_generation.generate()`](dpsynth/data_generation.py)

Built for **large-scale data** that may not fit on a single machine. This code
path:

*   Runs on distributed frameworks (Apache Beam) via
    [`pipeline_dp.PipelineBackend`](https://github.com/OpenMined/pipeline-dp).
*   Uses
    [`pipeline_transformations/`](dpsynth/pipeline_transformations/README.md)
    for all DP operations.
*   Requires a
    [`DatasetDescriptor`](dpsynth/dataset_descriptors/README.md) to bridge
    format-specific data (CSV, TFRecord) with internal representations.
*   Also works in local settings (`pipeline_dp.LocalBackend`), but follows a
    different code path than the in-memory mode above.

**CLI binary:**
[`bin/run_data_generation.py`](dpsynth/bin/README.md)

### 3. Post-Processing Mode

**Entry point:**
[`postprocessing.generate_synthetic_data_from_marginals()`
](dpsynth/postprocessing.py)

For situations where **noisy marginals are already computed** by an
external system (e.g., a SQL pipeline or a custom DP mechanism), you
can bypass the measurement step entirely and use DPSynth purely for
post-processing. This code path takes pre-computed noisy marginals as
Pandas DataFrames. It automatically infers the domain from the
marginals, but requires categorical data.

### Differences Between the Two Main Code Paths

We have made efforts to align the APIs and behavior between the in-memory and
pipeline code paths, but because they were developed independently, there are
some differences:

*   **API surface:** The in-memory API accepts domains as a plain `dict[str,
    AttributeType]`, while the pipeline API uses the `DatasetDescriptor`
    abstraction.
*   **Budget accounting:** There may be small differences in how
    the privacy budget is split across sub-operations (derivation,
    compression, measurement).
*   **Feature availability:** The in-memory mode may support more experimental
    mechanisms or features that have not yet been ported to the pipeline mode.

> [!NOTE]
> If you observe significant differences in behavior or utility between the two
> code paths on the same dataset and parameters, please open an issue.

## Project Structure

### Shared Modules (Both Code Paths)

These modules are used by both the in-memory and pipeline code paths:

*   **[`domain.py`](dpsynth/domain.py)**: Public API for defining attribute
    domains (`CategoricalAttribute`, `NumericalAttribute`,
    `OpenSetCategoricalAttribute`). Users construct these objects to describe
    their data schema.
*   **[`constraints.py`](dpsynth/constraints.py)**: Definition and validation of
    cross-attribute constraints, provided by users to enforce structural
    properties.
*   **[`transformations.py`](dpsynth/transformations.py)**: Internal logic for
    encoding, discretization, and mapping values between domains.

### In-Memory Mode Only

*   **[`discrete_mechanisms/`](dpsynth/discrete_mechanisms/README.md)**: Local,
    single-machine DP mechanisms (AIM, MST, etc.) and shared mathematical
    utilities like domain compression.
*   **[`data_generation_v2.py`](dpsynth/data_generation_v2.py)**: The end-to-end
    in-memory generation pipeline. This is what `dpsynth.generate()` calls.
*   **[`local_mode/`](dpsynth/local_mode/)**: Locally-optimized DP primitives
    for quantiles and partition selection (NumPy/SciPy-based).
*   **[`pydantic_api.py`](dpsynth/pydantic_api.py)**: API for synthesizing
    collections of Pydantic models directly.

### Pipeline Mode Only

*   **[`dataset_descriptors/`](dpsynth/dataset_descriptors/README.md)**: The
    central orchestration layer. Bridges format-specific data (CSV, TFRecord)
    with internal mathematical representations.
*   **[`pipeline_transformations/`
    ](dpsynth/pipeline_transformations/README.md)**:
    Distributed Beam implementations of DP primitives, derivations, and final
    sample synthesis.
*   **[`data_generation.py`](dpsynth/data_generation.py)**: High-level API for
    generating synthetic data in data pipelines using `pipeline_dp`.
*   **[`diagnostic_info.proto`](dpsynth/diagnostic_info.proto)**: Proto
    definition for tracking DP accounting and utility metrics during pipeline
    execution.

### Post-Processing

*   **[`postprocessing.py`](dpsynth/postprocessing.py)**: Utilities for
    post-processing pre-computed noisy marginals into synthetic data via
    Private-PGM, without running DPSynth's own DP measurement step.

### Binaries & Tools

*   **[`bin/`](dpsynth/bin/README.md)**: Entry points for local prototyping
    (`main.py`) and distributed production jobs (`run_data_generation.py`).
*   **[`eval/`](dpsynth/eval/)**: Tabular evaluation engine for comparing real
    and synthetic data distributions.

## Which Code Path Should I Use?

| Scenario | Recommended |
|---|---|
| Fits in memory, Pandas workflow | **In-Memory** (`dpsynth.generate`) |
| Discrete data, precomputed marginals | **In-Memory** (`discrete_mechanisms`) |
| Large-scale, distributed processing | **Pipeline** (`data_generation`) |
| Marginals from an external system | **Post-Processing** |
| Prototyping / experimental features | **In-Memory** (more flexible) |

## Supported Synthesis Algorithms

Both code paths support the following DP mechanisms:

*   **AIM (Adaptive Iterative Mechanism)**: An MWEM-style algorithm that
    iteratively selects and measures low-dimensional marginals
    ([arXiv:2201.12677](https://arxiv.org/abs/2201.12677)).
*   **AIM-GDP**: A variant of AIM using Gaussian Differential Privacy accounting
    for tighter budget composition.
*   **MST (Maximum Spanning Tree)**: Computes an approximate maximum spanning
    tree over pairwise attribute correlations using the exponential mechanism
    ([arXiv:2108.04978](https://arxiv.org/abs/2108.04978)).
*   **SWIFT (Scalable Workload-Informed Factor Tree)**: An unpublished mechanism
    that operates on discrete data and improves over AIM for
    higher-dimensional datasets by supporting denser sets of marginal
    measurements. Numerical attributes can be handled via the existing
    discretization wrappers.
*   **INDEPENDENT**: A baseline mechanism that measures 1-way marginals and
    models all attributes independently.

## Further Documentation

Detailed guides are available in the [`documentation/`](dpsynth/documentation/)
directory:

*   **In-Memory DataFrame API Guide** (`documentation/in_memory_api.md`):
    Detailed guide to using the Pandas-based API and local CLI.
*   **Scalable Pipeline API Guide** (`documentation/scalable_beam_api.md`):
    Guide for distributed data generation.
*   **Data Model & Terminology** (`documentation/data_and_terminology.md`):
    Attributes, schema specifications, and `domain.yaml` format.
*   **Processing Lifecycle** (`documentation/processing_lifecycle.md`):
    The 5-stage mathematical lifecycle shared by both code paths.
*   **Contributor Guide** (`documentation/contributors_guide.md`):
    Architecture, PipelineBackend programming rules, and evaluation framework.

*This is not an officially supported Google product. This project is
not eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).*
