# DPSynth: Differentially Private Synthetic Tabular Data

DPSynth is a library for differentially private synthetic tabular data
generation. Given a sensitive dataset of records defined w.r.t. a single-table
schema, our library can generate a synthetic version of the dataset, preserving
the structure and statistical properties of the source data while satisfying
differential privacy.

## Core Concepts & Synthesis Lifecycle

The library orchestrates synthesis through the
**[DatasetDescriptor](dpsynth/dataset_descriptors/README.md)**, which manages
the lifecycle of a task:

1.  **Initialization**: A descriptor is created from a schema (CSV, Proto,
    TFRecord).
2.  **Population**: Missing domain metadata (like bounds or categories) is
    privately derived from the sensitive data using mechanisms in
    `pipeline_transformations/`.
3.  **Execution**: The fully populated descriptor transforms raw input records
    into discrete indices, which are then modeled by DP mechanisms to generate
    synthetic data.

## Execution Models

DPSynth supports two primary execution models depending on the dataset scale:

*   **Local (In-Memory)**: Optimized for small datasets (e.g., Pandas
    DataFrames). Uses
    [discrete_mechanisms/](dpsynth/discrete_mechanisms/README.md) for
    processing.
*   **Distributed (Massive Scale)**: Powered by Apache Beam and
    [`pipeline_dp`](https://github.com/google/pipeline-dp). Uses
    [pipeline_transformations/](dpsynth/pipeline_transformations/README.md) to
    scale algorithms across large computing clusters.

## Project Structure & Modules

The library is organized into the following functional layers:

### 1. Binaries ([bin/](dpsynth/bin/README.md))

High-level entry points for local prototyping (`main.py`) or launching
distributed production jobs (`run_data_generation.py`).

### 2. Library API for generating data in data pipelines ([data_generation.py](dpsynth/data_generation.py))

High-level API for generating synthetic data in data pipelines using
`pipeline_dp`. Supports local and distributed (Beam) execution for large-scale
datasets with mechanisms like AIM, MST, and INDEPENDENT.

### 3. Orchestration ([dataset_descriptors/](dpsynth/dataset_descriptors/README.md))

The central abstraction layer. Bridges format-specific data (CSV, Protos) with
internal mathematical representations.

### 4. Execution Engines

*   **[discrete_mechanisms/](dpsynth/discrete_mechanisms/README.md)**: Local,
    single-machine DP mechanisms (AIM, MST, etc.) and shared mathematical
    utilities like domain compression.
*   **[pipeline_transformations/](dpsynth/pipeline_transformations/README.md)**:
    Distributed Beam implementations of DP primitives, derivations, and final
    sample synthesis.

### 5. Data Modeling (Public API)

*   **[domain.py](dpsynth/domain.py)**: Represents categorical and numerical
    attribute domains. Users construct these objects to define their data
    schema.
*   **[constraints.py](dpsynth/constraints.py)**: Definition and validation of
    cross-attribute constraints, provided by users to enforce structural
    properties.

### 6. Internal Implementations

*   **[transformations.py](dpsynth/transformations.py)**: Internal logic for
    encoding, discretization, and mapping values between domains.

### 7. Diagnostics & Lifecycle

*   **[diagnostic_info.proto](dpsynth/diagnostic_info.proto)**: Proto definition
    for tracking accounting and utility metrics.
*   **[postprocessing.py](dpsynth/postprocessing.py)**: Optimization utilities
    (e.g., Private-PGM) for refining noisy marginals.

*This is not an officially supported Google product. This project is not
eligible for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).*
