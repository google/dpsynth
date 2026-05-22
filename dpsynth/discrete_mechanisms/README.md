# Discrete Mechanisms for Synthetic Tabular Data

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

This directory contains implementations of several Differentially Private (DP)
mechanisms that operate over **discrete** synthetic tabular data. These
mechanisms follow the SELECT-MEASURE-GENERATE paradigm to produce synthetic data
that approximates the true distribution while satisfying differential privacy.

In addition to providing standalone mechanisms, this package serves as a
critical **utility provider** for the rest of the project.

## Core Roles & Module Interactions

### 1. Local Mechanism Execution

This directory provides the "brains" for local, in-memory data synthesis. The
primary entry point is `run_mechanism` (in [api.py](api.py)), which orchestrates
specific algorithms (AIM, MST, etc.) on single-machine datasets (typically
loaded via Pandas or NumPy).

### 2. Common Utility Provider

The [common.py](common.py) module provides essential logic used project-wide:

*   **Domain Compression**: Implements the logic to calculate DP one-way
    marginals and merge rare values into an "Other" category. This is heavily
    relied upon by the `DatasetDescriptor.compress()` logic and distributed
    pipelines.
*   **DP Primitives**: Implements the Exponential mechanism and Gaussian
    mechanism logic used by various components.

### 3. Integration with `mbi`

All mechanisms in this package are built on top of the `mbi` library:

*   **Inputs**: Mechanisms consume `mbi.Projectable` objects (discrete data).
*   **Outputs**: Mechanisms produce `mbi.MarkovRandomField` objects, which model
    the approximated distribution and are used to sample synthetic records.

### 4. Privacy Accounting

The [accounting.py](accounting.py) module serves as the bridge between
high-level user privacy parameters (epsilon, delta) and the internal `sigma`
values required by the Gaussian mechanisms.

## Relationship to Other Packages

*   **[dataset_descriptors/](../dataset_descriptors/)**: Provides the
    `DatasetDescriptor.encode()` method which produces the discrete
    integer-encoded data that these mechanisms require.
*   **[pipeline_transformations/](../pipeline_transformations/)**: Contains
    distributed, production-scale implementations of these same algorithms
    (using Apache Beam). Many pipeline transformations reuse the mathematical
    utilities defined here.

## Mechanisms Implemented

*   **Adaptive+Iterative Mechanism (AIM)** (`aim.py`, `aim_gdp.py`): An
    MWEM-style algorithm that iteratively improves the estimation of the data
    distribution.
*   **Maximum Spanning Tree (MST)** (`mst.py`): Computes an approximate maximum
    spanning tree to model pairwise correlations privately.
*   **Direct Mechanism** (`direct.py`): Measures user-prespecified two-way
    marginals and fits a distribution via mirror descent.
*   **Independent Mechanism** (`independent.py`): A baseline approach that
    models all attributes as independent.

## Utilities and Privacy Accounting

*   **`common.py`**: Shared logic for exponential mechanisms, measurement
    noising, and domain compression.
*   **`accounting.py`**: Translation layer for DP budget accounting (zCDP, GDP,
    Approximate DP).
