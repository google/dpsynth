# DPSynth: Differentially Private Synthetic Tabular Data

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

DPSynth is a comprehensive library and toolset for generating **Differentially
Private (DP) synthetic tabular data**. Given a sensitive dataset of records
defined with respect to a tabular schema (such as CSV, Protobuf, TFRecord, or
SQL tables), DPSynth generates a synthetic version of the dataset. The synthetic
data preserves the multi-dimensional structure, relational correlations, and
statistical properties of the original data while rigorously satisfying
mathematical differential privacy (`(epsilon, delta)-DP`).

--------------------------------------------------------------------------------

## Why DPSynth?

In modern data science, sharing sensitive tabular data for analytics, testing,
or model training presents significant privacy risks. Standard anonymization
techniques (like masking or `k`-anonymity) are notoriously vulnerable to linkage
attacks.

DPSynth solves this by employing state-of-the-art Differential Privacy
mechanisms (including **AIM**, **MST**, and **SWIFT**) paired with Graphical
Model (Private-PGM) inference. The resulting synthetic records can be freely
shared, analyzed, or published without compromising the privacy of any
individual present in the source data.

--------------------------------------------------------------------------------

## Core APIs and Execution Models

DPSynth provides two unified execution models designed to scale from small local
dataframes to massive distributed datasets across computing clusters:

```
               ┌────────────────────────────────────────┐
               │            Sensitive Dataset           │
               └───────────────────┬────────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │ Choose Execution Environment │
                    └──────────────┬───────────────┘
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
┌─────────────────────────┐                    ┌─────────────────────────┐
│ In-Memory DataFrame API │                    │  Scalable Pipeline API  │
├─────────────────────────┤                    ├─────────────────────────┤
│ • Single-machine RAM    │                    │ • Massive cluster scale │
│ • Pandas DataFrame I/O  │                    │ • Apache Beam / Spark   │
│ • Python scripts & CLI  │                    │ • Sharded files / SQL   │
└───────────┬─────────────┘                    └───────────┬─────────────┘
            └──────────────────────┬───────────────────────┘
                                   ▼
               ┌────────────────────────────────────────┐
               │   Differentially Private Synthetic Data │
               └────────────────────────────────────────┘
```

### 1. In-Memory DataFrame API (`dpsynth.generate`)

Optimized for rapid prototyping, research experimentation, and datasets that
easily fit within single-machine memory.

*   **Inputs/Outputs**: Pandas DataFrames.
*   **CLI
    Binary**: [`bin/main.py`](../bin/main.py).
*   **Documentation**: [In-Memory DataFrame API Guide](in_memory_api.md).

### 2. Scalable Pipeline API (`dpsynth.data_generation`)

Built for production-grade data pipelines operating on massive distributed
datasets. Written against the framework-agnostic `pipeline_dp.PipelineBackend`
interface.

*   **Execution**: Runs
    on Apache Beam
    (or Apache Spark in open-source deployments).
[`bin/run_data_generation.py`](../bin/run_data_generation.py). *   **Documentation**:
    [Scalable PipelineBackend API Guide](scalable_beam_api.md).

--------------------------------------------------------------------------------

## Documentation Sitemap & Navigation

Whether you are a software engineer looking to integrate synthetic data into
your workflow, a researcher investigating graphical model synthesis, or a
contributor expanding the library, explore the documentation below:

*   **[Documentation Sitemap](sitemap.md)**: Complete table of contents and
    layout of the DPSynth documentation.
*   **[Data Model & Terminology](data_and_terminology.md)**: Attributes
    (categorical vs. numerical), schema deduction, and `domain.yaml`
    specifications.
*   **[In-Memory DataFrame API Guide](in_memory_api.md)**: Detailed guide to
    using the Pandas-based API and local CLI. *   **[Scalable PipelineBackend API Guide](scalable_beam_api.md)**: Guide for
    distributed data generation using Beam and cluster execution.
*   **[Processing & Synthesis Lifecycle](processing_lifecycle.md)**: Deep dive
    into the 5-stage mathematical lifecycle: Derivation, Encoding, Compression,
    PGM Modeling, and Decoding.
*   **[Contributor & Architecture Guide](contributors_guide.md)**: Architectural
    separation, core abstractions, PipelineBackend programming rules,
    diagnostics, and the Tabular Evaluation framework.

--------------------------------------------------------------------------------

## Supported Synthesis Algorithms

DPSynth supports the following cutting-edge DP mechanisms across both of its
APIs:

*   **AIM (Adaptive Iterative Mechanism)**: An MWEM-style algorithm that
    iteratively selects and measures low-dimensional marginals based on
    workload, privacy budget, and data characteristics
    ([arXiv:2201.12677](https://arxiv.org/abs/2201.12677)).
*   **MST (Maximum Spanning Tree)**: Computes an approximate maximum spanning
    tree over pairwise attribute correlations using the exponential mechanism to
    model bivariate dependencies privately
    ([arXiv:2108.04978](https://arxiv.org/abs/2108.04978)).
*   **SWIFT**: An advanced, highly optimized mechanism for scalable synthesis of
    complex categorical and numerical distributions.
*   **INDEPENDENT**: A robust baseline mechanism that measures 1-way marginals
    via Gaussian noise and models all attributes independently.

--------------------------------------------------------------------------------

## Contact

--------------------------------------------------------------------------------
