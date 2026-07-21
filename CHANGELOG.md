# Changelog

All notable changes to the `dpsynth` library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

-   **Breaking**: The base install now contains only the dependencies required
    for in-memory synthesis via `dpsynth.TabularSynthesizer`. Heavyweight and
    feature-specific dependencies moved to optional extras to keep the base
    install small:
    -   `[pipeline]` — scalable Apache Beam / `pipeline_dp` execution path and
        its TensorFlow-based TFRecord I/O.
    -   `[text]` — free-form text features (`dpsynth.text`): GenAI batch feature
        extraction and DP fine-tuning of Gemma. `google-genai` moved here from
        the base install.
    -   `[examples]` — dependencies used only by the example notebooks
        (`kagglehub`, `scikit-learn`, `sdmetrics`).
    -   `[all]` — the `pipeline` and `text` feature extras.

    DPSynth is installed from GitHub rather than PyPI, so request extras with
    the direct-reference syntax, e.g.:

    ```
    pip install "dpsynth[pipeline] @ git+https://github.com/google/dpsynth.git"
    ```

    Users of the Beam pipeline should include the `pipeline` extra.

## [0.1.0] - 2026-06-15

Initial public release of DP Synth — a library for generating differentially
private synthetic data.

### Added

This first release contains code for generating differentially private synthetic
tabular data using marginal measurement and Private-PGM inference, including:

-   **Two execution modes**: In-memory local mode
    (via `dpsynth.TabularSynthesizer`, tested up to ~100M rows) and a
    workloads.
-   **Marginal-based mechanisms**: AIM, MST, Independent, and Direct mechanisms
    for selecting and measuring marginals under differential privacy.
-   **Closed-domain categorical attributes**: Standard categorical columns
    where the full domain is known upfront.
-   **Open-domain categorical attributes**: DP partition selection to privately
    discover significant categories when the domain is not known in advance.
-   **Numerical attributes**: Discretization with configurable
    `interval_handling` to control how intervals are converted back to values
    (`midpoint`, `sample`, or raw `pd.Interval`).
-   **Quickstart notebook**: Interactive Colab notebook demonstrating basic
    usage of the library.
-   **Documentation**: README with architecture overview, module-level READMEs,
    and work-in-progress notice.

[0.1.0]: https://github.com/google/dpsynth/releases/tag/v0.1.0
