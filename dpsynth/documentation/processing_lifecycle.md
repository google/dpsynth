# Processing and Synthesis Lifecycle

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

At the core of **DPSynth** is a robust mathematical transformation engine.
Transforming sensitive, multi-dimensional tabular records into differentially
private synthetic data requires navigating complex challenges: continuous
floating-point ranges, unbounded string categories, high-dimensional state
spaces, and strict privacy budget accounting.

This page details the 5-stage mathematical lifecycle executed across both the
[In-Memory](in_memory_api.md) and [Scalable Pipeline](scalable_beam_api.md)
APIs.

```
┌────────────────────────────────────────────────────────┐
│ 1. Initialization                                      │
│    • DP Quantiles for continuous numericals            │
│    • DP Partition Selection for open categorical values│
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 2. Integer Encoding                                    │
│    • Raw tuples -> Dense integer indices [0, K-1]      │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Domain Compression                                  │
│    • Measure 1-way marginals with DP noise             │
│    • Merge rare values & bins into "Other" bucket      │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. Graphical Model Inference (Private-PGM)             │
│    • Select & measure low-dimensional marginals        │
│    • Fit Markov Random Field via mirror descent        │
└───────────────────────────┬────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. Sampling, Uncompression, & Decoding                 │
│    • Sample synthetic integer records from model       │
│    • Unpack "Other" bins probabilistically             │
│    • Decode integers -> Native strings, floats, & ints │
└────────────────────────────────────────────────────────┘
```

--------------------------------------------------------------------------------

## Mathematical Modeling & Discretization

To understand why DPSynth processes data in distinct encoding and discretization
stages, it is important to understand the underlying mathematical
representation:

### Markov Random Field (MRF) Representation

DPSynth models the joint probability distribution of your multi-dimensional
tabular dataset as a
[Markov Random Field (MRF)](https://en.wikipedia.org/wiki/Markov_random_field),
also referred to as an undirected probabilistic graphical model (PGM).

*   Instead of storing the massive, joint probability table of all columns
    combined, the MRF factorizes the joint distribution into a product of
    low-dimensional factors (or cliques) representing the correlations between
    small subsets of columns (e.g., pairwise relationships).
*   This factorization allows the mathematical engine (`mbi` and Private-PGM) to
    optimize and store the distribution efficiently, scaling to datasets with
    high dimensionalities.

### The Strict Discretization Requirement

> [!IMPORTANT] **Markov Random Fields are defined strictly over finite, discrete
> spaces.**
>
> In graphical models, variables must represent a countable set of discrete
> categories. * Continuous values (like continuous floating-point numbers)
> cannot be represented as discrete nodes without infinite dimensions. *
> Unbounded, open-set strings cannot be mapped to consistent indices without
> defining a finite, fixed vocabulary.
>
> **Therefore, all continuous numerical columns and open-set string columns must
> be discretized.** Continuous continuous features are partitioned into finite
> quantile buckets, and unbounded string lists are selection-filtered and mapped
> to standard zero-indexed integers `[0, K-1]`.
>
> Every raw input record must become a vector of discrete integer tokens before
> it is ingested by the measurement and modeling engine. Stage 1 and Stage 2
> below detail how DPSynth manages this discretization privately and
> mathematically.

--------------------------------------------------------------------------------

## Stage 1: Derivation and Initialization

When raw datasets are ingested without an exhaustive, explicit schema, DPSynth
must privately deduce domain boundaries before any modeling can begin.

*   **Numerical
    Derivation**: [`numerical_values_derivation.py`](../pipeline_transformations/numerical_values_derivation.py)
*   **Categorical
    Derivation**: [`categorical_values_derivation.py`](../pipeline_transformations/categorical_values_derivation.py)

### Continuous Numerical Discretization

Continuous floating-point columns (like transaction amounts or timestamps)
cannot be modeled directly by discrete graphical models.

1.  DPSynth runs distributed Differentially Private Quantile algorithms over the
    data distribution.
2.  The continuous range is split into equal-frequency buckets (e.g., `K = 32`
    bins by default). Bins are clustered where data is dense and expanded where
    data is sparse.

### Open-Set Categorical Discovery

For unbounded string columns (like occupation or city), the total set of unique
categories is unknown.

1.  DPSynth runs Differentially Private Partition Selection over the column
    entries.
2.  String categories whose appearance count exceeds the private threshold are
    added to the allowed schema domain. All rare strings failing the threshold
    are mapped to a fallback `"Unknown"` category.

--------------------------------------------------------------------------------

## Stage 2: Integer Encoding

Once domain metadata is fully populated in the `DatasetDescriptor`, raw record
tuples are mapped into standardized mathematical tensors.

*   **Distributed
    Encoding**: [`dataset_encoding.py`](../pipeline_transformations/dataset_encoding.py)
*   **Discrete
    Encoders**: [`transformations.py`](../transformations.py)

Every column is assigned a discrete encoder that maps native values to dense
zero-indexed integers `0 ... K-1`. * A boolean column maps to `[0, 1]`. * A
50-state categorical column maps to `[0 ... 49]`. * A 32-bucket numerical column
maps to `[0 ... 31]`.

Raw records become uniform integer vectors: e.g., `("Private", 45.2, True) ->
(1, 18, 1)`.

--------------------------------------------------------------------------------

## Stage 3: Domain Compression

A major hurdle in generating synthetic data is high-dimensional state space
explosion. If a dataset has 20 columns, each with 1,000 categories, the joint
probability space has `10^60` cells—making direct measurement impossible under
strict `(epsilon, delta)` budgets.

*   **Distributed
    Compression**: [`dataset_compression.py`](../pipeline_transformations/dataset_compression.py)
*   **Mathematical Compression
    Utilities**: [`common.py`](../discrete_mechanisms/common.py)

DPSynth solves this through **Domain Compression**:

1.  The library allocates a fraction of the privacy budget (typically 10%) to
    measure 1-way marginals (univariate counts) for every column with Gaussian
    or Exponential noise.
2.  For each column, categories or bins whose noisy frequency falls below a
    calculated significance threshold are merged together into a single
    composite `"Other"` category index.
3.  **Result**: A string column that originally had 10,000 unique categories
    might be compressed down to the top 40 most frequent categories plus 1
    `"Other"` index (`K = 41`). This drastically reduces the state space,
    allowing downstream mechanisms to measure complex multi-column correlations
    with high precision.

--------------------------------------------------------------------------------

## Stage 4: Graphical Model Inference (Private-PGM)

With the data encoded and compressed into a compact discrete domain
(`mbi.Dataset`), DPSynth executes the core synthesis mechanism (`AIM`, `MST`,
`SWIFT`, or `INDEPENDENT`).

*   **Discrete
    Mechanisms**: [`discrete_mechanisms/`](../discrete_mechanisms/README.md)
*   **Distributed
    Modeling**: [`model.py`](../pipeline_transformations/model.py)

### 1. Workload Measurement

Rather than attempting to measure the massive joint distribution directly,
mechanisms select and measure specific low-dimensional marginal queries (e.g.,
2-way or 3-way cross-tabulations between correlated attributes).

*   **MST ([Maximum Spanning Tree](https://arxiv.org/abs/2108.04978))**: Uses
    the exponential mechanism to privately select attribute pairs exhibiting
    high mutual information, forming a spanning tree of pairwise dependencies.
    It then measures 2-way marginals across these pairs.
*   **AIM ([Adaptive Iterative Mechanism](https://arxiv.org/abs/2201.12677))**:
    Iteratively selects the most informative low-dimensional marginals to
    measure based on current model errors and remaining privacy budget.

### 2. Private-PGM Inference

All noisy low-dimensional measurements (`mbi.LinearMeasurement` objects) are
passed into the **Private-PGM** inference engine (`mbi`).

*   Private-PGM uses mirror descent optimization to construct a high-dimensional
    Markov Random Field (`mbi.MarkovRandomField`) / Graphical Model.
*   This graphical model represents a joint probability distribution over all
    attributes that closely approximates all measured noisy marginals
    simultaneously, while rigorously enforcing non-negativity and consistency
    across overlapping measurements.

--------------------------------------------------------------------------------

## Stage 5: Sampling, Uncompression, & Decoding

Once the graphical model is fitted, the synthesis pipeline transitions to
generating output data.

*   **Decoding
    Pipeline**: [`data_generation.py`](../data_generation.py)

1.  **Exact Sampling**: Synthetic integer records are sampled in parallel
    batches directly from the Markov Random Field.
2.  **Uncompression**: For any synthetic record where a column value was sampled
    into the composite `"Other"` index, the uncompression layer
    probabilistically unpacks it back into one of the original rare integer
    indices. The assignment is weighted proportionally to prior one-way marginal
    distributions or uniform priors.
3.  **Decoding**: Finally, the uncompressed integer indices are passed through
    the `DatasetDescriptor.decode()` layer. Integers are mapped back to native
    strings, un-bucketed back to continuous floating-point numbers (sampled
    uniformly within the specific quantile bucket bounds), and formatted into
    the destination format (CSV rows, Protobuf messages, or TFRecord tensors).
