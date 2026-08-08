# DPSynth Adapters

This module wraps local mode mechanisms (`TabularSynthesizer`) to support other
data formats beyond the basic `pd.DataFrame` (or a dictionary of numpy arrays,
which could also probably be supported).

In the future, this module can be extended to support other distributed backends
and data formats like `tensorflow_federated` (for TEE execution), Spark,
Polars, etc. One can follow the basic pattern established by `adapters/beam.py`,
and these can be added on an as-needed basis (making sure the dependencies they
require remain optional for the core library to keep it lightweight).

> **Note:** This does NOT act as a replacement for the `pipeline_dp` code.
> Adapters provide a lightweight bridge for `TabularSynthesizer` but are not a
> substitute for a hardened, pipeline-native DP framework such as PipelineDP,
> which should be preferred for production pipelines.
>
> Compared to the pipeline DP approach, these adapters do not go through the
> hardened privacy-verification path that PipelineDP provides, which offers
> stronger guarantees around DP primitive correctness and audited randomness.
> This module may serve as a temporary stopgap until there is better alignment
> between the pipeline DP implementations and the local-mode NumPy-based
> implementations. How it fits within the broader ecosystem long-term is an
> open question.
