# Scalable Beam API Guide

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

[TOC]

When your dataset is large enough to be processed on one machine (>1M records)
[`pipeline_transformations/`](../pipeline_transformations/README.md)
to run seamlessly on distributed computation engines
like **Apache Beam**
(or Apache Spark in open-source deployments).

--------------------------------------------------------------------------------

## Python API: `data_generation.generate`

The core entry point for distributed pipeline execution is `generate()` located
in [`dpsynth.data_generation`](../data_generation.py).
It operates directly on collections (such as Apache Beam `PCollection`s or
standard Python iterables) and returns a distributed collection of synthetic
tuples or records.

--------------------------------------------------------------------------------

## Creating a `DatasetDescriptor`

Before configuring the data generation job, you must build a `DatasetDescriptor`
corresponding to your input data format. DPSynth provides dedicated builder
utilities under
the [`dataset_descriptors/`](../dataset_descriptors/README.md)
module:

### 1. For CSV Datasets (`csv_descriptor.py`)

To build a descriptor from a CSV file, load a small representative sample (e.g.
first 100 rows) into a Pandas DataFrame. DPSynth will automatically deduce the
data type (`INT`, `STR`, `FLOAT`, `BOOL`) for each column.

```python
from dpsynth.dataset_descriptors import csv_descriptor

# Load sample from CSV file to deduce schema types
sample_df = csv_descriptor.read_csv_sample("/path/to/my_dataset.csv")

# Deduces types and instantiates AttributeDescriptors automatically
descriptor = csv_descriptor.get_dataset_descriptor_for_csv(
    dataframe=sample_df,
    field_names=["age", "occupation", "capital_gain"] # Optional: restrict columns to synthesize
)
```

### 3. For TFRecord Datasets (`tfrecord_descriptor.py`)

To build a descriptor from `tf.train.Example` records inside TFRecord files,
parse a representative sample list of records.

```python
from dpsynth.dataset_descriptors import tfrecord_descriptor

# Load first 1000 sample tf.train.Example records
sample_records = tfrecord_descriptor.read_tfrecords_sample(
    path="/path/to/sharded_data*.tfrecord",
    sample_size=1000
)

# Deduces types and validates that all sample records conform to same schema
descriptor = tfrecord_descriptor.get_dataset_descriptor_for_tfrecord(
    sample_records=sample_records,
    attributes=["age", "label"] # Optional: filter attributes
)
```

--------------------------------------------------------------------------------

### Configuration: `DataGenerationConfig`

Before launching a job, instantiate a `DataGenerationConfig` dataclass to define
your privacy parameters, input/output formats, and mechanism preferences.

```python
from dpsynth import data_generation
from dpsynth.pipeline_transformations import types

config = data_generation.DataGenerationConfig(
    epsilon=1.0,
    delta=1e-7,
    mechanism=data_generation.Mechanism.MST, # or AIM, SWIFT, INDEPENDENT
    dataset_descriptor=descriptor_object,    # Schema descriptor (CSV, TFRecord)
    [`bin/run_data_generation.py`](../bin/run_data_generation.py)
orchestrates the complete distributed lifecycle. It handles format deduction,
launches the Beam job on cluster infrastructure, and manages sink connectors.

### Example: Local Run
```bash
python3 bin/run_data_generation.py \
  --dataset=/path/to/sharded_data@100.tfrecord \
  --epsilon=1.0 \
  --delta=1e-8 \
  --data_format=TFRECORD \
  --output_format=TFRECORD \
  --mechanism=mst \
  --use_beam=false \
  --output_path=/tmp/synthetic_out@100.tfrecord
```

### Comprehensive Flag Reference

#### Core Privacy & I/O Flags

*   `--dataset`: Path (or glob pattern) to input sharded files or database
    query.
*   `--domain_file`: Optional path to explicit `domain.yaml`. If omitted, the
    schema is deduced and populated privately on-the-fly.
*   `--epsilon`, `--delta`: Differential privacy budget.

--------------------------------------------------------------------------------

## Specifying Attribute Domains (`--domain_file`)

While DPSynth can dynamically deduce column schemas and categorical listings,
you can explicitly configure attribute types, categories, and bounds using a
YAML domain specification file via the `--domain_file` flag.

### ⚠️ Critical Requirement for Numerical Columns

> [!IMPORTANT] **For now, you MUST use a `domain.yaml` file to specify
> boundaries for all `NumericalAttributes`.**
>
> Currently, if a numerical column is not defined in the domain file, DPSynth
> attempts to derive its `min_value` and `max_value` boundaries dynamically from
> the raw dataset. However, this dynamic derivation is currently a **non-private
> operation** (marked as a developmental `TODO`).
>
> To ensure your pipeline satisfies strict mathematical Differential Privacy
> guarantees, you **must** pre-specify the absolute `min_value` and `max_value`
> bounds for every numerical column in your `--domain_file`. These bounds should
> be determined using public information or analytical heuristics without
> looking at the sensitive data.

[`synthetic_model.proto`](../pipeline_transformations/synthetic_model.proto).

#### A. From the CLI (`bin/run_data_generation.py`)

Pass the `--model_save_path` flag to serialize the model during a standard data
generation run.
* Saves the model locally as a single binary file.

#### B. From the Library API (Python)

If building a custom pipeline, use the unified helper functions
in [`google_input_output.py`](../pipeline_transformations/google_input_output.py)
to serialize and save the MRF and descriptor collections.

--------------------------------------------------------------------------------

### 2. How to Load and Sample from a Saved Model

Once saved, you can reload the model and use the mathematical sampling layer to
generate fresh synthetic records.

#### A. From the CLI (`bin/run_data_generation_from_model.py`)

Use the dedicated
binary [`bin/run_data_generation_from_model.py`](../bin/run_data_generation_from_model.py)
to sample records at scale on cluster infrastructure.

#### B. From the Library API (Python)

You can load and sample directly within your pipeline scripts, allowing you to
easily plug synthetic generators into downstream ETL steps.
