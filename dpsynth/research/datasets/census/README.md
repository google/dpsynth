<!-- disableFinding(LINE_OVER_80) -->
# 1940 US Census Dataset Benchmark Pipeline

This directory contains the end-to-end preprocessing pipeline and benchmark
schema for the **1940 US Decennial Census Full-Count Database** (~137 million
person records, ~54 GB uncompressed fixed-width microdata). It provides a
standardized, large-scale, high-dimensional benchmark dataset for
Differentially Private Synthetic Data (`dpsynth`).

The domain definition (`domain.yaml`) and codebook lookup (`codebook.json`)
are checked directly into this repository.

--------------------------------------------------------------------------------

## Accessing the Data: Three Pathways

Depending on whether you are evaluating synthesized data, generating synthetic
data with preprocessed Parquet, or building from scratch, choose from three
pathways:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Option A: Direct Download of Precomputed Marginals (For Evaluation)         │
│ Download `marginals.npz` directly to evaluate synthetic data utility        │
│ against ground truth using the checked-in `domain.yaml`.                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      OR
┌─────────────────────────────────────────────────────────────────────────────┐
│ Option B: Request Preprocessed Data (Recommended for low compute)          │
│ 1. Register at IPUMS USA (usa.ipums.org)                                    │
│ 2. Email IPUMS (ipums@umn.edu) for permission to receive the preprocessed   │
│    extract from the project maintainers                                     │
│ 3. Contact maintainers with IPUMS approval to receive the Parquet files     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      OR
┌─────────────────────────────────────────────────────────────────────────────┐
│ Option C: Build from Raw Extract (Full pipeline)                            │
│ 1. Download raw extract (census_full_1940.dat + command_file_sas.txt)       │
│ 2. Run the preprocessing pipeline scripts locally (convert_to_parquet, etc.)│
└─────────────────────────────────────────────────────────────────────────────┘
```

--------------------------------------------------------------------------------

## Option A: Direct Download of Precomputed Marginals

If your research focus is evaluating synthetic data generation mechanisms or
fitting models directly to precomputed marginals:

1. **Precomputed Aggregate Marginals (`marginals.npz`):** Serialized as an
   `mbi.CliqueVector` via `mbi.save`, containing exact 2-way joint marginal
   frequency tables for all 9,045 pairs across 135 analysis attributes.
   As aggregate statistical counts, these are not subject to microdata
   redistribution restrictions.
   - **Loading:**
     ```python
     import mbi

     # Loads the full typed CliqueVector in 1 line
     clique_vector = mbi.load("marginals.npz")
     age_sex_factor = clique_vector[("AGE", "SEX")]
     ```
2. **Domain Specification (`domain.yaml`):** Checked in directly in this
   directory, defining the exact binning and categorical index ordering.

`marginals.npz` can be downloaded directly from the project release assets or
storage repository.

--------------------------------------------------------------------------------

## Option B: Requesting Preprocessed Data

Due to [IPUMS USA terms of use](https://usa.ipums.org/), redistribution of raw
or preprocessed full-count microdata is restricted without prior authorization.

If you do not have the local compute or memory infrastructure required to parse
the full 54 GB fixed-width extract, preprocessed Parquet shards can be shared
directly upon request, provided you complete the following steps:

1. **Register at IPUMS USA:** Create a research account at
   [usa.ipums.org](https://usa.ipums.org/usa/) and agree to the IPUMS Data Use
   Agreement.
2. **Obtain Written Approval from IPUMS:** Send an email to
   **`ipums@umn.edu`** stating:
   > *"I am a registered IPUMS USA user conducting research with DP Synth. I
   > would like permission to receive the preprocessed 1940 Full-Count Census
   > benchmark dataset directly from the DP Synth project maintainers for my
   > research."*
3. **Contact Project Maintainers:** Once IPUMS grants written approval, forward
   the approval confirmation along with your IPUMS registration details to the
   maintainers to receive access to the preprocessed Parquet files.

--------------------------------------------------------------------------------

## Option C: Building from Raw Extract

### Step 1: Register at IPUMS USA

Visit [usa.ipums.org](https://usa.ipums.org/usa/) and register for an IPUMS USA
research account.

### Step 2: Create a Data Extract

1. Navigate to **Select Data** $\rightarrow$ **Extract System**.
2. **Select Samples:**
   - Under **USA Full Count**, select **1940 100% Database** (Sample ID:
     `1940b`).
   - Deselect all other sample years.
   - *(Tip: For quick local testing on a standard laptop, select the **1940 1%
     Sample** (`1940a`, ~1.3M rows) instead).*
3. **Select Variables:**
   - Select all demographic, geographic, household, employment, and economic
     variables corresponding to the 147 attributes in `column_types.py`.
4. **Extract Formatting Options:**
   - **Data structure:** Rectangular (person-level records).
   - **Data format:** Fixed-width text (`.dat`).
5. **Submit Extract:**
   - Submit the request and download:
     - The raw data file: `census_full_1940.dat` (or `usa_XXXXX.dat.gz`).
     - The accompanying SAS command file: `command_file_sas.txt` (or
       `usa_XXXXX.sas`).

### Step 3: Local Directory Setup

Place the downloaded extract files into a local directory (e.g., `./data/`):

```bash
mkdir -p ./data/
mv /path/to/downloaded/census_full_1940.dat ./data/census_full_1940.dat
mv /path/to/downloaded/command_file_sas.txt ./data/command_file_sas.txt
```

--------------------------------------------------------------------------------

## Pipeline Execution

The pipeline transforms the raw fixed-width extract into sharded Parquet files,
a machine-readable domain specification (`domain.yaml`), and precomputed 2-way
evaluation marginals:

```
raw .dat + .sas
       │
       ▼ (1) convert_to_parquet.py
  sharded Parquet + stats.json
       │
       ├────────────────────────┬────────────────────────┐
       ▼ (2) build_domain.py    ▼ (3) shuffle_parquet.py
  domain.yaml + codebook.json  shuffled Parquet
       │
       ▼ (4) precompute_marginals.py
  2-way marginal histograms (.npz)
```

### Step 1: Convert Fixed-Width Extract to Sharded Parquet

The parser uses a fast, vectorized chunked reader that converts raw ASCII byte
slices to integer representations via matrix multiplications:

```bash
python3 convert_to_parquet.py \
    --input_path=./data/census_full_1940.dat \
    --output_dir=./data/parquet/ \
    --stats_dir=./data/stats/ \
    --chunk_size=500000
```

### Step 2: Build Domain Specification and Codebooks

Merges column statistics with value labels parsed from the SAS command file to
construct the domain configuration (`domain.yaml`) and label mapping
(`codebook.json`):

```bash
python3 build_domain.py \
    --command_file_path=./data/command_file_sas.txt \
    --domain_path=domain.yaml \
    --codebook_path=codebook.json
```

### Step 3 (Optional): Uniform Shuffle

Evenly scatters and gathers rows to produce uniformly sampled Parquet shards
(useful for streaming and subsampling benchmarks):

```bash
# Scatter phase:
python3 shuffle_parquet.py \
    --phase=scatter \
    --input_dir=./data/parquet/ \
    --tmp_dir=./data/shuffled_temp/

# Gather phase:
python3 shuffle_parquet.py \
    --phase=gather \
    --tmp_dir=./data/shuffled_temp/ \
    --output_dir=./data/shuffled_parquet/
```

### Step 4: Precompute Evaluation Marginals

Computes all 2-way marginal histograms across categorical and discretized
numerical features in a single pass using JIT-compiled GPU/CPU kernels:

```bash
python3 precompute_marginals.py \
    --parquet_dir=./data/parquet/ \
    --domain_path=domain.yaml \
    --output_path=./data/marginals.npz
```
