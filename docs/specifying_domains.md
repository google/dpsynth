<!-- Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Specifying Attribute Domains

<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

Every synthesis mechanism in DPSynth requires an explicit **domain
specification** describing the attributes (columns) of your dataset and the set
of valid values each attribute can take. This guide explains why domains must be
defined before touching private data, how to express domains directly in the
DPSynth domain language, and how to derive them automatically from structured
schema definitions like Pydantic models, Protocol Buffers, and Pandas dtypes.

## Why Domains Must Be Specified Up Front

In differential privacy, the domain of an attribute defines the sample space of
noisy measurements and bounds the sensitivity of each record's contribution.
Crucially, **the domain must be specified before inspecting the sensitive
dataset**:

*   **Data-dependent bounds leak privacy**: Computing `df["income"].min()` and
    `df["income"].max()` or extracting `df["category"].unique()` directly from
    private data is not differentially private. Adding or removing a single
    outlier record can change the observed range or reveal the presence of a
    rare category.
*   **Calibration precedes execution**: In DPSynth's [three-step mechanism
    lifecycle](mechanism_api.md#three-step-pipeline), you pass the domain schema
    to `config.calibrate(schema, epsilon=..., delta=...)` to allocate privacy
    budgets and construct per-column initializers *before* the calibrated
    mechanism ever sees private data.
*   **Handling unknown domains privately**: When the valid categories of a
    string column are not known from public metadata, use
    {class}`~dpsynth.domain.OpenSetCategoricalAttribute` rather than scanning
    the raw data. DPSynth will automatically spend a fraction of the privacy
    budget on differentially private partition selection to discover common
    categories.

## Supported Attribute Types

DPSynth provides four attribute classes in {mod}`dpsynth.domain` to model
tabular columns:

| Attribute Type | When to Use | Required Public Metadata | DP Initialization |
| :--- | :--- | :--- | :--- |
| {class}`~dpsynth.domain.CategoricalAttribute` | Finite set of known values (`bool`, `int`, `float`, or `str`), such as booleans, enum values, US states, or status codes. | `possible_values`: sequence of valid categories. | None (domain is already discrete and fixed). |
| {class}`~dpsynth.domain.NumericalAttribute` | Ordered integer or continuous floating-point numbers, such as age, balance, or duration. | `min_value` and `max_value` (inclusive bounds), plus `dtype` (`"int"` or `"float"`). | Discretizes into bins via DP quantiles (or uses fixed `bin_edges` if provided). |
| {class}`~dpsynth.domain.OpenSetCategoricalAttribute` | Categorical strings whose valid values are unknown in advance, such as free-entered city names or job titles. | None (`default_value` defaults to `"<OOD>"`; optional `public_possible_values`). | Discovers frequent categories via DP Gaussian thresholding (requires $\delta > 0$). |
| {class}`~dpsynth.domain.FreeFormTextAttribute` | Unstructured text fields synthesized by a language model. | `max_tokens` (defaults to `256`). | Handled by text-generation mechanisms. |

### Out-of-Domain Handling

Real-world datasets frequently contain missing values (`None`, `NaN`) or records
that fall outside expected bounds. Because raising a runtime error on unexpected
records would leak information about the private data, DPSynth handles
out-of-domain values deterministically during encoding:

*   **Categorical attributes**: Any value not present in `possible_values` is
    mapped to `possible_values[out_of_domain_index]` (default `0`). If missing
    or unexpected values may occur, place a typed sentinel (such as `"Unknown"`
    for strings or `-1` for integers) at index `0` of `possible_values`. If all
    records are guaranteed to be in-domain, omit the sentinel so the synthesizer
    never generates it.
*   **Numerical attributes**: Controlled by `clip_to_range`:
    *   `clip_to_range=True` (default): Values outside `[min_value, max_value]`
        are clipped to the nearest bound, and `NaN` or non-numeric entries map
        to `min_value`.
    *   `clip_to_range=False`: Values outside `[min_value, max_value]` and `NaN`
        entries are routed to a dedicated out-of-domain bin during
        discretization and decoded back to `sentinel` (`np.nan` by default, or a
        custom numeric/string sentinel).

## Specifying Domains in the DPSynth Language

You can define a domain directly in Python as a dictionary mapping column names
to attribute instances, or wrap it in a {class}`~dpsynth.domain.Schema` when
pairing the domain with cross-attribute constraints:

```python
import dpsynth
from dpsynth import domain

domains = {
    "age": domain.NumericalAttribute(min_value=18, max_value=100, dtype="int"),
    "tier": domain.CategoricalAttribute(possible_values=["FREE", "PRO"]),
    "city": domain.OpenSetCategoricalAttribute(),
}

# Optional: bundle with cross-attribute constraints in a Schema
schema = domain.Schema(domains)
```

Domain specifications and `Schema` objects can also be saved to and loaded from
YAML via `dpsynth.to_yaml` and `dpsynth.from_yaml`:

```python
yaml_str = dpsynth.to_yaml(schema)
restored_schema = dpsynth.from_yaml(yaml_str)
```

## Deriving Domains from Structured Containers

In many applications, your dataset schema is already captured in a structured
container such as a **Pydantic model**, a **Protocol Buffer descriptor**, or
**Pandas column dtypes** (for example, from a Parquet schema). Manually
rewriting tens or hundreds of fields in the DPSynth domain language creates
unnecessary boilerplate and risks schema drift.

DPSynth provides `infer_domain` helpers under `dpsynth.adapters` that
automatically translate these schema definitions into `dict[str,
domain.AttributeType]`. Because the output is a standard Python dictionary, you
can inspect or override individual attributes before calibrating your mechanism.

### 1. From Pydantic Models (`dpsynth.adapters.pydantic`)

Pydantic models can encode both categorical types (`bool`, `enum.Enum`,
`typing.Literal`) and numerical bounds (`pydantic.Field(ge=..., le=...)`)
directly on the class definition. Optional fields (`T | None`) automatically
include `"None"` in `CategoricalAttribute` or set `clip_to_range=False` on
`NumericalAttribute`:

```python
from typing import Literal
from dpsynth.adapters import pydantic as pydantic_adapter
import pydantic


class UserRecord(pydantic.BaseModel):
  age: int = pydantic.Field(ge=18, le=100)
  tier: Literal["FREE", "PRO"]
  city: str


domains = pydantic_adapter.infer_domain(UserRecord)
```

If numeric fields on the Pydantic model do not carry `Field(ge=..., le=...)`
annotations, you can supply their bounds via `numerical_bounds={"age": (18,
100)}`. Nested `BaseModel` fields are also supported as long as they are
non-optional, and are flattened into dot-separated attribute names (such as
`"profile.age"`).

### 2. From Protocol Buffers (`dpsynth.adapters.protobuf`)

Protocol Buffer descriptors define field types (`int32`, `float`, `bool`,
`string`) and enum values, but do not store numeric min/max ranges:

```protobuf
message UserRecord {
  enum Tier {
    FREE = 0;
    PRO = 1;
  }
  int32 age = 1;
  Tier tier = 2;
  string city = 3;
}
```

Pass your Protobuf message class or `Descriptor` along with `numerical_bounds`
to `protobuf.infer_domain`:

```python
from dpsynth.adapters import protobuf

domains = protobuf.infer_domain(
    UserRecord,
    numerical_bounds={"age": (18, 100)},
)
```

Enums and booleans map to `CategoricalAttribute`, numeric fields map to
`NumericalAttribute` using `numerical_bounds`, and strings map to
`OpenSetCategoricalAttribute`. Nested submessages are also supported as long as
they are non-repeated (flattened into dot-separated keys such as
`"profile.age"`), and setting `ignore_unsupported_fields=True` skips repeated,
unsupported, or unbounded fields.

### 3. From Pandas Dtypes (`dpsynth.adapters.pandas`)

When working with typed tabular formats such as Parquet or Arrow, or when a
public schema is already represented as Pandas dtypes (`pd.CategoricalDtype`,
`bool`, `int64`, `float64`, `string`), use `pandas.infer_domain` on a
`df.dtypes` `pd.Series` or a `{column: dtype}` mapping:

```python
from dpsynth.adapters import pandas as pandas_adapter
import pandas as pd

schema_dtypes = {
    "age": "int64",
    "tier": pd.CategoricalDtype(["FREE", "PRO"]),
    "city": "string",
}

domains = pandas_adapter.infer_domain(
    schema_dtypes,
    numerical_bounds={"age": (18, 100)},
)
```

```{warning}
When inferring a `CategoricalAttribute` from a `pd.CategoricalDtype`, the list
of categories in `dtype.categories` **must** come from a public schema. Never
call `df[col].astype("category")` on sensitive data and pass the resulting
dtypes to `infer_domain`, as doing so extracts the exact active support of the
private dataset without differential privacy. For string columns with unknown
categories, leave the dtype as `"string"` (or `object`) so it maps to
`OpenSetCategoricalAttribute`.
```

### Customizing Inferred Domains

Because `infer_domain` returns a plain `dict[str, domain.AttributeType]`, you
can easily customize specific columns after inference when a schema container
does not capture a nuance of your domain (for example, replacing an inferred
`OpenSetCategoricalAttribute` on a string column with a known closed-set
`CategoricalAttribute`, or specifying custom `bin_edges` on a numerical column):

```python
domains = protobuf.infer_domain(
    UserRecord,
    numerical_bounds={"age": (18, 100)},
)

# Override an open-set string field when valid values are publicly known:
domains["city"] = domain.CategoricalAttribute(
    possible_values=["OTHER", "NYC", "LA", "SF"],
)
```
