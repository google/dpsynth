# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end DP synthetic tabular data generation using local mode primitives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import warnings

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
import numpy as np
import pandas as pd


def create_initializers(
    domains: domain.Schema | Mapping[str, domain.AttributeType],
    numerical_bins: int,
) -> dict[str, api.MechanismConfig]:
  """Creates per-column initializers from the domain specification.

  Args:
    domains: Mapping from column names to attribute domain specifications.
    numerical_bins: Number of bins for numerical discretization.

  Returns:
    A dictionary mapping column names to uncalibrated initializer configs.

  Raises:
    ValueError: If a column has an unsupported attribute type.
  """
  initializers = {}
  attrs = domains.attributes if isinstance(domains, domain.Schema) else domains
  for col, attr in attrs.items():
    if isinstance(attr, domain.NumericalAttribute):
      initializers[col] = initialization.NumericalInitializerConfig(
          num_partitions=numerical_bins,
      )
    elif isinstance(attr, domain.CategoricalAttribute):
      initializers[col] = initialization.CategoricalInitializerConfig()
    elif isinstance(attr, domain.OpenSetCategoricalAttribute):
      initializers[col] = initialization.OpenSetInitializerConfig()
    else:
      raise ValueError(
          f'Unsupported attribute type for column {col!r}: {type(attr)}'
      )
  return initializers


@dataclasses.dataclass(frozen=True)
class ColumnCodec:
  """Maps one column between raw values and discrete ids, and back.

  Attributes:
    column_measurement: Per-column initialization result defining the discrete
      domain (and bin edges for numerical columns).
    attribute: The original attribute spec; only numerical (un)discretization
      uses it, for the value range.
  """

  column_measurement: initialization.ColumnMeasurement
  attribute: domain.AttributeType

  def encode(self, values: np.ndarray) -> np.ndarray:
    """Encodes raw column values to discrete integer ids."""
    if isinstance(self.column_measurement, initialization.NumericalMeasurement):
      return vtx.discretize(
          # pyrefly: ignore[bad-argument-type]
          values, self.column_measurement.bin_edges, self.attribute
      )
    return vtx.discrete_encode(
        values, self.column_measurement.categorical_attribute
    )

  def decode(self, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Decodes synthetic discrete ids back to the original domain."""
    if isinstance(self.column_measurement, initialization.NumericalMeasurement):
      return vtx.undiscretize(
          # pyrefly: ignore[bad-argument-type]
          ids, self.column_measurement.bin_edges, self.attribute, rng=rng
      )
    return vtx.discrete_decode(
        ids, self.column_measurement.categorical_attribute
    )


@dataclasses.dataclass(frozen=True)
class TabularCodec:
  """Encodes a table to the discrete domain and decodes synthetic output back.

  Attributes:
    columns: Per-column codecs, keyed by column name.
  """

  columns: Mapping[str, ColumnCodec]

  @classmethod
  def from_measurements(
      cls,
      results: Mapping[str, initialization.ColumnMeasurement],
      domains: domain.Schema | Mapping[str, domain.AttributeType],
  ) -> TabularCodec:
    """Builds a codec from initialization results and the original domains."""
    columns = {col: ColumnCodec(m, domains[col]) for col, m in results.items()}
    return cls(columns=columns)

  @property
  def mbi_domain(self) -> mbi.Domain:
    """The discrete mbi.Domain induced by the per-column categorical attributes."""
    cats = {
        col: c.column_measurement.categorical_attribute
        for col, c in self.columns.items()
    }
    return mbi.Domain(
        attributes=tuple(cats.keys()),
        shape=tuple(a.size for a in cats.values()),
        labels=tuple(tuple(a.possible_values) for a in cats.values()),
    )

  def one_way_measurements(self) -> list[mbi.LinearMeasurement]:
    """Constructs one-way marginal measurements from per-column noisy counts."""
    measurements = []
    for col, c in self.columns.items():
      cm = c.column_measurement
      if cm.noisy_counts is None:
        continue
      elif isinstance(cm, initialization.NumericalMeasurement):
        query = mbi.DatavectorQuery(use_for_total_estimation=False)
      elif isinstance(cm, initialization.OpenSetMeasurement):
        query = mbi.SlicedQuery(start=1)
      else:
        query = mbi.DatavectorQuery()
      measurement = mbi.LinearMeasurement(
          cm.noisy_counts, (col,), stddev=cm.stddev, query=query
      )
      measurements.append(measurement)
    return measurements

  def encode(self, data: pd.DataFrame) -> mbi.Dataset:
    """Encodes ``data`` into an mbi.Dataset over the discrete domain."""
    discrete = {
        col: c.encode(data[col].values) for col, c in self.columns.items()
    }
    # pyrefly: ignore[bad-argument-type]
    return mbi.Dataset(discrete, self.mbi_domain)

  def decode(
      self,
      synthetic: mbi.Dataset,
      rng: np.random.Generator,
      column_order: Sequence[str],
  ) -> pd.DataFrame:
    """Decodes synthetic discrete data back to a DataFrame."""
    ids = synthetic.to_dict()
    decoded = {col: c.decode(ids[col], rng) for col, c in self.columns.items()}
    return pd.DataFrame(decoded)[list(column_order)]


@dataclasses.dataclass
class DataGenerationResult:
  """Result of end-to-end DP synthetic data generation."""

  synthetic_data: pd.DataFrame
  discrete_mechanism_result: dm_common.DiscreteMechanismResult
  codec: TabularCodec


@dataclasses.dataclass
class TabularMechanism(api.CalibratedMechanism):
  """End-to-end DP synthetic tabular data generation, calibrated and runnable.

  Attributes:
    config: The original configuration.
    schema: The dataset schema and constraints.
    base_mechanism: The calibrated discrete mechanism.
    initializers: Per-column calibrated initializers.
    total_count_sigma: Sigma for the total-count mechanism.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes.
  """

  config: TabularConfig
  schema: domain.Schema
  base_mechanism: discrete_mechanisms.CalibratedMechanism
  initializers: dict[str, api.CalibratedMechanism]
  total_count_sigma: float = dataclasses.field(repr=False)
  max_records_per_user: int = 1

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent for all sub-mechanisms."""
    events = [init.dp_event for init in self.initializers.values()]
    events.append(
        dp_accounting.GaussianDpEvent(noise_multiplier=self.total_count_sigma)
    )
    events.append(self.base_mechanism.dp_event)
    events = [e for e in events if not isinstance(e, dp_accounting.NoOpDpEvent)]

    if not events:
      return dp_accounting.NoOpDpEvent()
    if len(events) == 1:
      return events[0]
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self,
      rng: np.random.Generator,
      data: pd.DataFrame,
      *,
      cross_attribute_constraints: Sequence[constraints.Constraint] = (),
  ) -> DataGenerationResult:
    """Generates differentially private synthetic data.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for. Must contain all columns
        specified in ``schema``.
      cross_attribute_constraints: Constraints to enforce on generated data.

    Returns:
      A DataGenerationResult containing the synthetic DataFrame.

    Raises:
      ValueError: If required columns are missing from the input data.
    """
    for col in self.schema:
      if col not in data.columns:
        raise ValueError(
            f'{col=} not found in dataset. Available: {list(data.columns)}'
        )
    if not cross_attribute_constraints:
      cross_attribute_constraints = (
          self.schema.constraints or self.config.cross_attribute_constraints
      )

    mbi_constraints = tuple(c.to_mbi() for c in cross_attribute_constraints)

    # Phase 1: Per-column initialization.
    # Measure total count first, then run per-column initializers.

    noisy_total = primitives.add_gaussian_noise(
        rng,
        len(data),
        self.total_count_sigma,
        self.max_records_per_user,
    )
    total = max(1.0, noisy_total)
    total_measurement = mbi.LinearMeasurement(
        noisy_measurement=np.array([total]),
        clique=(),
        stddev=self.max_records_per_user * self.total_count_sigma,
    )

    results: dict[str, initialization.ColumnMeasurement] = {}
    for col, init in self.initializers.items():
      if isinstance(init, initialization.NumericalInitializer):
        results[col] = init(rng, data[col].values, estimated_total=float(total))
      else:
        results[col] = init(rng, data[col].values)

    # Phase 2: Encode data to the discrete domain.
    codec = TabularCodec.from_measurements(results, self.schema)
    discrete = codec.encode(data)
    logging.info('[DPSynth]: Finished encoding data.')

    # Phase 3: Run the discrete mechanism and decode back to the input domain.
    # Feed the noisy total (clique ()) and one-way column measurements as
    # initial measurements so the mechanism does not re-measure them.
    column_order = [col for col in data.columns if col in self.schema]
    one_way_measurements = codec.one_way_measurements()
    initial_measurements = [total_measurement, *one_way_measurements]

    if self.config.compress_columns:
      candidate_cols = [
          col
          for col in self.schema
          if isinstance(self.schema[col], domain.CategoricalAttribute)
      ]

      mappings = dm_common.compression_mappings(
          one_way_measurements,
          compress_columns=candidate_cols,
          constraints=mbi_constraints,
      )
    else:
      mappings = {}

    if mappings:
      discrete = discrete.compress(mappings)  # pyrefly: ignore[bad-argument-type]
      initial_measurements = [
          m.compress(mappings, discrete.domain)  # pyrefly: ignore[bad-argument-type]
          for m in initial_measurements
      ]
    logging.info('[DPSynth]: Compressed discrete domain:\n%s', discrete.domain)

    cfg = self.config.discrete_mechanism
    if hasattr(cfg, 'supporting_cliques'):
      cliques = cfg.supporting_cliques(discrete.domain)
      discrete = dm_common.precompute_marginals(
          discrete,
          cliques,  # pyrefly: ignore[bad-argument-type]
          use_jax=self.config.use_jax_for_bincount,
      )

    mechanism_result = self.base_mechanism(
        rng,
        data=discrete,
        initial_measurements=initial_measurements,
        constraints=mbi_constraints,
    )
    logging.info('[DPSynth]: Generated discrete synthetic data.')

    if mechanism_result.synthetic_data is not None:
      synthetic_discrete = mechanism_result.synthetic_data
    else:
      synthetic_discrete = dm_common.generate_synthetic_data(
          mechanism_result.model,
          rng,
          use_jax=self.config.use_jax_for_generation,
      )
    if mappings:
      synthetic_discrete = synthetic_discrete.decompress(mappings)  # pyrefly: ignore[bad-argument-type]
      mechanism_result = dataclasses.replace(
          mechanism_result,
          synthetic_data=synthetic_discrete,
          mappings=mappings,
      )
    else:
      mechanism_result = dataclasses.replace(
          mechanism_result,
          synthetic_data=synthetic_discrete,
      )

    synthetic_data = codec.decode(synthetic_discrete, rng, column_order)
    logging.info('[DPSynth]: Converted data back to original domain.')

    return DataGenerationResult(
        synthetic_data=synthetic_data,
        discrete_mechanism_result=mechanism_result,
        codec=codec,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TabularConfig(api.MechanismConfig):
  """Configures end-to-end DP synthetic data generation.

  Usage::

      config = TabularConfig(discrete_mechanism=MSTConfig())
      calibrated = config.configure(schema, zcdp_rho=1.0)
      result = calibrated(rng, df)
      synthetic_df = result.synthetic_data

  Attributes:
    domains: Mapping from column names to attribute domain specifications.
    discrete_mechanism: The mechanism to run on the discretized data.
    numerical_bins: Number of bins for numerical attribute discretization.
    init_budget_fraction: Fraction of total zCDP budget allocated to per-column
      initialization (the rest goes to the discrete mechanism).
    cross_attribute_constraints: Constraints to enforce on generated data.
    compress_columns: Whether to compress rare categories (< 3*sigma) for
      CategoricalAttribute columns not present in constraints.
    use_jax_for_bincount: Whether to use JAX-accelerated bincount (via
      mbi.extensions.precompute_marginals) to compute marginals from Dataset.
    use_jax_for_generation: Whether to use JAX-accelerated generation (via
      mbi.extensions.synthetic_data) to generate synthetic data from the model.
    working_dir: Base directory path for intermediate checkpoints (passed down
      to the underlying discrete mechanism). If None, checkpointing is disabled.
  """

  domains: Mapping[str, domain.AttributeType] | None = None
  discrete_mechanism: api.MechanismConfig = discrete_mechanisms.MSTConfig()
  numerical_bins: int = 32
  init_budget_fraction: float = 0.1
  cross_attribute_constraints: Sequence[constraints.Constraint] = ()
  compress_columns: bool = False
  use_jax_for_bincount: bool = False
  use_jax_for_generation: bool = False
  working_dir: str | None = None

  def _compute_per_col_deltas(self, domains, delta):
    # Split delta across open-set columns, analogous to splitting zcdp_rho.
    # Under calibrate(), any delta not consumed here is automatically
    # available for the zCDP-to-(epsilon, delta) conversion, so this
    # simple additive split is tight.
    num_open_set = sum(
        isinstance(attr, domain.OpenSetCategoricalAttribute)
        for attr in domains.values()
    )
    if num_open_set > 0 and delta <= 0:
      raise ValueError(
          'delta must be positive when open-set categorical attributes are'
          ' present. It is used for Gaussian partition selection.'
      )

    thresholding_delta = self.init_budget_fraction * delta

    per_col_deltas = {}
    for col in domains:
      if isinstance(domains[col], domain.OpenSetCategoricalAttribute):
        per_col_deltas[col] = thresholding_delta / num_open_set
      else:
        per_col_deltas[col] = 0.0
    return per_col_deltas

  def configure(
      self,
      schema: domain.Schema | Mapping[str, domain.AttributeType] | None = None,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> TabularMechanism:
    """Returns a calibrated mechanism configured with the given privacy budget.

    Splits the budget additively, just as it does for ``zcdp_rho``:

    - ``init_budget_fraction`` of ``zcdp_rho`` goes to per-column initializers
      (split evenly, including a total-count mechanism); the remainder goes to
      the discrete mechanism.
    - ``init_budget_fraction`` of ``delta`` is reserved for open-set partition
      selection (split evenly across open-set columns); the remaining delta is
      unused by pure-zCDP sub-mechanisms.

    When ``calibrate(epsilon, delta)`` is called, the base class binary search
    passes the guarantee delta here. Because the thresholding delta is honestly
    reported in the composite ``dp_event``, the binary search automatically
    ensures the overall (epsilon, delta) guarantee is tight.

    Args:
      schema: Dataset schema or mapping from column names to attribute domain.
      zcdp_rho: The zCDP privacy budget.
      delta: Overall approximate DP delta for the mechanism. A fraction
        (``init_budget_fraction``) is allocated to partition selection for
        open-set columns. Must be positive when open-set categorical attributes
        are present.
      max_records_per_user: Assumed upper bound on the number of records a
        single user contributes. Values greater than 1 scale the added noise
        (and mechanism sensitivity) to provide user-level rather than
        record-level DP; the privacy accounting is unchanged. This bound is NOT
        enforced -- soundness relies on the caller guaranteeing it via
        preprocessing.

    Returns:
      A calibrated TabularMechanism ready to be run on tabular data.

    Raises:
      ValueError: If open-set attributes exist but delta is 0.
    """
    if schema is not None and isinstance(schema, domain.Schema):
      pass
    elif schema is not None:
      schema = domain.Schema(
          schema, constraints=self.cross_attribute_constraints
      )
    elif self.domains is not None:
      schema = domain.Schema(
          self.domains, constraints=self.cross_attribute_constraints
      )
    else:
      raise ValueError(
          'No schema provided. Pass schema to configure() or set domains at'
          ' construction time.'
      )

    api.validate_max_records_per_user(max_records_per_user)
    per_col_deltas = self._compute_per_col_deltas(schema, delta)

    inits = create_initializers(schema, self.numerical_bins)
    init_rho = self.init_budget_fraction * zcdp_rho
    # +1 for the DPGaussianCount that always measures the total.
    per_col_rho = init_rho / (len(inits) + 1)
    discrete_rho = (1 - self.init_budget_fraction) * zcdp_rho
    total_count_sigma = (0.5 / per_col_rho) ** 0.5

    calibrated_inits: dict[str, api.CalibratedMechanism]

    calibrated_inits = {
        col: init.configure(
            schema[col],
            zcdp_rho=per_col_rho,
            delta=per_col_deltas[col],
            max_records_per_user=max_records_per_user,
        )
        for col, init in inits.items()
    }

    discrete_mechanism = self.discrete_mechanism.with_working_dir(
        self.working_dir
    )

    calibrated_discrete = discrete_mechanism.configure(
        max_records_per_user=max_records_per_user,
        zcdp_rho=discrete_rho,
    )

    return TabularMechanism(
        config=self,
        schema=schema,
        base_mechanism=calibrated_discrete,
        initializers=calibrated_inits,
        total_count_sigma=total_count_sigma,
        max_records_per_user=max_records_per_user,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TabularSynthesizer(TabularConfig):
  """Deprecated. Use TabularConfig and TabularMechanism instead."""

  def __post_init__(self):
    warnings.warn(
        'TabularSynthesizer is deprecated. Use TabularConfig for configuration '
        'and TabularMechanism for the calibrated runnable mechanism.',
        DeprecationWarning,
        stacklevel=2,
    )
