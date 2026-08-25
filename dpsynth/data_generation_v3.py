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
import typing
import warnings

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import constraints as constraints_mod
from dpsynth import discrete_mechanisms
from dpsynth import domain as domain_mod
from dpsynth.discrete_mechanisms import common as dm_common
from dpsynth.domain import Schema
from dpsynth.local_mode import initialization
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
import numpy as np
import pandas as pd


def create_initializers(
    domains: Mapping[str, domain_mod.AttributeType],
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
  for col, attr in domains.items():
    if isinstance(attr, domain_mod.NumericalAttribute):
      initializers[col] = initialization.NumericalInitializerConfig(
          name=col,
          num_partitions=numerical_bins,
          attribute=attr,
      )
    elif isinstance(attr, domain_mod.CategoricalAttribute):
      initializers[col] = initialization.CategoricalInitializerConfig(
          name=col,
          attribute=attr,
      )
    elif isinstance(attr, domain_mod.OpenSetCategoricalAttribute):
      initializers[col] = initialization.OpenSetInitializerConfig(
          name=col,
          attribute=attr,
      )
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
  attribute: domain_mod.AttributeType

  def encode(self, values: np.ndarray) -> np.ndarray:
    """Encodes raw column values to discrete integer ids."""
    if self.column_measurement.bin_edges is not None:
      return vtx.discretize(
          values,
          self.column_measurement.bin_edges,
          typing.cast(domain_mod.NumericalAttribute, self.attribute),
      )
    return vtx.discrete_encode(
        values, self.column_measurement.categorical_attribute
    )

  def decode(self, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Decodes synthetic discrete ids back to the original domain."""
    if self.column_measurement.bin_edges is not None:
      return vtx.undiscretize(
          ids,
          self.column_measurement.bin_edges,
          typing.cast(domain_mod.NumericalAttribute, self.attribute),
          rng=rng,
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
      domains: Mapping[str, domain_mod.AttributeType],
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
    """Returns the non-None one-way marginal measurements, in column order."""
    return [
        c.column_measurement.measurement
        for c in self.columns.values()
        if c.column_measurement.measurement is not None
    ]

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
    domains: Mapping from column names to attribute domain specifications.
    base_mechanism: The calibrated discrete mechanism.
    initializers: Per-column calibrated initializers.
    total_count_sigma: Sigma for the total-count mechanism.
    cross_attribute_constraints: Constraints to enforce on generated data.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes.
  """

  config: TabularConfig
  domains: Mapping[str, domain_mod.AttributeType]
  base_mechanism: discrete_mechanisms.CalibratedMechanism
  initializers: dict[str, api.CalibratedMechanism]
  total_count_sigma: float = dataclasses.field(repr=False)
  cross_attribute_constraints: Sequence[constraints_mod.Constraint] = ()
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
      cross_attribute_constraints: Sequence[constraints_mod.Constraint] = (),
  ) -> DataGenerationResult:
    """Generates differentially private synthetic data.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for. Must contain all columns
        specified in ``domains``.
      cross_attribute_constraints: Constraints to enforce on generated data.

    Returns:
      A DataGenerationResult containing the synthetic DataFrame.

    Raises:
      ValueError: If required columns are missing from the input data.
    """
    for col in self.domains:
      if col not in data.columns:
        raise ValueError(
            f'{col=} not found in dataset. Available: {list(data.columns)}'
        )
    if not cross_attribute_constraints:
      cross_attribute_constraints = self.cross_attribute_constraints

    mbi_constraints = tuple(
        c.to_mbi(self.domains) for c in cross_attribute_constraints
    )

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
    codec = TabularCodec.from_measurements(results, self.domains)
    discrete = codec.encode(data)
    logging.info('[DPSynth]: Finished encoding data.')

    # Phase 3: Run the discrete mechanism and decode back to the input domain.
    # Feed the noisy total (clique ()) and one-way column measurements as
    # initial measurements so the mechanism does not re-measure them.
    column_order = [col for col in data.columns if col in self.domains]
    initial_measurements = [total_measurement, *codec.one_way_measurements()]
    mechanism_result = self.base_mechanism(
        rng,
        data=discrete,
        initial_measurements=initial_measurements,
        constraints=mbi_constraints,
    )
    logging.info('[DPSynth]: Generated discrete synthetic data.')

    synthetic_data = codec.decode(
        mechanism_result.synthetic_data, rng, column_order
    )
    logging.info('[DPSynth]: Converted data back to original domain.')

    return DataGenerationResult(
        synthetic_data=synthetic_data,
        discrete_mechanism_result=mechanism_result,
        codec=codec,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TabularConfig(api.MechanismConfig):
  """Configures end-to-end DP synthetic data generation.

  This config encodes input categorical and numerical data into a discrete
  domain using local mode primitives, runs a discrete mechanism on the
  discretized data, and converts the synthetic output back to the original
  domain.

  Usage::

      config = TabularConfig()
      calibrated = config.calibrate(schema=schema, epsilon=1.0, delta=1e-5)
      result = calibrated(rng, df)
      synthetic_df = result.synthetic_data

  Attributes:
    discrete_mechanism: The mechanism to run on the discretized data.
    numerical_bins: Number of bins for numerical attribute discretization.
    init_budget_fraction: Fraction of total zCDP budget allocated to per-column
      initialization (the rest goes to the discrete mechanism).
    cross_attribute_constraints: Constraints to enforce on generated data.
    schema: Schema or mapping from column names to attribute domain
      specifications. Optional; can be supplied at calibration time via
      ``configure(schema=...)`` or ``calibrate(schema=...)``.
    domains: Alias for ``schema`` for backward compatibility.
  """

  discrete_mechanism: api.MechanismConfig = discrete_mechanisms.MSTConfig()
  numerical_bins: int = 32
  init_budget_fraction: float = 0.1
  cross_attribute_constraints: Sequence[constraints_mod.Constraint] = ()
  schema: domain_mod.Schema | Mapping[str, domain_mod.AttributeType] | None = (
      None
  )
  domains: Mapping[str, domain_mod.AttributeType] | None = None

  def _compute_per_col_deltas(
      self, schema: Mapping[str, domain_mod.AttributeType], delta: float
  ) -> dict[str, float]:
    # Split delta across open-set columns, analogous to splitting zcdp_rho.
    # Under calibrate(), any delta not consumed here is automatically
    # available for the zCDP-to-(epsilon, delta) conversion, so this
    # simple additive split is tight.
    num_open_set = sum(
        isinstance(attr, domain_mod.OpenSetCategoricalAttribute)
        for attr in schema.values()
    )
    if num_open_set > 0 and delta <= 0:
      raise ValueError(
          'delta must be positive when open-set categorical attributes are'
          ' present. It is used for Gaussian partition selection.'
      )

    thresholding_delta = self.init_budget_fraction * delta

    per_col_deltas = {}
    for col in schema:
      if isinstance(schema[col], domain_mod.OpenSetCategoricalAttribute):
        per_col_deltas[col] = thresholding_delta / num_open_set
      else:
        per_col_deltas[col] = 0.0
    return per_col_deltas

  def configure(
      self,
      *,
      zcdp_rho: float,
      schema: (
          domain_mod.Schema | Mapping[str, domain_mod.AttributeType] | None
      ) = None,
      domain: (
          domain_mod.Schema | Mapping[str, domain_mod.AttributeType] | None
      ) = None,
      constraints: Sequence[constraints_mod.Constraint] | None = None,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> TabularMechanism:
    """Returns a calibrated mechanism configured with the given privacy budget."""
    api.validate_max_records_per_user(max_records_per_user)
    if schema is not None:
      resolved_schema = schema
    elif domain is not None:
      resolved_schema = domain
    elif self.schema is not None:
      resolved_schema = self.schema
    elif self.domains is not None:
      resolved_schema = self.domains
    else:
      raise ValueError(
          "Must provide 'schema' to configure() or in TabularConfig"
          ' constructor.'
      )

    attr_schema = (
        resolved_schema.attributes
        if isinstance(resolved_schema, Schema)
        else resolved_schema
    )
    schema_constraints = (
        resolved_schema.constraints
        if isinstance(resolved_schema, Schema)
        else ()
    )

    resolved_constraints = (
        constraints
        if constraints is not None
        else (self.cross_attribute_constraints or schema_constraints)
    )

    per_col_deltas = self._compute_per_col_deltas(attr_schema, delta)
    inits = create_initializers(attr_schema, self.numerical_bins)
    init_rho = self.init_budget_fraction * zcdp_rho
    per_col_rho = init_rho / (len(inits) + 1)
    discrete_rho = zcdp_rho - init_rho

    calibrated_inits: dict[str, api.CalibratedMechanism] = {
        col: init.configure(
            zcdp_rho=per_col_rho,
            delta=per_col_deltas[col],
            max_records_per_user=max_records_per_user,
        )
        for col, init in inits.items()
    }
    total_count_sigma = (0.5 / per_col_rho) ** 0.5

    calibrated_discrete = self.discrete_mechanism.configure(
        max_records_per_user=max_records_per_user,
        zcdp_rho=discrete_rho,
    )

    return TabularMechanism(
        config=self,
        domains=attr_schema,
        base_mechanism=calibrated_discrete,
        initializers=calibrated_inits,
        total_count_sigma=total_count_sigma,
        cross_attribute_constraints=resolved_constraints,
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
