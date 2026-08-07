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


def _create_initializers(
    domains: Mapping[str, domain.AttributeType],
    numerical_bins: int,
    init_delta: float,
    max_records_per_user: int = 1,
) -> dict[str, primitives.DPMechanism]:
  """Creates per-column initializers from the domain specification.

  Args:
    domains: Mapping from column names to attribute domain specifications.
    numerical_bins: Number of bins for numerical discretization.
    init_delta: Delta for open-set categorical partition selection.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Sensitivity (and hence added noise) is scaled by this
      factor to provide user-level rather than record-level DP; the privacy
      analysis is unchanged. Soundness relies on the caller enforcing the bound.

  Returns:
    A dictionary mapping column names to uncalibrated initializer instances.

  Raises:
    ValueError: If a column has an unsupported attribute type.
  """
  initializers = {}
  for col, attr in domains.items():
    if isinstance(attr, domain.NumericalAttribute):
      initializers[col] = initialization.NumericalInitializer(
          name=col,
          num_partitions=numerical_bins,
          attribute=attr,
          max_records_per_user=max_records_per_user,
      )
    elif isinstance(attr, domain.CategoricalAttribute):
      initializers[col] = initialization.CategoricalInitializer(
          name=col, attribute=attr, max_records_per_user=max_records_per_user
      )
    elif isinstance(attr, domain.OpenSetCategoricalAttribute):
      initializers[col] = initialization.OpenSetCategoricalInitializer(
          name=col,
          attribute=attr,
          delta=init_delta,
          max_records_per_user=max_records_per_user,
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
  attribute: domain.AttributeType

  def encode(self, values: np.ndarray) -> np.ndarray:
    """Encodes raw column values to discrete integer ids."""
    if self.column_measurement.bin_edges is not None:
      return vtx.discretize(
          # pyrefly: ignore[bad-argument-type]
          values, self.column_measurement.bin_edges, self.attribute
      )
    return vtx.discrete_encode(
        values, self.column_measurement.categorical_attribute
    )

  def decode(self, ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Decodes synthetic discrete ids back to the original domain."""
    if self.column_measurement.bin_edges is not None:
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
      domains: Mapping[str, domain.AttributeType],
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
class TabularSynthesizer(primitives.DPMechanism):
  """End-to-end DP synthetic data generation mechanism.

  This mechanism encodes input categorical and numerical data into a discrete
  domain using local mode primitives, runs a discrete mechanism on the
  discretized data, and converts the synthetic output back to the original
  domain.

  Usage::

      synth = TabularSynthesizer(domains=domains)
      calibrated = synth.configure(zcdp_rho=1.0)
      result = calibrated(rng, df)
      synthetic_df = result.synthetic_data

  Attributes:
    domains: Mapping from column names to attribute domain specifications.
    discrete_mechanism: The mechanism to run on the discretized data.
    numerical_bins: Number of bins for numerical attribute discretization.
    init_budget_fraction: Fraction of total zCDP budget allocated to per-column
      initialization (the rest goes to the discrete mechanism).
    initializers: Per-column initializer mechanisms. If None, created
      automatically from ``domains`` during ``configure()``.
    skip_compression: Whether to skip domain compression.
    cross_attribute_constraints: Constraints to enforce on generated data.
    experimental_max_records_per_user: EXPERIMENTAL and subject to change.
      Assumed upper bound on the number of records a single user contributes.
      Values greater than 1 scale the added noise (and mechanism sensitivity) to
      provide user-level rather than record-level DP; the privacy accounting is
      unchanged. This bound is NOT enforced -- soundness relies on the caller
      guaranteeing it via preprocessing.
  """

  domains: Mapping[str, domain.AttributeType]
  discrete_mechanism: discrete_mechanisms.DiscreteMechanism = dataclasses.field(
      default_factory=discrete_mechanisms.MSTMechanism
  )
  numerical_bins: int = 32
  init_budget_fraction: float = 0.1
  initializers: dict[str, primitives.DPMechanism] | None = None
  total_count_mechanism: primitives.DPGaussianCount | None = None
  cross_attribute_constraints: Sequence[constraints.Constraint] = ()
  experimental_max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.experimental_max_records_per_user)

  def configure(  # pyrefly: ignore[bad-override]
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
  ) -> TabularSynthesizer:
    """Returns a copy configured with the given privacy budget.

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
      zcdp_rho: The zCDP privacy budget.
      delta: Overall approximate DP delta for the mechanism. A fraction
        (``init_budget_fraction``) is allocated to partition selection for
        open-set columns. Must be positive when open-set categorical attributes
        are present.

    Returns:
      A new TabularSynthesizer with calibrated sub-mechanisms.

    Raises:
      ValueError: If open-set attributes exist but delta is 0.
    """
    num_open_set = sum(
        isinstance(attr, domain.OpenSetCategoricalAttribute)
        for attr in self.domains.values()
    )
    if num_open_set > 0 and delta <= 0:
      raise ValueError(
          'delta must be positive when open-set categorical attributes are'
          ' present. It is used for Gaussian partition selection.'
      )
    # Split delta across open-set columns, analogous to splitting zcdp_rho.
    # Under calibrate(), any delta not consumed here is automatically
    # available for the zCDP-to-(epsilon, delta) conversion, so this
    # simple additive split is tight.
    thresholding_delta = self.init_budget_fraction * delta
    per_col_delta = (
        thresholding_delta / num_open_set if num_open_set > 0 else 0.0
    )

    inits = self.initializers
    if inits is None:
      inits = _create_initializers(
          self.domains,
          self.numerical_bins,
          per_col_delta,
          self.experimental_max_records_per_user,
      )
    elif self.experimental_max_records_per_user > 1:
      # The synthesizer's experimental_max_records_per_user is the single
      # source of truth: the total-count and discrete mechanisms already use it,
      # so propagate it to caller-supplied initializers too.
      propagated = {}
      for col, init in inits.items():
        propagated[col] = dataclasses.replace(  # pytype: disable=wrong-arg-types
            init, max_records_per_user=self.experimental_max_records_per_user
        )
      inits = propagated
    init_rho = self.init_budget_fraction * zcdp_rho
    # +1 for the DPGaussianCount that always measures the total.
    per_col_rho = init_rho / (len(inits) + 1)
    discrete_rho = zcdp_rho - init_rho

    calibrated_inits = {
        col: init.configure(zcdp_rho=per_col_rho) for col, init in inits.items()
    }
    calibrated_total = primitives.DPGaussianCount(
        max_records_per_user=self.experimental_max_records_per_user
    ).configure(zcdp_rho=per_col_rho)
    calibrated_discrete = dataclasses.replace(
        self.discrete_mechanism,
        max_records_per_user=self.experimental_max_records_per_user,
    ).configure(zcdp_rho=discrete_rho)
    return dataclasses.replace(
        self,
        initializers=calibrated_inits,
        discrete_mechanism=calibrated_discrete,
        total_count_mechanism=calibrated_total,
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed DpEvent for all sub-mechanisms.

    Returns:
      A ComposedDpEvent combining all initializer and discrete mechanism events.

    Raises:
      ValueError: If configure() has not been called.
    """
    if self.initializers is None or self.total_count_mechanism is None:
      raise ValueError(
          'Must call configure() or calibrate() before accessing dp_event.'
      )
    events = [init.dp_event for init in self.initializers.values()]
    events.append(self.total_count_mechanism.dp_event)
    events.append(self.discrete_mechanism.dp_event)
    return dp_accounting.ComposedDpEvent(events)

  def __call__(
      self, rng: np.random.Generator, data: pd.DataFrame
  ) -> DataGenerationResult:
    """Generates differentially private synthetic data.

    Args:
      rng: A numpy random number generator.
      data: The dataset to generate synthetic data for. Must contain all columns
        specified in ``domains``.

    Returns:
      A DataGenerationResult containing the synthetic DataFrame.

    Raises:
      ValueError: If configure() has not been called or if required columns are
        missing from the input data.
    """
    if self.initializers is None or self.total_count_mechanism is None:
      raise ValueError(
          'Must call configure() or calibrate() before running the mechanism.'
      )
    for col in self.domains:
      if col not in data.columns:
        raise ValueError(
            f'{col=} not found in dataset. Available: {list(data.columns)}'
        )

    # Phase 1: Per-column initialization.
    # Measure total count first, then run per-column initializers.
    any_col = next(iter(self.domains))
    total = max(1.0, self.total_count_mechanism(rng, data[any_col].values))
    k = self.experimental_max_records_per_user
    total_measurement = mbi.LinearMeasurement(
        np.array([total]),  # pyrefly: ignore[bad-argument-type]
        (),
        stddev=k * self.total_count_mechanism.sigma,  # pyrefly: ignore[unsupported-operation]
    )

    results: dict[str, initialization.ColumnMeasurement] = {}
    for col, init in self.initializers.items():
      if isinstance(init, initialization.NumericalInitializer):
        results[col] = init(rng, data[col].values, estimated_total=total)
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
    mbi_constraints = tuple(
        c.to_mbi() for c in self.cross_attribute_constraints
    )
    mechanism_result = self.discrete_mechanism(
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
