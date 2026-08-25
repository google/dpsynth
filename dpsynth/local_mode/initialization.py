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

"""Utilities for measuring and integer-encoding single columns."""

from __future__ import annotations

import dataclasses
import math
from typing import TypeVar

import dp_accounting
from dpsynth import api
from dpsynth import domain
from dpsynth.local_mode import _quantiles
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
import numpy as np
import scipy.stats

_M = TypeVar('_M')


def encode_to_grid(values, lower, upper, delta, **_):
  """Maps finite value(s) to quantile-grid index/indices in [0, grid_size - 1].

  Clipping to ``[lower, upper]`` folds out-of-grid values into the boundary bins
  and guarantees the returned index stays in range. Polymorphic over scalars and
  NumPy arrays; callers must handle NaN / out-of-domain filtering beforehand.

  Args:
    values: A finite scalar or NumPy array of standardized numerical values.
    lower: The inclusive lower bound of the candidate grid.
    upper: The inclusive upper bound of the candidate grid.
    delta: The spacing between adjacent grid points.

  Returns:
    The nearest grid index (or array of indices) as ``np.int64``.
  """
  clamped = np.clip(values, lower, upper)
  return np.round((clamped - lower) / delta).astype(np.int64)


@dataclasses.dataclass
class ColumnMeasurement:
  """Result of running a column initializer on raw data.

  Attributes:
    categorical_attribute: The discovered or constructed CategoricalAttribute
      defining the discrete domain for this column.
    bin_edges: Inner bin edges for numerical columns (used for
      discretize/undiscretize). None for categorical columns.
    measurement: A noisy one-way marginal measurement, or None if the
      initializer does not produce one (e.g. NumericalInitializer).
  """

  categorical_attribute: domain.CategoricalAttribute
  bin_edges: np.ndarray | None = None
  measurement: mbi.LinearMeasurement | None = None


@dataclasses.dataclass(frozen=True, kw_only=True)
class NumericalInitializerConfig(api.MechanismConfig):
  """Configuration for initializing numerical attributes."""

  num_partitions: int
  max_grid_size: int = 10_000_000
  epsilon_ratio: float = 2.0
  attribute: domain.NumericalAttribute | None = None

  def __post_init__(self):
    if self.max_grid_size < 2:
      raise ValueError(f'max_grid_size must be >= 2, got {self.max_grid_size}.')
    if self.num_partitions >= self.max_grid_size:
      raise ValueError(f'{self.num_partitions=} >= {self.max_grid_size=}')

  def grid_spec(
      self, attribute: domain.NumericalAttribute
  ) -> tuple[float, float, int]:
    """Returns (lower, upper, grid_size) for the quantile candidate grid."""
    min_value = float(attribute.min_value)
    if attribute.dtype == 'int':
      m = _quantiles.jitter_factor(self.num_partitions)
      budget = max(2, self.max_grid_size // m)
      int_range = int(attribute.max_value - attribute.min_value + 1)
      step = max(1, math.ceil(int_range / budget))
      gs = math.ceil(int_range / step)
      return min_value, min_value + (gs - 1) * step, gs

    return min_value, float(attribute.exclusive_max_value), self.max_grid_size

  def configure(
      self,
      attribute: domain.NumericalAttribute | None = None,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> NumericalInitializer:
    api.validate_max_records_per_user(max_records_per_user)
    attr = attribute if attribute is not None else self.attribute
    if attr is None:
      raise ValueError('NumericalInitializerConfig requires attribute.')

    levels = int(np.log2(self.num_partitions))
    if 2**levels != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')

    rho_ratio = self.epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    bound_config = (
        self
        if self.attribute is attr
        else dataclasses.replace(self, attribute=attr)
    )
    return NumericalInitializer(
        config=bound_config,
        attribute=attr,
        epsilon_levels=tuple(eps.tolist()),
        max_records_per_user=max_records_per_user,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class NumericalInitializer(api.CalibratedMechanism):
  """Calibrated mechanism for initializing numerical attributes."""

  config: NumericalInitializerConfig
  attribute: domain.NumericalAttribute
  epsilon_levels: tuple[float, ...]
  max_records_per_user: int = 1

  def __post_init__(self):
    if self.max_records_per_user != 1:
      raise NotImplementedError('max_records_per_user != 1 not yet supported.')

  @property
  def _num_levels(self) -> int:
    return int(np.log2(self.config.num_partitions))

  @property
  def grid_spec(self) -> tuple[float, float, int]:
    return self.config.grid_spec(self.attribute)

  @property
  def grid_size(self) -> int:
    return self.grid_spec[2]

  @property
  def zcdp_rho(self) -> float:
    return sum(e**2 / 8.0 for e in self.epsilon_levels)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    return dp_accounting.ComposedDpEvent([
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self.epsilon_levels
    ])

  def __call__(
      self,
      rng: np.random.Generator,
      data: np.ndarray,
      *,
      estimated_total: float | None = None,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the discretization transform."""
    counts = self._grid_histogram(data)
    return self.from_summary(rng, counts, estimated_total=estimated_total)

  def _grid_histogram(self, data):
    """Returns the quantile candidate-grid histogram (length grid_size)."""
    # Applies NumericalAttribute.standardize semantics in a vectorized manner.
    lower, upper, gs = self.grid_spec
    delta = (upper - lower) / (gs - 1)
    attr = self.attribute
    values = np.asarray(data, dtype=float)
    if attr.clip_to_range:
      values = np.where(np.isnan(values), attr.min_value, values)
    else:
      in_domain = (values >= attr.min_value) & (values <= attr.max_value)
      values = values[in_domain]
    if attr.dtype == 'int':
      values = np.round(values)
    indices = encode_to_grid(values, lower, upper, delta)
    return np.bincount(indices, minlength=gs)

  def from_summary(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
      *,
      estimated_total: float | None = None,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated histogram counts."""
    jitter_strategy = 'refine' if self.attribute.dtype == 'int' else 'symmetric'
    indices = _quantiles.quantiles_from_histogram(
        rng,
        counts,
        epsilon_levels=np.asarray(self.epsilon_levels),
        jitter_strategy=jitter_strategy,
        max_records_per_user=self.max_records_per_user,
    )
    lower, upper, _ = self.grid_spec
    delta = (upper - lower) / max(1, np.asarray(counts).size - 1)
    raw_edges = [lower + i * delta for i in indices]

    return edges_to_column_measurement(
        raw_edges=raw_edges,
        attribute=self.attribute,
        zcdp_rho=self.zcdp_rho,
        estimated_total=estimated_total,
        max_records_per_user=self.max_records_per_user,
    )


def edges_to_column_measurement(
    raw_edges,
    attribute,
    zcdp_rho,
    estimated_total=None,
    max_records_per_user=1,
):
  """Converts raw quantile edges into a ColumnMeasurement.

  Handles edge deduplication, degenerate-bin removal, and categorical
  attribute construction.  Shared between the data-based
  ``NumericalInitializer`` and the histogram-based
  ``HistogramNumericalInitializer``.

  Args:
    raw_edges: Quantile edge values (unsorted duplicates are fine).
    attribute: The ``NumericalAttribute`` defining the data domain.
    zcdp_rho: Total zCDP rho consumed by the quantile mechanism.
    estimated_total: If provided, a heuristic one-way measurement is included.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.

  Returns:
    A ``ColumnMeasurement`` with bin edges and optionally a measurement.
  """
  raw_edges = np.asarray(raw_edges, dtype=float)
  bin_edges, edge_counts = np.unique(raw_edges, return_counts=True)
  # Edges at or above max_value produce a degenerate empty tail bin;
  # absorb their weight into the last real bin.
  max_val = attribute.max_value
  if len(bin_edges) > 0 and bin_edges[-1] >= max_val:
    tail_count = edge_counts[-1]
    bin_edges = bin_edges[:-1]
    edge_counts = edge_counts[:-1]
    bin_weights = np.append(edge_counts, tail_count + 1)
  else:
    bin_weights = np.append(edge_counts, 1)
  cat_attr = vtx.categorical_attribute_from_edges(bin_edges, attribute)

  measurement = None
  if estimated_total is not None:
    if not attribute.clip_to_range:
      # Prepend zero weight for the OUT_OF_DOMAIN slot at index 0.
      bin_weights = np.r_[0, bin_weights]
    counts = estimated_total * bin_weights / bin_weights.sum()
    stddev = max_records_per_user / np.sqrt(zcdp_rho)
    measurement = mbi.LinearMeasurement(
        counts,
        (),
        stddev=stddev,
        query=mbi.DatavectorQuery(use_for_total_estimation=False),
    )

  return ColumnMeasurement(cat_attr, bin_edges, measurement=measurement)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CategoricalInitializerConfig(api.MechanismConfig):
  """Configuration for initializing categorical attributes."""

  attribute: domain.CategoricalAttribute | None = None

  def configure(
      self,
      attribute: domain.CategoricalAttribute | None = None,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> CategoricalInitializer:
    api.validate_max_records_per_user(max_records_per_user)
    attr = attribute if attribute is not None else self.attribute
    if attr is None:
      raise ValueError('CategoricalInitializerConfig requires attribute.')
    bound_config = (
        self
        if self.attribute is attr
        else dataclasses.replace(self, attribute=attr)
    )
    return CategoricalInitializer(
        config=bound_config,
        attribute=attr,
        sigma=math.sqrt(0.5 / zcdp_rho),
        max_records_per_user=max_records_per_user,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CategoricalInitializer(api.CalibratedMechanism):
  """Calibrated mechanism for initializing categorical attributes."""

  config: CategoricalInitializerConfig
  attribute: domain.CategoricalAttribute
  sigma: float
  max_records_per_user: int = 1

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the noisy histogram."""
    encoded = vtx.discrete_encode(data, self.attribute)
    counts = np.bincount(encoded, minlength=self.attribute.size)
    return self.from_summary(rng, counts)

  def from_summary(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated counts."""
    noisy = primitives.add_gaussian_noise(
        rng, counts, self.sigma, self.max_records_per_user
    )
    noisy_counts = np.asarray(noisy)
    measurement = mbi.LinearMeasurement(
        noisy_counts,
        (),
        stddev=self.max_records_per_user * self.sigma,
    )
    return ColumnMeasurement(self.attribute, measurement=measurement)


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenSetInitializerConfig(api.MechanismConfig):
  """Configuration for initializing open-set categorical attributes."""

  min_count: int = 1
  attribute: domain.OpenSetCategoricalAttribute | None = None

  def configure(
      self,
      attribute: domain.OpenSetCategoricalAttribute | None = None,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      max_records_per_user: int = 1,
  ) -> OpenSetInitializer:
    api.validate_max_records_per_user(max_records_per_user)
    attr = attribute if attribute is not None else self.attribute
    if attr is None:
      raise ValueError('OpenSetInitializerConfig requires attribute.')
    bound_config = (
        self
        if self.attribute is attr
        else dataclasses.replace(self, attribute=attr)
    )
    return OpenSetInitializer(
        config=bound_config,
        attribute=attr,
        max_records_per_user=max_records_per_user,
        sigma=math.sqrt(0.5 / zcdp_rho),
        delta=delta,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenSetInitializer(api.CalibratedMechanism):
  """Calibrated mechanism for initializing open-set categorical attributes."""

  config: OpenSetInitializerConfig
  attribute: domain.OpenSetCategoricalAttribute
  max_records_per_user: int = 1
  sigma: float
  delta: float

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    main_event = dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)
    failure_event = dp_accounting.dp_event.EpsilonDeltaDpEvent(0, self.delta)
    return dp_accounting.ComposedDpEvent([main_event, failure_event])

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a differentially private measurement of the given data."""
    unique_values, inverse = np.unique(data, return_inverse=True)
    counts = np.bincount(inverse)
    return self.from_summary(rng, unique_values, counts)

  def from_summary(
      self,
      rng: np.random.Generator,
      unique_values: np.ndarray,
      counts: np.ndarray,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated value counts."""
    above_min = counts >= self.config.min_count
    eligible_idx = np.where(above_min)[0]
    eligible_counts = counts[above_min].astype(float)

    noisy = primitives.add_gaussian_noise(
        rng, eligible_counts, self.sigma, self.max_records_per_user
    )
    noisy_counts = np.asarray(noisy)

    stddev = self.max_records_per_user * self.sigma
    base = float(self.max_records_per_user + self.config.min_count - 1)
    threshold = base + stddev * scipy.stats.norm.ppf(1.0 - self.delta)
    passed = noisy_counts >= threshold

    selected_partitions = eligible_idx[passed]
    estimated_counts = noisy_counts[passed]

    selected_values = np.array(
        [str(v) for v in unique_values[selected_partitions]]
    )

    if self.attribute.public_possible_values:
      pub = np.array(self.attribute.public_possible_values)
      selected_values, estimated_counts = primitives.ensure_public_partitions(
          rng,
          selected_values,
          estimated_counts,
          stddev,
          pub,
      )

    # Build the discovered domain: default first, then selected values.
    default = self.attribute.default_value
    possible_values = [default] + selected_values.tolist()
    cat_attr = domain.CategoricalAttribute(possible_values)

    # The measurement covers only the discovered partitions (indices 1:),
    # not the unmeasured default at index 0.
    measurement = mbi.LinearMeasurement(
        estimated_counts,  # pyrefly: ignore[bad-argument-type]
        (),
        stddev=stddev,
        query=mbi.SlicedQuery(start=1),
    )
    return ColumnMeasurement(cat_attr, measurement=measurement)
