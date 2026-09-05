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

import dp_accounting
from dpsynth import api
from dpsynth import domain
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import numpy as np
import scipy.stats


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


@dataclasses.dataclass(frozen=True)
class NumericalMeasurement:
  """Measurement from a numerical initializer."""

  categorical_attribute: domain.CategoricalAttribute
  bin_edges: np.ndarray
  noisy_counts: np.ndarray | None = None
  stddev: float = np.nan


@dataclasses.dataclass(frozen=True)
class CategoricalMeasurement:
  """Measurement from a categorical initializer."""

  categorical_attribute: domain.CategoricalAttribute
  noisy_counts: np.ndarray
  stddev: float


@dataclasses.dataclass(frozen=True)
class OpenSetMeasurement:
  """Measurement from an open-set categorical initializer."""

  categorical_attribute: domain.CategoricalAttribute
  noisy_counts: np.ndarray
  stddev: float


ColumnMeasurement = (
    NumericalMeasurement | CategoricalMeasurement | OpenSetMeasurement
)


def compute_grid_spec(
    attribute: domain.NumericalAttribute,
    num_partitions: int,
    max_grid_size: int = 10_000_000,
) -> tuple[float, float, int]:
  """Returns (lower, upper, grid_size) for the quantile candidate grid."""
  min_value = float(attribute.min_value)
  if attribute.dtype == 'int':
    m = primitives.jitter_factor(num_partitions)
    budget = max(2, max_grid_size // m)
    int_range = int(attribute.max_value - attribute.min_value + 1)
    step = max(1, math.ceil(int_range / budget))
    gs = math.ceil(int_range / step)
    return min_value, min_value + (gs - 1) * step, gs

  return min_value, float(attribute.exclusive_max_value), max_grid_size


@dataclasses.dataclass(frozen=True, kw_only=True)
class NumericalInitializerConfig(api.MechanismConfig):
  """Configuration for initializing numerical attributes."""

  num_partitions: int
  max_grid_size: int = 10_000_000
  epsilon_ratio: float = 2.0

  def __post_init__(self):
    if self.max_grid_size < 2:
      raise ValueError(f'max_grid_size must be >= 2, got {self.max_grid_size}.')
    if self.num_partitions >= self.max_grid_size:
      raise ValueError(f'{self.num_partitions=} >= {self.max_grid_size=}')

  def configure(
      self, attribute=None, *, zcdp_rho, delta=0, max_records_per_user=1
  ):
    assert attribute is not None
    api.validate_max_records_per_user(max_records_per_user)

    levels = int(np.log2(self.num_partitions))
    if 2**levels != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')

    rho_ratio = self.epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    return NumericalInitializer(
        config=self,
        attribute=attribute,
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
    return compute_grid_spec(
        self.attribute, self.config.num_partitions, self.config.max_grid_size
    )

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
  ) -> NumericalMeasurement:
    """Returns a NumericalMeasurement with the discretization transform."""
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
  ) -> NumericalMeasurement:
    """Returns a NumericalMeasurement from pre-aggregated histogram counts."""
    jitter_strategy = 'refine' if self.attribute.dtype == 'int' else 'symmetric'
    indices = primitives.quantiles_from_histogram(
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
) -> NumericalMeasurement:
  """Converts raw quantile edges into a NumericalMeasurement.

  Handles edge deduplication, degenerate-bin removal, and categorical
  attribute construction.

  Args:
    raw_edges: Quantile edge values (unsorted duplicates are fine).
    attribute: The ``NumericalAttribute`` defining the data domain.
    zcdp_rho: Total zCDP rho consumed by the quantile mechanism.
    estimated_total: If provided, a heuristic one-way measurement is included.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes.

  Returns:
    A ``NumericalMeasurement`` with bin edges and optionally noisy counts.
  """
  raw_edges = np.asarray(raw_edges, dtype=float)
  bin_edges, edge_counts = np.unique(raw_edges, return_counts=True)
  max_val = attribute.max_value
  if len(bin_edges) > 0 and bin_edges[-1] >= max_val:
    tail_count = edge_counts[-1]
    bin_edges = bin_edges[:-1]
    edge_counts = edge_counts[:-1]
    bin_weights = np.append(edge_counts, tail_count + 1)
  else:
    bin_weights = np.append(edge_counts, 1)
  cat_attr = vtx.categorical_attribute_from_edges(bin_edges, attribute)

  noisy_counts = None
  stddev = np.nan
  if estimated_total is not None:
    if not attribute.clip_to_range:
      bin_weights = np.r_[0, bin_weights]
    noisy_counts = estimated_total * bin_weights / bin_weights.sum()
    stddev = max_records_per_user / np.sqrt(zcdp_rho)

  return NumericalMeasurement(
      cat_attr, bin_edges, noisy_counts=noisy_counts, stddev=stddev
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CategoricalInitializerConfig(api.MechanismConfig):
  """Configuration for initializing categorical attributes."""

  def configure(
      self, attribute=None, *, zcdp_rho, delta=0, max_records_per_user=1
  ):
    assert attribute is not None
    api.validate_max_records_per_user(max_records_per_user)
    return CategoricalInitializer(
        config=self,
        attribute=attribute,
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
  ) -> CategoricalMeasurement:
    """Returns a CategoricalMeasurement with the noisy histogram."""
    encoded = vtx.discrete_encode(data, self.attribute)
    counts = np.bincount(encoded, minlength=self.attribute.size)
    return self.from_summary(rng, counts)

  def from_summary(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> CategoricalMeasurement:
    """Returns a CategoricalMeasurement from pre-aggregated counts."""
    noisy = primitives.add_gaussian_noise(
        rng, counts, self.sigma, self.max_records_per_user
    )
    stddev = self.max_records_per_user * self.sigma
    return CategoricalMeasurement(
        self.attribute, noisy_counts=np.asarray(noisy), stddev=stddev
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class OpenSetInitializerConfig(api.MechanismConfig):
  """Configuration for initializing open-set categorical attributes."""

  min_count: int = 1

  def configure(
      self, attribute=None, *, zcdp_rho, delta=0, max_records_per_user=1
  ):
    assert attribute is not None
    api.validate_max_records_per_user(max_records_per_user)
    return OpenSetInitializer(
        config=self,
        attribute=attribute,
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
  ) -> OpenSetMeasurement:
    """Returns a differentially private measurement of the given data."""
    data = np.asarray(data, dtype=str)
    unique_values, inverse = np.unique(data, return_inverse=True)
    counts = np.bincount(inverse)
    return self.from_summary(rng, unique_values, counts)

  def from_summary(
      self,
      rng: np.random.Generator,
      unique_values: np.ndarray,
      counts: np.ndarray,
  ) -> OpenSetMeasurement:
    """Returns an OpenSetMeasurement from pre-aggregated value counts."""
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

    default = self.attribute.default_value
    possible_values = [default] + selected_values.tolist()
    cat_attr = domain.CategoricalAttribute(possible_values)

    return OpenSetMeasurement(
        cat_attr, noisy_counts=estimated_counts, stddev=stddev
    )
