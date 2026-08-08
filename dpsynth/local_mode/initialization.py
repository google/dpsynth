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


@dataclasses.dataclass(frozen=True)
class NumericalInitializerConfig(api.MechanismConfig):
  """Configuration for a numerical data encoding mechanism."""

  name: str
  num_partitions: int
  attribute: domain.NumericalAttribute
  max_grid_size: int = 10_000_000
  epsilon_ratio: float = 2.0

  def __post_init__(self):
    if self.max_grid_size < 2:
      raise ValueError(f'max_grid_size must be >= 2, got {self.max_grid_size}.')

  @property
  def grid_spec(self) -> tuple[float, float, int]:
    """Returns (lower, upper, grid_size) for the quantile candidate grid."""
    attr = self.attribute
    if attr.dtype == 'int':
      # Reserve budget for the m-fold refinement so the refined grid fits.
      m = _quantiles.jitter_factor(self.num_partitions)
      budget = max(2, self.max_grid_size // m)
      int_range = int(attr.max_value - attr.min_value + 1)
      step = max(1, math.ceil(int_range / budget))
      gs = math.ceil(int_range / step)
      return (attr.min_value, attr.min_value + (gs - 1) * step, gs)
    return (attr.min_value, attr.exclusive_max_value, self.max_grid_size)

  @property
  def grid_size(self) -> int:
    """Grid size used for histogram construction."""
    return self.grid_spec[2]

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):
    """Returns a runnable mechanism calibrated to the given zCDP budget."""
    lower, upper, _ = self.grid_spec
    mechanism = primitives.DPQuantilesConfig(
        num_partitions=self.num_partitions,
        lower=lower,
        upper=upper,
        jitter_strategy=(
            'refine' if self.attribute.dtype == 'int' else 'symmetric'
        ),
        epsilon_ratio=self.epsilon_ratio,
    ).configure(
        zcdp_rho=zcdp_rho,
        max_records_per_user=max_records_per_user,
    )
    return NumericalInitializer(config=self, mechanism=mechanism)


@dataclasses.dataclass(frozen=True)
class NumericalInitializer(api.CalibratedMechanism):
  """Mechanism that creates the data encoding transform for numerical data."""

  config: NumericalInitializerConfig
  mechanism: primitives.DPQuantiles

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    return self.mechanism.dp_event

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
    lower, upper, gs = self.config.grid_spec
    delta = (upper - lower) / (gs - 1)
    attr = self.config.attribute
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
    raw_edges = self.mechanism(rng, counts)
    return edges_to_column_measurement(
        raw_edges=raw_edges,
        attribute=self.config.attribute,
        name=self.config.name,
        zcdp_rho=self.mechanism.zcdp_rho,
        estimated_total=estimated_total,
        max_records_per_user=self.mechanism.max_records_per_user,
    )


def edges_to_column_measurement(
    raw_edges,
    attribute,
    name,
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
    name: Attribute name used as the clique key in any measurement.
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
        (name,),
        stddev=stddev,
        query=mbi.DatavectorQuery(use_for_total_estimation=False),
    )

  return ColumnMeasurement(cat_attr, bin_edges, measurement=measurement)


@dataclasses.dataclass(frozen=True)
class CategoricalInitializerConfig(api.MechanismConfig):
  """Configuration for measuring a noisy histogram for categorical data."""

  name: str
  attribute: domain.CategoricalAttribute

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):  # pyrefly: ignore[bad-override]
    """Returns a runnable mechanism calibrated to the given zCDP budget."""
    mechanism = primitives.DPGaussianHistogramConfig(
        domain_size=self.attribute.size,
    ).configure(zcdp_rho=zcdp_rho, max_records_per_user=max_records_per_user)
    return CategoricalInitializer(config=self, mechanism=mechanism)


@dataclasses.dataclass(frozen=True)
class CategoricalInitializer(api.CalibratedMechanism):
  """Mechanism that measures a noisy histogram for categorical data."""

  config: CategoricalInitializerConfig
  mechanism: primitives.DPGaussianHistogram

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return self.mechanism.dp_event

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the noisy histogram."""
    encoded = vtx.discrete_encode(data, self.config.attribute)
    counts = np.bincount(encoded, minlength=self.config.attribute.size)
    return self.from_summary(rng, counts)

  def from_summary(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated counts."""
    result = self.mechanism(rng, counts)
    measurement = mbi.LinearMeasurement(
        result.counts,
        (self.config.name,),
        stddev=self.mechanism.max_records_per_user * self.mechanism.sigma,
    )
    return ColumnMeasurement(self.config.attribute, measurement=measurement)


@dataclasses.dataclass(frozen=True)
class OpenSetCategoricalInitializerConfig(api.MechanismConfig):
  """Configuration for discovering an open-set categorical domain."""

  name: str
  attribute: domain.OpenSetCategoricalAttribute
  delta: float
  min_count: int = 1

  def configure(self, *, zcdp_rho, delta=0.0, max_records_per_user=1):  # pyrefly: ignore[bad-override]
    """Returns a runnable mechanism calibrated to the given zCDP budget."""
    mechanism = primitives.DPPartitionSelectionConfig(
        delta=self.delta,
        min_count=self.min_count,
    ).configure(zcdp_rho=zcdp_rho, max_records_per_user=max_records_per_user)
    return OpenSetCategoricalInitializer(config=self, mechanism=mechanism)


@dataclasses.dataclass(frozen=True)
class OpenSetCategoricalInitializer(api.CalibratedMechanism):
  """Mechanism that discovers and measures an open-set categorical domain."""

  config: OpenSetCategoricalInitializerConfig
  mechanism: primitives.DPPartitionSelection

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    return self.mechanism.dp_event

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
    result = self.mechanism.from_summary(rng, counts)
    selected_values = [
        str(v) for v in unique_values[result.selected_partitions]
    ]

    # Build the discovered domain: default first, then selected values.
    possible_values = [self.config.attribute.default_value] + selected_values
    cat_attr = domain.CategoricalAttribute(
        possible_values=possible_values,  # pyrefly: ignore[unexpected-keyword]
        out_of_domain_index=0,  # pyrefly: ignore[unexpected-keyword]
    )

    # The measurement covers only the discovered partitions (indices 1:),
    # not the unmeasured default at index 0.
    measurement = mbi.LinearMeasurement(
        result.estimated_counts,  # pyrefly: ignore[bad-argument-type]
        (self.config.name,),
        stddev=self.mechanism.max_records_per_user * self.mechanism.sigma,
        query=mbi.SlicedQuery(start=1),
    )
    return ColumnMeasurement(cat_attr, measurement=measurement)
