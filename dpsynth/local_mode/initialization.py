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
from typing import TypeVar

import dp_accounting
from dpsynth import domain
from dpsynth.local_mode import primitives
from dpsynth.local_mode import vectorized_transformations as vtx
import mbi
import numpy as np

_M = TypeVar('_M')


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


def _validate_mechanism(mechanism: _M | None) -> _M:
  """Validates that the mechanism has been calibrated and returns it."""
  if mechanism is None:
    raise ValueError('Must call calibrate() before using the mechanism.')
  return mechanism


@dataclasses.dataclass
class NumericalInitializer(primitives.DPMechanism):
  """Mechanism that creates the data encoding transform for numerical data.

  Internally delegates to a ``DPQuantiles`` mechanism for privacy accounting
  and quantile computation.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    num_partitions: Number of quantile partitions (must be a power of 2).
    attribute: The NumericalAttribute defining the data domain.
  """

  name: str
  num_partitions: int
  attribute: domain.NumericalAttribute
  grid_size: int = 10_000_000
  mechanism: primitives.DPQuantiles | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(
      self, *, zcdp_rho: float, epsilon_ratio: float = 2.0
  ) -> NumericalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPQuantiles(
        num_partitions=self.num_partitions,
        lower=self.attribute.min_value,
        upper=self.attribute.exclusive_max_value,
        grid_size=self.grid_size,
    ).calibrate(zcdp_rho=zcdp_rho, epsilon_ratio=epsilon_ratio)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    return _validate_mechanism(self.mechanism).dp_event

  def __call__(
      self,
      rng: np.random.Generator,
      data: np.ndarray,
      *,
      estimated_total: float | None = None,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the discretization transform.

    Args:
      rng: A numpy random number generator.
      data: 1D array of numerical data.
      estimated_total: If provided, a heuristic one-way measurement is included
        assuming a uniform distribution over the original bins.

    Returns:
      A ColumnMeasurement with bin edges and optionally a heuristic measurement.
    """
    _validate_mechanism(self.mechanism)
    lower = self.attribute.min_value
    upper = self.attribute.exclusive_max_value
    finite_data = data[np.isfinite(data.astype(float))]
    clamped = np.clip(finite_data, lower, upper)
    delta = (upper - lower) / (self.grid_size - 1)
    indices = np.round((clamped - lower) / delta).astype(np.int64)
    counts = np.bincount(indices, minlength=self.grid_size)
    return self.from_summary(rng, counts, estimated_total=estimated_total)

  def from_summary(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
      *,
      estimated_total: float | None = None,
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement from pre-aggregated histogram counts."""
    mechanism = _validate_mechanism(self.mechanism)
    raw_edges = mechanism(rng, counts)
    return edges_to_column_measurement(
        raw_edges=raw_edges,
        attribute=self.attribute,
        name=self.name,
        zcdp_rho=mechanism.zcdp_rho,
        estimated_total=estimated_total,
    )


def edges_to_column_measurement(
    raw_edges,
    attribute,
    name,
    zcdp_rho,
    estimated_total=None,
):
  """Converts raw quantile edges into a ColumnMeasurement.

  Handles integer snapping, edge deduplication, degenerate-bin removal, and
  categorical attribute construction.  Shared between the data-based
  ``NumericalInitializer`` and the histogram-based
  ``HistogramNumericalInitializer``.

  Args:
    raw_edges: Quantile edge values (unsorted duplicates are fine).
    attribute: The ``NumericalAttribute`` defining the data domain.
    name: Attribute name used as the clique key in any measurement.
    zcdp_rho: Total zCDP rho consumed by the quantile mechanism.
    estimated_total: If provided, a heuristic one-way measurement is included.

  Returns:
    A ``ColumnMeasurement`` with bin edges and optionally a measurement.
  """
  raw_edges = np.asarray(raw_edges, dtype=float)
  if attribute.dtype == 'int':
    # Snap edges to the integer lattice.  Bins are right-closed (left,
    # right] and discretize uses searchsorted with side='left', so
    # floor preserves the partition: edge 4.7 → floor 4 gives the
    # same integer split {≤4} | {≥5} via (…, 4] | (4, …].
    raw_edges = np.floor(raw_edges)
  bin_edges, edge_counts = np.unique(raw_edges, return_counts=True)
  # For integer data with upper=max_value+1, edges can land at max_value
  # after floor.  Remove such edges and absorb their count into the last
  # bin's weight so categorical_attribute_from_edges doesn't create a
  # degenerate (max_value, max_value] tail bin.
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
    normalized = bin_weights / bin_weights.sum()
    stddev = 1.0 / (np.sqrt(zcdp_rho) * estimated_total)
    measurement = mbi.LinearMeasurement(
        normalized,
        (name,),
        stddev=stddev,
        query=lambda f: f.normalize(1.0).datavector(),
    )

  return ColumnMeasurement(cat_attr, bin_edges, measurement=measurement)


@dataclasses.dataclass
class CategoricalInitializer(primitives.DPMechanism):
  """Mechanism that measures a noisy histogram for categorical data.

  Internally delegates to a ``DPGaussianHistogram`` mechanism for privacy
  accounting and noise addition.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The CategoricalAttribute defining the closed domain.
  """

  name: str
  attribute: domain.CategoricalAttribute
  mechanism: primitives.DPGaussianHistogram | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> CategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPGaussianHistogram(
        domain_size=self.attribute.size,
    ).calibrate(zcdp_rho=zcdp_rho)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return _validate_mechanism(self.mechanism).dp_event

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
    mechanism = _validate_mechanism(self.mechanism)
    result = mechanism(rng, counts)
    measurement = mbi.LinearMeasurement(
        result.counts, (self.name,), stddev=mechanism.sigma
    )
    return ColumnMeasurement(self.attribute, measurement=measurement)


@dataclasses.dataclass
class OpenSetCategoricalInitializer(primitives.DPMechanism):
  """Mechanism that discovers and measures an open-set categorical domain.

  Uses Gaussian Thresholding (Algorithm 2 from the DP-SIPS paper) to privately
  select significant partitions from the data and simultaneously obtain noisy
  counts for each discovered partition. The discovered partitions, together
  with the attribute's default_value (used as a catch-all for undiscovered
  values), form a CategoricalAttribute used for downstream synthesis.

  Attributes:
    name: Attribute name used as the clique key in the measurement.
    attribute: The OpenSetCategoricalAttribute specifying the default value.
    delta: Failure probability for the partition selection threshold.
    min_count: Minimum true count for a partition to be discovered.
  """

  name: str
  attribute: domain.OpenSetCategoricalAttribute
  delta: float
  min_count: int = 1
  mechanism: primitives.DPPartitionSelection | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> OpenSetCategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPPartitionSelection(
        delta=self.delta,
        min_count=self.min_count,
    ).calibrate(zcdp_rho=zcdp_rho)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    return _validate_mechanism(self.mechanism).dp_event

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
    mechanism = _validate_mechanism(self.mechanism)
    result = mechanism.from_summary(rng, counts)
    selected_values = list(unique_values[result.selected_partitions])

    # Build the discovered domain: default first, then selected values.
    possible_values = [self.attribute.default_value] + selected_values
    cat_attr = domain.CategoricalAttribute(
        possible_values=possible_values,
        out_of_domain_index=0,
    )

    # The measurement covers only the discovered partitions (indices 1:),
    # not the unmeasured default at index 0.
    measurement = mbi.LinearMeasurement(
        result.estimated_counts,
        (self.name,),
        stddev=mechanism.sigma,
        query=lambda x: x.datavector()[1:],
    )
    return ColumnMeasurement(cat_attr, measurement=measurement)
