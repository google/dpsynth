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
  mechanism: primitives.DPQuantiles | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> NumericalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    if zcdp_rho <= 0:
      raise ValueError(f'zcdp_rho must be positive, got {zcdp_rho}.')
    mechanism = primitives.DPQuantiles(
        lower=self.attribute.min_value,
        upper=self.attribute.exclusive_max_value,
        num_partitions=self.num_partitions,
        # Infer from attribute, not data.dtype: NaN promotes int to float.
        integer_jitter=self.attribute.dtype == 'int',
    ).calibrate(zcdp_rho=zcdp_rho)
    return dataclasses.replace(self, mechanism=mechanism)

  @property
  def _zcdp_rho(self) -> float:
    """Total zCDP rho, derived as sum(eps_i^2 / 8) over composed events."""
    event = self.dp_event  # raises if not calibrated
    assert isinstance(event, dp_accounting.ComposedDpEvent)
    return sum(e.epsilon**2 / 8.0 for e in event.events)

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
    # Dedup: concentrated data can make quantiles return duplicate edges.
    raw_edges = _validate_mechanism(self.mechanism)(rng, data).quantiles
    raw_edges = np.asarray(raw_edges, dtype=float)
    if self.attribute.dtype == 'int':
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
    # At most one edge can equal max_value: DPQuantiles clamps outputs to
    # [lower, upper), so after floor + unique only the last edge can hit it.
    max_val = self.attribute.max_value
    if len(bin_edges) > 0 and bin_edges[-1] >= max_val:
      tail_count = edge_counts[-1]
      bin_edges = bin_edges[:-1]
      edge_counts = edge_counts[:-1]
      bin_weights = np.append(edge_counts, tail_count + 1)
    else:
      bin_weights = np.append(edge_counts, 1)
    cat_attr = vtx.categorical_attribute_from_edges(bin_edges, self.attribute)

    measurement = None
    if estimated_total is not None:
      rho = self._zcdp_rho
      uniform_counts = bin_weights * (estimated_total / self.num_partitions)
      stddev = 1.0 / np.sqrt(rho)
      measurement = mbi.LinearMeasurement(
          uniform_counts, (self.name,), stddev=stddev
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
    mechanism = _validate_mechanism(self.mechanism)
    encoded = vtx.discrete_encode(data, self.attribute)
    noisy_counts = mechanism(rng, encoded).counts
    measurement = mbi.LinearMeasurement(
        noisy_counts, (self.name,), stddev=mechanism.sigma
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
  """

  name: str
  attribute: domain.OpenSetCategoricalAttribute
  delta: float
  mechanism: primitives.DPPartitionSelection | None = dataclasses.field(
      default=None, repr=False
  )

  def calibrate(self, *, zcdp_rho: float) -> OpenSetCategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    mechanism = primitives.DPPartitionSelection(
        delta=self.delta,
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
    mechanism = _validate_mechanism(self.mechanism)
    # Map raw values to integer partition IDs for thresholding.
    unique_values, inverse = np.unique(data, return_inverse=True)
    result = mechanism(rng, inverse)
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
