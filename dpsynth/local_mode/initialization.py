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
import functools

import dp_accounting
from dpsynth import domain
from dpsynth import transformations
from dpsynth.local_mode import primitives
import mbi
import numpy as np


@dataclasses.dataclass
class ColumnMeasurement:
  categorical_attribute: domain.CategoricalAttribute
  transform_fn: transformations.DataTransformation
  measurement: mbi.LinearMeasurement | None


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
  _zcdp_rho: float | None = dataclasses.field(default=None, repr=False)

  @functools.cached_property
  def _mechanism(self) -> primitives.DPQuantiles:
    if self._zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    return primitives.DPQuantiles(
        lower=self.attribute.min_value,
        upper=self.attribute.max_value,
        num_partitions=self.num_partitions,
    ).calibrate(zcdp_rho=self._zcdp_rho)

  def calibrate(self, *, zcdp_rho: float) -> NumericalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(self, _zcdp_rho=zcdp_rho)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    return self._mechanism.dp_event

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the discretization transform."""
    bucket_edges = self._mechanism(rng, data)
    attr, discretize_fn = transformations.create_discretize_transformation(
        self.attribute, bucket_edges
    )
    transform_fn = transformations.discrete_encoder(attr) @ discretize_fn
    return ColumnMeasurement(attr, transform_fn, None)


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
  _zcdp_rho: float | None = dataclasses.field(default=None, repr=False)

  @functools.cached_property
  def _mechanism(self) -> primitives.DPGaussianHistogram:
    if self._zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    return primitives.DPGaussianHistogram(
        domain_size=self.attribute.size,
    ).calibrate(zcdp_rho=self._zcdp_rho)

  def calibrate(self, *, zcdp_rho: float) -> CategoricalInitializer:
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(self, _zcdp_rho=zcdp_rho)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    return self._mechanism.dp_event

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> ColumnMeasurement:
    """Returns a ColumnMeasurement with the noisy histogram."""
    transform_fn = transformations.discrete_encoder(self.attribute)
    encoded = np.array([transform_fn(v) for v in data])
    noisy_counts = self._mechanism(rng, encoded)
    measurement = mbi.LinearMeasurement(
        noisy_counts, (self.name,), stddev=self._mechanism.sigma
    )
    return ColumnMeasurement(self.attribute, transform_fn, measurement)
