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

import dataclasses

import dp_accounting
from dpsynth import domain
from dpsynth import transformations
from dpsynth.discrete_mechanisms import accounting
from dpsynth.local_mode import primitives
import mbi
import numpy as np


@dataclasses.dataclass
class ColumnMeasurement:
  categorical_attribute: domain.CategoricalAttribute
  transform_fn: transformations.DataTransformation
  measurement: mbi.LinearMeasurement | None


@dataclasses.dataclass
class NumericalInitializer:
  """Mechanism that creates the data encoding transform for numerical data."""

  name: str
  num_partitions: int
  attribute: domain.NumericalAttribute
  rng: np.random.Generator

  def dp_event(self, zcdp_rho: float) -> dp_accounting.DpEvent:
    levels = int(np.log2(self.num_partitions))
    budget_weights = 4 ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    epsilons = [accounting.zcdp_exponential_eps(rho) for rho in rho_levels]

    return dp_accounting.ComposedDpEvent(
        [dp_accounting.ExponentialMechanismDpEvent(epsilon=e) for e in epsilons]
    )

  def __call__(self, zcdp_rho: float, data: np.ndarray) -> ColumnMeasurement:
    """Returns a differentially private measurement of the given data."""
    bucket_edges = primitives.quantiles(
        self.rng,
        data,
        self.attribute.min_value,
        self.attribute.max_value,
        self.num_partitions,
        zcdp_rho,
    )
    attr, discretize_fn = transformations.create_discretize_transformation(
        self.attribute, bucket_edges
    )
    transform_fn = transformations.discrete_encoder(attr) @ discretize_fn
    return ColumnMeasurement(attr, transform_fn, None)
