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

"""Common utility functions for synthetic data mechanisms."""

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import functools
import itertools
from typing import Any, TypeAlias

from dpsynth import transformations
import mbi
import more_itertools
import numpy as np
import scipy
import scipy.special
import tqdm


@dataclasses.dataclass
class DiscreteMechanismResult:
  """Result of running a discrete mechanism.

  Attributes:
    model: The estimated graphical model (Markov random field).
    synthetic_data: A synthetic dataset whose marginals closely match the model.
    measurements: The noisy marginal measurements made by the mechanism.
    diagnostics: Optional mechanism-specific diagnostic information.
  """

  model: mbi.Model
  synthetic_data: mbi.Dataset
  measurements: list[mbi.LinearMeasurement] = dataclasses.field(
      default_factory=list
  )
  diagnostics: Any | None = None


def exponential_mechanism(
    quality_scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
    rng: np.random.Generator,
    monotonic: bool = False,
) -> int:
  """Returns an index chosen by the exponential mechanism."""
  coef = 1.0 if monotonic else 0.5
  scores = coef * epsilon / sensitivity * quality_scores
  probas = scipy.special.softmax(scores)
  return rng.choice(quality_scores.size, p=probas)


def measure_marginals_with_noise(
    rng: np.random.Generator,
    data: mbi.Projectable,
    marginal_queries: list[tuple[str, ...]],
    gdp_sigma: float,
    weights: np.ndarray | None = None,
) -> list[mbi.LinearMeasurement]:
  """Measures the given marginal queries with the Gaussian mechanism.

  Note that this function is a **single** instance of the Gaussian mechanism,
  even when multiple marginal queries are provided as input. The input gdp_sigma
  will be divided between the marginal queries evenly, unless weights are
  sigma is divided proportionally to the weights.

  Args:
    rng: A numpy random number generator.
    data: The sensitive dataset whose marginals are to be measured.
    marginal_queries: The list of marginal queries to measure, represented as a
      list of tuples of column names.
    gdp_sigma: The parameter of the Gaussian mechanism.
    weights: The weights to use for each marginal query. If None, use uniform
      weights.

  Returns:
    The list of LinearMeasurements.
  """
  if weights is None:
    weights = np.ones(len(marginal_queries))
  weights = np.array(weights) / np.linalg.norm(weights)
  if len(weights) != len(marginal_queries):
    raise ValueError(
        'The number of weights must be equal to the number of marginal queries.'
    )
  measurements = []
  for proj, wgt in zip(marginal_queries, weights):
    x = data.project(proj).datavector()
    y = x + rng.normal(loc=0, scale=gdp_sigma / wgt, size=x.size)
    measurements.append(mbi.LinearMeasurement(y, proj, gdp_sigma / wgt))
  return measurements


def _weighted_identity(weights, x: mbi.Factor):
  # We make this a global function so that it can be pickle-serialized.
  return x.datavector() * weights


def compressed_measurement(
    one_way_measurement: mbi.LinearMeasurement,
    size: int,
    transform_fn: transformations.DataTransformation[int, int],
) -> mbi.LinearMeasurement:
  """Returns a measurement defined over the compressed domain.

  Args:
    one_way_measurement: The measurement to compress.
    size: The size of the compressed domain.
    transform_fn: The domain compression transformation.

  Returns:
    A measurement defined over the compressed domain.
  """
  if len(one_way_measurement.clique) != 1:
    raise ValueError(
        'The measurement must be defined with respect to a one-way marginal,'
        f' got {one_way_measurement.clique}.'
    )
  y = one_way_measurement.noisy_measurement
  mapping = np.array([transform_fn(i) for i in range(y.size)])
  y2 = np.bincount(mapping, weights=y, minlength=size)
  coefs = np.sqrt(np.bincount(mapping, minlength=size))
  return mbi.LinearMeasurement(
      y2 / coefs,
      one_way_measurement.clique,
      one_way_measurement.stddev,
      query=functools.partial(_weighted_identity, 1.0 / coefs),
  )


def compression_transformation(
    measurement: mbi.LinearMeasurement,
) -> tuple[int, transformations.DataTransformation[int, int]]:
  """Returns a domain compression transformation for the given measurement."""
  mask = measurement.noisy_measurement < 3 * measurement.stddev
  size, transform_fn = transformations.create_rare_value_merging_transformation(
      mask
  )
  return size, transform_fn


def convert_to_total_measurement(
    measurements: list[mbi.LinearMeasurement],
) -> mbi.LinearMeasurement:
  """Converts a list of measurements to a measurement of total records."""
  # Note: This is a hack to get around the fact that
  # mbi.estimation.minimum_variance_unbiased_total does not work on compressed
  # measurements.
  total = mbi.estimation.minimum_variance_unbiased_total(measurements)
  return mbi.LinearMeasurement(
      noisy_measurement=total,
      clique=(),
      stddev=1.0,  # ideally we'd get this from minimum_variance_unbiased_total.
  )


def get_domain_compression_transformations(
    one_way_measurements: list[mbi.LinearMeasurement],
) -> tuple[
    mbi.Domain,
    list[mbi.LinearMeasurement],
    dict[str, transformations.DataTransformation[int, int]],
]:
  """Returns a new domain and transformations for compressing the domain.

  Args:
    one_way_measurements: List of one-way measurements over the original domain.

  Returns: A tuple of three elements:
    - The new (compressed) domain.
    - The list of measurements defined over the compressed domain.
    - A dictionary mapping each column of the original domain to a
      transformation that maps values in that column to values in the
      compressed domain.
  """
  column_transforms = {}
  sizes = {}
  new_measurements = []
  for measurement in one_way_measurements:
    col = measurement.clique[0]
    size, transform_fn = compression_transformation(measurement)
    sizes[col] = size
    column_transforms[col] = transform_fn
    new_measurements.append(
        compressed_measurement(measurement, size, transform_fn)
    )
  return mbi.Domain.fromdict(sizes), new_measurements, column_transforms


def downward_closure(
    marginal_queries: Iterable[mbi.Clique],
) -> Iterable[mbi.Clique]:
  """Returns the downward closure of the given marginal queries.

  Given a collection of sets, the downward closure is the set of all sets that
  are subsets of any of the given sets.

  Example Usage:
  >>> downward_closure([('a', 'b'), ('a', 'c')])
  [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c')]

  Args:
    marginal_queries: The marginal queries to compute the downward closure of.

  Returns:
    The downward closure of the given marginal queries, without the empty tuple.
  """
  ans = set()
  for proj in marginal_queries:
    ans.update(more_itertools.powerset(proj))
  return list(sorted(ans - {()}, key=len))


Workload: TypeAlias = Mapping[mbi.Clique, float]
Workload2: TypeAlias = Iterable[mbi.Clique]


def compiled_workload(
    domain: mbi.Domain,
    workload: Workload | Workload2 | None,
    max_marginal_size: float = float('inf'),
) -> Workload:
  """Compiles an input workload into a set of candidate measurements for AIM.

  Args:
    domain: The domain of the dataset.
    workload: A dictionary mapping marginal queries to weights representing the
      importance of each marginal query.
    max_marginal_size: The maximum size of a marginal query to consider.

  Returns:
    A dictionary mapping marginal queries in the downward closure of the
    workload to weights representing the importance of each marginal query.
  """

  if workload is None:
    workload = list(itertools.combinations(domain.attributes, 3))

  if not isinstance(workload, Mapping):
    workload = {cl: 1.0 for cl in workload}

  def score(cl):
    return sum(
        workload[workload_cl] * len(set(cl) & set(workload_cl))
        for workload_cl in workload
    )

  return {
      cl: score(cl)
      for cl in downward_closure(workload.keys())
      if domain.size(cl) <= max_marginal_size
  }


def compute_independence_errors(
    data: mbi.Projectable,
    model: mbi.MarkovRandomField,
    cliques: Sequence[mbi.Clique],
) -> dict[mbi.Clique, float]:
  """Computes L1 errors between actual marginals and the independence model."""
  total = float(model.total)

  # Pure numpy to avoid XLA recompilation: each distinct clique shape triggers
  # a separate compilation, which dominates runtime for large candidate sets.
  oneway = {
      a: np.asarray(model.project((a,)).datavector()) / total
      for a in data.domain
  }

  errors = {}
  for cl in tqdm.tqdm(cliques, desc='Computing independence errors'):
    estimate = functools.reduce(np.multiply.outer, (oneway[a] for a in cl))
    actual = np.asarray(data.project(cl).datavector(flatten=False))
    error = np.abs(total * estimate - actual).sum()
    errors[cl] = error
  return errors
