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

"""Implementation of the Maximum Spanning Tree mechanism."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import itertools
import typing

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
import mbi
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import DisjointSet  # pylint: disable=g-importing-member


def dp_maximum_spanning_tree(
    rng: np.random.Generator,
    weights: dict[tuple[str, str], float],
    zcdp_rho: float | None = None,
    exponential_mechanism_epsilon: float | None = None,
    initial_marginal_queries: Sequence[tuple[str, str]] = (),
) -> list[tuple[str, str]]:
  """Computes an approximate maximum spanning tree with differential privacy.

  This is a differentially-private version of Kruskal's algorithm, where the
  best edge in each round is selected privately by the exponential mechanism.

  The differential privacy guarantees:
    1. zcdp_rho-zCDP if zcdp_rho is given.
    2. otherwise, it has the same privacy guarantees as the len(weights)-1
     Exponential Mechanism with parameter exponential_mechanism_epsilon.

  It is assumed that the weights are obtained from sensitivity 1 functions of
  the data (i.e., L1 norm between true and estimated marginal).

  Args:
    rng: A numpy random number generator.
    weights: A dictionary mapping pairs of attributes to the sensitivity 1
      measure of correlation between them.
    zcdp_rho: the zCDP budget to use for this mechanism.
    exponential_mechanism_epsilon: The epsilon parameter for the exponential
      mechanism. If None, the value is computed from zcdp_rho.
    initial_marginal_queries: The list of initial attribute pairs to include in
      the tree.

  Returns:
    A list of attribute pairs that constitute an approximate maximum spanning
    tree.
  """
  if (zcdp_rho is None) == (exponential_mechanism_epsilon is None):
    raise ValueError(
        'zcdp_rho or exponential_mechanism_epsilon must be set, but not both.'
    )
  tree = nx.Graph()
  attributes = set()
  for key in weights.keys():
    for attribute in key:
      attributes.add(attribute)
  tree.add_nodes_from(attributes)
  ds = DisjointSet(attributes)

  for e in initial_marginal_queries:
    tree.add_edge(*e)
    ds.merge(*e)

  candidates = list(weights.keys())
  r = len(list(nx.connected_components(tree)))
  if exponential_mechanism_epsilon is None:
    exponential_mechanism_epsilon = np.sqrt(8 * zcdp_rho / (r - 1))  # pyrefly: ignore[unsupported-operation]
  for _ in range(r - 1):
    candidates = [e for e in candidates if not ds.connected(*e)]
    wgts = np.array([weights[e] for e in candidates])
    idx = common.exponential_mechanism(
        wgts, exponential_mechanism_epsilon, sensitivity=1.0, rng=rng
    )
    e = candidates[idx]
    tree.add_edge(*e)
    ds.merge(*e)

  return list(tree.edges)


def _select_two_way_marginal_queries(
    rng: np.random.Generator,
    data: mbi.Projectable,
    zcdp_rho: float,
    one_way_measurements: list[mbi.LinearMeasurement],
    initial_marginal_queries: Sequence[tuple[str, ...]] = (),
    maximum_marginal_size: int = 10_000_000,
) -> list[tuple[str, ...]]:
  """Selects a set of two-way marginal queries with DP to form a spanning tree.

  This mechanism satisfies rho-zCDP.

  Args:
    rng: A numpy random number generator.
    data: The sensitive dataset to use to determine the quality scores of each
      two-way marginal query.
    zcdp_rho: The zCDP privacy parameter.
    one_way_measurements: The initial one-way measurements already made.
    initial_marginal_queries: The list of cliques to start with.
    maximum_marginal_size: The maximum size of a marginal query.

  Returns:
    A list of two-way marginal queries over highly correlated attributes.
  """

  independent_model = mbi.estimation.MirrorDescent().estimate(
      data.domain, one_way_measurements, iters=2500
  )
  independent_model = typing.cast(mbi.MarkovRandomField, independent_model)

  # Construct a complete graph where nodes=attributes and weight of edge
  # (a, b) is a sensitivity 1 measure of correlation between a and b.
  candidates = [
      cl
      for cl in itertools.combinations(data.domain.attributes, 2)
      if data.domain.size(cl) <= maximum_marginal_size
  ]
  logging.info('[MST]: Computing Quality Scores')
  weights = common.compute_independence_errors(
      data, independent_model, candidates
  )

  return dp_maximum_spanning_tree(  # pyrefly: ignore[bad-return]
      rng,
      weights,  # pyrefly: ignore[bad-argument-type]
      zcdp_rho=zcdp_rho,
      initial_marginal_queries=initial_marginal_queries,  # pyrefly: ignore[bad-argument-type]
  )


@dataclasses.dataclass
class MSTMechanism(api.DPMechanism):
  """Configuration for the maximum spanning tree mechanism.

  Details are described in the paper:
  [Winning the NIST Contest: A scalable and general approach to differentially
  private synthetic data](https://arxiv.org/abs/2108.04978)

  Attributes:
    pgm_iters: The number of iterations for the mirror descent algorithm.
    maximum_marginal_size: The maximum size of a marginal query.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    one_way_budget_fraction: The fraction of the total budget to use for one-way
      marginal queries.
    compress_columns: Controls domain compression. True compresses all columns,
      False disables compression, or a sequence of column names to compress
      selectively.
    select_budget_fraction: The fraction of the total budget to use for
      selecting two-way marginal queries.
  """

  pgm_iters: int = 5000
  maximum_marginal_size: int = 10_000_000
  marginal_oracle: mbi.MarginalOracle | None = None
  one_way_budget_fraction: float = 1 / 3
  compress_columns: bool | Sequence[str] = False
  select_budget_fraction: float = 1 / 3
  zcdp_rho: float | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns all pairwise marginals within the size limit."""
    return common.supporting_cliques(
        domain,
        itertools.combinations(domain.attributes, 2),
        self.maximum_marginal_size,
    )

  def configure(self, *, zcdp_rho: float, delta: float = 0.0) -> MSTMechanism:
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(self, zcdp_rho=zcdp_rho)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the MST mechanism."""
    if self.zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    # exponential mechanisms and (d-1) Gaussian mechanisms.
    return dp_accounting.ZCDpEvent(self.zcdp_rho)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: list[mbi.LinearMeasurement] | None = None,
      constraints: tuple[mbi.Constraint, ...] = (),
  ) -> common.DiscreteMechanismResult:
    """Runs the MST mechanism on the given data.

    Args:
      rng: A numpy random number generator.
      data: The sensitive dataset. Must be an mbi.Dataset for domain
        compression; mbi.CliqueVector is supported but compression will be
        skipped.
      initial_measurements: Optional pre-existing one-way measurements.
      constraints: Structural constraints for the estimation.

    Returns:
      A DiscreteMechanismResult containing the estimated data distribution.

    Raises:
      ValueError: If calibrate() has not been called.
    """
    if self.zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    logging.info('[MST]: Starting MST mechanism.')
    phase_times = {}
    budget_remaining = self.zcdp_rho

    with common.timed(phase_times, 'measurement'):
      if initial_measurements is None:
        budget_remaining -= self.one_way_budget_fraction * self.zcdp_rho
        one_way_rho = self.zcdp_rho * self.one_way_budget_fraction
        one_way_sigma = accounting.zcdp_gaussian_sigma(one_way_rho)
        one_way_measurements = common.measure_marginals_with_noise(
            rng,
            data,
            marginal_queries=[(a,) for a in data.domain],
            gdp_sigma=one_way_sigma,
        )
      else:
        one_way_measurements = initial_measurements

    mappings = common.compression_mappings(
        one_way_measurements, self.compress_columns, constraints
    )
    if mappings:
      data = data.compress(mappings)
      one_way_measurements = [
          m.compress(mappings, data.domain) for m in one_way_measurements
      ]

    exponential_rho = self.select_budget_fraction * self.zcdp_rho
    budget_remaining -= exponential_rho
    # Select and measure 2-way marginals using rho/3 budget for each step.
    with common.timed(phase_times, 'selection'):
      two_way_marginal_queries = _select_two_way_marginal_queries(
          rng,
          data,
          exponential_rho,
          one_way_measurements,
          maximum_marginal_size=self.maximum_marginal_size,
      )
      logging.info('[MST]: Selected two-way marginal queries.')
    with common.timed(phase_times, 'measurement'):
      gaussian_rho = budget_remaining
      sigma = accounting.zcdp_gaussian_sigma(gaussian_rho)
      two_way_measurements = common.measure_marginals_with_noise(
          rng, data, two_way_marginal_queries, sigma
      )
      logging.info('[MST]: Measured two-way marginals.')
    all_measurements = one_way_measurements + two_way_measurements
    # Fit a distribution to the noisy measurements using Private-PGM.
    model_size = mbi.junction_tree.hypothetical_model_size(
        data.domain, [m.clique for m in all_measurements]
    )
    logging.info('[MST]: Model size: %d MB', model_size)
    logging.info(
        '[MST]:\n%s',
        mbi.summarize(data.domain, [m.clique for m in all_measurements]),
    )
    with common.timed(phase_times, 'estimation'):
      estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
      model = estimator.estimate(
          data.domain,
          all_measurements,
          iters=self.pgm_iters,
          callback_fn=mbi.callbacks.default(all_measurements, data.domain),
          constraints=constraints,
      )
      logging.info('[MST]: Fit distribution to the noisy measurements.')
    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times
    synthetic_data = model.synthetic_data()
    if mappings:
      synthetic_data = synthetic_data.decompress(mappings)
    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=synthetic_data,
        measurements=all_measurements,
        diagnostics=diagnostics,
        mappings=mappings,
    )
