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

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import typing

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import base
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
    sensitivity: float = 1.0,
) -> list[tuple[str, str]]:
  """Computes an approximate maximum spanning tree with differential privacy.

  This is a differentially-private version of Kruskal's algorithm, where the
  best edge in each round is selected privately by the exponential mechanism.

  The differential privacy guarantees:
    1. zcdp_rho-zCDP if zcdp_rho is given.
    2. otherwise, it has the same privacy guarantees as the len(weights)-1
     Exponential Mechanism with parameter exponential_mechanism_epsilon.

  It is assumed that the weights are obtained from sensitivity ``sensitivity``
  functions of the data (i.e., L1 norm between true and estimated marginal,
  scaled by the maximum number of records a single user contributes).

  Args:
    rng: A numpy random number generator.
    weights: A dictionary mapping pairs of attributes to the sensitivity 1
      measure of correlation between them.
    zcdp_rho: the zCDP budget to use for this mechanism.
    exponential_mechanism_epsilon: The epsilon parameter for the exponential
      mechanism. If None, the value is computed from zcdp_rho.
    initial_marginal_queries: The list of initial attribute pairs to include in
      the tree.
    sensitivity: The sensitivity of the quality scores in ``weights``.

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
    exponential_mechanism_epsilon = np.sqrt(8 * zcdp_rho / max(r - 1, 1))  # pyrefly: ignore[unsupported-operation]
  for _ in range(r - 1):
    candidates = [e for e in candidates if not ds.connected(*e)]
    wgts = np.array([weights[e] for e in candidates])
    idx = common.exponential_mechanism(
        wgts, exponential_mechanism_epsilon, sensitivity=sensitivity, rng=rng
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
    max_records_per_user: int = 1,
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
    max_records_per_user: The assumed maximum number of records a single user
      contributes; scales the sensitivity of the correlation quality scores.

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
      sensitivity=max_records_per_user,
  )


@dataclasses.dataclass
class MSTMechanism(api.DPMechanism):
  """Configuration for the maximum spanning tree mechanism."""

  marginal_oracle: mbi.MarginalOracle | None = None
  pgm_iters: int = 5000
  max_records_per_user: int = 1

  def __post_init__(self):
    api.validate_max_records_per_user(self.max_records_per_user)

  select_budget_fraction: float = 1 / 2
  maximum_marginal_size: int = 10_000_000
  zcdp_rho: float | None = None
  _select_rho: float | None = dataclasses.field(default=None, repr=False)
  _measurement_rho: float | None = dataclasses.field(default=None, repr=False)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return common.supporting_cliques(
        domain,
        itertools.combinations(domain.attributes, 2),
        self.maximum_marginal_size,
    )

  def configure(self, *, zcdp_rho: float, **kwargs) -> MSTMechanism:
    select_rho = zcdp_rho * self.select_budget_fraction
    return dataclasses.replace(
        self,
        zcdp_rho=zcdp_rho,
        _select_rho=select_rho,
        _measurement_rho=zcdp_rho - select_rho,
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    if self.zcdp_rho is None:
      raise ValueError('Must call configure() before using the mechanism.')
    assert self._select_rho is not None
    assert self._measurement_rho is not None
    return dp_accounting.ZCDpEvent(self.zcdp_rho)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    if self.zcdp_rho is None:
      raise ValueError('Must call configure() before using the mechanism.')
    assert self._select_rho is not None
    assert self._measurement_rho is not None
    phase_times = {}

    with common.timed(phase_times, 'selection'):
      selected = _select_two_way_marginal_queries(
          rng,
          data,  # pyrefly: ignore[bad-argument-type]
          self._select_rho,
          list(initial_measurements or []),
          maximum_marginal_size=self.maximum_marginal_size,
          max_records_per_user=self.max_records_per_user,
      )

    all_cliques = [m.clique for m in initial_measurements or []] + list(
        selected
    )
    logging.info(
        '[%s]:\n%s',
        type(self).__name__,
        mbi.summarize(data.domain, all_cliques),
    )

    estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
    futures = None
    try:
      futures = estimator.precompile(
          data.domain,
          list(initial_measurements or []),
          extra_cliques=list(selected),
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Precompile failed (non-fatal): %s', e)

    measurements = list(initial_measurements or [])
    if selected:
      with common.timed(phase_times, 'measurement'):
        sigma = accounting.zcdp_gaussian_sigma(self._measurement_rho)
        measurements.extend(
            common.measure_marginals_with_noise(
                rng,
                data,  # pyrefly: ignore[bad-argument-type]
                selected,
                sigma,
                max_records_per_user=self.max_records_per_user,
            )
        )

    with common.timed(phase_times, 'estimation'):
      if futures is not None:
        try:
          futures.result()
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning('Precompile wait failed (non-fatal): %s', e)
      model = estimator.estimate(
          data.domain,
          measurements,
          iters=self.pgm_iters,
          callback_fn=mbi.callbacks.default(measurements, data.domain),
          constraints=constraints,
      )

      model = typing.cast(mbi.MarkovRandomField, model)

    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times

    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=model.synthetic_data(),
        measurements=measurements,
        diagnostics=diagnostics,
    )
