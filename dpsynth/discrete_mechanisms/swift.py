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

"""SWIFT: Scalable Workload-Informed Factor Tree.

SWIFT is designed to work best in large-scale scenarios where AIM does not scale
well, and MST provides sub-optimal utility. It works by building the largest
clique tree it can subject to configurable size constraints, based on the
provided workload and data distribution. Then it allocates the budget to a
a subset of marginal queries supported by the clique tree and answers them all
at once using the Gaussian mechanism. It then estimates a MarkovRandomField
that maximizes the likelihood of the noisy marginals measured.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import functools
import math

from absl import logging
import dp_accounting
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import clique_tree
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import swift_utils
from dpsynth.local_mode import primitives
import jax
import mbi
import networkx as nx
import numpy as np
import tqdm


@dataclasses.dataclass
class SWIFTMechanismResult:
  """Result of running the SWIFT mechanism."""

  model: mbi.MarkovRandomField


@dataclasses.dataclass
class SWIFTMechanism(primitives.DPMechanism):
  """Configuration for the SWIFT mechanism.

  Attributes:
    workload: The set of marginals to consider for the mechanism. Can be a
      mapping from cliques to their weights or just an iterable of cliques.
    max_clique_size: The maximum size (domain product) allowed for any clique in
      the junction tree.
    max_marginal_size: The maximum size (domain product) of any marginal
      considered in the workload.
    marginal_oracle: An optional oracle for marginal computations.
    pgm_iters: Number of iterations for the PGM estimation.
    select_budget_frac: Fraction of the total budget used for selecting which
      marginals to measure.
    one_way_budget_frac: Fraction of the total budget used for measuring one-way
      marginals initially.
    gdp_sigma: The GDP sigma of the end-to-end mechanism. Privacy budget is
      split across measurement steps internally.
  """

  workload: Mapping[mbi.Clique, float] | Iterable[mbi.Clique] | None = None
  max_clique_size: float = 1e9
  max_marginal_size: float = 1e7
  marginal_oracle: mbi.MarginalOracle | None = None
  pgm_iters: int = 25_000
  select_budget_frac: float = 0.1
  one_way_budget_frac: float = 0.1
  gdp_sigma: float | None = None

  def calibrate(self, *, zcdp_rho: float) -> SWIFTMechanism:
    """Returns a copy calibrated to the given zCDP budget."""
    return dataclasses.replace(
        self, gdp_sigma=accounting.zcdp_gaussian_sigma(zcdp_rho)
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the SWIFT mechanism."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.gdp_sigma)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Projectable,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      initial_potentials: mbi.CliqueVector | None = None,
  ) -> SWIFTMechanismResult:
    """Runs the SWIFT mechanism on the given data.

    Args:
      rng: A numpy random number generator.
      data: The sensitive dataset.
      initial_measurements: Optional pre-existing one-way measurements.
      initial_potentials: Optional initial potentials for constrained
        estimation.

    Returns:
      A SWIFTMechanismResult containing the estimated data distribution.

    Raises:
      ValueError: If calibrate() has not been called.
    """
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

    logging.info('[SWIFT] Starting Mechanism.')
    constraints = initial_potentials is not None
    marginal_oracle = common.default_oracle(self.marginal_oracle, constraints)

    #########################################################################
    # Compile workload into candidate measurements, and precompute answers. #
    #########################################################################
    candidates = common.compiled_workload(
        data.domain, self.workload, self.max_marginal_size
    )
    answers = mbi.CliqueVector.from_projectable(data, candidates)
    logging.info('[SWIFT] Calculated workload-query answers.')

    # Convert end-to-end GDP sigma to budget for internal allocation.
    gdp_budget = 1.0 / self.gdp_sigma**2
    budget_remaining = gdp_budget
    domain = data.domain
    if initial_measurements is None:
      budget_oneway = self.one_way_budget_frac * gdp_budget
      sigma_oneway = accounting.gdp_gaussian_sigma(budget_oneway)
      budget_remaining -= budget_oneway
      # measure_marginals_with_noise splits sigma_oneway across queries.
      measurements = common.measure_marginals_with_noise(
          rng, data, [(a,) for a in domain], gdp_sigma=sigma_oneway
      )
    else:
      measurements = list(initial_measurements)

    potentials = initial_potentials
    if potentials is not None:
      potentials = potentials.expand([m.clique for m in measurements])

    model = mbi.estimation.mirror_descent(
        domain,
        measurements,
        iters=self.pgm_iters,
        potentials=potentials,
        marginal_oracle=marginal_oracle,
    )
    logging.info('[SWIFT] Estimated initial model.')

    ###########################################
    # Select subset of candidates to measure. #
    ###########################################
    assert 0 < self.select_budget_frac < 1
    l1_error_budget = self.select_budget_frac * gdp_budget
    budget_remaining -= l1_error_budget

    errors = _compute_initial_errors(
        rng, answers, model, list(candidates), l1_error_budget
    )
    logging.info('[SWIFT] Computed initial errors.')

    selected, jtree = select_queries(
        errors, candidates, domain, self.max_clique_size, budget_remaining
    )

    ##########################################
    # Measure the selected marginal queries. #
    ##########################################
    new_measurements, _ = _measure_selected_marginals(
        rng, answers, selected, budget_remaining
    )
    measurements.extend(new_measurements)

    ########################################################
    # Estimate the model using all measurements            #
    ########################################################

    closed_oracle = functools.partial(
        mbi.marginal_oracles.message_passing_stable, jtree=jtree
    )

    callback_fn = mbi.callbacks.default(measurements)
    model = mbi.estimation.mirror_descent(
        domain,
        measurements,
        iters=self.pgm_iters,
        potentials=potentials,
        marginal_oracle=closed_oracle,
        callback_fn=callback_fn,
    )
    logging.info('[SWIFT] Estimated final model.')

    return SWIFTMechanismResult(model=model)


def _is_supported(clique: mbi.Clique, tree: nx.Graph) -> bool:
  """Returns whether the clique is supported by the clique tree."""
  return any(set(clique) <= set(n) for n in tree.nodes)


def build_clique_tree(
    domain: mbi.Domain,
    errors: Mapping[mbi.Clique, float],
    max_clique_size: float,
    penalty: float = 0.0,
) -> nx.Graph:
  """Greedily construct a clique tree using the SWIFT heuristic.

  We iteratively build a clique tree by iteratively incorporating attribute
  pairs with high error, subject to a constraint on the size of the largest
  clique/node in the tree.

  Args:
    domain: The domain of the data.
    errors: A dictionary mapping cliques to the DP error of the corresponding
      marginal in the workload.
    max_clique_size: The maximum size of a clique in the clique tree.
    penalty: Penalize scores by the domain size of the clique times this factor.

  Returns:
    A clique tree whose nodes (cliques) support a subset of the workload with
    high error.
  """
  result = nx.Graph()
  result.add_nodes_from([(a,) for a in domain.attributes])

  # We only consider 2-way cliques for this greedy algorithm, although the
  # resulting clique tree will generally contain larger cliques.
  errors = {
      key: value - penalty * domain.size(key)
      for key, value in errors.items()
      if len(key) == 2
  }
  prev_size = None

  while prev_size != len(errors):
    prev_size = len(errors)
    supporting_edges = clique_tree.derive_supporting_edges(result)

    for cl in sorted(errors, key=lambda x: errors[x], reverse=True):
      edge, cost = clique_tree.best_supporting_edge(
          cl, supporting_edges, domain
      )
      is_supported = edge is not None
      is_small_enough = cost <= max_clique_size
      if is_supported and is_small_enough:
        result = clique_tree.local_update(result, cl, domain)
        errors = {c: errors[c] for c in errors if not _is_supported(c, result)}
        break

      elif math.isfinite(cost) and not is_small_enough:
        del errors[cl]

  return result


def build_best_clique_tree(
    domain: mbi.Domain,
    errors: Mapping[mbi.Clique, float],
    max_clique_size: float,
    penalties: Sequence[float] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> nx.Graph:
  """Builds the best clique tree by trying different penalties."""
  best_tree = None
  best_score = 0.0
  for penalty in penalties:
    tree = build_clique_tree(domain, errors, max_clique_size, penalty)
    # By measuring these cliques, we will be able to greatly reduce the error.
    # Therefore, selecting clique with high total error is beneficial.
    score = sum(errors[cl] for cl in errors if _is_supported(cl, tree))

    if score > best_score:
      best_score = score
      best_tree = tree
  assert best_tree is not None
  return best_tree


def _compute_initial_errors(
    rng: np.random.Generator,
    data: mbi.Projectable,
    model: mbi.MarkovRandomField,
    cliques: Sequence[mbi.Clique],
    gdp_mu: float,
) -> dict[mbi.Clique, float]:
  """Computes the initial errors for the SWIFT mechanism."""
  gdp_per_clique = gdp_mu / len(cliques)
  sigma_per_clique = accounting.gdp_gaussian_sigma(gdp_per_clique)
  errors = {}
  total = float(model.total)
  oneway = {a: model.project((a,)) / total for a in data.domain}
  oneway = jax.tree.map(np.asarray, oneway)
  for cl in tqdm.tqdm(cliques, desc='Computing initial errors'):
    estimate = functools.reduce(mbi.Factor.__mul__, (oneway[a] for a in cl))
    actual = data.project(cl)
    diff = (total * estimate - actual).datavector()
    error = np.abs(diff).sum()
    errors[cl] = error + rng.normal(loc=0.0, scale=sigma_per_clique)
  return errors


def select_queries(
    errors: Mapping[mbi.Clique, float],
    candidates: Mapping[mbi.Clique, float],
    domain: mbi.Domain,
    max_clique_size: float,
    gdp_budget: float,
) -> tuple[dict[mbi.Clique, float], nx.Graph]:
  """Selects queries to measure and returns a supporting junction tree."""
  jtree = build_best_clique_tree(domain, errors, max_clique_size)
  eligible_subset = [cl for cl in candidates if _is_supported(cl, jtree)]

  logging.info('[SWIFT] Built clique tree.')
  logging.info(
      '[SWIFT] %d of %d candidates are supported.',
      len(eligible_subset),
      len(candidates),
  )

  swift_candidates = [
      swift_utils.Candidate(
          id=cl,
          error=errors[cl],
          size=domain.size(cl),
          weight=candidates[cl],
      )
      for cl in eligible_subset
  ]
  selected = swift_utils.best_subset_and_allocation(
      swift_candidates, gdp_budget
  )
  logging.info('[SWIFT] Allocated budget to %d candidates.', len(selected))

  assert all(b >= 0 for b in selected.values())
  budget_to_spend = sum(selected.values())
  logging.info(
      '[SWIFT] Budget to spend/remaining: %f / %f', budget_to_spend, gdp_budget
  )
  assert math.isclose(budget_to_spend, gdp_budget, abs_tol=1e-6)

  jtree2, _ = mbi.junction_tree.make_junction_tree(domain, list(selected))
  size1 = max(domain.size(cl) for cl in jtree.nodes)
  size2 = max(domain.size(cl) for cl in jtree2.nodes)
  if size2 < size1:
    jtree = jtree2
  logging.info('[SWIFT] Max clique size: %d (before) %d (after)', size1, size2)

  return selected, jtree


def _measure_selected_marginals(
    rng: np.random.Generator,
    data: mbi.Projectable,
    selected: dict[mbi.Clique, float],
    budget_remaining: float,
) -> tuple[list[mbi.LinearMeasurement], float]:
  """Measures the selected marginal queries."""
  measurements = []
  for cl in selected:
    budget_remaining -= selected[cl]
    sigma = accounting.gdp_gaussian_sigma(selected[cl])
    x = data.project(cl).datavector()
    y = x + rng.normal(loc=0.0, scale=sigma, size=x.size)
    measurements.append(mbi.LinearMeasurement(y, cl, sigma))
    logging.info('[SWIFT] Measured %s with sigma %f', cl, sigma)

  logging.info('[SWIFT] Budget remaining: %f', budget_remaining)
  logging.info('[SWIFT] Measured selected marginals.')
  logging.info('[SWIFT] Selected %d marginals.', len(selected))

  return measurements, budget_remaining
