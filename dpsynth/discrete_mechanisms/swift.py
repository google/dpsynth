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
import concurrent.futures
import dataclasses
import functools
import itertools
import math
import time
from typing import Any

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth import checkpoint as checkpoint_lib
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import clique_tree
from dpsynth.discrete_mechanisms import common
from dpsynth.discrete_mechanisms import swift_utils
import mbi
import networkx as nx
import numpy as np


@dataclasses.dataclass(frozen=True)
class SWIFTConfig(api.MechanismConfig):
  """Configuration for the SWIFT mechanism.

  Attributes:
    workload: The set of marginals to consider for the mechanism. Can be a
      mapping from cliques to their weights or just an iterable of cliques.
    max_clique_size: The maximum size (domain product) allowed for any clique in
      the junction tree. This is the main knob to tune to improve utility for a
      given compute cost.
    max_marginal_size: The maximum size (domain product) of any marginal
      considered in the workload.
    pgm_iters: Number of mirror descent iterations for PGM estimation.
    select_budget_frac: Fraction of the total budget used for selecting which
      marginals to measure.
    working_dir: Base directory path for intermediate checkpoints (e.g. exact
      marginals, noisy measurements, model). If None, checkpointing is disabled.
  """

  workload: Mapping[mbi.Clique, float] | Iterable[mbi.Clique] | None = None
  max_clique_size: float = 1e7
  max_marginal_size: float = 1e6
  pgm_iters: int = 10_000
  marginal_oracle: mbi.MarginalOracle | None = None
  select_budget_frac: float = 0.1
  working_dir: str | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the workload cliques filtered by max_marginal_size."""
    return common.supporting_cliques(
        domain, self.workload, self.max_marginal_size
    )

  def configure(self, _=None, *, zcdp_rho, delta=0, max_records_per_user=1):
    api.validate_max_records_per_user(max_records_per_user)
    return SWIFT(
        config=self,
        gdp_budget=accounting.zcdp_to_gdp(zcdp_rho),
        max_records_per_user=max_records_per_user,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class SWIFT(api.CalibratedMechanism):
  """Calibrated SWIFT instance."""

  config: SWIFTConfig
  gdp_budget: float
  max_records_per_user: int = 1

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the SWIFT mechanism."""
    return dp_accounting.GaussianDpEvent(
        accounting.gdp_gaussian_sigma(self.gdp_budget)
    )

  def _select_and_measure(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      checkpointer: checkpoint_lib.Checkpointer,
      phase_times: dict[str, float],
      initial_measurements: Sequence[mbi.LinearMeasurement],
      constraints: Sequence[mbi.Constraint],
  ) -> tuple[
      list[mbi.LinearMeasurement],
      nx.Graph,
      concurrent.futures.Future[Any] | None,
      concurrent.futures.Future[Any] | None,
  ]:
    """Selects and measures candidate marginals, returning measurements and jtree."""
    select_gdp_budget = self.gdp_budget * self.config.select_budget_frac
    measure_gdp_budget = self.gdp_budget - select_gdp_budget

    with common.timed(phase_times, 'compiled_workload'):
      candidates = common.compiled_workload(
          data.domain,
          self.config.workload,
          self.config.max_marginal_size,
      )
    logging.info('[SWIFT] %d candidates.', len(candidates))

    with common.timed(phase_times, 'initial_mirror_descent'):
      estimator = mbi.estimation.MirrorDescent(self.config.marginal_oracle)
      model = estimator.estimate(
          data.domain,
          list(initial_measurements),  # pyrefly: ignore[bad-argument-type]
          iters=self.config.pgm_iters,
          constraints=constraints,
      )

    with common.timed(phase_times, 'selection'):
      with common.timed(phase_times, 'compute_initial_errors'):
        noisy_errors = _compute_initial_errors(
            rng,
            data,
            model,  # pyrefly: ignore[bad-argument-type]
            list(candidates),
            select_gdp_budget,
            max_records_per_user=self.max_records_per_user,
        )

      with common.timed(phase_times, 'select_queries'):
        selected, jtree = select_queries(
            noisy_errors,
            candidates,
            data.domain,
            self.config.max_clique_size,
            measure_gdp_budget,
        )

    all_cliques = [m.clique for m in initial_measurements] + list(selected)
    logging.info(mbi.summarize(data.domain, all_cliques, jtree))

    oracle = self.config.marginal_oracle or mbi.marginal_oracles.default_oracle(
        all_cliques, data.domain, has_constraints=bool(constraints)
    )
    closed_oracle = functools.partial(oracle, jtree=jtree)
    estimator = mbi.estimation.MirrorDescent(marginal_oracle=closed_oracle)
    rows = mbi.estimation.minimum_variance_unbiased_total(initial_measurements)  # pyrefly: ignore[bad-argument-type]
    rows = int(max(rows, 1))

    pgm_future = estimator.precompile(
        data.domain, list(initial_measurements), extra_cliques=list(selected)  # pyrefly: ignore[bad-argument-type]
    )
    synth_future = mbi.extensions.precompile(
        data.domain, list(jtree.nodes), rows
    )
    logging.info('[SWIFT] Started precompilation of MirrorDescent + synth.')

    with common.timed(phase_times, 'measurement'):
      logging.info('[SWIFT] Starting measurements.')
      new_measurements = _measure_selected_marginals(
          rng,
          data,
          selected,
          measure_gdp_budget,
          max_records_per_user=self.max_records_per_user,
      )
      measurements = list(initial_measurements) + new_measurements
      checkpointer.save('measurements.npz', measurements)
      logging.info('[SWIFT] Finished measurements.')

    return measurements, jtree, pgm_future, synth_future

  def _estimate_model(
      self,
      domain: mbi.Domain,
      measurements: Sequence[mbi.LinearMeasurement],
      jtree: nx.Graph,
      checkpointer: checkpoint_lib.Checkpointer,
      phase_times: dict[str, float],
      constraints: Sequence[mbi.Constraint],
      pgm_future: concurrent.futures.Future[Any] | None = None,
  ) -> mbi.Model:
    """Estimates the MRF model from measurements using MirrorDescent."""
    with common.timed(phase_times, 'estimation'):
      if pgm_future is not None:
        t0 = time.time()
        pgm_future.result()
        logging.info('[SWIFT] PGM precompile wait: %.2fs', time.time() - t0)

      all_cliques = list(jtree.nodes)
      oracle = (
          self.config.marginal_oracle
          or mbi.marginal_oracles.default_oracle(
              all_cliques, domain, has_constraints=bool(constraints)
          )
      )
      closed_oracle = functools.partial(oracle, jtree=jtree)
      estimator = mbi.estimation.MirrorDescent(marginal_oracle=closed_oracle)
      final_model = estimator.estimate(
          domain,
          list(measurements),
          iters=self.config.pgm_iters,
          callback_fn=mbi.callbacks.default(list(measurements), domain),
          constraints=constraints,
      )
      checkpointer.save('model.npz', final_model)
      logging.info('[SWIFT] Estimated final model.')
      return final_model

  def _synthesize_result(
      self,
      final_model: mbi.Model,
      measurements: Sequence[mbi.LinearMeasurement],
      initial_measurements: Sequence[mbi.LinearMeasurement],
      phase_times: dict[str, float],
      synth_future: concurrent.futures.Future[Any] | None = None,
  ) -> common.DiscreteMechanismResult:
    """Synthesizes dataset records from model and builds mechanism result."""
    if synth_future is not None:
      t0 = time.time()
      synth_future.result()
      logging.info('[SWIFT] Synth precompile wait: %.2fs', time.time() - t0)

    total_src = initial_measurements if initial_measurements else measurements
    rows = mbi.estimation.minimum_variance_unbiased_total(total_src)  # pyrefly: ignore[bad-argument-type]
    rows = int(round(max(rows, 1)))
    syn = mbi.extensions.synthetic_data(final_model, rows)  # pyrefly: ignore[bad-argument-type]
    logging.info('[SWIFT] Generated %d synthetic records.', rows)

    diagnostics = common.clique_stats(final_model)
    diagnostics.phase_times = phase_times
    return common.DiscreteMechanismResult(
        synthetic_data=syn,
        measurements=list(measurements),
        model=final_model,
        diagnostics=diagnostics,
    )

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] = (),
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    common.validate_initial_measurements(initial_measurements)
    phase_times = {}
    checkpointer = checkpoint_lib.Checkpointer(self.config.working_dir)

    # 1. Full resume: if model and measurements exist, skip to synthesis.
    if checkpointer.exists('model.npz') and checkpointer.exists(
        'measurements.npz'
    ):
      logging.info('[SWIFT] Resuming from checkpointed model and measurements.')
      final_model = checkpointer.load('model.npz')
      measurements = checkpointer.load('measurements.npz')
      assert final_model is not None and measurements is not None
      return self._synthesize_result(
          final_model, measurements, initial_measurements, phase_times
      )

    # 2. Stage 1: Measurements
    if checkpointer.exists('measurements.npz'):
      logging.info('[SWIFT] Resuming from checkpointed measurements.')
      measurements = checkpointer.load('measurements.npz')
      assert measurements is not None
      jtree, _ = mbi.junction_tree.make_junction_tree(
          data.domain, [m.clique for m in measurements]
      )
      pgm_future, synth_future = None, None
    else:
      measurements, jtree, pgm_future, synth_future = self._select_and_measure(
          rng,
          data,
          checkpointer,
          phase_times,
          initial_measurements,
          constraints,
      )

    # 3. Stage 2: Model Estimation
    if checkpointer.exists('model.npz'):
      logging.info('[SWIFT] Resuming from checkpointed model.')
      final_model = checkpointer.load('model.npz')
      assert final_model is not None
    else:
      final_model = self._estimate_model(
          data.domain,
          measurements,
          jtree,
          checkpointer,
          phase_times,
          constraints,
          pgm_future,
      )

    # 4. Stage 3: Synthesis & Diagnostics
    return self._synthesize_result(
        final_model,
        measurements,
        initial_measurements,
        phase_times,
        synth_future=synth_future,
    )


def _is_supported(clique: mbi.Clique, tree: nx.Graph) -> bool:
  """Returns whether the clique is supported by the clique tree."""
  return any(set(clique) <= set(n) for n in tree.nodes)


def _supported_pairs(tree: nx.Graph) -> frozenset[tuple[str, str]]:
  """Returns all attribute pairs supported by the clique tree."""
  pairs = set()
  for node in tree.nodes:
    for pair in itertools.combinations(sorted(node), 2):
      pairs.add(pair)
  return frozenset(pairs)


def build_clique_tree(
    domain: mbi.Domain,
    errors: Mapping[mbi.Clique, float],
    max_clique_size: float,
    penalty: float = 0.0,
    max_candidates: int = 5000,
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
    max_candidates: Cap on the number of top-error candidates to keep before the
      greedy loop. Set to 0 to disable.

  Returns:
    A clique tree whose nodes (cliques) support a subset of the workload with
    high error.
  """
  result = nx.Graph()
  result.add_nodes_from([(a,) for a in domain.attributes])

  # We only consider 2-way cliques for this greedy algorithm, although the
  # resulting clique tree will generally contain larger cliques.
  # The penalty is a surrogate for the noise sigma that will be determined
  # later by best_subset_and_allocation once the budget is split.
  errors = {
      key: value - penalty * domain.size(key)
      for key, value in errors.items()
      if len(key) == 2
  }

  # Sort once — values never change, only entries get removed.
  sorted_cliques = sorted(errors, key=errors.get, reverse=True)  # pyrefly: ignore[no-matching-overload]

  # Cap candidates to keep the greedy loop tractable.
  if max_candidates > 0 and len(sorted_cliques) > max_candidates:
    dropped = set(sorted_cliques[max_candidates:])
    sorted_cliques = sorted_cliques[:max_candidates]
    errors = {c: errors[c] for c in errors if c not in dropped}

  prev_size = None
  while prev_size != len(errors):
    prev_size = len(errors)
    supporting_edges = clique_tree.derive_supporting_edges(result)

    for cl in sorted_cliques:
      if cl not in errors:
        continue
      edge, cost = clique_tree.best_supporting_edge(
          cl, supporting_edges, domain
      )
      is_supported = edge is not None
      is_small_enough = cost <= max_clique_size
      if is_supported and is_small_enough:
        result = clique_tree.local_update(result, cl, domain)
        supported = _supported_pairs(result)
        errors = {
            c: errors[c] for c in errors if tuple(sorted(c)) not in supported
        }
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
  best_score = float('-inf')
  for penalty in penalties:
    tree = build_clique_tree(domain, errors, max_clique_size, penalty)
    # By measuring these cliques, we will be able to greatly reduce the error.
    # Therefore, selecting clique with high total error is beneficial.
    supported = _supported_pairs(tree)
    score = sum(
        errors[cl]
        for cl in errors
        if len(cl) == 2 and tuple(sorted(cl)) in supported
    )

    if score > best_score or best_tree is None:
      best_score = score
      best_tree = tree
  assert best_tree is not None
  return best_tree


def _compute_initial_errors(
    rng: np.random.Generator,
    data: mbi.Dataset | mbi.CliqueVector,
    model: mbi.MarkovRandomField,
    cliques: Sequence[mbi.Clique],
    gdp_budget: float,
    max_records_per_user: int = 1,
) -> dict[mbi.Clique, float]:
  """Computes DP initial errors for the SWIFT mechanism."""
  if not cliques:
    return {}
  budget_per_clique = gdp_budget / len(cliques)
  sigma_per_clique = max_records_per_user * accounting.gdp_gaussian_sigma(
      budget_per_clique
  )
  errors = common.compute_independence_errors(data, model, cliques)  # pyrefly: ignore[bad-argument-type]
  for cl in errors:
    errors[cl] += rng.normal(loc=0.0, scale=sigma_per_clique)
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
    data: mbi.Dataset | mbi.CliqueVector,
    selected: dict[mbi.Clique, float],
    budget_remaining: float,
    max_records_per_user: int = 1,
) -> list[mbi.LinearMeasurement]:
  """Measures the selected marginal queries."""
  measurements = []
  for cl in selected:
    budget_remaining -= selected[cl]
    sigma = max_records_per_user * accounting.gdp_gaussian_sigma(selected[cl])
    x = data.project(cl).datavector()
    y = x + rng.normal(loc=0.0, scale=sigma, size=x.size)
    measurements.append(mbi.LinearMeasurement(y, cl, sigma))
    logging.info('[SWIFT] Measured %s with sigma %f', cl, sigma)

  logging.info('[SWIFT] Budget remaining: %f', budget_remaining)
  logging.info('[SWIFT] Measured selected marginals.')
  logging.info('[SWIFT] Selected %d marginals.', len(selected))

  return measurements
