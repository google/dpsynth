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

"""Implementation of the Adaptive+Iterative Mechanism (AIM)."""

from collections.abc import Iterable, Mapping
import dataclasses

from absl import logging
import dp_accounting
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import base
from dpsynth.discrete_mechanisms import common
import jax.numpy as jnp
import mbi
import mbi.junction_tree
import numpy as np


def _filter_candidates(
    candidates: Mapping[mbi.Clique, float],
    model: mbi.MarkovRandomField,
    size_limit: float,
) -> Mapping[mbi.Clique, float]:
  """Filters the given candidates that lead to tractable graphical models.

  Args:
    candidates: The candidate marginal queries.
    model: The current graphical model.
    size_limit: The size limit in megabytes for the new graphical model, if a
      given candidate is selected.

  Returns:
    A collection of new candidates that pass the size_limit filter.
  """
  ans = {}
  free_cliques = common.downward_closure(model.cliques)
  domain = model.domain
  for cl in candidates:
    cliques = [*model.cliques, cl]
    cond1 = (
        mbi.junction_tree.hypothetical_model_size(domain, cliques) <= size_limit
    )
    cond2 = cl in free_cliques
    if cond1 or cond2:
      ans[cl] = candidates[cl]
  return ans


def _worst_approximated(
    rng: np.random.Generator,
    candidates: Mapping[mbi.Clique, float],
    answers: mbi.CliqueVector,
    estimates: mbi.CliqueVector,
    eps: float,
    sigma: float,
    domain: mbi.Domain,
) -> mbi.Clique:
  """Returns the worst approximated candidate in the given candidates."""
  errors = {}
  for cl in candidates:
    wgt = candidates[cl]
    diff = answers[cl].datavector() - estimates[cl].datavector()
    bias = jnp.sqrt(2 / jnp.pi) * sigma * domain.size(cl)
    errors[cl] = wgt * (jnp.linalg.norm(diff, ord=1) - bias)

  max_sensitivity = max(
      candidates.values(),
  )  # if all weights are 0, could be a problem
  keys, values = list(errors.keys()), np.array(list(errors.values()))
  idx = common.exponential_mechanism(
      values, eps, max_sensitivity, rng, monotonic=True
  )
  return keys[idx]


@dataclasses.dataclass
class AIMMechanism(base.DiscreteMechanism):
  """Configuration for the AIM mechanism.

  Details are described in the paper:
  [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic
  Data](https://arxiv.org/abs/2201.12677). This mechanism is a competitive
  algorithm within the broader SELECT-MEASURE-GENERATE paradigm. It is an
  MWEM-style algorithm (Multiplicative Weights + Exponential Mechanism), that
  iteratively improves the estimate of the data distribution by selecting
  marginal queries that are poorly approximated by the current model. It is a
  scalable algorithm that can handle high-dimensional datasets, but it can be
  time consuming to run (hours). The runtime/utility trade-off can be controlled
  by the max_model_size parameter. For quick experimentation, we recommend
  setting max_model_size = 1, for production use cases, we recommend setting
  max_model_size >= 80.

  Attributes:
    workload: A collection of marginal queries (and weights) the synthetic data
      should be tailored to.
    max_rounds: The maximum number of rounds to run the mechanism.
    max_model_size: The maximum size of the graphical model in megabytes.
      Controls the utility/runtime trade-off.
    max_marginal_size: The maximum size of a marginal query to consider.
    anneal_factor: The factor by which to anneal the privacy.
    select_budget_fraction: The fraction of the total budget to use for
      selecting two-way marginal queries.
  """

  workload: Mapping[mbi.Clique, float] | Iterable[mbi.Clique] | None = None
  max_rounds: int | None = None
  max_model_size: int = 80
  max_marginal_size: float = 1e6
  anneal_factor: float = 4.0
  select_budget_fraction: float = 0.1
  _loop_rho: float | None = dataclasses.field(default=None, repr=False)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the workload cliques filtered by max_marginal_size."""
    return common.supporting_cliques(
        domain, self.workload, self.max_marginal_size
    )

  def _one_way_cliques(self, data):
    """Returns only the workload-specified one-way cliques."""
    return common.one_way_cliques(self.workload, data.domain)

  def _allocate_budget(self, remaining_rho: float) -> Mapping[str, float]:
    """Allocates the entire remaining budget to the adaptive loop."""
    return {'_loop_rho': remaining_rho}

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the AIM mechanism."""
    self._check_calibration()
    events = self._one_way_dp_event()
    events.append(dp_accounting.ZCDpEvent(self._loop_rho))
    return dp_accounting.ComposedDpEvent(events)

  def _run(self, rng, data, measurements, constraints, phase_times):
    """Adaptively selects, measures, and estimates in an annealed loop."""
    logging.info('[AIM]: Starting Mechanism.')
    zcdp_rho = self.zcdp_rho
    terminate = False
    rho_remaining = self._loop_rho
    max_rounds = self.max_rounds or 16 * len(data.domain)
    rho_per_round = self._loop_rho / max_rounds

    #########################################################################
    # Compile workload into candidate measurements, and precompute answers. #
    #########################################################################
    candidates = common.compiled_workload(
        data.domain, self.workload, self.max_marginal_size
    )
    answers = mbi.CliqueVector.from_projectable(data, list(candidates))
    logging.info('[AIM]: Calculated workload-query answers.')

    estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
    model = estimator.estimate(
        data.domain, measurements, iters=self.pgm_iters, constraints=constraints
    )
    assert isinstance(model, mbi.MarkovRandomField)

    t = 0
    while not terminate:
      t += 1
      if rho_remaining < 2 * rho_per_round:
        logging.info('[AIM] Final round, Using all remaining privacy budget.')
        rho_per_round = rho_remaining
        terminate = True

      ########################################################################
      # Select a marginal query worst approximated by the current model.     #
      ########################################################################
      with common.timed(phase_times, 'selection'):
        rho_remaining -= rho_per_round
        fraction = self.select_budget_fraction
        sigma = accounting.zcdp_gaussian_sigma((1 - fraction) * rho_per_round)
        epsilon = accounting.zcdp_exponential_eps(fraction * rho_per_round)
        size_limit = self.max_model_size * (zcdp_rho - rho_remaining) / zcdp_rho
        small_candidates = _filter_candidates(candidates, model, size_limit)

        estimates = mbi.marginal_oracles.bulk_variable_elimination(
            model.potentials, list(small_candidates), total=model.total  # pyrefly: ignore[bad-argument-type]
        )
        marginal_query = _worst_approximated(
            rng,
            small_candidates,
            answers,
            estimates,
            epsilon,
            sigma,
            data.domain,
        )

      summary = mbi.summarize(
          data.domain, [m.clique for m in measurements] + [marginal_query]
      )
      logging.info(
          '[AIM] Round %d, Budget used: %.4f, Measuring: %s, Candidates: %d,'
          ' cliques: %d, treewidth: %d, memory: %d bytes',
          t,
          (zcdp_rho - rho_remaining) / zcdp_rho,
          marginal_query,
          len(small_candidates),
          summary.num_cliques,
          summary.treewidth,
          summary.memory_bytes,
      )

      ######################################################################
      # Measure the marginal query privately using the Gaussian mechanism. #
      ######################################################################
      with common.timed(phase_times, 'measurement'):
        measurement = common.measure_marginals_with_noise(
            rng, data, [marginal_query], sigma
        )[0]
        measurements.append(measurement)
        old_estimate = model.project(marginal_query).datavector()

      #####################################################
      # Estimate the data distribution using Private-PGM. #
      #####################################################
      with common.timed(phase_times, 'estimation'):
        callback_fn = mbi.callbacks.default(measurements, data.domain)
        measured_cliques = list(set(m.clique for m in measurements))
        warm_start = model.potentials.expand(measured_cliques)
        model = estimator.estimate(
            data.domain,
            measurements,
            potentials=warm_start,
            iters=self.pgm_iters,
            callback_fn=callback_fn,
            constraints=constraints,
        )
        assert isinstance(model, mbi.MarkovRandomField)

      new_estimate = model.project(marginal_query).datavector()

      ##########################################
      # Anneal epsilon and sigma if necessary. #
      ##########################################
      threshold = sigma * np.sqrt(2 / np.pi) * data.domain.size(marginal_query)
      if np.linalg.norm(new_estimate - old_estimate, ord=1) <= threshold:
        # No useful information at this noise level, increase budget per round.
        rho_per_round *= self.anneal_factor
        fraction = self.select_budget_fraction
        sigma = accounting.zcdp_gaussian_sigma((1 - fraction) * rho_per_round)
        logging.info('[AIM] Reducing sigma: %.1f', sigma)

    synthetic_data = model.synthetic_data()
    return model, synthetic_data, measurements
