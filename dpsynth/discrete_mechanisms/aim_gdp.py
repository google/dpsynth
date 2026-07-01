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

"""Variant of the Adaptive+Iterative Mechanism (AIM) that satisfies Gaussian DP."""

from collections.abc import Iterable, Mapping
import dataclasses
import typing

from absl import logging
import dp_accounting
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.local_mode import primitives
import jax.numpy as jnp
import mbi
import mbi.junction_tree
import numpy as np

MarginalQuery: typing.TypeAlias = tuple[str, ...]


def _filter_candidates(
    candidates: Mapping[MarginalQuery, float],
    model: mbi.MarkovRandomField,
    size_limit: float,
) -> Mapping[MarginalQuery, float]:
  """Filters the given candidates that lead to tractable graphical models.

  Args:
    candidates: The candidate marginal queries.
    model: The current graphical model.
    size_limit: The size limit in megabytes for the new graphical model, if a
      given candidate is selected.

  Returns:
    A collection of new candidates that pass the size_limit filter.
  """

  def expected_size(cl):
    return mbi.junction_tree.hypothetical_model_size(
        model.domain, [*model.cliques, cl]
    )

  ans = {}
  free_cliques = common.downward_closure(model.cliques)
  for cl in candidates:
    if expected_size(cl) <= size_limit or cl in free_cliques:
      ans[cl] = candidates[cl]
  return ans


def _compute_dp_errors(
    rng: np.random.Generator,
    answers: mbi.CliqueVector,
    estimates: mbi.CliqueVector,
    gdp_budget: float,
    subset: Iterable[MarginalQuery] | None = None,
) -> dict[MarginalQuery, float]:
  """Compute L1 error between the model answers and the true answers with DP."""
  if subset is None:
    subset = answers.cliques

  # sensitivity is 1 because it is an L1 norm of a vector that changes by
  # at most +/- 1 in one entry.
  per_candidate_sigma = accounting.gdp_gaussian_sigma(gdp_budget / len(subset))
  result = {}
  for cl in subset:
    actual = answers[cl].datavector(flatten=True)
    estimate = estimates[cl].datavector(flatten=True)
    error = jnp.linalg.norm(actual - estimate, ord=1)
    noise = rng.normal(loc=0, scale=per_candidate_sigma)
    result[cl] = error + noise
  return result


def _worst_approximated(
    rng: np.random.Generator,
    candidates: Mapping[MarginalQuery, float],
    errors: dict[MarginalQuery, float],  # will be updated in-place.
    answers: mbi.CliqueVector,  # derived from sensitive data.
    model: mbi.MarkovRandomField,
    select_budget: float,  # satisfies select_budget-GDP.
    measure_sigma: float,
    max_new_evals: int,
) -> MarginalQuery:
  """Returns the worst approximated candidate in the given candidates."""
  current_score_estimates = {}
  for cl in candidates:
    weight = candidates[cl]
    bias = (2 / np.pi) ** 0.5 * measure_sigma * model.domain.size(cl)
    current_score_estimates[cl] = weight * (errors[cl] - bias)

  subset = sorted(current_score_estimates, key=current_score_estimates.get)
  subset = subset[-max_new_evals:]

  estimates = mbi.marginal_oracles.bulk_variable_elimination(
      model.potentials, subset, model.total
  )
  # Only step that uses "answers", satisfies DP.
  current_errors = _compute_dp_errors(
      rng, answers, estimates, select_budget, subset
  )
  errors.update(current_errors)

  current_scores = {}
  for cl in subset:
    weight = candidates[cl]
    bias = (2 / np.pi) ** 0.5 * measure_sigma * model.domain.size(cl)
    current_scores[cl] = weight * (errors[cl] - bias)

  return max(current_scores, key=current_scores.get)


@dataclasses.dataclass
class AIMGDPMechanism(primitives.DPMechanism):
  """Configuration for the AIM mechanism with Gaussian DP.

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
      should be tailored to. The weights determine the relative importance of
      each marginal query. A default value of 1.0 will be assigned if the
      workload is provided as a list.
    max_rounds: The maximum number of rounds to run the mechanism.
    pgm_iters: The number of iterations for the mirror descent algorithm.
    max_model_size: The maximum size of the graphical model in megabytes.
      Controls the utility/runtime trade-off.
    max_marginal_size: The maximum size of a marginal query to consider.
    max_candidates_per_round: The maximum number of candidates to consider per
      round. This can improve privacy budget utilization as well as speed up the
      "Select" step, which in some settings is the main bottelneck of the
      mechanism.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    anneal_factor: The factor by which to anneal the privacy budget.
    one_way_budget_fraction: The fraction of the privacy budget to use for
      one-way marginals.
    select_budget_fraction: The fraction of the privacy budget to use for the
      "Select" step.
    gdp_sigma: The GDP sigma of the end-to-end mechanism. Privacy budget is
      split across rounds internally.
  """

  workload: Mapping[MarginalQuery, float] | Iterable[MarginalQuery] | None = (
      None
  )
  max_rounds: int | None = None
  pgm_iters: int = 1000
  max_model_size: int = 80
  max_marginal_size: float = 1e6
  max_candidates_per_round: int = 16
  marginal_oracle: mbi.MarginalOracle | None = None
  anneal_factor: float = 4.0
  one_way_budget_fraction: float = 0.1
  select_budget_fraction: float = 0.1
  gdp_sigma: float | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the workload cliques filtered by max_marginal_size."""
    return common.supporting_cliques(
        domain, self.workload, self.max_marginal_size
    )

  def calibrate(self, *, zcdp_rho: float) -> 'AIMGDPMechanism':
    """Returns a new instance calibrated to the given zCDP budget."""
    return dataclasses.replace(
        self, gdp_sigma=accounting.zcdp_gaussian_sigma(zcdp_rho)
    )

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the AIM-GDP mechanism."""
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.gdp_sigma)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Projectable,
      *,
      initial_measurements: list[mbi.LinearMeasurement] | None = None,
      initial_potentials: mbi.CliqueVector | None = None,
  ) -> common.DiscreteMechanismResult:
    """Runs the AIM-GDP mechanism on the given data.

    Args:
      rng: A numpy random number generator.
      data: The input data to the mechanism.
      initial_measurements: Optional initial measurements to start from.
      initial_potentials: Optional initial potentials (constraints).

    Returns:
      A DiscreteMechanismResult containing the estimated data distribution.
    """
    if self.gdp_sigma is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

    logging.info('[AIM] Starting Mechanism.')
    phase_times = {}

    # Convert end-to-end GDP sigma to budget for internal allocation.
    gdp_budget = 1.0 / self.gdp_sigma**2

    #########################################################################
    # Compile workload into candidate measurements, and precompute answers. #
    #########################################################################
    candidates = common.compiled_workload(
        data.domain, self.workload, self.max_marginal_size
    )
    answers = mbi.CliqueVector.from_projectable(data, candidates)

    logging.info('[AIM] Calculated workload-query answers.')
    terminate = False
    budget_remaining = gdp_budget
    domain = data.domain
    max_rounds = self.max_rounds or 16 * len(domain)
    budget_per_round = budget_remaining / max_rounds

    if initial_measurements is None:
      one_way_budget = self.one_way_budget_fraction * gdp_budget
      one_way_gdp_sigma = accounting.gdp_gaussian_sigma(one_way_budget)
      budget_remaining -= one_way_budget
      marginal_queries = [cl for cl in candidates.keys() if len(cl) == 1]
      # measure_marginals_with_noise splits one_way_gdp_sigma across queries.
      measurements = common.measure_marginals_with_noise(
          rng,
          data,
          marginal_queries=marginal_queries,
          gdp_sigma=one_way_gdp_sigma,
      )
    else:
      measurements = list(initial_measurements)

    potentials = initial_potentials
    if potentials is not None:
      potentials = potentials.expand([m.clique for m in measurements])

    model = mbi.estimation.MirrorDescent(
        marginal_oracle=self.marginal_oracle,
    ).estimate(
        domain,
        measurements,
        iters=self.pgm_iters,
        potentials=potentials,
    )
    assert isinstance(model, mbi.MarkovRandomField)
    logging.info('[AIM] Estimated initial model.')

    # The initial model is fitted from 1-way measurements only, so it IS the
    # independence model. compute_independence_errors is much faster than
    # bulk_variable_elimination for this case (pure numpy, no XLA compilation).
    budget_remaining -= 0.5 * budget_per_round
    per_candidate_sigma = accounting.gdp_gaussian_sigma(
        0.5 * budget_per_round / len(candidates)
    )
    errors = common.compute_independence_errors(data, model, list(candidates))
    for cl in errors:
      errors[cl] += rng.normal(loc=0.0, scale=per_candidate_sigma)
    logging.info('[AIM] Computed initial errors.')

    t = 0
    while not terminate:
      t += 1
      if budget_remaining < 2 * budget_per_round:
        logging.info('[AIM] Final round, Using all remaining privacy budget.')
        budget_per_round = budget_remaining
        terminate = True

      ########################################################################
      # Select a marginal query worst approximated by the current model.     #
      ########################################################################
      with common.timed(phase_times, 'selection'):
        budget_remaining -= budget_per_round
        measure_budget = budget_per_round * (1 - self.select_budget_fraction)
        select_budget = budget_per_round * self.select_budget_fraction
        measure_sigma = accounting.gdp_gaussian_sigma(measure_budget)
        percent_used = (gdp_budget - budget_remaining) / gdp_budget
        size_limit = self.max_model_size * percent_used
        small_candidates = _filter_candidates(candidates, model, size_limit)

        marginal_query = _worst_approximated(
            rng,
            candidates=small_candidates,
            errors=errors,
            answers=answers,
            model=model,
            select_budget=select_budget,
            measure_sigma=measure_sigma,
            max_new_evals=self.max_candidates_per_round,
        )

      summary = mbi.summarize(
          domain, [m.clique for m in measurements] + [marginal_query]
      )
      logging.info(
          '[AIM-GDP] Round %d, Budget used: %.4f, Measuring: %s,'
          ' Candidates: %d, cliques: %d, treewidth: %d, memory: %d bytes',
          t,
          percent_used,
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
            rng, data, [marginal_query], measure_sigma
        )[0]
        measurements.append(measurement)
        old_estimate = model.project(marginal_query).datavector()

      #####################################################
      # Estimate the data distribution using Private-PGM. #
      #####################################################
      with common.timed(phase_times, 'estimation'):
        callback_fn = mbi.callbacks.default(measurements, domain)
        measured_cliques = list(set(m.clique for m in measurements))
        warm_start = model.potentials.expand(measured_cliques)
        model = mbi.estimation.MirrorDescent(
            marginal_oracle=self.marginal_oracle,
        ).estimate(
            domain,
            measurements,
            potentials=warm_start,
            iters=self.pgm_iters,
            callback_fn=callback_fn,
        )
        model = typing.cast(mbi.MarkovRandomField, model)

      new_estimate = model.project(marginal_query).datavector()

      ##########################################
      # Anneal epsilon and sigma if necessary. #
      ##########################################
      # See Alg 4 of https://arxiv.org/pdf/2201.12677.
      # of just the largest error candidate), we can maybe simplify this logic.
      threshold = (
          measure_sigma * (2 / np.pi) ** 0.5 * domain.size(marginal_query)
      )
      if np.linalg.norm(new_estimate - old_estimate, ord=1) <= threshold:
        # No useful information at this noise level, increase budget per round.
        budget_per_round *= self.anneal_factor
        logging.info(
            '[AIM] Increasing budget per round: %.5f', budget_per_round
        )

    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times
    diagnostics.num_rounds = t
    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=model.synthetic_data(),
        measurements=measurements,
        diagnostics=diagnostics,
    )
