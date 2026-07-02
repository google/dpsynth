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

from collections.abc import Iterable, Mapping, Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import _api
from dpsynth.discrete_mechanisms import accounting
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
class AIMMechanism(_api.DPMechanism):
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
    pgm_iters: The number of iterations for the mirror descent algorithm.
    max_model_size: The maximum size of the graphical model in megabytes.
      Controls the utility/runtime trade-off.
    max_marginal_size: The maximum size of a marginal query to consider.
    marginal_oracle: The marginal oracle to use for the mirror descent
      algorithm.
    anneal_factor: The factor by which to anneal the privacy.
    one_way_budget_fraction: The fraction of the total budget to use for one-way
      marginal queries.
    compress_columns: Controls domain compression. True compresses all columns,
      False disables compression, or a list of column names to compress.
    select_budget_fraction: The fraction of the total budget to use for
      selecting two-way marginal queries.
  """

  workload: Mapping[mbi.Clique, float] | Iterable[mbi.Clique] | None = None
  max_rounds: int | None = None
  pgm_iters: int = 1000
  max_model_size: int = 80
  max_marginal_size: float = 1e6
  marginal_oracle: mbi.MarginalOracle | None = None
  anneal_factor: float = 4.0
  one_way_budget_fraction: float = 0.1
  compress_columns: bool | Sequence[str] = False
  select_budget_fraction: float = 0.1
  zcdp_rho: float | None = None

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    """Returns the workload cliques filtered by max_marginal_size."""
    return common.supporting_cliques(
        domain, self.workload, self.max_marginal_size
    )

  def configure(self, *, zcdp_rho: float, delta: float = 0.0) -> 'AIMMechanism':
    """Returns a new instance calibrated to the given zCDP budget."""
    return dataclasses.replace(self, zcdp_rho=zcdp_rho)

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the DP event for the AIM mechanism."""
    if self.zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')
    return dp_accounting.ZCDpEvent(self.zcdp_rho)

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: list[mbi.LinearMeasurement] | None = None,
      constraints: tuple[mbi.Constraint, ...] = (),
  ) -> common.DiscreteMechanismResult:
    """Runs the AIM mechanism on the given data.

    Args:
      rng: A numpy random number generator.
      data: The input data. Must be an mbi.Dataset for domain compression;
        mbi.CliqueVector is supported but compression will be skipped.
      initial_measurements: Optional initial measurements to start from.
      constraints: Structural constraints for the estimation.

    Returns:
      A DiscreteMechanismResult containing the estimated data distribution.
    """
    if self.zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

    logging.info('[AIM]: Starting Mechanism.')
    phase_times = {}

    zcdp_rho = self.zcdp_rho
    terminate = False
    rho_remaining = zcdp_rho
    max_rounds = self.max_rounds or 16 * len(data.domain)
    rho_per_round = zcdp_rho / max_rounds

    if initial_measurements is None:
      rho_remaining -= self.one_way_budget_fraction * zcdp_rho
      marginal_queries = common.one_way_cliques(self.workload, data.domain)
      measurements = common.measure_marginals_with_noise(
          rng,
          data,
          marginal_queries=marginal_queries,
          gdp_sigma=zcdp_rho * self.one_way_budget_fraction,
      )
    else:
      measurements = list(initial_measurements)

    mappings = common.compression_mappings(
        measurements, self.compress_columns, constraints
    )
    if mappings:
      data = data.compress(mappings)
      measurements = [m.compress(mappings, data.domain) for m in measurements]

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
            model.potentials, list(small_candidates), total=model.total
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

    diagnostics = common.clique_stats(model)
    diagnostics.phase_times = phase_times
    diagnostics.num_rounds = t
    synthetic_data = model.synthetic_data()
    if mappings:
      synthetic_data = synthetic_data.decompress(mappings)
    return common.DiscreteMechanismResult(
        model=model,
        synthetic_data=synthetic_data,
        measurements=measurements,
        diagnostics=diagnostics,
        mappings=mappings,
    )
