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

"""Base classes for the select-measure-estimate paradigm.

This module defines the ``DiscreteMechanism`` base class, which implements the
select-measure-estimate paradigm from `McKenna et al. (2021)
<https://arxiv.org/abs/2108.04978>`_. Each step of the paradigm is a separate
method that subclasses can override independently, enabling code reuse across
mechanisms that differ primarily in the *select* step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses

from absl import logging
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
import mbi
import mbi.callbacks
import mbi.estimation
import numpy as np


@dataclasses.dataclass
class DiscreteMechanism(api.DPMechanism):
  """Base class for mechanisms following the select-measure-estimate paradigm.

  Subclasses implement ``_select`` to define which marginals to measure.
  The base ``__call__`` orchestrates the full pipeline::

      check_calibration → measure_one_way → compress → run → result

  where ``_run`` performs select → measure → estimate → generate.  One-shot
  mechanisms need only override ``_select``; adaptive mechanisms (e.g. AIM) or
  those needing a custom estimator (e.g. SWIFT) override ``_run`` directly.

  Attributes:
    marginal_oracle: Oracle for marginal inference in Private-PGM.
    pgm_iters: Number of mirror descent iterations for estimation.
    compress_columns: Domain compression config. True = all, list = specific.
    one_way_budget_fraction: Fraction of zCDP budget for one-way marginals.
    zcdp_rho: Total zCDP budget (set by configure).
    one_way_rho: zCDP budget for one-way measurements (set by configure).
    measurement_rho: zCDP budget for selected marginal measurements.
  """

  marginal_oracle: mbi.MarginalOracle | None = None
  pgm_iters: int = 5000
  compress_columns: bool | Sequence[str] = False
  one_way_budget_fraction: float = 1 / 3
  zcdp_rho: float | None = None
  one_way_rho: float | None = dataclasses.field(default=None, repr=False)
  measurement_rho: float | None = dataclasses.field(default=None, repr=False)

  def configure(
      self,
      *,
      zcdp_rho: float,
      delta: float = 0.0,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      **kwargs,
  ) -> DiscreteMechanism:
    """Configures the mechanism with a zCDP budget."""
    if initial_measurements is not None or self.one_way_budget_fraction <= 0:
      one_way_rho = None
    else:
      one_way_rho = zcdp_rho * self.one_way_budget_fraction
    remaining_rho = zcdp_rho - (one_way_rho or 0.0)
    return dataclasses.replace(
        self,
        zcdp_rho=zcdp_rho,
        one_way_rho=one_way_rho,
        **self._allocate_budget(remaining_rho),
    )

  def _allocate_budget(self, remaining_rho: float) -> Mapping[str, float]:
    """Splits the post-one-way budget into mechanism-specific rho fields.

    Subclasses override this to distribute ``remaining_rho`` across their own
    budget fields (e.g. ``measurement_rho``, ``_select_rho``); the returned
    mapping is applied as field overrides in ``configure``.

    Args:
      remaining_rho: zCDP budget left after the shared one-way measurement.

    Returns:
      A mapping from dataclass field name to allocated zCDP budget.
    """
    return {}

  @property
  def remaining_rho(self):
    """zCDP budget remaining after one-way measurements."""
    one_way_rho = 0.0 if self.one_way_rho is None else self.one_way_rho
    return self.zcdp_rho - one_way_rho

  def _one_way_dp_event(self):
    """DpEvents for the shared one-way measurement ([] if there is none)."""
    if self.one_way_rho is None:
      return []
    return [
        dp_accounting.GaussianDpEvent(
            noise_multiplier=accounting.zcdp_gaussian_sigma(self.one_way_rho)
        )
    ]

  def _check_calibration(self):
    """Raises ValueError if the mechanism has not been configured."""
    if self.zcdp_rho is None:
      raise ValueError('Must call calibrate() before using the mechanism.')

  def _one_way_cliques(self, data):
    """Returns the one-way cliques to measure."""
    cliques = [(a,) for a in data.domain]
    if hasattr(data, 'cliques'):
      supported = common.downward_closure(data.cliques)  # pytype: disable=attribute-error
      cliques = [cl for cl in cliques if cl in supported]
    return cliques

  def _measure_one_way(
      self, rng, data, phase_times, *, initial_measurements=None
  ):
    """Measures one-way marginals or returns pre-measured ones."""
    if initial_measurements is not None:
      return list(initial_measurements)
    if self.one_way_rho is None:
      return []
    with common.timed(phase_times, 'measurement'):
      sigma = accounting.zcdp_gaussian_sigma(self.one_way_rho)
      cliques = self._one_way_cliques(data)
      return common.measure_marginals_with_noise(rng, data, cliques, sigma)

  def _compress(self, data, measurements, constraints):
    """Compresses the domain by merging rare values."""
    mappings = common.compression_mappings(
        measurements, self.compress_columns, constraints
    )
    if mappings and hasattr(data, 'compress'):
      data = data.compress(mappings)  # pytype: disable=attribute-error
      measurements = [m.compress(mappings, data.domain) for m in measurements]
    return data, measurements, mappings

  def _select(self, rng, data, measurements, phase_times):
    """Selects which marginals to measure.  Mechanism-specific."""
    raise NotImplementedError

  def __call__(
      self,
      rng: np.random.Generator,
      data: mbi.Dataset | mbi.CliqueVector,
      *,
      initial_measurements: Sequence[mbi.LinearMeasurement] | None = None,
      constraints: Sequence[mbi.Constraint] = (),
  ) -> common.DiscreteMechanismResult:
    """Runs the select-measure-estimate pipeline."""
    self._check_calibration()
    phase_times = {}
    measurements = self._measure_one_way(
        rng, data, phase_times, initial_measurements=initial_measurements
    )
    data, measurements, mappings = self._compress(
        data, measurements, constraints
    )
    model, synthetic_data, measurements = self._run(
        rng, data, measurements, constraints, phase_times
    )
    if mappings:
      synthetic_data = synthetic_data.decompress(mappings)
    diagnostics = common.clique_stats(model)  # pytype: disable=wrong-arg-types
    diagnostics.phase_times = phase_times
    return common.DiscreteMechanismResult(
        model=model,  # pytype: disable=wrong-arg-types
        synthetic_data=synthetic_data,
        measurements=measurements,
        diagnostics=diagnostics,
        mappings=mappings,
    )

  def _run(self, rng, data, measurements, constraints, phase_times):
    """Selects, measures, estimates, and generates in the compressed domain."""
    # Adaptive mechanisms (e.g. AIM) override this to interleave selection and
    # measurement in a loop; SWIFT overrides it to use a junction-tree oracle.
    selected = self._select(rng, data, measurements, phase_times)
    all_cliques = [m.clique for m in measurements] + list(selected)
    logging.info(
        '[%s]:\n%s',
        type(self).__name__,
        mbi.summarize(data.domain, all_cliques),
    )

    # Kick off async AOT compilation of the estimator while we measure.
    estimator = mbi.estimation.MirrorDescent(self.marginal_oracle)
    futures = None
    try:
      futures = estimator.precompile(
          data.domain, measurements, extra_cliques=list(selected)
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning('Precompile failed (non-fatal): %s', e)

    if selected:
      with common.timed(phase_times, 'measurement'):
        sigma = accounting.zcdp_gaussian_sigma(self.measurement_rho)
        measurements = measurements + common.measure_marginals_with_noise(
            rng, data, selected, sigma
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
      assert isinstance(model, mbi.MarkovRandomField)

    synthetic_data = model.synthetic_data()
    return model, synthetic_data, measurements
