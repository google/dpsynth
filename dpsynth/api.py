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

"""Core abstractions for differentially private mechanisms.

This module defines the ``DPMechanism`` base class, which is the primary
building block for all differentially private algorithms in DP Synth.

Example usage::

  mechanism = dpsynth.discrete_mechanisms.AIMMechanism(pgm_iters=500)

  # Option 1: Calibrate to (epsilon, delta)-DP (tight PLD accounting).
  calibrated = mechanism.calibrate(epsilon=1.0, delta=1e-5)

  # Option 2: Configure with a zCDP budget directly.
  calibrated = mechanism.configure(zcdp_rho=0.5)

  # Run the mechanism.
  result = calibrated(rng, data)
"""

from __future__ import annotations

import abc
from collections.abc import Callable
import dataclasses
import functools
from typing import Any

import dp_accounting


class CalibratedMechanism(abc.ABC):
  """A privacy-calibrated, runnable differentially private mechanism.

  Produced by ``MechanismConfig.configure()`` / ``.calibrate()``: its natural
  privacy parameter (e.g. Gaussian sigma) is populated and it is ready to run.
  It exposes the exact ``DpEvent`` characterizing its privacy cost and is
  directly callable on data.

  Subclasses must implement:

  - ``dp_event``: return the exact ``DpEvent`` characterizing the mechanism.
  - ``__call__``: run the mechanism on data.
  """

  @property
  @abc.abstractmethod
  def dp_event(self) -> dp_accounting.DpEvent:
    """The DpEvent characterizing the privacy cost of this mechanism."""

  @abc.abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Runs the mechanism on the given data.

    Subclass signatures vary, but typically accept at least the data to operate
    on and a source of randomness.

    Args:
      *args: Positional arguments (subclass-specific).
      **kwargs: Keyword arguments (subclass-specific).
    """


class MechanismConfig(abc.ABC):
  """A recipe that produces a calibrated, runnable mechanism.

  A config holds the mechanism's structural and hyperparameter fields and knows
  how to turn a privacy budget into a runnable ``CalibratedMechanism``.
  Usage generally follows a three-phase pattern:

  1. **Construct**: Create the config with algorithm-specific parameters
     (e.g., ``AIMConfig(pgm_iters=500)``).
  2. **Calibrate**: Call ``calibrate(epsilon=..., delta=...)`` or
     ``configure(zcdp_rho=...)`` to bind a privacy budget, returning a
     ``CalibratedMechanism`` with the mechanism's natural privacy parameter set.
  3. **Run**: Call the calibrated mechanism on data via ``__call__``.

  **Design: configure vs calibrate.**  The API separates two concerns:

  - ``configure(zcdp_rho, **kwargs)`` is the low-level primitive that each
    config must implement. It maps a zCDP budget to the mechanism's natural
    privacy parameter (e.g., Gaussian sigma) and returns a runnable mechanism.
    This is lightweight — just arithmetic — and produces reasonably tight
    parameter settings for most mechanisms.

  - ``calibrate(epsilon, delta, **kwargs)`` is the high-level entry point
    defined once on the base class. When called with ``(epsilon, delta)``,
    it performs a binary search over zCDP budgets using
    ``dp_accounting.calibrate_dp_mechanism``, calling ``configure`` at each
    candidate and inspecting the resulting ``dp_event`` for tight PLD-based
    accounting. This gives each mechanism the maximum possible budget that
    still satisfies the target (epsilon, delta) guarantee. The (epsilon, delta)
    path is more precise but more expensive than the direct ``zcdp_rho`` path.

  **Why zCDP as the intermediate.**  Calibrating to zCDP rho makes it easy to
  split a privacy budget across a heterogeneous composition of mechanisms:
  simply divide rho additively in any ratio and each share is a valid zCDP
  guarantee.

  **Tight accounting via dp_events.**  Mechanisms may be tighter than their
  zCDP guarantee implies (e.g., GDP mechanisms). The ``calibrate`` binary
  search exploits this: it evaluates each candidate's raw ``dp_event`` rather
  than relying on the zCDP conversion, so the final calibration is as tight
  as the mechanism's own privacy characterization allows.
  """

  _registry: dict[str, type[MechanismConfig]] = {}

  @property
  def working_dir(self) -> str | None:
    """Base directory path for checkpointing intermediate mechanism state."""
    return None

  def with_working_dir(self, working_dir: str | None) -> MechanismConfig:
    """Returns a copy of the config with working_dir set if supported and unset."""
    if self.working_dir is not None or working_dir is None:
      return self
    if dataclasses.is_dataclass(self):
      try:
        return dataclasses.replace(  # pyrefly: ignore[bad-specialization]
            self, working_dir=working_dir
        )
      except (TypeError, ValueError):
        return self
    return self

  def __init_subclass__(cls, **kwargs: Any):
    super().__init_subclass__(**kwargs)
    MechanismConfig._registry[cls.__name__] = cls

  @classmethod
  def get_subclass(cls, name: str) -> type[MechanismConfig] | None:
    """Returns the registered MechanismConfig subclass by name."""
    return cls._registry.get(name)

  @abc.abstractmethod
  def configure(
      self, domain=None, *, zcdp_rho, delta=0, max_records_per_user=1
  ) -> CalibratedMechanism:
    """Returns a calibrated mechanism for the given zCDP budget.

    Converts the zCDP budget into the mechanism's natural privacy parameter
    (e.g., Gaussian sigma) and returns a runnable ``CalibratedMechanism`` with
    that parameter set.

    Most mechanisms are pure zCDP and ignore ``delta``. Mechanisms that
    consume approximate DP budget (e.g., partition selection with Gaussian
    thresholding) should raise ``ValueError`` if ``delta`` is required but
    not provided (i.e., is 0).

    Args:
      domain: Optional domain specification over which the mechanism operates
        (e.g., a dataset ``dpsynth.Schema`` or mapping from column names to
        attribute domain specifications for tabular mechanisms, an individual
        ``AttributeType`` for initializers, or a relational domain mapping).
        Mechanisms that do not need a domain ignore this argument.
      zcdp_rho: The zCDP privacy budget (rho).
      delta: Approximate DP delta consumed by the mechanism itself (e.g., for
        thresholding). Defaults to 0 (pure zCDP). Mechanisms that need delta
        should raise if it is 0.
      max_records_per_user: Assumed upper bound on the number of records a
        single user contributes. Values greater than 1 scale the added noise
        (and mechanism sensitivity) to provide user-level rather than
        record-level DP; the privacy accounting is unchanged. This bound is NOT
        enforced -- soundness relies on the caller guaranteeing it via
        preprocessing.

    Returns:
      A calibrated, runnable mechanism.
    """

  def _find_optimal_rho(
      self,
      make_event_fn: Callable[[float], dp_accounting.DpEvent],
      target_epsilon: float,
      target_delta: float,
  ) -> float:
    """Binary-search for the tightest zCDP rho within an (ε, δ) guarantee.

    Tries both RDP and PLD accountants and returns whichever gives the
    highest rho (more budget = better utility). Neither accountant
    universally dominates in tightness.

    Args:
      make_event_fn: Maps a candidate rho to the mechanism's DpEvent.
      target_epsilon: Target epsilon for (epsilon, delta)-DP.
      target_delta: Target delta for (epsilon, delta)-DP.

    Returns:
      The optimal zCDP rho.

    Raises:
      UnsupportedEventError: If no accountant supports the DpEvent.
    """
    rho = float('inf')
    try:
      # This is a heuristic to avoid excessively fine discretization in PLD
      # accounting, which can cause OOM at extremely small target epsilons.
      value_discretization_interval = max(1e-4, 1e-4 / (target_epsilon + 1e-5))
      accountant_fn = functools.partial(
          dp_accounting.pld.PLDAccountant,
          value_discretization_interval=value_discretization_interval,
      )
      rho = dp_accounting.calibrate_dp_mechanism(
          make_fresh_accountant=accountant_fn,
          make_event_from_param=make_event_fn,
          target_epsilon=target_epsilon,
          target_delta=target_delta,
      )
    except (dp_accounting.UnsupportedEventError, NotImplementedError):
      # If PLD accounting is not supported, fall back to RDP accounting.
      pass

    rho2 = dp_accounting.calibrate_dp_mechanism(
        make_fresh_accountant=dp_accounting.rdp.RdpAccountant,
        make_event_from_param=make_event_fn,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )

    # RDP can also be better than PLD in some cases due to looseness in the
    # handling of certain DpEvents like the ExponentialMechanismDpEvent.
    return min(rho, rho2)

  def calibrate(
      self,
      domain=None,
      /,
      *,
      epsilon: float,
      delta: float,
      poisson_sampling_prob: float = 1.0,
      max_records_per_user: int = 1,
  ) -> CalibratedMechanism:
    """Calibrate the mechanism to a target (epsilon, delta)-DP guarantee.

    Performs a binary search over zCDP budgets, calling ``configure`` at each
    candidate and inspecting the resulting ``dp_event``. Tries both RDP and
    PLD accounting and picks whichever gives the tightest result.

    Args:
      domain: Optional domain specification, forwarded to ``configure()``.
      epsilon: Target epsilon for (epsilon, delta)-DP.
      delta: Target delta for (epsilon, delta)-DP.
      poisson_sampling_prob: If specified, calibrate the mechanism assuming the
        input data is subsampled with the given probability. The actual sampling
        is **NOT** handled internally by the calibrated mechanism.
      max_records_per_user: Assumed upper bound on the number of records a
        single user contributes. Added noise (and mechanism sensitivity) is
        scaled by this factor to provide user-level rather than record-level DP;
        the privacy accounting is unchanged. Soundness relies on the caller
        enforcing this bound.

    Returns:
      A calibrated, runnable mechanism.
    """

    def make_event_fn(rho: float) -> dp_accounting.DpEvent:
      base = self.configure(
          domain,
          zcdp_rho=rho,
          delta=delta,
          max_records_per_user=max_records_per_user,
      ).dp_event
      sampled = dp_accounting.PoissonSampledDpEvent(poisson_sampling_prob, base)
      return base if poisson_sampling_prob == 1.0 else sampled

    optimal_rho = self._find_optimal_rho(
        make_event_fn=make_event_fn,
        target_epsilon=epsilon,
        target_delta=delta,
    )
    return self.configure(
        domain,
        zcdp_rho=optimal_rho,
        delta=delta,
        max_records_per_user=max_records_per_user,
    )


class DPMechanism(MechanismConfig, CalibratedMechanism, abc.ABC):
  """Transitional monolithic base: both a config and a runnable mechanism.

  Historically every mechanism was a single class that was constructed, then
  ``configure``d in place, then run. That design is being split into a
  ``MechanismConfig`` recipe and a ``CalibratedMechanism`` runnable (so a
  calibrated instance can never be un-calibrated and needs no nullable
  privacy-parameter fields or guards). Mechanisms are migrated one layer at a
  time; those not yet split still subclass this monolith, which exposes exactly
  the historical abstract surface (``configure`` + ``dp_event`` + ``__call__``,
  with ``calibrate`` inherited). Remove once every mechanism is split.
  """


def validate_max_records_per_user(value: int) -> None:
  """Raises ValueError if the per-user record bound is not a positive int."""
  if value < 1:
    raise ValueError(f'max_records_per_user must be >= 1, got {value}.')
