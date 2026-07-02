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
from typing import Any
import warnings

import dp_accounting


class DPMechanism(abc.ABC):
  """Abstract base class for differentially private mechanisms.

  A DPMechanism encapsulates a randomized algorithm that satisfies differential
  privacy. Usage follows a three-phase pattern:

  1. **Construct**: Create the mechanism with algorithm-specific parameters
     (e.g., ``AIMMechanism(pgm_iters=500)``).
  2. **Calibrate**: Call ``calibrate(epsilon=..., delta=...)`` or
     ``calibrate(zcdp_rho=...)`` to bind a privacy budget, returning a new
     frozen instance with the mechanism's natural privacy parameter set.
  3. **Run**: Call the calibrated mechanism on data via ``__call__``.

  **Design: configure vs calibrate.**  The API separates two concerns:

  - ``configure(zcdp_rho, **kwargs)`` is the low-level primitive that each
    mechanism must implement. It maps a zCDP budget to the mechanism's natural
    privacy parameter (e.g., Gaussian sigma) and returns a new frozen instance.
    This is lightweight — just arithmetic — and produces reasonably tight
    parameter settings for most mechanisms.

  - ``calibrate(epsilon, delta | zcdp_rho, **kwargs)`` is the high-level
    entry point defined once on the base class. When called with ``zcdp_rho``,
    it simply forwards to ``configure``. When called with ``(epsilon, delta)``,
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

  Subclasses must implement:

  - ``configure(zcdp_rho, **kwargs)``: set the mechanism's natural privacy
    parameter (e.g., Gaussian sigma) from a zCDP budget.
  - ``dp_event``: return the exact ``DpEvent`` characterizing the mechanism.
  - ``__call__``: run the mechanism on data.
  """

  @abc.abstractmethod
  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0, **kwargs: Any
  ) -> DPMechanism:
    """Returns a new mechanism configured with the given zCDP budget.

    Converts the zCDP budget into the mechanism's natural privacy parameter
    (e.g., Gaussian sigma) and returns a new frozen instance with that
    parameter set.

    Most mechanisms are pure zCDP and ignore ``delta``. Mechanisms that
    consume approximate DP budget (e.g., partition selection with Gaussian
    thresholding) should raise ``ValueError`` if ``delta`` is required but
    not provided (i.e., is 0).

    Args:
      zcdp_rho: The zCDP privacy budget (rho).
      delta: Approximate DP delta consumed by the mechanism itself (e.g., for
        thresholding). Defaults to 0 (pure zCDP). Mechanisms that need delta
        should raise if it is 0.
      **kwargs: Mechanism-specific hyperparameters.

    Returns:
      A new DPMechanism instance configured for the given budget.
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
    accountants = [
        dp_accounting.rdp.RdpAccountant,
        dp_accounting.pld.PLDAccountant,
    ]
    best_rho = None
    for make_accountant in accountants:
      try:
        rho = dp_accounting.calibrate_dp_mechanism(
            make_fresh_accountant=make_accountant,
            make_event_from_param=make_event_fn,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
        )
        if best_rho is None or rho > best_rho:
          best_rho = rho
      except (dp_accounting.UnsupportedEventError, NotImplementedError):
        continue

    if best_rho is None:
      raise dp_accounting.UnsupportedEventError(
          f'No accountant supports the dp_event: {make_event_fn(1.0)}'
      )
    return best_rho

  def calibrate(
      self,
      *,
      epsilon: float | None = None,
      delta: float | None = None,
      zcdp_rho: float | None = None,
      **kwargs: Any,
  ) -> DPMechanism:
    """Calibrate the mechanism to a target (epsilon, delta)-DP guarantee.

    Performs a binary search over zCDP budgets, calling ``configure`` at each
    candidate and inspecting the resulting ``dp_event``. Tries both RDP and
    PLD accounting and picks whichever gives the tightest result.

    .. deprecated::
      Passing ``zcdp_rho`` to ``calibrate`` is deprecated. Use
      ``configure(zcdp_rho=...)`` directly instead.

    Args:
      epsilon: Target epsilon for (epsilon, delta)-DP.
      delta: Target delta for (epsilon, delta)-DP.
      zcdp_rho: Deprecated. Direct zCDP budget. Use ``configure()`` instead.
      **kwargs: Forwarded to ``configure()``.

    Returns:
      A new calibrated DPMechanism instance.

    Raises:
      ValueError: If neither (epsilon, delta) nor zcdp_rho is specified, or
        if both are specified simultaneously.
    """
    if zcdp_rho is not None:
      if epsilon is not None or delta is not None:
        raise ValueError(
            'Specify either zcdp_rho or (epsilon, delta), not both.'
        )
      warnings.warn(
          'Passing zcdp_rho to calibrate() is deprecated. Use'
          ' configure(zcdp_rho=...) directly instead.',
          DeprecationWarning,
          stacklevel=2,
      )
      return self.configure(zcdp_rho=zcdp_rho, **kwargs)

    if epsilon is None or delta is None:
      raise ValueError('Must specify both epsilon and delta, or zcdp_rho.')

    optimal_rho = self._find_optimal_rho(
        make_event_fn=lambda rho: self.configure(
            zcdp_rho=rho, **kwargs
        ).dp_event,
        target_epsilon=epsilon,
        target_delta=delta,
    )
    return self.configure(zcdp_rho=optimal_rho, **kwargs)
