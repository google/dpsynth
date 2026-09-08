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

"""Calibration utilities for differentially private mechanisms."""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

import dp_accounting


def with_group_size(
    event: dp_accounting.DpEvent, group_size: int
) -> dp_accounting.DpEvent:
  """Lifts a DpEvent from record-level (group_size=1) to the given group_size.

  Args:
    event: The record-level ``DpEvent`` to transform.
    group_size: Positive integer bound on the number of records per group/user.

  Returns:
    A ``DpEvent`` characterizing the privacy guarantee for groups of size
    ``group_size``.

  Raises:
    ValueError: If ``group_size < 1``.
    UnsupportedEventError: If ``group_size > 1`` and ``event`` (or a nested
      sub-event) does not support group-size scaling.
  """
  if group_size < 1:
    raise ValueError(f'group_size must be >= 1, got {group_size}.')
  if group_size == 1:
    return event
  if isinstance(
      event, (dp_accounting.NoOpDpEvent, dp_accounting.NonPrivateDpEvent)
  ):
    return event
  if isinstance(event, dp_accounting.GaussianDpEvent):
    return dp_accounting.GaussianDpEvent(event.noise_multiplier / group_size)
  if isinstance(event, dp_accounting.LaplaceDpEvent):
    return dp_accounting.LaplaceDpEvent(event.noise_multiplier / group_size)
  if isinstance(event, dp_accounting.ExponentialMechanismDpEvent):
    return dp_accounting.ExponentialMechanismDpEvent(event.epsilon * group_size)
  if isinstance(event, dp_accounting.ZCDpEvent):
    return dp_accounting.ZCDpEvent(
        rho=event.rho * group_size**2, xi=event.xi * group_size
    )
  if isinstance(event, dp_accounting.ComposedDpEvent):
    scaled = [with_group_size(e, group_size) for e in event.events]
    return dp_accounting.ComposedDpEvent(scaled)
  if isinstance(event, dp_accounting.SelfComposedDpEvent):
    inner = with_group_size(event.event, group_size)
    return dp_accounting.SelfComposedDpEvent(inner, event.count)
  # (EpsilonDeltaDpEvent) for group_size > 1.
  raise dp_accounting.UnsupportedEventError(
      f'with_group_size does not support {type(event).__name__} for'
      f' group_size={group_size}.'
  )


def calibrate(
    config: Any,
    domain: Any = None,
    /,
    *,
    epsilon: float,
    delta: float,
    poisson_sampling_prob: float = 1.0,
    max_records_per_user: int = 1,
    accountant_fn: Callable[[], dp_accounting.PrivacyAccountant] | None = None,
) -> Any:
  """Calibrates a mechanism config to a target (epsilon, delta)-DP guarantee.

  Performs a binary search over zCDP budgets, calling ``config.configure`` at
  each candidate and inspecting the resulting ``dp_event``. If ``accountant_fn``
  is provided, calibrates with that accountant. Otherwise tries both RDP and PLD
  accounting and picks whichever gives the tightest result.

  Args:
    config: A ``MechanismConfig`` instance to calibrate.
    domain: Optional domain specification, forwarded to ``config.configure()``.
    epsilon: Target epsilon for (epsilon, delta)-DP.
    delta: Target delta for (epsilon, delta)-DP.
    poisson_sampling_prob: If specified, calibrate the mechanism assuming the
      input data is subsampled with the given probability. The actual sampling
      is **NOT** handled internally by the calibrated mechanism.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Lifts the candidate mechanism's ``dp_event`` via
      ``with_group_size`` during calibration so the resulting mechanism provides
      user-level rather than record-level DP. Soundness relies on the caller
      enforcing this bound.
    accountant_fn: Optional zero-argument callable returning a fresh
      ``PrivacyAccountant``. If specified, calibrate using this accountant.

  Returns:
    A calibrated, runnable mechanism.

  Raises:
    ValueError: If epsilon is not positive or max_records_per_user < 1.
    UnsupportedEventError: If no accountant supports the mechanism.
  """
  if epsilon <= 0:
    raise ValueError(f'Target epsilon must be positive, got {epsilon}.')
  if max_records_per_user < 1:
    raise ValueError(
        f'max_records_per_user must be >= 1, got {max_records_per_user}.'
    )

  def make_event_fn(rho: float) -> dp_accounting.DpEvent:
    base = config.configure(domain, zcdp_rho=rho, delta=delta).dp_event
    base = with_group_size(base, max_records_per_user)
    sampled = dp_accounting.PoissonSampledDpEvent(poisson_sampling_prob, base)
    return base if poisson_sampling_prob == 1.0 else sampled

  init_guess = epsilon**2
  if accountant_fn is not None:
    accountants = {'accountant': accountant_fn}
  else:
    accountants = {
        'PLDAccountant': functools.partial(
            dp_accounting.pld.PLDAccountant,
            value_discretization_interval=min(1e-4, 1e-1 * epsilon),
        ),
        'RdpAccountant': dp_accounting.rdp.RdpAccountant,
    }

  rhos = {}
  errors = {}
  for name, fn in accountants.items():
    try:
      rhos[name] = dp_accounting.calibrate_dp_mechanism(
          make_fresh_accountant=fn,
          make_event_from_param=make_event_fn,
          target_epsilon=epsilon,
          target_delta=delta,
          bracket_interval=dp_accounting.LowerEndpointAndGuess(0.0, init_guess),  # pyrefly: ignore[bad-argument-count]
      )
    except (dp_accounting.UnsupportedEventError, NotImplementedError) as e:
      errors[name] = e

  if not rhos:
    if accountant_fn is not None:
      raise errors['accountant']
    raise dp_accounting.UnsupportedEventError(
        'No accountant supports the mechanism:\n'
        f'  PLDAccountant error: {errors.get("PLDAccountant")}\n'
        f'  RdpAccountant error: {errors.get("RdpAccountant")}'
    )

  optimal_rho = max(rhos.values())
  return config.configure(
      domain,
      zcdp_rho=optimal_rho,
      delta=delta,
  )
