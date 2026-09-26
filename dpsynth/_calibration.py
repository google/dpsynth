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
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.
    accountant_fn: Optional zero-argument callable returning a fresh
      ``PrivacyAccountant``. If specified, calibrate using this accountant.

  Returns:
    A calibrated, runnable mechanism.

  Raises:
    ValueError: If epsilon is not positive.
    UnsupportedEventError: If no accountant supports the mechanism.
  """
  if epsilon <= 0:
    raise ValueError(f'Target epsilon must be positive, got {epsilon}.')

  def make_event_fn(rho: float) -> dp_accounting.DpEvent:
    base = config.configure(
        domain,
        zcdp_rho=rho,
        delta=delta,
        max_records_per_user=max_records_per_user,
    ).dp_event
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
      max_records_per_user=max_records_per_user,
  )
