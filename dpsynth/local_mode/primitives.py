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

"""Differentially private primitives for quantiles and partition selection.

These implementations only depened on numpy and scipy and utilize vectorized
operations for efficiency in single-machine environments.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import dp_accounting
import numpy as np
import scipy.special
import scipy.stats


class DPMechanism(abc.ABC):
  """Abstract base class for differentially private mechanisms.

  A DPMechanism encapsulates a randomized algorithm that satisfies differential
  privacy. Usage follows a three-phase pattern:

  1. **Configure**: Create the mechanism with algorithm-specific parameters by
     calling the mechanism's class constructor.
  2. **Calibrate**: Call ``calibrate(zcdp_rho=...)`` to bind a privacy budget,
     returning a new mechanism instance whose natural privacy parameter (e.g.,
     Gaussian sigma, exponential mechanism epsilon) has been set accordingly.
  3. **Run**: Call the calibrated mechanism on data via ``__call__``.

  Subclasses should be parameterized by their **natural** privacy parameter
  (e.g., ``sigma`` for the Gaussian mechanism, ``epsilon`` for the exponential
  mechanism). The ``calibrate`` method converts from the universal zCDP budget
  to the mechanism's natural parameter.

  **Why zCDP for calibration.** Calibrating to zCDP rho makes it easy to split
  a privacy budget across a heterogeneous composition of mechanisms: simply
  divide rho in any ratio and each share is a valid zCDP guarantee.

  **Tight accounting via DpEvents.** For tight privacy accounting, do not rely
  on the zCDP guarantee directly. Instead, use the ``dp_event`` property, which
  precisely characterizes the exact DP properties of the underlying mechanism.
  Compose the raw ``DpEvent`` objects using ``dp_accounting`` for tight
  PLD-based accounting.

  **Typical pattern.** Calibrate early mechanisms to zCDP fractions of the
  total budget. For the last mechanism, perform a tight PLD-based calibration
  of its privacy parameters to target whatever overall privacy guarantee is
  desired (which may or may not be zCDP).
  """

  @abc.abstractmethod
  def calibrate(self, *, zcdp_rho: float) -> DPMechanism:
    """Returns a new mechanism calibrated to the given zCDP budget.

    Converts the zCDP budget ``rho`` into the mechanism's natural privacy
    parameter and returns a new instance with that parameter set.

    Args:
      zcdp_rho: The zCDP privacy budget (rho).

    Returns:
      A new DPMechanism instance calibrated to the given budget.
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


_UNCALIBRATED_MSG = (
    '{param} has not been set. Set it directly or call calibrate().'
)


def _median(
    rng: np.random.Generator,
    data: np.ndarray,
    lower: float,
    upper: float,
    epsilon: float,
    jitter_multiple: float = 1e-4,
) -> float:
  """Computes a differentially private median using the exponential mechanism.

  This function implements the continuous rank-based DP median. Candidates are
  the intervals between sorted data points. The utility of an interval is based
  on the distance of its rank from N/2.

  Args:
    rng: A numpy random number generator.
    data: 1D array of numerical data.
    lower: Lower bound for the data.
    upper: Upper bound for the data.
    epsilon: Exponential mechanism privacy parameter.
    jitter_multiple: Multiplier for the jitter scale, relative to upper-lower.

  Returns:
    A differentially private median estimate.
  """
  if lower > upper:
    raise ValueError(f'{lower=} cannot be greater than {upper=}.')

  clamped_data = np.clip(data, lower, upper)
  n = clamped_data.size

  if epsilon == np.inf:
    if n == 0:
      return (lower + upper) / 2
    return float(np.median(clamped_data))

  # Jitter size proportional to range. A small jitter makes duplicates unique
  # and gives them non-zero length intervals, allowing them to be sampled.
  jitter_scale = (upper - lower) * jitter_multiple
  jitter = rng.uniform(-jitter_scale, jitter_scale, size=clamped_data.size)
  jittered_data = np.clip(clamped_data + jitter, lower, upper)

  sorted_data = np.sort(jittered_data)
  n = sorted_data.size
  x = np.r_[lower, sorted_data, upper]
  lengths = np.diff(x)
  ranks = np.arange(n + 1)
  utilities = -np.abs(ranks - n / 2)

  # Compute output probabilities for each interval.
  probs = scipy.special.softmax(np.log(lengths) + epsilon * utilities)

  # Sample an interval index, and a value uniformly from the interval.
  interval_idx = rng.choice(n + 1, p=probs)
  v_min = x[interval_idx]
  v_max = x[interval_idx + 1]
  return rng.uniform(v_min, v_max)


def _quantile_epsilon_levels(
    zcdp_rho: float, num_levels: int, epsilon_ratio: float = 2.0
) -> np.ndarray:
  """Computes per-level exponential mechanism epsilons for DP quantiles.

  At each level of the recursive bisection, each data point participates in
  exactly one median computation (parallel composition), so the privacy cost
  at each level is that of a single exponential mechanism invocation.  Deeper
  levels operate on half the data of the level above, halving the signal.  To
  keep the noise proportional to the signal, epsilon grows by a factor of
  ``epsilon_ratio`` at each deeper level.  Since rho = epsilon^2 / 8 for the
  exponential mechanism under zCDP, scaling epsilon by ``epsilon_ratio``
  scales rho by ``epsilon_ratio**2``.

  Args:
    zcdp_rho: Total zCDP privacy budget.
    num_levels: Number of levels in the quantile tree. Number of buckets is ``2
      ** num_levels``.
    epsilon_ratio: Factor by which epsilon grows at each deeper level. A value
      of 2 means epsilon doubles per level (noise halves), preserving the
      signal-to-noise ratio as data is split.

  Returns:
    A length ``num_levels`` array of per-level epsilons, ordered from the
    deepest (finest) level to the shallowest (coarsest).
  """
  if num_levels == 0:
    return np.array([])
  # Since rho = eps^2 / 8, scaling epsilon by r scales rho by r^2.
  rho_ratio = epsilon_ratio**2
  budget_weights = rho_ratio ** np.arange(num_levels)[::-1]
  rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
  return np.sqrt(8 * rho_levels)


def _quantiles(
    rng: np.random.Generator,
    data: np.ndarray,
    lower: float,
    upper: float,
    epsilon_levels: np.ndarray,
) -> list[float]:
  """Computes uniformly spaced differentially private quantiles.

  This function is a ``len(epsilon_levels)``-level composition of the
  exponential mechanism.  The number of partitions is inferred as
  ``2 ** len(epsilon_levels)``.

  Args:
    rng: A numpy random number generator.
    data: 1D array of numerical data.
    lower: Lower bound for the data.
    upper: Upper bound for the data.
    epsilon_levels: Per-level exponential mechanism epsilons, as returned by
      ``_quantile_epsilon_levels``.

  Returns:
    A list of ``2 ** len(epsilon_levels) - 1`` sorted private quantile
    estimates.
  """
  levels = len(epsilon_levels)
  if levels == 0:
    return []

  def quantiles_rec(current_data, curr_lower, curr_upper, current_depth):
    if current_depth == 0:
      return []

    eps = epsilon_levels[current_depth - 1]
    med = _median(rng, current_data, curr_lower, curr_upper, eps)

    left_mask = current_data <= med
    left_data = current_data[left_mask]
    right_data = current_data[~left_mask]

    left_points = quantiles_rec(left_data, curr_lower, med, current_depth - 1)
    right_points = quantiles_rec(right_data, med, curr_upper, current_depth - 1)

    return left_points + [med] + right_points

  return quantiles_rec(data, lower, upper, levels)


def _contribution_bound(prng, user_ids, max_part):
  """Return array idx where all ids appear <=max_part times in user_ids[idx]."""
  # Sort by ID + noise to shuffle within groups. Then find where
  # groups start/end, and select the first max_part elements of each group.
  # Use lexsort with random keys to shuffle string/object IDs safely.
  random_keys = prng.uniform(size=user_ids.size)
  idx = np.lexsort((random_keys, user_ids))
  sorted_ids = user_ids[idx]
  diff = np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]
  kernel = np.ones(max_part, dtype=bool)
  # This convolution determines if any of previous max_part elements are True.
  mask = np.convolve(diff, kernel, mode='full')[: user_ids.size]
  return idx[mask]


def _get_threshold(delta, sigma, max_part):
  ks = np.arange(1, max_part + 1)
  failure_prob = (1 - delta) ** (1 / ks)
  thresholds = 1 / np.sqrt(ks) + sigma * scipy.stats.norm.ppf(failure_prob)
  return thresholds.max()


def _select_partitions_sips(
    rng: np.random.Generator,
    data: np.ndarray,
    gdp_budget: float,
    delta: float,
    num_rounds: int | None = None,
    user_ids: np.ndarray | None = None,
    max_part: int = 1,
    allocation_factor: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Implements the DP-SIPS mechanism for partition selection.

  Args:
    rng: A numpy random number generator.
    data: 1D array of integers, where each element is a partition ID.
    gdp_budget: Total privacy budget in terms of squared Gaussian DP mu
      parameter (gdp_budget = mu^2 = 1 / sigma^2).
    delta: Failure probability (false positive bound per empty partition).
    num_rounds: Number of rounds to run the mechanism. Defaults to 1 if user_ids
      is None, and 3 otherwise.
    user_ids: Optional 1D array of user IDs corresponding to data. If provided,
      user-level DP is guaranteed. If None, item-level DP is guaranteed
      (assuming each record is a unique user).
    max_part: Maximum number of partitions any single user can contribute to in
      a single round.
    allocation_factor: Factor by which to increase the budget each round.

  Returns:
    A tuple containing:
      - selected_partitions: 1D array of unique partition IDs that passed the
        threshold.
      - estimated_counts: 1D array of noisy (or weighted noisy) counts for each
        selected partition in the round it was discovered.
      - standard_deviation: A single float representing the uniform standard
        deviation of the noise added to the estimated counts.
  """
  if num_rounds is None:
    num_rounds = 1 if user_ids is None else 3
  if num_rounds <= 0:
    raise ValueError(f'num_rounds ({num_rounds}) must be greater than 0.')
  if gdp_budget <= 0 or delta <= 0:
    raise ValueError(f'{gdp_budget=} and {delta=} must be positive.')

  fractions = allocation_factor ** np.arange(num_rounds)[::-1]
  fractions /= fractions.sum()
  gdp_rounds, delta_rounds = gdp_budget * fractions, delta * fractions
  sigma_rounds = 1.0 / np.sqrt(gdp_rounds)
  max_sigma = float(np.max(sigma_rounds))

  if data.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), max_sigma

  if user_ids is None:
    user_ids = np.arange(data.size)
  if user_ids.size != data.size:
    raise ValueError('user_ids must have the same size as data.')

  combined = np.stack((user_ids, data), axis=1)
  unique_combined = np.unique(combined, axis=0)
  rem_user_ids = unique_combined[:, 0]
  rem_partitions = unique_combined[:, 1]

  selected_partitions = []
  selected_counts = []
  for i in range(num_rounds):
    if rem_partitions.size == 0:
      break

    threshold = _get_threshold(delta_rounds[i], sigma_rounds[i], max_part)

    mask = _contribution_bound(rng, rem_user_ids, max_part)
    curr_user_ids = rem_user_ids[mask]
    curr_partitions = rem_partitions[mask]

    unique_users, user_counts = np.unique(curr_user_ids, return_counts=True)
    user_to_count = dict(zip(unique_users, user_counts))
    weights = np.array([1.0 / user_to_count[u] ** 0.5 for u in curr_user_ids])

    unique_parts, inverse_indices = np.unique(
        curr_partitions, return_inverse=True
    )
    weighted_counts = np.bincount(inverse_indices, weights=weights)
    noised_counts = rng.normal(weighted_counts, scale=sigma_rounds[i])

    passed_mask = noised_counts >= threshold
    round_selections = unique_parts[passed_mask]
    round_counts = noised_counts[passed_mask]
    if round_selections.size > 0:
      selected_partitions.append(round_selections)
      selected_counts.append(round_counts)

      mask = ~np.isin(rem_partitions, round_selections)
      rem_user_ids = rem_user_ids[mask]
      rem_partitions = rem_partitions[mask]

  if not selected_partitions:
    return (
        np.empty(0, dtype=data.dtype),
        np.empty(0, dtype=float),
        max_sigma,
    )
  selected_partitions = np.concatenate(selected_partitions)
  selected_counts = np.concatenate(selected_counts)
  return selected_partitions, selected_counts, max_sigma


# ---------------------------------------------------------------------------
# DPMechanism subclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DPQuantiles(DPMechanism):
  """Differentially private quantiles via composed exponential mechanisms.

  This is a ``log2(num_partitions)``-level composition of the exponential
  mechanism.  Use ``calibrate`` to set privacy parameters before calling.

  Attributes:
    lower: Lower bound for the data domain.
    upper: Upper bound for the data domain.
    num_partitions: Number of partitions (must be a power of 2).
  """

  lower: float
  upper: float
  num_partitions: int
  _epsilon_levels: Sequence[float] | None = dataclasses.field(
      default=None, repr=False
  )

  def __post_init__(self):
    if self._epsilon_levels is not None:
      if len(self._epsilon_levels) != self._num_levels:
        raise ValueError(
            f'len(epsilon_levels)={len(self._epsilon_levels)} must equal'
            f' log2(num_partitions)={self._num_levels}.'
        )

  @property
  def _num_levels(self) -> int:
    result = int(np.log2(self.num_partitions))
    if 2**result != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')
    return result

  def calibrate(
      self, *, zcdp_rho: float, epsilon_ratio: float = 2.0
  ) -> DPQuantiles:
    """Returns a copy calibrated to the given zCDP budget.

    Args:
      zcdp_rho: The zCDP privacy budget (rho).
      epsilon_ratio: Factor by which epsilon grows at each deeper level. A value
        of 2 (default) means epsilon doubles per level, preserving the
        signal-to-noise ratio as data is halved at each split.

    Returns:
      A new calibrated ``DPQuantiles`` instance.
    """
    eps = _quantile_epsilon_levels(zcdp_rho, self._num_levels, epsilon_ratio)
    return dataclasses.replace(self, _epsilon_levels=tuple(eps.tolist()))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for this mechanism."""
    if self._epsilon_levels is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='_epsilon_levels'))
    return dp_accounting.ComposedDpEvent([
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self._epsilon_levels
    ])

  def __call__(self, rng: np.random.Generator, data: np.ndarray) -> list[float]:
    """Computes differentially private quantiles."""
    if self._epsilon_levels is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='_epsilon_levels'))
    return _quantiles(
        rng, data, self.lower, self.upper, np.asarray(self._epsilon_levels)
    )


@dataclasses.dataclass
class DPGaussianHistogram(DPMechanism):
  """Differentially private histogram via the Gaussian mechanism.

  The natural privacy parameter is ``sigma``, the noise standard deviation.
  The conversion from zCDP is ``sigma = sqrt(0.5 / zcdp_rho)``.

  Attributes:
    domain_size: Number of categories in the histogram domain.
    sigma: Gaussian noise standard deviation. Set directly or via ``calibrate``.
  """

  domain_size: int
  sigma: float | None = None

  def calibrate(self, *, zcdp_rho: float) -> DPGaussianHistogram:
    """Returns a copy with sigma derived from the zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(self, rng: np.random.Generator, data: np.ndarray) -> np.ndarray:
    """Computes a differentially private histogram."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    return np.bincount(data, minlength=self.domain_size) + rng.normal(
        scale=self.sigma, size=self.domain_size
    )


@dataclasses.dataclass
class DPGaussianCount(DPMechanism):
  """Differentially private count via the Gaussian mechanism.

  Returns a noisy count of the number of records in the input data. The
  conversion from zCDP is ``sigma = sqrt(0.5 / zcdp_rho)``.

  Attributes:
    sigma: Gaussian noise standard deviation. Set directly or via ``calibrate``.
  """

  sigma: float | None = None

  def calibrate(self, *, zcdp_rho: float) -> DPGaussianCount:
    """Returns a copy with sigma derived from the zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(self, rng: np.random.Generator, data: np.ndarray) -> float:
    """Returns a differentially private count of records.

    Args:
      rng: A numpy random number generator.
      data: 1D array of data records.

    Returns:
      The noisy count (true count + Gaussian noise).
    """
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    return float(len(data) + rng.normal(scale=self.sigma))


@dataclasses.dataclass
class DPPartitionSelection(DPMechanism):
  """Differentially private partition selection via Gaussian Thresholding.

  Because partition selection is an approximate (delta > 0) mechanism, the
  ``dp_event`` composes the Gaussian event with an ``EpsilonDeltaDpEvent``
  representing the additive thresholding delta.

  Attributes:
    delta: Failure probability for the thresholding step.
    sigma: Gaussian noise standard deviation. Set directly or via ``calibrate``.
  """

  delta: float
  sigma: float | None = None

  def calibrate(self, *, zcdp_rho: float) -> DPPartitionSelection:
    """Returns a copy with sigma derived from the zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the privacy event including thresholding delta."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    main_event = dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)
    failure_event = dp_accounting.dp_event.EpsilonDeltaDpEvent(0, self.delta)
    return dp_accounting.ComposedDpEvent([main_event, failure_event])

  def __call__(
      self, rng: np.random.Generator, data: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Runs partition selection on integer-encoded partition IDs.

    Args:
      rng: A numpy random number generator.
      data: 1D array of integer partition IDs.

    Returns:
      A tuple of (selected_partitions, noisy_counts, sigma).
    """
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    if data.size == 0:
      return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), self.sigma
    unique_parts, counts = np.unique(data, return_counts=True)
    noisy_counts = counts + rng.normal(scale=self.sigma, size=counts.size)
    # Threshold ensures an empty partition passes with probability <= delta.
    threshold = 1.0 + self.sigma * scipy.stats.norm.ppf(1.0 - self.delta)
    passed = noisy_counts >= threshold
    return unique_parts[passed], noisy_counts[passed], self.sigma
