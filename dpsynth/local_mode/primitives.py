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

These implementations only depend on numpy and scipy and utilize vectorized
operations for efficiency in single-machine environments.
"""

from __future__ import annotations

import dataclasses
import math

import dp_accounting
from dpsynth import api
from dpsynth.local_mode import _quantiles
import numpy as np
import scipy.stats

DPMechanism = api.DPMechanism


@dataclasses.dataclass
class HistogramResult:
  """Result of a differentially private histogram computation."""

  counts: np.ndarray


@dataclasses.dataclass
class PartitionSelectionResult:
  """Result of differentially private partition selection."""

  selected_partitions: np.ndarray
  estimated_counts: np.ndarray


_UNCALIBRATED_MSG = (
    '{param} has not been set. Set it directly or call calibrate().'
)


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


def select_partitions_gaussian_thresholding(
    rng: np.random.Generator,
    data: np.ndarray,
    gdp_budget: float,
    delta: float,
    min_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Selects partitions using Gaussian Thresholding (Weighted Gaussian).

  This implements Algorithm 2 from the DP-SIPS paper (Swanberg et al., 2023)
  under item-level DP. It is the simplest partition selection mechanism:

    1. Compute the histogram of partition counts.
    2. Add Gaussian noise calibrated to the privacy budget.
    3. Return partitions whose noisy count exceeds a threshold chosen to
       bound the false-positive probability per empty partition at delta.

  Under item-level DP each record is treated as a distinct user contributing
  to exactly one partition, so the histogram has L2 sensitivity 1.  The
  threshold is T = min_count + sigma * Phi^{-1}(1 - delta), following the
  paper's formula with max_part = 1 and a shift of (min_count - 1) to
  account for the minimum count guarantee.

  When ``min_count > 1``, partitions with true count below ``min_count``
  are pre-filtered and the threshold shifts up accordingly. The privacy
  guarantee is preserved: partitions where both neighboring datasets are
  above ``min_count`` are covered by the Gaussian mechanism, and the
  boundary case (one dataset at ``min_count - 1``, the other at
  ``min_count``) is covered by the same additive delta.

  Args:
    rng: A numpy random number generator.
    data: 1D array of integers, where each element is a partition ID.
    gdp_budget: Privacy budget in terms of squared Gaussian DP mu parameter
      (gdp_budget = mu^2 = 1 / sigma^2).
    delta: Failure probability (false positive bound per empty partition).
    min_count: Minimum true count for a partition to be eligible. Partitions
      with fewer occurrences in the data are never returned. Must be >= 1.

  Returns:
    A tuple containing:
      - selected_partitions: 1D array of partition IDs that passed the
        threshold.
      - estimated_counts: 1D array of noisy counts for each selected
        partition.
      - sigma: The standard deviation of the Gaussian noise added.
  """
  if gdp_budget <= 0 or delta <= 0:
    raise ValueError(f'{gdp_budget=} and {delta=} must be positive.')
  if min_count < 1:
    raise ValueError(f'{min_count=} must be >= 1.')

  sigma = 1.0 / np.sqrt(gdp_budget)

  if data.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), sigma

  unique_parts, counts = np.unique(data, return_counts=True)

  # Filter partitions below the minimum count before adding noise.
  above_min = counts >= min_count
  unique_parts, counts = unique_parts[above_min], counts[above_min]
  if unique_parts.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), sigma

  noisy_counts = counts + rng.normal(scale=sigma, size=counts.size)

  # Threshold shifted by (min_count - 1) relative to the base formula.
  # Base: T = 1 + sigma * ppf(1 - delta) bounds Pr[N(0, sigma^2) >= T] <= delta.
  # With min_count, worst-case non-eligible count is (min_count - 1), so
  # T' = min_count + sigma * ppf(1 - delta).
  threshold = float(min_count) + sigma * scipy.stats.norm.ppf(1.0 - delta)
  passed = noisy_counts >= threshold

  return unique_parts[passed], noisy_counts[passed], sigma


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

  Computes quantile edges by recursive median bisection on a dense histogram.
  The ``__call__`` method takes a 1D histogram of counts and returns the
  quantile edge values.

  Attributes:
    num_partitions: Number of quantile partitions (must be a power of 2).
    lower: Lower bound of the data domain.
    upper: Upper bound of the data domain (exclusive).
    grid_size: Number of uniformly spaced grid points.
  """

  num_partitions: int
  lower: float
  upper: float
  grid_size: int = 10_000_000
  _epsilon_levels: tuple[float, ...] | None = dataclasses.field(
      default=None, repr=False
  )

  @property
  def _num_levels(self) -> int:
    result = int(np.log2(self.num_partitions))
    if 2**result != self.num_partitions:
      raise ValueError(f'{self.num_partitions=} must be a power of 2.')
    return result

  @property
  def zcdp_rho(self) -> float:
    """Total zCDP rho consumed, derived from the per-level epsilons."""
    if self._epsilon_levels is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='_epsilon_levels'))
    return sum(e**2 / 8.0 for e in self._epsilon_levels)

  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0, epsilon_ratio: float = 2.0
  ) -> DPQuantiles:
    """Returns a copy calibrated to the given zCDP budget.

    Args:
      zcdp_rho: The zCDP privacy budget (rho).
      delta: Unused. Accepted for interface compatibility.
      epsilon_ratio: Factor by which epsilon grows at each deeper level.
    """
    if zcdp_rho <= 0:
      raise ValueError(f'zcdp_rho must be positive, got {zcdp_rho}.')
    levels = self._num_levels
    if levels == 0:
      return dataclasses.replace(self, _epsilon_levels=())
    rho_ratio = epsilon_ratio**2
    budget_weights = rho_ratio ** np.arange(levels)[::-1]
    rho_levels = zcdp_rho * budget_weights / budget_weights.sum()
    eps = np.sqrt(8.0 * rho_levels)
    return dataclasses.replace(self, _epsilon_levels=tuple(eps.tolist()))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the composed privacy event for the quantile computation."""
    if self._epsilon_levels is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='_epsilon_levels'))
    return dp_accounting.ComposedDpEvent([
        dp_accounting.ExponentialMechanismDpEvent(epsilon=float(eps))
        for eps in self._epsilon_levels
    ])

  def __call__(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> list[float]:
    """Returns quantile edges from a dense histogram of counts."""
    if self._epsilon_levels is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='_epsilon_levels'))
    return _quantiles.quantiles_from_histogram(
        rng,
        counts,
        self.lower,
        self.upper,
        epsilon_levels=np.asarray(self._epsilon_levels),
        grid_size=self.grid_size,
    )


@dataclasses.dataclass
class DPGaussianHistogram(DPMechanism):
  """Differentially private histogram via the Gaussian mechanism.

  The natural privacy parameter is ``sigma``, the noise standard deviation.
  The conversion from zCDP is ``sigma = sqrt(0.5 / zcdp_rho)``.

  Attributes:
    domain_size: Number of categories in the histogram domain.
    sigma: Gaussian noise standard deviation. Set directly or via ``configure``.
  """

  domain_size: int
  sigma: float | None = None

  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0
  ) -> DPGaussianHistogram:
    """Returns a copy with sigma derived from the zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(
      self, rng: np.random.Generator, counts: np.ndarray
  ) -> HistogramResult:
    """Adds Gaussian noise to the given counts."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    noise = rng.normal(scale=self.sigma, size=self.domain_size)
    return HistogramResult(counts=counts.astype(float) + noise)


@dataclasses.dataclass
class DPGaussianCount(DPMechanism):
  """Differentially private count via the Gaussian mechanism."""

  sigma: float | None = None

  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0
  ) -> DPGaussianCount:
    """Returns a copy with sigma derived from the zCDP budget."""
    return dataclasses.replace(self, sigma=math.sqrt(0.5 / zcdp_rho))

  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    """Returns the Gaussian privacy event for this mechanism."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    return dp_accounting.GaussianDpEvent(noise_multiplier=self.sigma)

  def __call__(self, rng: np.random.Generator, data: np.ndarray) -> float:
    """Returns a noisy count of len(data) + Gaussian noise."""
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
    min_count: Minimum true count for a partition to be returned.
    sigma: Gaussian noise standard deviation. Set directly or via ``configure``.
  """

  delta: float
  min_count: int = 1
  sigma: float | None = None

  def configure(
      self, *, zcdp_rho: float, delta: float = 0.0
  ) -> DPPartitionSelection:
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
  ) -> PartitionSelectionResult:
    """Runs partition selection on integer-encoded partition IDs."""
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    gdp_budget = np.inf if self.sigma == 0.0 else 1.0 / (self.sigma**2)
    parts, counts, _ = select_partitions_gaussian_thresholding(
        rng, data, gdp_budget, self.delta, min_count=self.min_count
    )
    return PartitionSelectionResult(
        selected_partitions=parts, estimated_counts=counts
    )

  def from_summary(
      self,
      rng: np.random.Generator,
      counts: np.ndarray,
  ) -> PartitionSelectionResult:
    """Single-round partition selection from pre-aggregated counts.

    Args:
      rng: A numpy random number generator.
      counts: 1D array of per-partition counts.

    Returns:
      A PartitionSelectionResult with indices into `counts` as the
      selected_partitions and their noisy counts.
    """
    if self.sigma is None:
      raise ValueError(_UNCALIBRATED_MSG.format(param='sigma'))
    above_min = counts >= self.min_count
    eligible_idx = np.where(above_min)[0]
    eligible_counts = counts[above_min].astype(float)
    noisy_counts = eligible_counts + rng.normal(
        scale=self.sigma, size=len(eligible_counts)
    )
    threshold = float(self.min_count) + self.sigma * scipy.stats.norm.ppf(
        1.0 - self.delta
    )
    passed = noisy_counts >= threshold
    return PartitionSelectionResult(
        selected_partitions=eligible_idx[passed],
        estimated_counts=noisy_counts[passed],
    )
