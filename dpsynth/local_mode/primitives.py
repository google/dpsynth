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

"""Differentially private primitives for local mode synthetic data generation.

Design Decisions:

1) Summary Statistics: All primitives are defined strictly in terms of summary
   statistics (counts, histograms, etc.) rather than raw datasets. This ensures
   computational efficiency and separation of data aggregation from DP
   calibration.
2) Pure Functions: Primitives are pure functions that take numpy inputs and
return
   numpy outputs. State and DP accounting, when needed, are managed externally.

Privacy Characterizations:

- Quantiles (via `_quantiles.quantiles_from_histogram`): This mechanism is a
composition
  of exponential mechanisms with parameters taken from `epsilon_levels`.
- Partition Selection (`_select_partitions_sips`): DP-SIPS mechanism for
discovering
  open-set vocabulary, utilizing Gaussian Thresholding combined with privacy
  filtering.
- Gaussian Thresholding (`select_partitions_gaussian_thresholding`): This
mechanism adds
  Gaussian noise to counts and tests against a threshold that provides a (0,
  delta)
  bound on false positives for unpopulated partitions.
- Gaussian Noise (`add_gaussian_noise`): This is a standard Gaussian mechanism
applied
  to the input summary statistics (e.g. counts).
"""

from __future__ import annotations

from typing import overload

import numpy as np
import scipy.stats


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
    max_records_per_user: int = 1,
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

  When ``max_records_per_user > 1`` the mechanism switches to user-level DP
  via a naive, conservative reduction: a single user may place all ``k``
  records in one partition, so the histogram's L2 sensitivity grows to ``k``.
  Both the noise standard deviation and the threshold are scaled by ``k``,
  which is equivalent to running the item-level mechanism with ``k`` times the
  sigma and threshold. This is sound but suboptimal -- a user->record mapping
  would allow tighter per-user contribution bounding (e.g. capping the number
  of distinct partitions a user touches) and hence far better utility.

  Args:
    rng: A numpy random number generator.
    data: 1D array of integers, where each element is a partition ID.
    gdp_budget: Privacy budget in terms of squared Gaussian DP mu parameter
      (gdp_budget = mu^2 = 1 / sigma^2).
    delta: Failure probability (false positive bound per empty partition).
    min_count: Minimum true count for a partition to be eligible. Partitions
      with fewer occurrences in the data are never returned. Must be >= 1.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes. Added noise (and mechanism sensitivity) is scaled by
      this factor to provide user-level rather than record-level DP; the privacy
      accounting is unchanged. Soundness relies on the caller enforcing this
      bound.

  Returns:
    A tuple containing:
      - selected_partitions: 1D array of partition IDs that passed the
        threshold.
      - estimated_counts: 1D array of noisy counts for each selected
        partition.
      - stddev: The standard deviation of the Gaussian noise added
        (``max_records_per_user * sigma``).
  """
  if gdp_budget <= 0 or delta <= 0 or delta > 1:
    raise ValueError(f'{gdp_budget=} and {delta=} must be positive.')
  if min_count < 1:
    raise ValueError(f'{min_count=} must be >= 1.')

  stddev = max_records_per_user / np.sqrt(gdp_budget)

  if data.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), stddev

  unique_parts, counts = np.unique(data, return_counts=True)

  # Filter partitions below the minimum count before adding noise.
  above_min = counts >= min_count
  unique_parts, counts = unique_parts[above_min], counts[above_min]
  if unique_parts.size == 0:
    return np.empty(0, dtype=data.dtype), np.empty(0, dtype=float), stddev

  noisy_counts = counts + rng.normal(scale=stddev, size=counts.size)

  # A partition that is a candidate here but absent from a neighbor drives the
  # per-partition false-positive budget `delta`. One user contributes up to
  # k = max_records_per_user records, so (i) the noise std is
  # stddev = k / sqrt(gdp_budget), and (ii) such a partition's true count can
  # reach (min_count - 1) + k -- the neighbor sits just under the eligibility
  # cutoff at min_count - 1 and the user piles all k records into it. Bounding
  #   Pr[(min_count - 1 + k) + N(0, stddev^2) >= T] <= delta
  # gives T = (min_count + k - 1) + stddev * ppf(1 - delta).
  base = float(max_records_per_user + min_count - 1)
  threshold = base + stddev * scipy.stats.norm.ppf(1.0 - delta)
  passed = noisy_counts >= threshold
  # unique_parts is sorted (see np.unique), so the output order is determinstic.
  return unique_parts[passed], noisy_counts[passed], stddev


def ensure_public_partitions(
    rng: np.random.Generator,
    selected: np.ndarray,
    counts: np.ndarray,
    stddev: float,
    public: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Ensures public partition IDs appear in the selected set.

  For any public ID not already in ``selected``, appends it with a noisy
  count drawn from N(0, stddev²), consistent with the Gaussian mechanism
  applied to an empty partition.

  Args:
    rng: A numpy random number generator.
    selected: 1D array of already-selected partition IDs.
    counts: 1D array of noisy counts parallel to ``selected``.
    stddev: Gaussian noise standard deviation used by the mechanism.
    public: 1D array of public partition IDs to guarantee.

  Returns:
    A (selected, counts) tuple with missing public partitions appended.
  """
  missing_mask = ~np.isin(public, selected)
  missing = public[missing_mask]
  if missing.size == 0:
    return selected, counts
  noise = rng.normal(scale=stddev, size=missing.size)
  all_selected = np.concatenate([selected, missing])
  all_counts = np.concatenate([counts, noise])
  # Sort by partition key to ensure deterministic order and avoid leaking
  # which partitions were missing.
  order = np.argsort(all_selected)
  return all_selected[order], all_counts[order]


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
  if gdp_budget <= 0 or delta <= 0 or delta > 1:
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
# Simple noisy counting functions
# ---------------------------------------------------------------------------


@overload
def add_gaussian_noise(
    rng: np.random.Generator,
    counts: float | int,
    sigma: float,
    max_records_per_user: int = 1,
) -> float:
  ...


@overload
def add_gaussian_noise(
    rng: np.random.Generator,
    counts: np.ndarray,
    sigma: float,
    max_records_per_user: int = 1,
) -> np.ndarray:
  ...


@overload
def add_gaussian_noise(
    rng: np.random.Generator,
    counts: np.ndarray | float | int,
    sigma: float,
    max_records_per_user: int = 1,
) -> float | np.ndarray:
  ...


def add_gaussian_noise(
    rng: np.random.Generator,
    counts: np.ndarray | float | int,
    sigma: float,
    max_records_per_user: int = 1,
) -> float | np.ndarray:
  """Adds Gaussian noise to scalar, 1D array, or multi-dimensional array counts.

  Args:
    rng: A numpy random number generator.
    counts: The true count(s). Can be a scalar, 1D array, or multi-dimensional
      array.
    sigma: The Gaussian noise standard deviation.
    max_records_per_user: Assumed upper bound on the number of records a single
      user contributes, used to scale the noise for user-level DP.

  Returns:
    The noisy count(s) with the same shape as `counts`.
  """
  if sigma < 0:
    raise ValueError(f'sigma must be positive, got {sigma}')
  stddev = max_records_per_user * sigma

  if isinstance(counts, (int, float, np.generic)):
    noise = float(rng.normal(scale=stddev))
    return float(counts) + noise

  counts_arr = np.asarray(counts, dtype=float)
  noise = rng.normal(scale=stddev, size=counts_arr.shape)
  return counts_arr + noise
