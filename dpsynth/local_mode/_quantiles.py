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

"""DP quantiles from dense histograms via recursive median bisection.

This module computes differentially private quantile edges from a dense
histogram of counts, using the discrete exponential mechanism.  The primary
use case is a two-pass pipeline: a first pass computes a dense histogram
over a fine-grained grid, then ``quantiles_from_histogram`` finds DP
quantiles from that histogram without touching individual records.

Public API:
  - ``quantiles_from_histogram``: DP quantiles via recursive median splits.
"""

from __future__ import annotations

import numpy as np
import scipy.special


def _median_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    epsilon: float,
) -> int:
  """Returns the index of a DP median within a dense histogram.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram counts.
    epsilon: Exponential mechanism privacy parameter for this level.

  Returns:
    The index of the selected median grid point within ``counts``.
  """
  total_points = len(counts)
  if total_points == 0:
    return 0
  n = counts.sum()
  target = n / 2.0
  cumsum = np.cumsum(counts)

  # Infinite budget = exact median, useful for testing.
  if epsilon == np.inf:
    return int(np.searchsorted(cumsum, target))

  # Score u(v) = -dist(target, [L_v, R_v]), sensitivity 1/2.
  left_ranks = np.r_[0, cumsum[:-1]]
  scores = -np.maximum(0, np.maximum(left_ranks - target, target - cumsum))

  probs = scipy.special.softmax(epsilon * scores)
  return int(rng.choice(total_points, p=probs))


def quantiles_from_histogram(
    rng: np.random.Generator,
    counts: np.ndarray,
    lower: float,
    upper: float,
    epsilon_levels: np.ndarray,
    grid_size: int = 10_000_000,
) -> list[float]:
  """Computes DP quantiles from a dense histogram via recursive median splits.

  Uses the discrete exponential mechanism to recursively find medians,
  splitting the histogram at each level to produce ``num_buckets - 1``
  quantile edges.  The number of buckets is ``2 ** len(epsilon_levels)``.

  Args:
    rng: A numpy random number generator.
    counts: Dense 1D histogram of shape ``(grid_size,)``.
    lower: Lower bound of the data domain.
    upper: Upper bound of the data domain (exclusive).
    epsilon_levels: Per-level exponential mechanism epsilons, ordered from the
      deepest (finest) level to the shallowest (coarsest).
    grid_size: Number of uniformly spaced grid points spanning ``[lower,
      upper]``.

  Returns:
    A sorted list of ``2 ** len(epsilon_levels) - 1`` quantile edge values.
  """
  levels = len(epsilon_levels)
  if levels == 0:
    return []

  # Uniform grid: counts[i] corresponds to value lower + i * delta.
  delta = (upper - lower) / (grid_size - 1)

  def _rec(lo_idx, hi_idx, depth):
    if depth == 0 or lo_idx >= hi_idx:
      return []
    sub_counts = counts[lo_idx:hi_idx]
    median_local = _median_from_histogram(
        rng, sub_counts, epsilon_levels[depth - 1]
    )
    median_global_idx = lo_idx + median_local
    median_value = lower + median_global_idx * delta
    left = _rec(lo_idx, median_global_idx, depth - 1)
    right = _rec(median_global_idx, hi_idx, depth - 1)
    return left + [median_value] + right

  return _rec(0, len(counts), levels)
