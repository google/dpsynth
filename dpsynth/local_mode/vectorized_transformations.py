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

"""Vectorized transformations for the local_mode package.

This is a vectorized fork of ``dpsynth/transformations.py``, optimized for
single-machine (numpy-based) environments.  Functions operate on 1-D numpy
arrays rather than scalar values, yielding significant speedups by replacing
per-element Python loops with bulk numpy operations.

Covers:
  * Discrete encoding / decoding (categorical <-> integer index).
  * Discretization / undiscretization (numerical <-> bin index).
  * Rare-value merging / unmerging (domain compression).
"""

from __future__ import annotations

from collections.abc import Sequence

from dpsynth import domain
import numpy as np

# ---------------------------------------------------------------------------
# Discrete encoding / decoding
# ---------------------------------------------------------------------------


def discrete_encode(
    data: np.typing.ArrayLike,
    attribute_domain: domain.CategoricalAttribute,
) -> np.ndarray:
  """Maps categorical values to integer indices in ``attribute_domain``.

  Out-of-domain values are mapped to ``attribute_domain.out_of_domain_index``.

  Args:
    data: 1-D array of categorical values (any type that can appear in
      ``attribute_domain.possible_values``).
    attribute_domain: The categorical attribute defining the encoding.

  Returns:
    A 1-D integer array of indices into ``attribute_domain.possible_values``.

  Raises:
    ValueError: If *data* is not 1-D.
  """
  data = np.asarray(data)
  if data.ndim != 1:
    raise ValueError(f'data must be 1-D, got shape {data.shape}.')
  lookup = {v: i for i, v in enumerate(attribute_domain.possible_values)}
  default = attribute_domain.out_of_domain_index
  encoder = np.vectorize(lambda v: lookup.get(v, default), otypes=[int])
  return encoder(data)


def discrete_decode(
    encoded: np.typing.ArrayLike,
    attribute_domain: domain.CategoricalAttribute,
) -> np.ndarray:
  """Maps integer indices back to categorical values.

  Args:
    encoded: 1-D integer array of indices into
      ``attribute_domain.possible_values``.
    attribute_domain: The categorical attribute defining the decoding.

  Returns:
    A 1-D object-dtype array of categorical values.

  Raises:
    ValueError: If *encoded* is not 1-D.
  """
  encoded = np.asarray(encoded)
  if encoded.ndim != 1:
    raise ValueError(f'encoded must be 1-D, got shape {encoded.shape}.')
  values = np.array(attribute_domain.possible_values, dtype=object)
  return values[encoded]


# ---------------------------------------------------------------------------
# Discretization / undiscretization
# ---------------------------------------------------------------------------


def _validate_bin_edges(bin_edges, attribute_domain):
  """Validates bin_edges against the attribute domain."""
  if bin_edges.size == 0:
    raise ValueError(f'bin_edges must not be empty, got {bin_edges}.')
  if (
      bin_edges[0] < attribute_domain.min_value
      or bin_edges[-1] >= attribute_domain.max_value
  ):
    raise ValueError(
        'bin_edges must be within the range'
        f' [{attribute_domain.min_value}, {attribute_domain.max_value}),'
        f' got {list(bin_edges)}.'
    )
  if np.any(np.diff(bin_edges) <= 0):
    raise ValueError(
        f'bin_edges must be monotonically increasing, got {list(bin_edges)}.'
    )


def discretize(
    data: np.typing.ArrayLike,
    bin_edges: Sequence[int | float],
    attribute_domain: domain.NumericalAttribute,
) -> np.ndarray:
  """Maps numerical values to bin indices via ``np.searchsorted``.

  Mirrors the semantics of ``transformations.create_discretize_transformation``
  but operates on entire arrays at once.  Bin intervals are right-closed:
  ``(left, right]``, matching the ``pd.IntervalIndex`` convention used in the
  scalar implementation.

  Args:
    data: 1-D array of numerical values.
    bin_edges: Sorted inner bin edges (same convention as
      ``create_discretize_transformation``).  Must be monotonically increasing
      and within ``[min_value, max_value)``.
    attribute_domain: The ``NumericalAttribute`` describing the data domain.

  Returns:
    A 1-D integer array of 0-based bin indices.  When
    ``attribute_domain.clip_to_range`` is ``False``, index 0 represents the
    out-of-domain (``None``) bin and in-domain bins start at 1.

  Raises:
    ValueError: If *data* is not 1-D or *bin_edges* are invalid.
  """
  data = np.asarray(data, dtype=float)
  if data.ndim != 1:
    raise ValueError(f'data must be 1-D, got shape {data.shape}.')
  bin_edges = np.asarray(bin_edges, dtype=float)
  _validate_bin_edges(bin_edges, attribute_domain)

  if attribute_domain.clip_to_range:
    standardized = np.clip(
        data, attribute_domain.min_value, attribute_domain.max_value
    )
    # NaN values (from non-numeric inputs) become min_value after clip, but
    # np.clip does not handle NaN, so fix them explicitly.
    standardized = np.where(
        np.isnan(standardized), attribute_domain.min_value, standardized
    )
    # side='left' gives right-closed intervals: value == edge -> left bin.
    return np.searchsorted(bin_edges, standardized, side='left')
  else:
    standardized = data.copy()
    ood_mask = (
        np.isnan(standardized)
        | (standardized < attribute_domain.min_value)
        | (standardized > attribute_domain.max_value)
    )
    standardized[ood_mask] = attribute_domain.min_value
    indices = np.searchsorted(bin_edges, standardized, side='left')
    # Shift in-domain indices by 1 to leave room for the None bin at 0.
    indices += 1
    indices[ood_mask] = 0
    return indices


def undiscretize(
    bin_indices: np.typing.ArrayLike,
    bin_edges: Sequence[int | float],
    attribute_domain: domain.NumericalAttribute,
) -> np.ndarray:
  """Maps bin indices back to bin midpoints.

  This is the inverse of :func:`discretize`.

  Args:
    bin_indices: 1-D integer array of bin indices (as produced by
      :func:`discretize`).
    bin_edges: The same sorted inner bin edges used during discretization.
    attribute_domain: The ``NumericalAttribute`` describing the data domain.

  Returns:
    A 1-D float array of midpoints.  When
    ``attribute_domain.clip_to_range`` is ``False``, index 0 maps to ``NaN``
    (representing ``None`` / out-of-domain).  If ``attribute_domain.dtype``
    is ``'int'``, midpoints are rounded up via ``np.ceil`` and cast to int.

  Raises:
    ValueError: If *bin_indices* is not 1-D or *bin_edges* are invalid.
  """
  bin_indices = np.asarray(bin_indices, dtype=int)
  if bin_indices.ndim != 1:
    raise ValueError(f'bin_indices must be 1-D, got shape {bin_indices.shape}.')
  bin_edges = np.asarray(bin_edges, dtype=float)
  _validate_bin_edges(bin_edges, attribute_domain)

  full_edges = np.r_[
      attribute_domain.exclusive_min_value,
      bin_edges,
      attribute_domain.max_value,
  ]
  midpoints = (full_edges[:-1] + full_edges[1:]) / 2.0

  if attribute_domain.clip_to_range:
    result = midpoints[bin_indices]
  else:
    # Index 0 -> NaN (None bin); in-domain indices are shifted by 1.
    result = np.full(bin_indices.shape, np.nan)
    in_domain = bin_indices > 0
    result[in_domain] = midpoints[bin_indices[in_domain] - 1]

  if attribute_domain.dtype == 'int':
    valid = ~np.isnan(result)
    result[valid] = np.ceil(result[valid])
    if np.all(valid):
      result = result.astype(int)
  return result


# ---------------------------------------------------------------------------
# Rare-value merging / unmerging
# ---------------------------------------------------------------------------


def merge_rare_values(
    data: np.typing.ArrayLike,
    rare_value_mask: np.typing.ArrayLike,
) -> tuple[int, np.ndarray]:
  """Maps integer-encoded data to a compressed domain, merging rare values.

  Non-rare values are renumbered contiguously starting from 0; all rare values
  are mapped to the last index in the compressed domain.

  Args:
    data: 1-D integer array in the original domain.
    rare_value_mask: 1-D boolean array indicating which original-domain values
      are rare (``True`` means rare).

  Returns:
    A tuple ``(compressed_size, compressed_data)`` where *compressed_size* is
    the number of bins in the compressed domain and *compressed_data* is a 1-D
    integer array of the same length as *data*.

  Raises:
    ValueError: If inputs have incorrect shapes or dtypes.
  """
  data = np.asarray(data, dtype=int)
  rare_value_mask = np.asarray(rare_value_mask, dtype=bool)
  if data.ndim != 1:
    raise ValueError(f'data must be 1-D, got shape {data.shape}.')
  if rare_value_mask.ndim != 1:
    raise ValueError(
        f'rare_value_mask must be 1-D, got shape {rare_value_mask.shape}.'
    )

  num_rare = int(rare_value_mask.sum())
  num_common = rare_value_mask.size - num_rare
  compressed_size = num_common + (1 if num_rare >= 1 else 0)

  # Common values get contiguous indices; rare values all map to the last bin.
  mapping = np.empty(rare_value_mask.size, dtype=int)
  mapping[rare_value_mask] = compressed_size - 1
  mapping[~rare_value_mask] = np.arange(num_common)

  return compressed_size, mapping[data]


def unmerge_rare_values(
    data: np.typing.ArrayLike,
    rare_value_mask: np.typing.ArrayLike,
    rng: np.random.Generator,
) -> np.ndarray:
  """Maps compressed-domain integers back, randomly restoring rare values.

  This is the inverse of :func:`merge_rare_values`.  For the merged-rare bin,
  each element is randomly assigned to one of the original rare values.

  Args:
    data: 1-D integer array in the compressed domain.
    rare_value_mask: 1-D boolean array (same as used in
      :func:`merge_rare_values`).
    rng: A numpy random number generator used for sampling rare values.

  Returns:
    A 1-D integer array in the original domain.

  Raises:
    ValueError: If inputs have incorrect shapes or dtypes.
  """
  data = np.asarray(data, dtype=int)
  rare_value_mask = np.asarray(rare_value_mask, dtype=bool)
  if data.ndim != 1:
    raise ValueError(f'data must be 1-D, got shape {data.shape}.')
  if rare_value_mask.ndim != 1:
    raise ValueError(
        f'rare_value_mask must be 1-D, got shape {rare_value_mask.shape}.'
    )

  num_rare = int(rare_value_mask.sum())
  num_common = rare_value_mask.size - num_rare
  compressed_size = num_common + (1 if num_rare >= 1 else 0)
  rare_bin = compressed_size - 1

  common_indices = np.where(~rare_value_mask)[0]
  inv_mapping = np.empty(compressed_size, dtype=int)
  inv_mapping[:num_common] = common_indices
  if num_rare >= 1:
    inv_mapping[rare_bin] = -1  # Placeholder; overwritten below.

  result = inv_mapping[data]

  if num_rare >= 1:
    rare_indices = np.where(rare_value_mask)[0]
    rare_mask = data == rare_bin
    result[rare_mask] = rng.choice(rare_indices, size=int(rare_mask.sum()))

  return result
