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

"""Utilities for transforming data from one domain to another."""

import collections
from collections.abc import Callable, Mapping, Sequence
import math
from typing import Any, Generic, TypeAlias, TypeVar

import attr
from dpsynth import domain
import numpy as np
import pandas as pd

CategoricalValue: TypeAlias = None | bool | int | str | pd.Interval
R, T, S = TypeVar('R'), TypeVar('T'), TypeVar('S')


@attr.define(frozen=True)
class DataTransformation(Generic[R, T]):
  """Dataclass for transforming data from one domain to another.

  DataTransformations are both reversible (via inverse) and composable (via @).

  Example Usage:

  >>> transformer = DataTransformation(lambda x: x + 1, lambda x: x - 1)
  >>> transformer(1)
  2
  >>> transformer.inverse(2)
  1
  >>> transformer3 = transformer @ transformer @ transformer
  >>> transformer3(1)
  4
  >>> transformer3.inverse(4)
  1
  """

  transform: Callable[[R], T] | Mapping[R, T] = attr.field()
  inverse_transform: Callable[[T], R] | Mapping[T, R] = attr.field()

  def __call__(self, value: R) -> T:
    if isinstance(self.transform, Mapping):
      return self.transform[value]
    return self.transform(value)

  @property
  def inverse(self) -> 'DataTransformation[T, R]':
    """The reverse transformation of this instance."""
    return DataTransformation(self.inverse_transform, self.transform)

  def __matmul__(
      self, other: 'DataTransformation[T, S]'
  ) -> 'DataTransformation[R, S]':
    """Returns a DataTransformation that composes this instance with other.

    Example Usage:
    >>> f = DataTransformation(lambda x: x + 1, lambda x: x - 1)
    >>> g = DataTransformation(lambda x: 2 * x, lambda x: x / 2)
    >>> h = f @ g
    >>> h(3)
    7
    >>> f(g(3))
    7

    Args:
      other: The DataTransformation to compose with.

    Returns:
      A DataTransformation that composes this instance with other.
    """
    return DataTransformation(
        lambda x: self(other(x)),
        lambda x: other.inverse(self.inverse(x)),
    )


# Non-float values allowed, but will be mapped to None or a default interval.
DiscretizeTransformation = DataTransformation[Any, pd.Interval | None]


def discrete_encoder(
    attribute_domain: domain.CategoricalAttribute,
) -> DataTransformation[CategoricalValue, int]:
  """Returns a mapping from possible values to their index in the list.

  Example Usage:

  ```
  >>> grade = CategoricalAttribute([None, 'A', 'B', 'C', 'D', 'F'], 0)
  >>> transform_fn = discrete_encoder(grade)
  >>> transform_fn(None)
  0
  >>> transform_fn('A')
  1
  >>> transform_fn.inverse(0)
  None
  >>> transform_fn.inverse(3)
  'C'
  ```

  Args:
    attribute_domain: The CategoricalAttribute to encode.

  Returns:
    A DataTransformation that maps from possible values to their index in the
    list.
  """
  # vs. other implementations in downstream pandas replace applications.
  transform = collections.defaultdict(
      lambda: attribute_domain.out_of_domain_index,
      {value: i for i, value in enumerate(attribute_domain.possible_values)},
  )
  reverse = dict(enumerate(attribute_domain.possible_values))
  return DataTransformation(transform, reverse)


def create_discretize_transformation(
    attribute_domain: domain.NumericalAttribute,
    bin_edges: Sequence[int | float],
) -> tuple[domain.CategoricalAttribute, DiscretizeTransformation]:
  """Returns a mapping function and a CategoricalAttribute for discretization.

  Args:
    attribute_domain: The NumericalAttribute to discretize.
    bin_edges: A list of edges that characterizes the discretization. Must be
      monotonically increasing and each bin edge must be within the range
      (min_value, max_value).

  Returns:
    A tuple of (CategoricalAttribute, DiscretizeTransformation). The first
    element is a CategoricalAttribute that represents the possible values
    is a DiscretizeTransformation that can be used to map a numerical value
    to an interval in the CategoricalAttribute and vice versa.
  """
  if not bin_edges:
    raise ValueError(f'bin_edges must not be empty, got {bin_edges}.')
  if (
      bin_edges[0] < attribute_domain.min_value
      or bin_edges[-1] >= attribute_domain.max_value
  ):
    min_value = attribute_domain.min_value
    max_value = attribute_domain.max_value
    raise ValueError(
        f'bin_edges must be within the range [{min_value}, {max_value}), got'
        f' {bin_edges}.'
    )
  if any(bin_edges[i] >= bin_edges[i + 1] for i in range(len(bin_edges) - 1)):
    raise ValueError(
        f'bin_edges must be monotonically increasing, got {bin_edges}.'
    )

  bin_edges = np.r_[
      attribute_domain.exclusive_min_value,
      bin_edges,
      attribute_domain.max_value,
  ]
  intervals = pd.IntervalIndex.from_breaks(bin_edges)
  maybe_none = [] if attribute_domain.clip_to_range else [None]
  possible_values = maybe_none + list(intervals)

  def transform(value: Any) -> pd.Interval | None:
    value = attribute_domain.standardize(value)
    if value is None:
      return None
    return intervals[intervals.get_loc(value)]

  def reverse(value: pd.Interval | None) -> float | None:
    if value is None:
      return None
    if attribute_domain.dtype == 'int':
      return math.ceil(value.mid)
    return value.mid

  new_domain = domain.CategoricalAttribute(possible_values)
  transformation = DiscretizeTransformation(transform, reverse)
  return new_domain, transformation


def create_uniform_discretize_transformation(
    attribute_domain: domain.NumericalAttribute, num_bins: int
) -> tuple[domain.CategoricalAttribute, DiscretizeTransformation]:
  """Returns a mapping function and a CategoricalAttribute for discretization.

  Example Usage:

  ```
  >>> age = NumericalAttribute(min_value=0, max_value=100, clip_to_range=True)
  >>> _,transform_fn = create_uniform_discretize_transformation(age, num_bins=5)
  >>> transform_fn(50)
  Interval(40, 60, closed='right')
  >>> transform_fn(40)
  Interval(20, 40, closed='right')
  >>> transform_fn(105)
  Interval(80, 100, closed='right')
  ```

  Args:
    attribute_domain: The NumericalAttribute to discretize.
    num_bins: The number of equal-width bins to use for discretization.

  Returns:
    A tuple of (CategoricalAttribute, DiscretizeTransformation). The first
    element is a CategoricalAttribute that represents the possible values that
    can be returned by the transformation function. The second element is a
    DiscretizeTransformation that can be used to map a numerical value to an
    interval in the CategoricalAttribute and vice versa.
  """
  bin_edges = np.linspace(
      attribute_domain.exclusive_min_value,
      attribute_domain.max_value,
      num_bins + 1,
  )
  if attribute_domain.dtype == 'int':
    bin_edges = np.sort(np.unique(bin_edges.astype(int)))
  return create_discretize_transformation(
      attribute_domain, list(bin_edges[1:-1])
  )


def create_rare_value_merging_transformation(
    rare_value_mask: np.ndarray,
) -> tuple[int, DataTransformation[int, int]]:
  """Returns a DataTransformation that merges rare values.

  Example Usage:
  >>> rare_mask = np.array([True, False, True, False])
  >>> size, transform_fn = create_rare_value_merging_transformation(rare_mask)
  >>> size
  3
  >>> [transform_fn(i) for i in range(4)]
  [2, 0, 2, 1]

  Args:
    rare_value_mask: A 1D boolean array that indicates which values are rare.
      Values that are True in this array will be merged into a single value.
      Values that are False will be preserved as-is.

  Returns: A tuple of (int, DataTransformation). The first element is the
    compressed domain size. The second element is a DataTransformation that maps
    values to a compressed domain, and merges rare values into a single value.
  """
  rare_value_mask = np.array(rare_value_mask)
  if rare_value_mask.ndim != 1 or rare_value_mask.dtype != bool:
    raise ValueError(
        'rare_value_mask must be a 1D array of type bool, got'
        f' shape={rare_value_mask.shape} and dtype={rare_value_mask.dtype}.'
    )

  num_rare = rare_value_mask.sum()
  size = rare_value_mask.size - num_rare
  if num_rare >= 1:
    size += 1

  mapping = {}
  inv_mapping = {}
  idx = 0
  for i in range(rare_value_mask.size):
    if rare_value_mask[i]:
      mapping[i] = size - 1
    else:
      mapping[i] = idx
      inv_mapping[idx] = i
      idx += 1

  rare_values = np.where(rare_value_mask)[0]

  def reverse(value: int) -> int:
    if num_rare >= 1 and value == size - 1:
      return np.random.choice(rare_values)
    return inv_mapping[value]

  return size, DataTransformation(mapping, reverse)


def apply(
    data: pd.DataFrame,
    column_transforms: dict[str, DataTransformation],
    reverse: bool = False,
    drop_extra_columns: bool = True,
) -> pd.DataFrame:
  """Applies the given transformations to the data.

  Args:
    data: The DataFrame to transform.
    column_transforms: The dictionary of transformations to apply.
    reverse: Whether to apply the inverse transformations.
    drop_extra_columns: Whether to drop columns that are not in the
      transformations dictionary, or keep them as-is.

  Returns:
    A new DataFrame with the transformations applied columnwise.
  """
  df = pd.DataFrame()
  for col in column_transforms if drop_extra_columns else data.columns:
    if col not in column_transforms:
      df[col] = data[col]
      continue
    if reverse:
      transform_fn = column_transforms[col].inverse
    else:
      transform_fn = column_transforms[col]
    df[col] = data[col].map(transform_fn.transform)
  return df
