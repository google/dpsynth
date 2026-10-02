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

"""Cross-attribute constraints for categorical data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import itertools
from typing import Any

from dpsynth import domain
from dpsynth import transformations
import mbi
import numpy as np


def _validate_numerical(val, attr):
  """Validates a numerical constraint value at configuration time."""
  # Entries must be None (out-of-domain bin when clip_to_range=False) or a
  # (low, high) tuple whose endpoints match attr.bin_edges or the domain
  # bounds. Validating this in __post_init__ catches invalid bounds before
  # touching private data and guarantees exact alignment with discrete bins.
  if val is None:
    if attr.clip_to_range:
      raise ValueError('None requires clip_to_range=False.')
    return
  if not (isinstance(val, tuple) and len(val) == 2):
    raise ValueError(f'Expected (low, high) tuple or None, got {val!r}.')
  low, high = float(val[0]), float(val[1])
  if low >= high:
    raise ValueError(f'Interval ({low}, {high}) requires low < high.')
  edges = {attr.min_value, *attr.bin_edges, attr.max_value}
  if low not in edges or high not in edges:
    raise ValueError(f'Interval ({low}, {high}) must use {sorted(edges)}.')


def _domain_size(attr) -> int:
  """Returns the discrete MBI domain size for a categorical or numerical attribute."""
  if isinstance(attr, domain.NumericalAttribute):
    assert attr.bin_edges is not None
    return len(attr.bin_edges) + (1 if attr.clip_to_range else 2)
  return attr.size


def _encode_value(val, attr) -> list[int]:
  """Maps a categorical value or numerical interval/None to discrete bin indices."""
  # Categorical values map to a single index, None maps to out-of-domain bin 0,
  # and a validated (low, high) interval selects all in-domain bins inside
  # [low, high], shifted by 1 when index 0 is reserved for OOD.
  if not isinstance(attr, domain.NumericalAttribute):
    return [transformations.discrete_encoder(attr)(val)]
  if val is None:
    return [0]
  low, high = val
  full_edges = np.r_[attr.min_value, attr.bin_edges, attr.max_value]
  mask = (full_edges[:-1] >= low) & (full_edges[1:] <= high)
  offset = 0 if attr.clip_to_range else 1
  return (np.flatnonzero(mask) + offset).tolist()


def _validate(c: Constraint) -> None:
  """Validate a Constraint's fields."""
  if len(c.attribute_names) != len(c.attribute_domains):
    raise ValueError(
        'attribute_names and attribute_domains must have the same length, got'
        f' {len(c.attribute_names)} != {len(c.attribute_domains)}.'
    )
  modes = (
      c.possible_combinations,
      c.impossible_combinations,
      c.functional_dependency,
  )
  n_set = sum(x is not None for x in modes)
  if n_set != 1:
    raise ValueError(
        "Specify exactly one of 'possible_combinations',"
        " 'impossible_combinations', or 'functional_dependency'."
    )
  if c.functional_dependency is not None:
    if len(c.attribute_names) != 2:
      raise ValueError(
          'functional_dependency requires exactly 2 attributes (fine, coarse),'
          f' got {len(c.attribute_names)}.'
      )
    types = [type(d) for d in c.attribute_domains]
    if domain.NumericalAttribute in types:
      raise ValueError('functional_dependency requires CategoricalAttribute.')
  if c.functional_dependency is None:
    for attr in c.attribute_domains:
      if isinstance(attr, domain.NumericalAttribute) and attr.bin_edges is None:
        raise ValueError('NumericalAttribute in Constraint requires bin_edges.')
    combos = c.possible_combinations or c.impossible_combinations
    n_attrs = len(c.attribute_names)
    for combo in combos:  # pyrefly: ignore[not-iterable]
      if len(combo) != n_attrs:
        raise ValueError(
            'Each combination must have length equal to the number of'
            f' attributes ({n_attrs}), got {len(combo)}.'
        )
      for val, attr in zip(combo, c.attribute_domains):
        if isinstance(attr, domain.NumericalAttribute):
          _validate_numerical(val, attr)


@dataclasses.dataclass(frozen=True)
class Constraint:
  """A constraint on allowed value combinations across attributes.

  Mirrors :class:`mbi.Constraint` but accepts human-readable values from
  :class:`dpsynth.domain.CategoricalAttribute` instead of integer arrays.
  Exactly one of ``possible_combinations``, ``impossible_combinations``, or
  ``functional_dependency`` must be specified.

  Example Usage:

    >>> d1 = domain.CategoricalAttribute(['GameSuite', 'OfficePro', 'DevTool'])
    >>> d2 = domain.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])
    >>> constraint = Constraint(
    ...     attribute_names=('Software', 'Operating System'),
    ...     attribute_domains=(d1, d2),
    ...     possible_combinations=[
    ...         ('GameSuite', 'Windows'),
    ...         ('OfficePro', 'Windows'),
    ...         ('OfficePro', 'MacOS'),
    ...         ('DevTool', 'Linux'),
    ...         ('DevTool', 'MacOS'),
    ...     ],
    ... )

  Attributes:
    attribute_names: Names of the constrained attributes.
    attribute_domains: Categorical or numerical domain for each attribute.
    possible_combinations: Allowed value combinations.
    impossible_combinations: Forbidden value combinations.
    functional_dependency: Dict mapping fine attribute values to coarse
      attribute values. Requires exactly two attributes.
  """

  attribute_names: tuple[str, ...]
  attribute_domains: Sequence[Any]
  possible_combinations: Sequence[tuple[Any, ...]] | None = None
  impossible_combinations: Sequence[tuple[Any, ...]] | None = None
  functional_dependency: Mapping[Any, Any] | None = None

  def __post_init__(self):
    _validate(self)

  def to_mbi(self) -> mbi.Constraint:
    """Convert to an mbi.Constraint."""
    shape = tuple(_domain_size(d) for d in self.attribute_domains)
    mbi_domain = mbi.Domain(self.attribute_names, shape)

    if self.functional_dependency is not None:
      coarse_enc = transformations.discrete_encoder(self.attribute_domains[1])
      fine_values = self.attribute_domains[0].possible_values
      coarse_indices = [
          coarse_enc(self.functional_dependency[v]) for v in fine_values
      ]
      return mbi.Constraint(
          domain=mbi_domain, mapping=np.array(coarse_indices, dtype=np.int32)
      )

    combos = self.possible_combinations or self.impossible_combinations
    encoded = []
    for c in combos:  # pyrefly: ignore[not-iterable]
      bins = [_encode_value(v, d) for v, d in zip(c, self.attribute_domains)]
      encoded.extend(itertools.product(*bins))
    indices = np.array(encoded, dtype=np.int32).reshape(-1, len(shape))
    if self.possible_combinations is not None:
      return mbi.Constraint(domain=mbi_domain, valid=indices)
    return mbi.Constraint(domain=mbi_domain, invalid=indices)


@dataclasses.dataclass(frozen=True)
class InequalityConstraint:
  """A pairwise ordering constraint requiring lower_attribute <= upper_attribute.

  Attributes:
    lower_attribute: Name of the numerical attribute providing the lower bound.
    upper_attribute: Name of the numerical attribute providing the upper bound.
    attribute_domains: Pair of ``NumericalAttribute`` domains.
  """

  lower_attribute: str
  upper_attribute: str
  attribute_domains: tuple[domain.NumericalAttribute, domain.NumericalAttribute]

  def __post_init__(self):
    if self.lower_attribute == self.upper_attribute:
      raise ValueError('lower_attribute and upper_attribute must be distinct.')
    n_domains = len(self.attribute_domains)
    if n_domains != 2:
      raise ValueError(f'Expected 2 domains, got {n_domains}.')
    for d in self.attribute_domains:
      if not isinstance(d, domain.NumericalAttribute):
        raise ValueError(f'Expected NumericalAttribute, got {type(d)}.')
      if d.bin_edges is None:
        raise ValueError('InequalityConstraint requires bin_edges.')
    a, b = self.attribute_domains
    low_meta = (a.min_value, a.max_value, a.bin_edges, a.clip_to_range)
    high_meta = (b.min_value, b.max_value, b.bin_edges, b.clip_to_range)
    if low_meta != high_meta:
      raise ValueError('Domains must have matching discretizations.')

  @property
  def attribute_names(self) -> tuple[str, str]:
    """Returns the (lower_attribute, upper_attribute) pair."""
    return (self.lower_attribute, self.upper_attribute)

  def to_mbi(self) -> mbi.Constraint:
    """Convert to an mbi.Constraint marking strictly inverted bin pairs invalid."""
    attr = self.attribute_domains[0]
    assert attr.bin_edges is not None
    n_bins = len(attr.bin_edges) + 1
    offset = 0 if attr.clip_to_range else 1
    size = n_bins + offset
    mbi_domain = mbi.Domain(self.attribute_names, (size, size))
    inv_i, inv_j = np.tril_indices(n_bins, k=-1)
    invalid = np.column_stack((inv_i + offset, inv_j + offset)).astype(np.int32)
    return mbi.Constraint(domain=mbi_domain, invalid=invalid)


ConstraintType = Constraint | InequalityConstraint
