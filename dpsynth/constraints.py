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
  # (low, high) cutpoint tuple within [min_value, max_value]. Checking this in
  # __post_init__ catches invalid bounds before touching private data and
  # guarantees that to_mbi() will match at least one discrete bin.
  if val is None:
    if attr.clip_to_range:
      raise ValueError('None requires clip_to_range=False.')
    return
  if not (isinstance(val, tuple) and len(val) == 2):
    raise ValueError(f'Expected (low, high) tuple or None, got {val!r}.')
  low, high = float(val[0]), float(val[1])
  if not (attr.min_value <= low < high <= attr.max_value):
    raise ValueError(
        f'Interval ({low}, {high}) must satisfy'
        f' {attr.min_value} <= low < high <= {attr.max_value}.'
    )
  if attr.dtype == 'int' and not (low.is_integer() and high.is_integer()):
    raise ValueError(
        f'Integer attribute interval ({low}, {high}) requires integer bounds.'
    )


def _encode_numerical(val, attr, bin_edges):
  """Maps a validated numerical (low, high) interval or None to bin indices."""
  # None maps to the out-of-domain bin at index 0, while (low, high) selects
  # all in-domain bins [e_i, e_{i+1}] inside [low, high], shifted by 1 when
  # index 0 is reserved for OOD. Because pinned_edges includes low and high,
  # [low, high] aligns with bin boundaries and partitions into whole bins.
  if val is None:
    return [0]
  low, high = val
  full_edges = np.r_[attr.min_value, bin_edges, attr.max_value]
  mask = (full_edges[:-1] >= low) & (full_edges[1:] <= high)
  offset = 0 if attr.clip_to_range else 1
  return (np.flatnonzero(mask) + offset).tolist()


def _discretize_constraint(c, bin_edges):
  """Converts numerical attributes and intervals into discrete categorical bins."""
  # Replace each NumericalAttribute with a CategoricalAttribute over its bin
  # indices and expand each combination via Cartesian product, since a single
  # (low, high) interval can span multiple bins, before delegating to the
  # standard categorical to_mbi() path.
  cat_domains, encoders = [], []
  for name, attr in zip(c.attribute_names, c.attribute_domains):
    if isinstance(attr, domain.NumericalAttribute):
      edges = np.asarray(bin_edges[name], dtype=float)
      size = len(edges) + (1 if attr.clip_to_range else 2)
      cat_domains.append(domain.CategoricalAttribute(list(range(size))))
      encoders.append(lambda v, a=attr, e=edges: _encode_numerical(v, a, e))
    else:
      cat_domains.append(attr)
      encoders.append(lambda v: [v])

  combos = c.possible_combinations or c.impossible_combinations
  expanded = []
  for combo in combos:  # pyrefly: ignore[not-iterable]
    bins = [enc(v) for enc, v in zip(encoders, combo)]
    expanded.extend(itertools.product(*bins))
  possible = expanded if c.possible_combinations is not None else None
  impossible = expanded if c.impossible_combinations is not None else None
  return Constraint(c.attribute_names, tuple(cat_domains), possible, impossible)


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
      raise ValueError(
          'functional_dependency does not support NumericalAttribute.'
      )
  if c.functional_dependency is None:
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

  @property
  def pinned_edges(self) -> dict[str, tuple[float, ...]]:
    """Sorted inner bin edges required for constrained numerical attributes."""
    combos = self.possible_combinations or self.impossible_combinations or ()
    pinned = {}
    attrs = zip(self.attribute_names, self.attribute_domains)
    for i, (name, attr) in enumerate(attrs):
      if isinstance(attr, domain.NumericalAttribute):
        lo, hi = attr.min_value, attr.max_value
        edges = set()
        for c in combos:
          if c[i] is not None:
            edges.update(float(e) for e in c[i] if lo < e < hi)
        if edges:
          pinned.setdefault(name, set()).update(edges)
    return {col: tuple(sorted(edges)) for col, edges in pinned.items()}

  def to_mbi(
      self, bin_edges: Mapping[str, Sequence[float] | np.ndarray] | None = None
  ) -> mbi.Constraint:
    """Convert to an mbi.Constraint."""
    types = [type(d) for d in self.attribute_domains]
    if domain.NumericalAttribute in types:
      return _discretize_constraint(self, bin_edges or {}).to_mbi()
    shape = tuple(d.size for d in self.attribute_domains)
    mbi_domain = mbi.Domain(self.attribute_names, shape)
    encoders = [
        transformations.discrete_encoder(d) for d in self.attribute_domains
    ]

    if self.functional_dependency is not None:
      _, coarse_enc = encoders
      fine_values = self.attribute_domains[0].possible_values
      coarse_indices = [
          coarse_enc(self.functional_dependency[v]) for v in fine_values
      ]
      return mbi.Constraint(
          domain=mbi_domain, mapping=np.array(coarse_indices, dtype=np.int32)
      )

    combos = self.possible_combinations or self.impossible_combinations
    encoded = [[enc(v) for enc, v in zip(encoders, c)] for c in combos]  # pyrefly: ignore[not-iterable]
    indices = np.array(encoded, dtype=np.int32)
    if self.possible_combinations is not None:
      return mbi.Constraint(domain=mbi_domain, valid=indices)
    return mbi.Constraint(domain=mbi_domain, invalid=indices)
