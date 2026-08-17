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

"""Pure, deterministic relational data transformers and mathematical helpers."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import itertools
from typing import Literal

from dpsynth.relational import domain as rel_domain
import mbi
import numpy as np
import pandas as pd


def _compute_row_root_mappings(
    tables: Mapping[str, pd.DataFrame],
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    rng: np.random.Generator | None = None,
) -> dict[str, pd.Series]:
  """Maps each row in every table to its root parent row index (0..N-1) or None.

  Traverses the relational hierarchy top-down. For each child table, maps
  foreign key references to parent row positions and checks capacity limits.
  When a parent record exceeds its `max_children_per_parent` bound (s), exactly
  `s` children are selected uniformly at random without replacement.
  Truncated, orphaned, or descendant records under dropped parents map to None.

  Formal Guarantees:
    - Bounded Lineage: For each root entity H_i and relation with bound s_k, at
      most s_k children per parent (and at most prod_{j=1}^k s_j descendants at
      depth k) retain active root mappings to H_i.
    - Cascading Truncation Invariant: If an ancestor evaluates to None, all of
      its transitive descendant records in downstream tables evaluate to None.
    - Order-Agnostic Truncation: Child subsampling is uniform without
      replacement, preventing privacy leakage from input DataFrame row order.
    - Data-Dependent Error Immunity: Orphan foreign keys, duplicate primary
      keys, and NaNs evaluate to None without runtime errors.
    - Positional Alignment via None Preservation: Evaluates truncated, orphaned,
      or invalid records to None rather than deleting them, preserving strict
      1-to-1 positional alignment with input DataFrames. This avoids mutating
      table shapes, prevents index-shifting artifacts during downstream
      parent-to-child lookups, and cleanly maps to weight w_r = 0.0 in
      sensitivity weighting.
    - Downstream Ingestion: Zero-weighted records are stripped in the
      synthesizer (via `weights > 0.0`) before entering column initializers
      for bin discovery, encoding, and domain compression.

  Example:
    households: [H0, H1]
    persons (s1=2): [P0(H0), P1(H0), P2(H0), P3(H1)] -> P2 truncated (None)
    activities (s2=2): [A0(P0), A1(P1), A2(P2), A3(P3), A4(orphan)]
      -> A0 maps to 0
      -> A1 maps to 0
      -> A2 maps to None (parent P2 was truncated)
      -> A3 maps to 1
      -> A4 maps to None (orphan foreign key)

    Result:
      {
        'households': pd.Series([0, 1]),
        'persons':    pd.Series([0, 0, None, 1]),
        'activities': pd.Series([0, 0, None, 1, None]),
      }

  Args:
    tables: Mapping from table name to input DataFrame.
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.
    rng: Random number generator for uniform child record truncation.

  Returns:
    A dictionary mapping table name to a pd.Series of root row indices (int) or
    None, aligned 1-to-1 with DataFrame rows.

  Raises:
    ValueError: If required primary or foreign key columns are missing from
      schemas.
  """
  if rng is None:
    rng = np.random.default_rng()

  row_to_root: dict[str, pd.Series] = {}
  for depth, table_name, fk in hierarchy:
    child_df = tables[table_name]

    # Depth 0: Root privacy unit table (no incoming foreign key).
    # Each root record maps to its own 0-based integer row index.
    if depth == 0 or fk is None:
      row_to_root[table_name] = pd.Series(
          range(len(child_df)),
          index=child_df.index,
          dtype=object,
      )
      continue

    # Schema integrity validation (public schema check; safe to raise errors).
    if fk.parent_primary_key not in tables[fk.parent_table].columns:
      raise ValueError(
          f'Parent primary key column {fk.parent_primary_key!r} not in table'
          f' {fk.parent_table!r}.'
      )
    if fk.child_foreign_key not in child_df.columns:
      raise ValueError(
          f'Child foreign key column {fk.child_foreign_key!r} not in table'
          f' {table_name!r}.'
      )

    parent_df = tables[fk.parent_table]
    parent_roots = row_to_root[fk.parent_table]

    # Fast path for empty tables: returns all None without failing.
    if child_df.empty or parent_df.empty:
      row_to_root[table_name] = pd.Series(
          [None] * len(child_df), index=child_df.index, dtype=object
      )
      continue

    # 1. Parent lookup maps parent primary keys to row numbers in parent_df.
    # Ignores NaN primary keys and deduplicates repeated keys (keeping first).
    # parent_lookup: (Index = parent_pk, Value = parent_df row index).
    parent_pos = pd.Series(range(len(parent_df)), index=parent_df.index)
    parent_valid_mask = (
        parent_df[fk.parent_primary_key].notna()  # No NaN keys.
        & ~parent_df[fk.parent_primary_key].duplicated()  # Keep only first.
    )
    parent_keys = parent_df.loc[parent_valid_mask, fk.parent_primary_key]
    parent_lookup = pd.Series(
        parent_pos.loc[parent_valid_mask].values, index=parent_keys
    )

    # 2. Vectorized translation of child foreign keys to parent_df row indices.
    # Non-matching keys (orphans) and NaNs evaluate to NaN.
    # child_p_idx : (Index = child row, Value = parent row | NaN).
    child_p_idx = child_df[fk.child_foreign_key].map(parent_lookup)

    # 3. Discard unlinked children
    # valid_children: (Index = child row, Value = parent row).
    valid_children = child_p_idx.dropna().astype(int)
    if valid_children.empty:
      row_to_root[table_name] = pd.Series(
          [None] * len(child_df), index=child_df.index, dtype=object
      )
      continue

    # 4. Enforce cascading truncation: drop children whose parent root is None.
    # Map valid parent indices back to parent_roots positions
    parent_active_mask = parent_roots.notna().iloc[valid_children.values].values
    valid_children = valid_children[parent_active_mask]
    if valid_children.empty:
      row_to_root[table_name] = pd.Series(
          [None] * len(child_df), index=child_df.index, dtype=object
      )
      continue

    # 5. Intra-group uniform random ranking via Pandas, for uniform truncation.
    # Assigns random float to each child; ranks within each parent group.
    # random_scores: (Index = child row, Value = random float).
    # group_ranks: (Index = child row, Value = rank 1-to-n within parent group).
    random_scores = pd.Series(
        rng.random(len(valid_children)), index=valid_children.index
    )
    group_ranks = random_scores.groupby(valid_children.values).rank(
        method='first'
    )
    selected_children = valid_children[
        group_ranks <= fk.max_children_per_parent
    ]

    # 6. Assign root lineages to selected children in Pandas.
    # Initialize full child table to None; update only selected active rows.
    child_roots_series = pd.Series(
        [None] * len(child_df), index=child_df.index, dtype=object
    )
    child_roots_series.loc[selected_children.index] = parent_roots.iloc[
        selected_children.values
    ].values

    row_to_root[table_name] = child_roots_series
  return row_to_root


def compute_hierarchical_weights(
    tables: Mapping[str, pd.DataFrame],
    hierarchy: Sequence[tuple[int, str, rel_domain.ForeignKeyRelation | None]],
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
  """Computes standalone sensitivity weights (w = 1/k_eff) for Phase 1 initializers.

  Calculates a 1D weight array for each table such that the sum of weights
  associated with any single root entity (e.g. household) equals 1.0, ensuring
  global unit sensitivity (Delta = 1.0) without Cartesian joins or noise
  scaling.

  For each table, active records belonging to a root with k_eff active rows
  receive weight 1.0 / k_eff. Inactive rows (truncated or orphaned) receive 0.0.

  Formal Guarantees:
    - Unit Sensitivity (Delta = 1.0): For every table T and every root entity
      H_i, sum_{r in H_i} w_r <= 1.0, guaranteeing global ell_1-sensitivity
      Delta = 1.0 for weighted linear queries on T without Cartesian joins.
    - Equal Intra-Group Weighting: Each active record under root H_i with k_eff
      active rows receives identical weight w_r = 1.0 / k_eff.
    - Zero Inactive Weight: All truncated, orphaned, or unlinked records are
      assigned weight exactly w_r = 0.0.
    - Downstream Ingestion: Zero-weighted records (w_r = 0.0) are stripped at
      the synthesizer boundary (via `weights > 0.0`) before entering column
      initializers, bin discovery, discrete encoding, and domain compression.

  Args:
    tables: Mapping from table name to input DataFrame.
    hierarchy: Ordered topological synthesis levels from
      `topological_sort_hierarchy()`.
    rng: Random number generator for child record truncation.

  Returns:
    A dictionary mapping table name to a 1D float64 array of row weights.
  """
  row_to_root = _compute_row_root_mappings(tables, hierarchy, rng=rng)

  weights: dict[str, np.ndarray] = {}
  for depth, table_name, _ in hierarchy:
    if depth == 0:
      weights[table_name] = np.ones(len(tables[table_name]), dtype=np.float64)
    else:
      roots = row_to_root[table_name]
      root_counts = roots.value_counts()
      table_weights = (
          roots.map(1.0 / root_counts).fillna(0.0).to_numpy(dtype=np.float64)
      )
      weights[table_name] = table_weights

  return weights


def _build_exploration_domain(
    parent_domain: mbi.Domain,
    child_domain: mbi.Domain,
    max_group_size: int,
    num_permutation_slots: int,
    strategy: Literal['empty_token', 'size_sliced'],
) -> mbi.Domain:
  """Constructs the discrete mbi.Domain for the permuted exploration table.

  Formal Guarantees:
    - Zero Privacy Loss Domain Definition: The resulting domain shape is
      determined strictly by public metadata (input domain shapes, public
      `max_group_size` bound s, and exploration slot count o). No sensitive
      statistics or data-dependent domain dimensions are used, ensuring
      epsilon = 0 privacy loss for domain specification.
    - Deterministic Token Allocation: Under 'empty_token', the <EMPTY> token for
      each child attribute A is deterministically assigned to category index
      K_A (the original domain shape), maintaining a strictly disjoint state
      space from real child values [0, K_A - 1].

  Args:
    parent_domain: Domain of parent table attributes.
    child_domain: Domain of single-child attributes.
    max_group_size: Maximum observed or clipped child group size.
    num_permutation_slots: Permutation exploration slot count (o).
    strategy: 'empty_token' (extends domain by +1 for <EMPTY>) or 'size_sliced'.

  Returns:
    An mbi.Domain encompassing parent columns, group_size, and o child slots.
  """

  # Add group_size to parent domain (cardinality = max_group_size + 1).
  attrs = list(parent_domain.attributes) + ['group_size']
  shapes = list(parent_domain.shape) + [max_group_size + 1]

  # Add o slots, each with size+1 for empty_token or size for size_sliced.
  for i in range(1, num_permutation_slots + 1):
    for attr, size in zip(child_domain.attributes, child_domain.shape):
      slot_size = size + 1 if strategy == 'empty_token' else size
      attrs.append(f'slot_{i}.{attr}')
      shapes.append(slot_size)
  return mbi.Domain(tuple(attrs), tuple(shapes))


def _get_slot_permutation_patterns(
    k: int,
    num_permutation_slots: int,
    strategy: Literal['empty_token', 'size_sliced'],
) -> tuple[list[tuple[int, ...]], float]:
  """Generates slot index permutation patterns and row weight for (k, o).

  Formal Guarantees:
    - Weight Mass Conservation: For any household size k and slot count o, the
      sum of row weights over all generated permutation patterns is strictly
      equal to 1.0 (len(patterns) * weight == 1.0). This guarantees that
      expanding a parent household into multiple permutation rows does not
      scale or alter the total privacy unit mass.
    - Exchangeability & Unbiasedness: Symmetrically enumerates all distinct
      slot permutations, ensuring uniform marginal probability across all o
      slots (P(Slot_1 = v) == P(Slot_2 = v) == ... == P(Slot_o = v)).
    - Unit L1 Sensitivity per Household: Because the total emitted weight per
      parent is exactly 1.0, adding or removing a single parent entity changes
      the weighted exploration dataset histogram by at most L1 sensitivity
      Delta = 1.0.

  Args:
    k: Number of children in the household.
    num_permutation_slots: Permutation exploration slot count (o).
    strategy: 'empty_token' (permutes real and <EMPTY>) or 'size_sliced' (clone
      tiling).

  Returns:
    A tuple of (patterns, weight) where patterns is a list of o-tuples with
    child relative indices [0, k-1] (or -1 for <EMPTY>), and weight is the float
    weight for each emitted row such that len(patterns) * weight == 1.0.
  """
  if k == 0:
    empty_val = -1 if strategy == 'empty_token' else 0
    return [tuple(empty_val for _ in range(num_permutation_slots))], 1.0

  if k < num_permutation_slots:
    if strategy == 'size_sliced':
      return [tuple(i % k for i in range(num_permutation_slots))], 1.0
    items = list(range(k)) + [-1] * (num_permutation_slots - k)
    patterns = list(dict.fromkeys(itertools.permutations(items)))
    return patterns, 1.0 / len(patterns)

  patterns = list(itertools.permutations(range(k), num_permutation_slots))
  return patterns, 1.0 / len(patterns)


def build_permuted_exploration_dataset(
    parent_dataset: mbi.Dataset,
    child_dataset: mbi.Dataset,
    parent_primary_keys: Sequence[Hashable],
    child_foreign_keys: Sequence[Hashable],
    max_group_size: int,
    num_permutation_slots: int = 2,
    strategy: Literal['empty_token', 'size_sliced'] = 'empty_token',
) -> mbi.Dataset:
  """Constructs the permuted multi-slot exploration dataset for candidate selection.

  Formal Guarantees:
    - Inherited from `_get_slot_permutation_patterns`:
      - Mass: Total dataset weight mass strictly equals parent record count
        (sum(weights) == N_parents).
      - Sensitivity: Adding or removing a single parent entity (and its linked
        children) alters the weighted exploration table by at most Delta = 1.0.
      - Exact Slot Marginal Symmetry: Uniform permutation weighting guarantees
        identical marginal distributions across all child slots (P(Slot_i) ==
        P(Slot_j)), preventing slot bias during candidate query selection.

    - Inherited from `_build_exploration_domain`:
      - No Private Metadata Leakage: Exploration domain dimensions are fixed
        strictly by public metadata. No leaking private parent group sizes.

  Args:
    parent_dataset: Encoded discrete mbi.Dataset for the parent table.
    child_dataset: Encoded discrete mbi.Dataset for the child table.
    parent_primary_keys: Sequence of parent primary key identifiers.
    child_foreign_keys: Sequence of child foreign key references.
    max_group_size: Public upper bound for child group capacity (s >= 1).
    num_permutation_slots: Number of permutation slots (o) in exploration table,
      default 2.
    strategy: Exploration strategy ('empty_token' with <EMPTY> or
      'size_sliced').

  Returns:
    An mbi.Dataset instance representing the permuted exploration table.

  Raises:
    ValueError: If strategy is unsupported, num_permutation_slots < 1,
      max_group_size < 1, or key lengths do not match dataset record counts.
  """

  # This input validation can probably later be moved earlier in the pipeline.
  if max_group_size < 1:
    raise ValueError(f'max_group_size must be >= 1, got {max_group_size}')
  if num_permutation_slots < 1:
    raise ValueError(
        f'num_permutation_slots must be >= 1, got {num_permutation_slots}'
    )
  if strategy not in ('empty_token', 'size_sliced'):
    raise ValueError(
        f"strategy must be 'empty_token' or 'size_sliced', got {strategy!r}"
    )

  num_parents = parent_dataset.records
  if len(parent_primary_keys) != num_parents:
    raise ValueError(
        f'parent_primary_keys length ({len(parent_primary_keys)}) does not'
        f' match parent_dataset records ({num_parents})'
    )
  if len(child_foreign_keys) != child_dataset.records:
    raise ValueError(
        f'child_foreign_keys length ({len(child_foreign_keys)}) does not'
        f' match child_dataset records ({child_dataset.records})'
    )

  # Vectorized translation of child foreign keys to parent row indices.
  # parent_lookup: (Index = parent_pk, Value = parent row index [0, N_p-1]).
  parent_lookup = pd.Series(
      np.arange(num_parents), index=pd.Series(parent_primary_keys).values
  )
  parent_lookup = parent_lookup[~parent_lookup.index.duplicated(keep='first')]

  # child_parent_idx: (Index = child row, Value = parent row index [0, N_p-1]).
  child_parent_idx = (
      pd.Series(child_foreign_keys).map(parent_lookup).dropna().astype(int)
  )

  # Vectorized child counting and intra-parent ranking.
  # parent_group_sizes: (Index = parent row, Value = child count k).
  # child_ranks: 1D array of N_c intra-parent ranks (0, 1, ... k-1) per child.
  parent_group_sizes = pd.Series(0, index=np.arange(num_parents))
  if not child_parent_idx.empty:
    counts = child_parent_idx.value_counts()
    parent_group_sizes.loc[counts.index] = counts.values
    child_ranks = (
        child_parent_idx.groupby(child_parent_idx).cumcount().to_numpy()
    )
  else:
    child_ranks = np.empty(0, dtype=int)

  # Construct exploration domain fixed strictly by public max_group_size.
  exploration_domain = _build_exploration_domain(
      parent_domain=parent_dataset.domain,
      child_domain=child_dataset.domain,
      max_group_size=max_group_size,
      num_permutation_slots=num_permutation_slots,
      strategy=strategy,
  )

  parent_cols = list(parent_dataset.domain.attributes)
  child_cols = list(child_dataset.domain.attributes)

  # Example: real children have ages [0,9] -> empty_token: {'age': 10 = |[0,9]|}
  empty_tokens = dict(
      zip(child_dataset.domain.attributes, child_dataset.domain.shape)
  )

  # Vectorized block assembly grouped by unique family size k.
  # Avoids iterating over all N_p parent rows by processing all parents of the
  # same family size in bulk NumPy operations (at most s iterations total).
  block_arrays: dict[str | int, list[np.ndarray]] = {
      attr: [] for attr in exploration_domain.attributes
  }
  weights_blocks: list[np.ndarray] = []

  valid_child_parents = child_parent_idx.to_numpy()
  valid_child_rows = child_parent_idx.index.to_numpy()

  for k in np.unique(parent_group_sizes.values):
    parent_indices_k = np.where(parent_group_sizes.values == k)[0]
    n_k = len(parent_indices_k)
    if n_k == 0:
      continue

    # Get permutation template matrix (p_k, o) and row weight (w = 1 / p_k).
    # Examples for o=2 under strategy='empty_token':
    #   k=0: pattern_matrix = [[-1, -1]]                   (p_k=1, weight=1.0)
    #   k=1: pattern_matrix = [[ 0, -1], [-1, 0]]          (p_k=2, weight=0.5)
    #   k=2: pattern_matrix = [[ 0,  1], [ 1, 0]]          (p_k=2, weight=0.5)
    #   k=3: pattern_matrix = [[0,1],[0,2],[1,0],[1,2]...] (p_k=6, weight=1/6)
    patterns, weight = _get_slot_permutation_patterns(
        int(k), num_permutation_slots, strategy
    )
    p_k = len(patterns)
    pattern_matrix = np.array(patterns, dtype=np.int64)

    # Broadcast parent features, group_size, and weights (length = n_k * p_k).
    # Example: parent_indices_k=[0, 1], p_k=2 -> parent_rep=[0, 0, 1, 1]
    parent_rep = np.repeat(parent_indices_k, p_k)
    weights_blocks.append(np.full(n_k * p_k, weight, dtype=np.float64))
    block_arrays['group_size'].append(np.full(n_k * p_k, k, dtype=np.int64))
    for p_col in parent_cols:
      block_arrays[p_col].append(parent_dataset.data[p_col][parent_rep])

    # Broadcast child slot features across repeated parents.
    if k == 0:
      # Childless: fill all o slots with <EMPTY> (or 0 for size_sliced).
      # Example for o=2: empty_tokens={'age': 10} -> slot_1=[10], slot_2=[10]
      for slot_idx in range(1, num_permutation_slots + 1):
        for c_col in child_cols:
          val = empty_tokens[c_col] if strategy == 'empty_token' else 0
          block_arrays[f'slot_{slot_idx}.{c_col}'].append(
              np.full(n_k * p_k, val, dtype=np.int64)
          )
    else:
      # Multi-child: construct 2D index grid (n_k, k) mapping
      # (local_parent_idx, intra_group_rank) -> global child row index.
      # Example: H_A has children [10, 11], H_B has [20, 21]
      # -> child_grid = [[10, 11], [20, 21]] (row=household, col=sibling rank)
      child_mask_k = np.isin(valid_child_parents, parent_indices_k)
      local_parent_pos = (
          pd.Series(np.arange(n_k), index=parent_indices_k)
          .loc[valid_child_parents[child_mask_k]]
          .to_numpy()
      )

      child_grid = np.empty((n_k, k), dtype=np.int64)
      child_grid[local_parent_pos, child_ranks[child_mask_k]] = (
          valid_child_rows[child_mask_k]
      )

      # Broadcast relative permutation slot indices into global child rows.
      # Example for o=2, pattern_matrix=[[0, 1], [1, 0]]:
      #   Slot 1: rel_indices=tile([0, 1], 2) -> [0, 1, 0, 1]
      #           picks: (0,0)->10, (0,1)->11, (1,0)->20, (1,1)->21
      #   Slot 2: rel_indices=tile([1, 0], 2) -> [1, 0, 1, 0]
      #           picks: (0,1)->11, (0,0)->10, (1,1)->21, (1,0)->20
      local_parent_rep = np.repeat(np.arange(n_k), p_k)
      for slot_idx in range(1, num_permutation_slots + 1):
        rel_indices = np.tile(pattern_matrix[:, slot_idx - 1], n_k)
        is_empty = rel_indices == -1
        safe_rel_indices = np.maximum(rel_indices, 0)
        active_child_rows = child_grid[local_parent_rep, safe_rel_indices]

        for c_col in child_cols:
          slot_col_name = f'slot_{slot_idx}.{c_col}'
          child_vals = child_dataset.data[c_col][active_child_rows]
          if strategy == 'empty_token':
            # For empty slot (-1), replace dummy child row with <EMPTY> token.
            child_vals = np.where(is_empty, empty_tokens[c_col], child_vals)
          block_arrays[slot_col_name].append(child_vals)

  # Concatenate block arrays into final mbi.Dataset.
  data_arrays: dict[str | int, np.ndarray] = {
      attr: np.concatenate(arrs) if arrs else np.empty(0, dtype=np.int64)
      for attr, arrs in block_arrays.items()
  }
  weights_array = (
      np.concatenate(weights_blocks)
      if weights_blocks
      else np.empty(0, dtype=np.float64)
  )
  return mbi.Dataset(data_arrays, exploration_domain, weights=weights_array)


def create_slot_linear_chain_constraints(
    child_domain: mbi.Domain,
    num_permutation_slots: int = 2,
) -> list[mbi.Constraint]:
  """Creates adjacent pairwise mbi.Constraint objects for monolithic slot validity.

  For each slot, generates D-1 pairwise adjacent constraints ((S_i.A_1,
  S_i.A_2),
  (S_i.A_2, S_i.A_3), ...) setting log-potential to -inf on mixed states,
  ensuring sampled slots are 100% Real or 100% <EMPTY> with bounded treewidth <=
  2.

  Formal Guarantees:
    - Transitive Monolithic Slot Locking: By chaining pairwise constraints
      (A_1 = E <=> A_2 = E <=> ... <=> A_D = E), any mixed state containing both
      real values and <EMPTY> has joint log-potential = -inf (P = 0.0).
    - Treewidth <= 2 Bounded Complexity: Pairwise linear chains avoid star-graph
      hubs and high-dimensional cliques, keeping maximum constraint clique size
      to 2 (memory <= (K+1)^2 entries per factor) to prevent junction tree OOMs.
    - Zero Private Information: Constraints are constructed purely from public
      domain metadata and slot count, inducing zero DP privacy loss (eps = 0).

  Args:
    child_domain: Sub-domain representing attributes of a single child record.
    num_permutation_slots: Number of permutation slots (o), default 2.

  Returns:
    A list of mbi.Constraint instances enforcing monolithic slot locking.

  Raises:
    ValueError: If num_permutation_slots < 1.
  """
  if num_permutation_slots < 1:
    raise ValueError(
        f'num_permutation_slots must be >= 1, got {num_permutation_slots}'
    )

  child_attrs = child_domain.attributes
  child_shape = child_domain.shape
  num_attrs = len(child_attrs)

  # Single- or 0-attribute child domain requires no cross-attribute constraints.
  if num_attrs < 2:
    return []

  constraints: list[mbi.Constraint] = []
  for slot_idx in range(1, num_permutation_slots + 1):  # slot_1 to slot_o
    for i in range(num_attrs - 1):
      attr1, attr2 = child_attrs[i], child_attrs[i + 1]
      k1, k2 = child_shape[i], child_shape[i + 1]

      slot_attr1 = f'slot_{slot_idx}.{attr1}'
      slot_attr2 = f'slot_{slot_idx}.{attr2}'
      pair_domain = mbi.Domain((slot_attr1, slot_attr2), (k1 + 1, k2 + 1))

      # Mixed states: (k1, [0..k2-1]) and ([0..k1-1], k2).
      invalid_combos = [(k1, a2) for a2 in range(k2)] + [
          (a1, k2) for a1 in range(k1)
      ]
      invalid_arr = np.array(invalid_combos, dtype=np.int64)

      constraints.append(
          mbi.Constraint(domain=pair_domain, invalid=invalid_arr)
      )

  return constraints


def _extract_slot_indices(clique: Sequence[str | int]) -> list[int]:
  """Extracts unique sorted 1-based slot indices present in a measurement clique.

    - Returns sorted unique integer slot indices
      ('income', 'slot_2.age', 'slot_1.gender') -> [1, 2]).
    - Attributes not adhering to the 'slot_<idx>.<attr>' pattern
      (such as parent features or 'group_size') are cleanly ignored.

  Example:
    Household -> Person -> Activity running schema:
      >>> _extract_slot_indices(('income', 'region'))
      []
      >>> _extract_slot_indices(('income', 'slot_1.age', 'slot_1.gender'))
      [1]
      >>> _extract_slot_indices(('group_size', 'slot_2.gender', 'slot_1.age'))
      [1, 2]
      >>> _extract_slot_indices(('age', 'slot_1.amount', 'slot_2.type'))
      [1, 2]

  Args:
    clique: Sequence of attribute names defining a marginal measurement.

  Returns:
    A sorted list of unique integer slot indices present in the clique.
  """
  slot_indices: set[int] = set()
  for attr in clique:
    if isinstance(attr, str) and attr.startswith('slot_') and '.' in attr:
      prefix = attr.split('.', 1)[0]
      slot_str = prefix[len('slot_') :]
      if slot_str.isdigit():
        slot_indices.add(int(slot_str))
  return sorted(slot_indices)


def _remap_clique_slots(
    clique: Sequence[str | int],
    slot_mapping: Mapping[int, int],
) -> tuple[str | int, ...]:
  """Remaps slot index prefixes in a clique according to a given slot mapping.

  - Order & Non-Slot Invariance: Attributes not matching 'slot_<idx>.<attr>'
    and their relative positions in the clique tuple are strictly preserved.
  - Deterministic Substitution: Replaces 'slot_{orig_idx}.{col}' with
    'slot_{target_idx}.{col}' for any orig_idx in slot_mapping.

  Example:
    Household -> Person -> Activity running schema:
      >>> _remap_clique_slots(('income', 'slot_1.age'), {1: 3})
      ('income', 'slot_3.age')
      >>> _remap_clique_slots(('slot_1.age', 'slot_2.gender'), {1: 2, 2: 5})
      ('slot_2.age', 'slot_5.gender')
      >>> _remap_clique_slots(('age', 'slot_1.amount'), {1: 2})
      ('age', 'slot_2.amount')

  Args:
    clique: Sequence of attribute names defining a marginal measurement.
    slot_mapping: Dictionary mapping original 1-based slot indices to target
      slot indices.

  Returns:
    A tuple of attribute names with remapped slot indices.
  """
  new_attrs: list[str | int] = []
  for attr in clique:
    if isinstance(attr, str) and attr.startswith('slot_') and '.' in attr:
      prefix, col_name = attr.split('.', 1)
      slot_str = prefix[len('slot_') :]
      if slot_str.isdigit():
        orig_slot = int(slot_str)
        target_slot = slot_mapping.get(orig_slot, orig_slot)
        new_attrs.append(f'slot_{target_slot}.{col_name}')
        continue
    new_attrs.append(attr)
  return tuple(new_attrs)


def symmetrize_to_wide_domain(
    measurements: Sequence[mbi.LinearMeasurement],
    max_children_per_parent: int,
    num_permutation_slots: int = 2,
) -> list[mbi.LinearMeasurement]:
  """Replicates selected exploration measurements across all generation slots.

  Candidate selection (e.g. AIM, MST) explores dependencies in a compact
  o-slot exploration table (typically o = 2). In generation, families can have
  up to s = max_children_per_parent slots. Because child records within a parent
  are exchangeable, this function equivariantly replicates measurements across
  all single slots and all comb(s, r) multi-slot combinations:
    - Single-slot measurements (P, S_1) replicate symmetrically across all s
      slots: (P, S_1), ..., (P, S_s).
    - Multi-slot measurements (S_1, S_2) replicate symmetrically across all
      comb(s, 2) sibling pairs: (S_i, S_j) for 1 <= i < j <= s.
    - Parent-only and metadata measurements (e.g. ('income', 'region') or
      ('group_size',)) are passed through directly without duplication.
    - Preserves noisy_measurement datavector and stddev across all copies.

  Example:
    3-Tier Hierarchy: Household -> Person (s=3) -> Activity (s=2):

    Exploration Measurements (Household -> Person, o=2):
      - M1: ('income', 'group_size')
          -> ('income', 'group_size') [1 copy]
      - M2: ('income', 'slot_1.age')
          -> ('income', 'slot_1.age'), ('income', 'slot_2.age'),
             ('income', 'slot_3.age') [3 copies]
      - M3: ('slot_1.age', 'slot_1.gender')
          -> ('slot_1.age', 'slot_1.gender'), ('slot_2.age', 'slot_2.gender'),
             ('slot_3.age', 'slot_3.gender') [3 copies]
      - M4: ('slot_1.age', 'slot_2.age')
          -> ('slot_1.age', 'slot_2.age'), ('slot_1.age', 'slot_3.age'),
             ('slot_2.age', 'slot_3.age') [3 copies]

    Exploration Measurements (Person -> Activity, o=2, s=2):
      - M5: ('age', 'slot_1.amount')
          -> ('age', 'slot_1.amount'), ('age', 'slot_2.amount') [2 copies]

  Args:
    measurements: Noisy marginal measurements from exploration candidate
      selection.
    max_children_per_parent: Maximum group capacity bound (s >= 1).
    num_permutation_slots: Number of permutation exploration slots (o >= 1),
      default 2.

  Returns:
    A list of expanded LinearMeasurement objects for the wide generation MRF.

  Raises:
    ValueError: If max_children_per_parent < 1 or num_permutation_slots < 1.
  """
  if max_children_per_parent < 1:
    raise ValueError(
        f'max_children_per_parent must be >= 1, got {max_children_per_parent}'
    )
  if num_permutation_slots < 1:
    raise ValueError(
        f'num_permutation_slots must be >= 1, got {num_permutation_slots}'
    )

  s = max_children_per_parent
  expanded: list[mbi.LinearMeasurement] = []

  for m in measurements:
    slots = _extract_slot_indices(m.clique)
    r = len(slots)

    if r == 0:
      expanded.append(m)
    elif r <= s:
      for target_combo in itertools.combinations(range(1, s + 1), r):
        mapping = dict(zip(slots, target_combo))
        new_clique = _remap_clique_slots(m.clique, mapping)
        expanded.append(
            mbi.LinearMeasurement(
                noisy_measurement=m.noisy_measurement,
                clique=new_clique,
                stddev=m.stddev,
                query=m.query,
            )
        )

  return expanded


def quantile_copula_coupling(
    synth_parents: mbi.Dataset,
    synth_wide_children: mbi.Dataset,
    parent_columns: Sequence[str],
    rng: np.random.Generator | None = None,
) -> mbi.Dataset:
  """Couples synthetic parents and wide child records via Quantile Copula Matching.

  Applies randomized within-bin tie-breaking and lexicographical sorting along
  parent feature coordinates to align parent records with wide child records.

  Args:
    synth_parents: Discrete mbi.Dataset of synthesized parent records.
    synth_wide_children: Discrete mbi.Dataset of synthesized wide child records.
    parent_columns: Parent feature columns used as the coupling anchor.
    rng: Random number generator for within-bin tie-breaking permutation.

  Returns:
    The coupled wide child discrete mbi.Dataset aligned with synthetic parents.
  """
  del synth_parents, synth_wide_children, parent_columns, rng
  raise NotImplementedError('quantile_copula_coupling is not yet implemented.')


def unstack_wide_family_records(
    synth_wide_dataset: mbi.Dataset,
    child_domain: mbi.Domain,
    max_children_per_parent: int,
) -> tuple[mbi.Dataset, np.ndarray]:
  """Unstacks wide family records into a standard normalized child mbi.Dataset.

  Reads group_size = k on each wide row, emits the active child records, and
  returns the unstacked child dataset along with a 1D mapping array of parent
  row indices.

  Args:
    synth_wide_dataset: Discrete mbi.Dataset of wide family records.
    child_domain: Single-child mbi.Domain defining attribute sizes.
    max_children_per_parent: Maximum group capacity bound (s).

  Returns:
    A tuple of (unstacked_child_dataset, parent_row_indices) where
    parent_row_indices maps each unstacked child record to its parent row.
  """
  del synth_wide_dataset, child_domain, max_children_per_parent
  raise NotImplementedError(
      'unstack_wide_family_records is not yet implemented.'
  )
