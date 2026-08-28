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

"""Unit tests for dpsynth.relational.post_processing."""

import itertools
import math
from absl.testing import absltest
from dpsynth.relational import post_processing
import mbi
import numpy as np
import pandas as pd


class PostProcessingTest(absltest.TestCase):

  def test_create_slot_linear_chain_constraints_single_or_zero_attribute(self):
    dom_0 = mbi.Domain((), ())
    self.assertEmpty(
        post_processing.create_slot_linear_chain_constraints(
            dom_0, num_permutation_slots=2
        )
    )
    dom_1 = mbi.Domain(('age',), (10,))
    self.assertEmpty(
        post_processing.create_slot_linear_chain_constraints(
            dom_1, num_permutation_slots=2
        )
    )

  def test_create_slot_linear_chain_constraints_multi_attribute(self):
    # 3 child attributes: age (size 10), gender (size 2), status (size 5)
    # o = 2 slots -> 2 adjacent pairs per slot -> 4 constraints total
    child_dom = mbi.Domain(
        ('age', 'gender', 'status'),
        (10, 2, 5),
    )
    constraints = post_processing.create_slot_linear_chain_constraints(
        child_dom, num_permutation_slots=2
    )
    self.assertLen(constraints, 4)

    # Slot 1, Pair 1: (slot_1.age, slot_1.gender), domain (11, 3)
    c1 = constraints[0]
    self.assertEqual(c1.domain.attributes, ('slot_1.age', 'slot_1.gender'))
    self.assertEqual(c1.domain.shape, (11, 3))
    # Inv. combinations: (10, [0..1]) and ([0..9], 2) -> 2 + 10 = 12 inv. states
    self.assertLen(c1.invalid, 12)
    assert c1.invalid is not None
    # (10, 0) is mixed state -> must be in invalid
    self.assertTrue(np.any((c1.invalid == [10, 0]).all(axis=1)))
    # (0, 2) is mixed state -> must be in invalid
    self.assertTrue(np.any((c1.invalid == [0, 2]).all(axis=1)))
    # Valid real state (5, 1) -> must NOT be in invalid
    self.assertFalse(np.any((c1.invalid == [5, 1]).all(axis=1)))
    # Valid empty state (10, 2) -> must NOT be in invalid
    self.assertFalse(np.any((c1.invalid == [10, 2]).all(axis=1)))

    # Slot 1, Pair 2: (slot_1.gender, slot_1.status), domain (3, 6)
    c2 = constraints[1]
    self.assertEqual(c2.domain.attributes, ('slot_1.gender', 'slot_1.status'))
    self.assertEqual(c2.domain.shape, (3, 6))

    # Slot 2, Pair 1: (slot_2.age, slot_2.gender)
    c3 = constraints[2]
    self.assertEqual(c3.domain.attributes, ('slot_2.age', 'slot_2.gender'))

    # Slot 2, Pair 2: (slot_2.gender, slot_2.status)
    c4 = constraints[3]
    self.assertEqual(c4.domain.attributes, ('slot_2.gender', 'slot_2.status'))

  def test_create_slot_linear_chain_constraints_validation_errors(self):
    child_dom = mbi.Domain(('age', 'gender'), (10, 2))
    with self.assertRaises(ValueError):
      post_processing.create_slot_linear_chain_constraints(
          child_dom, num_permutation_slots=0
      )

  def test_extract_slot_indices(self):
    self.assertEqual(
        post_processing._extract_slot_indices(('income', 'region')), []
    )
    self.assertEqual(
        post_processing._extract_slot_indices(
            ('income', 'slot_1.age', 'slot_1.gender')
        ),
        [1],
    )
    self.assertEqual(
        post_processing._extract_slot_indices(
            ('group_size', 'slot_2.gender', 'slot_1.age')
        ),
        [1, 2],
    )
    self.assertEqual(
        post_processing._extract_slot_indices(
            ('age', 'slot_1.amount', 'slot_2.type')
        ),
        [1, 2],
    )

  def test_remap_clique_slots(self):
    self.assertEqual(
        post_processing._remap_clique_slots(('income', 'slot_1.age'), {1: 3}),
        ('income', 'slot_3.age'),
    )
    self.assertEqual(
        post_processing._remap_clique_slots(
            ('slot_1.age', 'slot_2.gender'), {1: 2, 2: 5}
        ),
        ('slot_2.age', 'slot_5.gender'),
    )
    self.assertEqual(
        post_processing._remap_clique_slots(('age', 'slot_1.amount'), {1: 2}),
        ('age', 'slot_2.amount'),
    )

  def test_symmetrize_to_wide_domain_running_example(self):
    # 3-Tier Hierarchy: Household -> Person (s=3, o=2) -> Activity (s=2, o=2)
    # 1. Household -> Person:
    #     Parent=('income',), Child=('age', 'gender'), s=3, o=2
    m1 = mbi.LinearMeasurement(
        np.array([1.0]), ('income', 'group_size'), stddev=1.0
    )
    m2 = mbi.LinearMeasurement(
        np.array([2.0]), ('income', 'slot_1.age'), stddev=1.0
    )
    m3 = mbi.LinearMeasurement(
        np.array([3.0]), ('slot_1.age', 'slot_1.gender'), stddev=1.0
    )
    m4 = mbi.LinearMeasurement(
        np.array([4.0]), ('slot_1.age', 'slot_2.age'), stddev=1.0
    )

    expanded = post_processing.symmetrize_to_wide_domain(
        measurements=[m1, m2, m3, m4],
        max_children_per_parent=3,
        num_permutation_slots=2,
    )

    # Expected:
    # m1 (parent-only/group_size) -> 1 copy: ('income', 'group_size')
    # m2 (single-slot) -> 3 copies: ('income', 'slot_1.age'),
    #     ('income', 'slot_2.age'), ('income', 'slot_3.age')
    # m3 (single-slot) -> 3 copies: ('slot_1.age', 'slot_1.gender'),
    #     ('slot_2.age', 'slot_2.gender'), ('slot_3.age', 'slot_3.gender')
    # m4 (pair-slot)   -> 3 copies: ('slot_1.age', 'slot_2.age'),
    #     ('slot_1.age', 'slot_3.age'), ('slot_2.age', 'slot_3.age')
    # Total = 1 + 3 + 3 + 3 = 10 measurements
    self.assertLen(expanded, 10)

    cliques = [m.clique for m in expanded]
    # m1
    self.assertIn(('income', 'group_size'), cliques)
    # m2
    self.assertIn(('income', 'slot_1.age'), cliques)
    self.assertIn(('income', 'slot_2.age'), cliques)
    self.assertIn(('income', 'slot_3.age'), cliques)
    # m3
    self.assertIn(('slot_1.age', 'slot_1.gender'), cliques)
    self.assertIn(('slot_2.age', 'slot_2.gender'), cliques)
    self.assertIn(('slot_3.age', 'slot_3.gender'), cliques)
    # m4
    self.assertIn(('slot_1.age', 'slot_2.age'), cliques)
    self.assertIn(('slot_1.age', 'slot_3.age'), cliques)
    self.assertIn(('slot_2.age', 'slot_3.age'), cliques)

    # 2. Person -> Activity: Parent=('age',), Child=('amount',), s=2, o=2
    m5 = mbi.LinearMeasurement(
        np.array([5.0]), ('age', 'slot_1.amount'), stddev=1.0
    )
    expanded_activity = post_processing.symmetrize_to_wide_domain(
        measurements=[m5],
        max_children_per_parent=2,
        num_permutation_slots=2,
    )
    self.assertLen(expanded_activity, 2)
    activity_cliques = [m.clique for m in expanded_activity]
    self.assertIn(('age', 'slot_1.amount'), activity_cliques)
    self.assertIn(('age', 'slot_2.amount'), activity_cliques)

  def test_symmetrize_to_wide_domain_various_clique_types(self):
    # s = 1 (boundary case): pair measurements should produce 0 copies
    m_pair = mbi.LinearMeasurement(
        np.array([1.0]), ('slot_1.a', 'slot_2.b'), stddev=1.0
    )
    expanded = post_processing.symmetrize_to_wide_domain(
        [m_pair], max_children_per_parent=1, num_permutation_slots=2
    )
    self.assertEmpty(expanded)

    # Single measurement with s=1
    m_single = mbi.LinearMeasurement(np.array([1.0]), ('slot_1.a',), stddev=1.0)
    expanded = post_processing.symmetrize_to_wide_domain(
        [m_single], max_children_per_parent=1, num_permutation_slots=2
    )
    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0].clique, ('slot_1.a',))

  def test_symmetrize_to_wide_domain_validation_errors(self):
    m = mbi.LinearMeasurement(np.array([1.0]), ('income',), stddev=1.0)
    with self.assertRaises(ValueError):
      post_processing.symmetrize_to_wide_domain(
          [m], max_children_per_parent=0, num_permutation_slots=2
      )
    with self.assertRaises(ValueError):
      post_processing.symmetrize_to_wide_domain(
          [m], max_children_per_parent=3, num_permutation_slots=0
      )

  def test_quantile_copula_coupling_running_example(self):
    # 3-Tier running example: Person (age, gender) -> Activity
    parent_dom = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    synth_parents = mbi.Dataset(
        {
            'age': np.array([9, 2], dtype=np.int64),
            'gender': np.array([0, 1], dtype=np.int64),
        },
        parent_dom,
    )

    child_dom = mbi.Domain.fromdict({
        'age': 10,
        'gender': 2,
        'group_size': 3,
        'slot_1.amount': 10,
    })
    synth_wide_children = mbi.Dataset(
        {
            'age': np.array([2, 9], dtype=np.int64),
            'gender': np.array([1, 0], dtype=np.int64),
            'group_size': np.array([2, 1], dtype=np.int64),
            'slot_1.amount': np.array([100, 200], dtype=np.int64),
        },
        child_dom,
    )

    coupled = post_processing.quantile_copula_coupling(
        synth_parents=synth_parents,
        synth_wide_children=synth_wide_children,
        parent_columns=['age', 'gender'],
    )

    # Row 0 of parents (age=9, gender=0) must align with child row
    # (age=9, gender=0, amt=200)
    self.assertEqual(int(coupled.data['age'][0]), 9)
    self.assertEqual(int(coupled.data['gender'][0]), 0)
    self.assertEqual(int(coupled.data['slot_1.amount'][0]), 200)
    self.assertEqual(int(coupled.data['group_size'][0]), 1)

    # Row 1 of parents (age=2, gender=1) must align with child row
    # (age=2, gender=1, amt=100)
    self.assertEqual(int(coupled.data['age'][1]), 2)
    self.assertEqual(int(coupled.data['gender'][1]), 1)
    self.assertEqual(int(coupled.data['slot_1.amount'][1]), 100)
    self.assertEqual(int(coupled.data['group_size'][1]), 2)

  def test_quantile_copula_coupling_multi_column_lexicographical_order(self):
    parent_dom = mbi.Domain.fromdict({'c1': 5, 'c2': 5})
    # Parents: (1, 2), (1, 1), (0, 4) -> Sorted order: (0, 4), (1, 1), (1, 2)
    synth_parents = mbi.Dataset(
        {
            'c1': np.array([1, 1, 0], dtype=np.int64),
            'c2': np.array([2, 1, 4], dtype=np.int64),
        },
        parent_dom,
    )

    child_dom = mbi.Domain.fromdict({'c1': 5, 'c2': 5, 'val': 100})
    # Children: (1, 1, val=11), (0, 4, val=4), (1, 2, val=12)
    synth_wide_children = mbi.Dataset(
        {
            'c1': np.array([1, 0, 1], dtype=np.int64),
            'c2': np.array([1, 4, 2], dtype=np.int64),
            'val': np.array([11, 4, 12], dtype=np.int64),
        },
        child_dom,
    )

    coupled = post_processing.quantile_copula_coupling(
        synth_parents=synth_parents,
        synth_wide_children=synth_wide_children,
        parent_columns=['c1', 'c2'],
    )

    # Parent 0 (1, 2) -> matched with val=12
    self.assertEqual(int(coupled.data['val'][0]), 12)
    # Parent 1 (1, 1) -> matched with val=11
    self.assertEqual(int(coupled.data['val'][1]), 11)
    # Parent 2 (0, 4) -> matched with val=4
    self.assertEqual(int(coupled.data['val'][2]), 4)

  def test_quantile_copula_coupling_within_bin_tie_breaking(self):
    # Multiple parents and children in the exact same bin (age=5)
    parent_dom = mbi.Domain.fromdict({'age': 10})
    synth_parents = mbi.Dataset(
        {'age': np.array([5, 5, 5, 5], dtype=np.int64)}, parent_dom
    )

    child_dom = mbi.Domain.fromdict({'age': 10, 'child_id': 10})
    synth_wide_children = mbi.Dataset(
        {
            'age': np.array([5, 5, 5, 5], dtype=np.int64),
            'child_id': np.array([0, 1, 2, 3], dtype=np.int64),
        },
        child_dom,
    )

    # With fixed seeds, verify reproducibility
    rng1 = np.random.default_rng(42)
    coupled1 = post_processing.quantile_copula_coupling(
        synth_parents,
        synth_wide_children,
        parent_columns=['age'],
        rng=rng1,
    )
    rng2 = np.random.default_rng(42)
    coupled2 = post_processing.quantile_copula_coupling(
        synth_parents,
        synth_wide_children,
        parent_columns=['age'],
        rng=rng2,
    )
    np.testing.assert_array_equal(
        coupled1.data['child_id'], coupled2.data['child_id']
    )
    # All 4 child IDs must be present exactly once
    self.assertCountEqual(coupled1.data['child_id'].tolist(), [0, 1, 2, 3])

  def test_quantile_copula_coupling_empty_and_edge_cases(self):
    parent_dom = mbi.Domain.fromdict({'p': 3})
    child_dom = mbi.Domain.fromdict({'p': 3, 'c': 4})

    # Empty datasets (N = 0)
    p_empty = mbi.Dataset({'p': np.empty(0, dtype=np.int64)}, parent_dom)
    c_empty = mbi.Dataset(
        {'p': np.empty(0, dtype=np.int64), 'c': np.empty(0, dtype=np.int64)},
        child_dom,
    )
    coupled_empty = post_processing.quantile_copula_coupling(
        p_empty, c_empty, parent_columns=['p']
    )
    self.assertEqual(coupled_empty.records, 0)

    # Empty parent_columns list -> returns child dataset as-is
    p_ds = mbi.Dataset({'p': np.array([1, 2], dtype=np.int64)}, parent_dom)
    c_ds = mbi.Dataset(
        {
            'p': np.array([1, 2], dtype=np.int64),
            'c': np.array([10, 20], dtype=np.int64),
        },
        child_dom,
    )
    coupled_no_cols = post_processing.quantile_copula_coupling(
        p_ds, c_ds, parent_columns=[]
    )
    np.testing.assert_array_equal(coupled_no_cols.data['c'], c_ds.data['c'])

  def test_quantile_copula_coupling_validation_errors(self):
    parent_dom = mbi.Domain.fromdict({'p': 3})
    child_dom = mbi.Domain.fromdict({'p': 3, 'c': 4})

    p_ds = mbi.Dataset({'p': np.array([1, 2], dtype=np.int64)}, parent_dom)
    c_ds_len3 = mbi.Dataset(
        {
            'p': np.array([1, 2, 0], dtype=np.int64),
            'c': np.array([1, 2, 3], dtype=np.int64),
        },
        child_dom,
    )

    # Mismatched records count (2 vs 3)
    with self.assertRaisesRegex(ValueError, 'does not match'):
      post_processing.quantile_copula_coupling(
          p_ds, c_ds_len3, parent_columns=['p']
      )

    c_ds_len2 = mbi.Dataset(
        {
            'p': np.array([1, 2], dtype=np.int64),
            'c': np.array([1, 2], dtype=np.int64),
        },
        child_dom,
    )
    # Missing column in parent domain
    with self.assertRaisesRegex(ValueError, 'not in synth_parents'):
      post_processing.quantile_copula_coupling(
          p_ds, c_ds_len2, parent_columns=['missing_col']
      )

    # Missing column in child domain
    p_dom_extra = mbi.Domain.fromdict({'p': 3, 'extra': 2})
    p_ds_extra = mbi.Dataset(
        {
            'p': np.array([1, 2], dtype=np.int64),
            'extra': np.array([0, 1], dtype=np.int64),
        },
        p_dom_extra,
    )
    with self.assertRaisesRegex(ValueError, 'not in synth_wide_children'):
      post_processing.quantile_copula_coupling(
          p_ds_extra, c_ds_len2, parent_columns=['extra']
      )

  def test_unstack_wide_family_records_running_example(self):
    # 3-Tier running example: Household -> Person (age [0..9], <EMPTY>=10)
    # s = 2
    # Parent 0: group_size=2, slot_1=5, slot_2=8
    # Parent 1: group_size=0, slot_1=10 (<EMPTY>), slot_2=10 (<EMPTY>)
    # Parent 2: group_size=1, slot_1=3, slot_2=10 (<EMPTY>)
    child_dom = mbi.Domain.fromdict({'age': 10})
    wide_dom = mbi.Domain.fromdict({
        'group_size': 3,
        'slot_1.age': 11,
        'slot_2.age': 11,
    })
    wide_ds = mbi.Dataset(
        {
            'group_size': np.array([2, 0, 1], dtype=np.int64),
            'slot_1.age': np.array([5, 10, 3], dtype=np.int64),
            'slot_2.age': np.array([8, 10, 10], dtype=np.int64),
        },
        wide_dom,
    )

    unstacked_ds, parent_indices = post_processing.unstack_wide_family_records(
        synth_wide_dataset=wide_ds,
        child_domain=child_dom,
        max_children_per_parent=2,
    )

    # 1. Total records: 2 from parent 0 + 0 from parent 1 + 1 from parent 2 = 3
    self.assertEqual(unstacked_ds.records, 3)
    self.assertLen(parent_indices, 3)

    # 2. Output values match active slots: [5, 8, 3]
    self.assertEqual(unstacked_ds.data['age'].tolist(), [5, 8, 3])

    # 3. Parent mappings match: [0, 0, 2]
    self.assertEqual(parent_indices.tolist(), [0, 0, 2])

    # 4. Target domain is strictly single-child domain
    self.assertEqual(unstacked_ds.domain.attributes, ('age',))
    self.assertEqual(unstacked_ds.domain.shape, (10,))

  def test_unstack_wide_family_records_size_sliced_strategy(self):
    # Strategy B: No <EMPTY> tokens (domain size 10), clone tiled slots > k
    # Parent 0: k=1, slot_1=7, slot_2=7 (clone) -> slot_2 ignored
    # Parent 1: k=2, slot_1=4, slot_2=9         -> both slots emitted
    child_dom = mbi.Domain.fromdict({'age': 10})
    wide_dom = mbi.Domain.fromdict({
        'group_size': 3,
        'slot_1.age': 10,
        'slot_2.age': 10,
    })
    wide_ds = mbi.Dataset(
        {
            'group_size': np.array([1, 2], dtype=np.int64),
            'slot_1.age': np.array([7, 4], dtype=np.int64),
            'slot_2.age': np.array([7, 9], dtype=np.int64),
        },
        wide_dom,
    )

    unstacked_ds, parent_indices = post_processing.unstack_wide_family_records(
        synth_wide_dataset=wide_ds,
        child_domain=child_dom,
        max_children_per_parent=2,
    )

    self.assertEqual(unstacked_ds.records, 3)
    self.assertEqual(unstacked_ds.data['age'].tolist(), [7, 4, 9])
    self.assertEqual(parent_indices.tolist(), [0, 1, 1])

  def test_unstack_wide_family_records_multi_attribute(self):
    # 2 child attributes: age (size 10), gender (size 2)
    # <EMPTY> tokens: age=10, gender=2
    child_dom = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    wide_dom = mbi.Domain.fromdict({
        'group_size': 3,
        'slot_1.age': 11,
        'slot_1.gender': 3,
        'slot_2.age': 11,
        'slot_2.gender': 3,
    })
    wide_ds = mbi.Dataset(
        {
            'group_size': np.array([1], dtype=np.int64),
            'slot_1.age': np.array([4], dtype=np.int64),
            'slot_1.gender': np.array([1], dtype=np.int64),
            'slot_2.age': np.array([10], dtype=np.int64),
            'slot_2.gender': np.array([2], dtype=np.int64),
        },
        wide_dom,
    )

    unstacked_ds, parent_indices = post_processing.unstack_wide_family_records(
        synth_wide_dataset=wide_ds,
        child_domain=child_dom,
        max_children_per_parent=2,
    )

    self.assertEqual(unstacked_ds.records, 1)
    self.assertEqual(unstacked_ds.data['age'].tolist(), [4])
    self.assertEqual(unstacked_ds.data['gender'].tolist(), [1])
    self.assertEqual(parent_indices.tolist(), [0])

  def test_unstack_wide_family_records_all_childless_and_empty(self):
    child_dom = mbi.Domain.fromdict({'c': 5})
    wide_dom = mbi.Domain.fromdict({
        'group_size': 3,
        'slot_1.c': 6,
    })

    # All childless (k=0)
    wide_ds_zeros = mbi.Dataset(
        {
            'group_size': np.array([0, 0], dtype=np.int64),
            'slot_1.c': np.array([5, 5], dtype=np.int64),
        },
        wide_dom,
    )
    unstacked_zeros, p_idx_zeros = post_processing.unstack_wide_family_records(
        wide_ds_zeros, child_dom, max_children_per_parent=1
    )
    self.assertEqual(unstacked_zeros.records, 0)
    self.assertEmpty(p_idx_zeros)

    # Empty parent dataset (N=0)
    wide_ds_empty = mbi.Dataset(
        {
            'group_size': np.empty(0, dtype=np.int64),
            'slot_1.c': np.empty(0, dtype=np.int64),
        },
        wide_dom,
    )
    unstacked_empty, p_idx_empty = post_processing.unstack_wide_family_records(
        wide_ds_empty, child_dom, max_children_per_parent=1
    )
    self.assertEqual(unstacked_empty.records, 0)
    self.assertEmpty(p_idx_empty)

  def test_unstack_wide_family_records_validation_errors(self):
    child_dom = mbi.Domain.fromdict({'c': 5})
    wide_dom = mbi.Domain.fromdict({'group_size': 2, 'slot_1.c': 6})
    wide_ds = mbi.Dataset(
        {'group_size': np.array([1]), 'slot_1.c': np.array([2])}, wide_dom
    )

    with self.assertRaisesRegex(
        ValueError, 'max_children_per_parent must be >= 1'
    ):
      post_processing.unstack_wide_family_records(
          wide_ds, child_dom, max_children_per_parent=0
      )

    # Missing slot_2 in wide dataset domain when s=2
    with self.assertRaisesRegex(ValueError, 'Required slot column'):
      post_processing.unstack_wide_family_records(
          wide_ds, child_dom, max_children_per_parent=2
      )


class PostProcessingFormalGuaranteesPropertyTest(absltest.TestCase):
  """Property-based tests verifying mathematical invariants and DP guarantees."""

  def test_property_slot_linear_chain_constraints_locking_and_treewidth(self):
    """Verifies monolithic slot locking (zero mixed state) and treewidth <= 2."""
    rng = np.random.default_rng(12345)
    for _ in range(15):
      d = int(rng.integers(2, 6))  # 2 to 5 child attributes
      o = int(rng.integers(1, 4))  # 1 to 3 slots
      shapes = tuple(int(x) for x in rng.integers(2, 8, size=d))
      child_dom = mbi.Domain(tuple(f'a{i}' for i in range(d)), shapes)

      constraints = post_processing.create_slot_linear_chain_constraints(
          child_dom, num_permutation_slots=o
      )

      # 1. Total constraints count is exactly o * (D - 1)
      self.assertLen(constraints, o * (d - 1))

      # 2. Maximum clique dimension is strictly 2 (Treewidth <= 2 guarantee)
      for c in constraints:
        self.assertLen(c.domain.attributes, 2)
        self.assertLen(c.domain.shape, 2)

      # 3. For any slot i, valid monolithic real and empty states are NOT
      # in invalid.
      for slot in range(1, o + 1):
        for c in constraints:
          if not all(
              str(attr).startswith(f'slot_{slot}.')
              for attr in c.domain.attributes
          ):
            continue
          k1, k2 = c.domain.shape[0] - 1, c.domain.shape[1] - 1
          # Monolithic empty: (k1, k2) -> must be VALID (not in invalid)
          assert c.invalid is not None
          self.assertFalse(np.any((c.invalid == [k1, k2]).all(axis=1)))
          # Monolithic real: (0, 0) -> must be VALID (not in invalid)
          self.assertFalse(np.any((c.invalid == [0, 0]).all(axis=1)))

  def test_property_symmetrize_to_wide_domain_invariants(self):
    """Verifies equivariant measurement replication invariants under arbitrary s and o."""
    rng = np.random.default_rng(12345)
    for _ in range(15):
      s = int(rng.integers(1, 7))  # Generation group size bound [1..6]
      o = int(rng.integers(1, 4))  # Exploration slots [1..3]

      # Synthetic input candidate measurements:
      # 1. Parent-only: ('p1', 'p2')
      # 2. Single-slot: ('p1', 'slot_1.c1')
      # 3. Pairwise-slot: ('slot_1.c1', 'slot_2.c2') if o >= 2
      m_parent = mbi.LinearMeasurement(
          np.array([1.0]), ('p1', 'p2'), stddev=1.0
      )
      m_single = mbi.LinearMeasurement(
          np.array([1.0, 2.0]), ('p1', 'slot_1.c1'), stddev=1.2
      )
      measurements = [m_parent, m_single]
      if o >= 2:
        m_pair = mbi.LinearMeasurement(
            np.array([3.0, 4.0]), ('slot_1.c1', 'slot_2.c2'), stddev=1.5
        )
        measurements.append(m_pair)

      expanded = post_processing.symmetrize_to_wide_domain(
          measurements=measurements,
          max_children_per_parent=s,
          num_permutation_slots=o,
      )

      # 1. Parent-only measurement: exactly 1 copy
      parent_copies = [m for m in expanded if m.clique == ('p1', 'p2')]
      self.assertLen(parent_copies, 1)

      # 2. Single-slot measurement: exactly s copies across slot_1 ... slot_s
      single_copies = [
          m
          for m in expanded
          if len(m.clique) == 2
          and 'p1' in m.clique
          and m.clique != ('p1', 'p2')
      ]
      self.assertLen(single_copies, s)
      expected_single_cliques = [
          ('p1', f'slot_{k}.c1') for k in range(1, s + 1)
      ]
      self.assertCountEqual(
          [m.clique for m in single_copies], expected_single_cliques
      )

      # 3. Pairwise slot measurement:
      # exactly comb(s, 2) copies (if o >= 2 and s >= 2)
      pair_copies = [
          m
          for m in expanded
          if any(str(a).startswith('slot_') for a in m.clique)
          and not any(str(a).startswith('p') for a in m.clique)
      ]
      expected_pair_count = math.comb(s, 2) if (o >= 2 and s >= 2) else 0
      self.assertLen(pair_copies, expected_pair_count)
      if o >= 2 and s >= 2:
        expected_pair_cliques = [
            (f'slot_{i}.c1', f'slot_{j}.c2')
            for i, j in itertools.combinations(range(1, s + 1), 2)
        ]
        self.assertCountEqual(
            [m.clique for m in pair_copies], expected_pair_cliques
        )

      # 4. Measurement vector and stddev are identical across all copies
      for m in single_copies:
        self.assertEqual(m.stddev, 1.2)
        np.testing.assert_allclose(m.noisy_measurement, [1.0, 2.0])
      for m in pair_copies:
        self.assertEqual(m.stddev, 1.5)
        np.testing.assert_allclose(m.noisy_measurement, [3.0, 4.0])

  def test_property_quantile_copula_coupling_bijection_and_preservation(self):
    """Verifies that copula coupling creates an exact 1-to-1 bijection and preserves values/weights."""
    for i in range(15):
      seed = 12345 + i
      rng = np.random.default_rng(seed)
      n = int(rng.integers(1, 100))
      parent_dom = mbi.Domain.fromdict({'p1': 4, 'p2': 3})
      synth_parents = mbi.Dataset(
          {
              'p1': rng.integers(0, 4, size=n),
              'p2': rng.integers(0, 3, size=n),
          },
          parent_dom,
      )

      child_dom = mbi.Domain.fromdict({'p1': 4, 'p2': 3, 'c1': 5, 'c2': 2})
      child_weights = rng.random(n)
      synth_wide_children = mbi.Dataset(
          {
              'p1': rng.integers(0, 4, size=n),
              'p2': rng.integers(0, 3, size=n),
              'c1': rng.integers(0, 5, size=n),
              'c2': rng.integers(0, 2, size=n),
          },
          child_dom,
          weights=child_weights,
      )

      coupling_seed = 99999 + i
      coupled = post_processing.quantile_copula_coupling(
          synth_parents=synth_parents,
          synth_wide_children=synth_wide_children,
          parent_columns=['p1', 'p2'],
          rng=np.random.default_rng(coupling_seed),
      )

      # 1. Output record count strictly preserved
      self.assertEqual(coupled.records, n)

      # 2. Histogram / Value distribution of wide children is 100% preserved
      for attr in child_dom.attributes:
        self.assertCountEqual(
            coupled.data[attr].tolist(),
            synth_wide_children.data[attr].tolist(),
        )

      # 3. Total weight mass strictly preserved
      assert coupled.weights is not None
      self.assertAlmostEqual(
          float(np.sum(coupled.weights)), float(np.sum(child_weights))
      )

      # 4. Invariant: Quantile ordering monotonicity along primary anchor 'p1'
      # Under the exact sort_order_parents generated by the tie-breaking RNG,
      # the aligned children appear in monotonically non-decreasing order.
      copula_replay = np.random.default_rng(coupling_seed)
      parent_rand = copula_replay.random(n)
      p_keys = (parent_rand, synth_parents.data['p2'], synth_parents.data['p1'])
      sort_order_p = np.lexsort(p_keys)

      c_p1_aligned = coupled.data['p1'][sort_order_p]
      self.assertTrue(
          np.all(np.diff(c_p1_aligned) >= 0),
          msg=(
              'Coupled child p1 is not monotonically increasing:'
              f' {c_p1_aligned}'
          ),
      )

  def test_property_unstack_wide_family_records_consistency(self):
    """Verifies that unstacking strictly bounds children to group_size and preserves contiguity."""
    rng = np.random.default_rng(54321)
    for _ in range(15):
      n_parents = int(rng.integers(1, 50))
      s = int(rng.integers(1, 5))
      child_dom = mbi.Domain.fromdict({'c1': 4, 'c2': 3})

      # Generate wide domain with <EMPTY> tokens (c1=4, c2=3)
      attrs = ['group_size']
      shapes = [s + 1]
      for slot in range(1, s + 1):
        attrs.extend([f'slot_{slot}.c1', f'slot_{slot}.c2'])
        shapes.extend([5, 4])
      wide_dom = mbi.Domain(tuple(attrs), tuple(shapes))

      group_sizes = rng.integers(0, s + 1, size=n_parents)
      wide_data: dict[str | int, np.ndarray] = {'group_size': group_sizes}
      for slot in range(1, s + 1):
        # Slot active if slot <= group_size
        is_active = slot <= group_sizes
        c1_vals = np.where(is_active, rng.integers(0, 4, size=n_parents), 4)
        c2_vals = np.where(is_active, rng.integers(0, 3, size=n_parents), 3)
        wide_data[f'slot_{slot}.c1'] = c1_vals
        wide_data[f'slot_{slot}.c2'] = c2_vals

      wide_ds = mbi.Dataset(wide_data, wide_dom)
      unstacked_ds, parent_indices = (
          post_processing.unstack_wide_family_records(
              synth_wide_dataset=wide_ds,
              child_domain=child_dom,
              max_children_per_parent=s,
          )
      )

      # 1. Total child count strictly equals sum of group_sizes
      expected_total = int(np.sum(group_sizes))
      self.assertEqual(unstacked_ds.records, expected_total)
      self.assertLen(parent_indices, expected_total)

      # 2. For each parent, child count matches group_size
      if expected_total > 0:
        counts = pd.Series(parent_indices).value_counts()
        for p_idx, k in enumerate(group_sizes):
          self.assertEqual(counts.get(p_idx, 0), k)

      # 3. No <EMPTY> tokens in unstacked records
      if expected_total > 0:
        self.assertTrue(np.all(unstacked_ds.data['c1'] < 4))
        self.assertTrue(np.all(unstacked_ds.data['c2'] < 3))

      # 4. Parent indices are monotonically non-decreasing (contiguous siblings)
      if expected_total > 1:
        self.assertTrue(np.all(np.diff(parent_indices) >= 0))


if __name__ == '__main__':
  absltest.main()
