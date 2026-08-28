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

"""Unit tests for dpsynth.relational.transformations."""
from typing import Literal

from absl.testing import absltest
from dpsynth.relational import domain as rel_domain
from dpsynth.relational import transformations
import mbi
import numpy as np
import pandas as pd


class TransformationsTest(absltest.TestCase):

  def test_build_exploration_domain_empty_token(self):
    parent_domain = mbi.Domain.fromdict({'income': 4, 'region': 3})
    child_domain = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    # s = 3, o = 2
    domain = transformations.build_exploration_domain(
        parent_domain=parent_domain,
        child_domain=child_domain,
        max_group_size=3,
        num_permutation_slots=2,
        strategy='empty_token',
    )
    expected_attrs = (
        'income',
        'region',
        'group_size',
        'slot_1.age',
        'slot_1.gender',
        'slot_2.age',
        'slot_2.gender',
    )
    expected_shape = (4, 3, 4, 11, 3, 11, 3)
    self.assertEqual(domain.attributes, expected_attrs)
    self.assertEqual(domain.shape, expected_shape)

  def test_build_exploration_domain_size_sliced(self):
    parent_domain = mbi.Domain.fromdict({'income': 4})
    child_domain = mbi.Domain.fromdict({'age': 10})
    domain = transformations.build_exploration_domain(
        parent_domain=parent_domain,
        child_domain=child_domain,
        max_group_size=2,
        num_permutation_slots=2,
        strategy='size_sliced',
    )
    expected_attrs = ('income', 'group_size', 'slot_1.age', 'slot_2.age')
    expected_shape = (4, 3, 10, 10)
    self.assertEqual(domain.attributes, expected_attrs)
    self.assertEqual(domain.shape, expected_shape)

  def test_compute_row_root_mappings_running_example(self):
    # 3-tier running example: Household -> Person (s1=2) -> Activity (s2=2)
    # H0: P0, P1, P2 (P2 truncated by s1=2 limit)
    # H1: P3
    # A0(P0), A1(P1), A2(P2 - truncated parent), A3(P3), A4(orphan)
    households = pd.DataFrame({'hh_id': ['H0', 'H1'], 'income': [50000, 75000]})
    persons = pd.DataFrame({
        'person_id': ['P0', 'P1', 'P2', 'P3'],
        'hh_id': ['H0', 'H0', 'H0', 'H1'],
        'age': [35, 32, 5, 28],
    })
    activities = pd.DataFrame({
        'activity_id': ['A0', 'A1', 'A2', 'A3', 'A4'],
        'person_id': ['P0', 'P1', 'P2', 'P3', 'P_orphan'],
        'amount': [100, 200, 50, 300, 400],
    })
    tables = {
        'households': households,
        'persons': persons,
        'activities': activities,
    }

    hierarchy = [
        (0, 'households', None),
        (
            1,
            'persons',
            rel_domain.ForeignKeyRelation(
                child_table='persons',
                child_foreign_key='hh_id',
                parent_table='households',
                parent_primary_key='hh_id',
                max_children_per_parent=2,
            ),
        ),
        (
            2,
            'activities',
            rel_domain.ForeignKeyRelation(
                child_table='activities',
                child_foreign_key='person_id',
                parent_table='persons',
                parent_primary_key='person_id',
                max_children_per_parent=2,
            ),
        ),
    ]

    rng = np.random.default_rng(42)
    mappings = transformations._compute_row_root_mappings(
        tables, hierarchy, rng=rng
    )

    # 1. Check shapes and series alignments
    self.assertLen(mappings['households'], 2)
    self.assertLen(mappings['persons'], 4)
    self.assertLen(mappings['activities'], 5)

    # 2. Households map to their own row indices [0, 1]
    self.assertEqual(mappings['households'].tolist(), [0, 1])

    # 3. Persons: Exactly 2 children from H0 retained
    #     (mapped to 0), 1 dropped (None), P3 mapped to 1
    p_roots = mappings['persons'].tolist()
    self.assertEqual(p_roots[3], 1)
    self.assertEqual(p_roots.count(0), 2)
    self.assertEqual(p_roots.count(None), 1)

    # 4. Activities: A3 mapped to 1, A4 (orphan) mapped to None
    a_roots = mappings['activities'].tolist()
    self.assertEqual(a_roots[3], 1)
    self.assertIsNone(a_roots[4])
    # A2 must evaluate to None if its parent P2 was truncated
    if p_roots[2] is None:
      self.assertIsNone(a_roots[2])

  def test_compute_row_root_mappings_orphans_and_missing_keys(self):
    households = pd.DataFrame({'hh_id': ['H0'], 'income': [100]})
    persons = pd.DataFrame({
        'person_id': ['P0', 'P1', 'P2'],
        'hh_id': ['H0', None, 'H_nonexistent'],
    })
    tables = {'households': households, 'persons': persons}
    hierarchy = [
        (0, 'households', None),
        (
            1,
            'persons',
            rel_domain.ForeignKeyRelation(
                child_table='persons',
                child_foreign_key='hh_id',
                parent_table='households',
                parent_primary_key='hh_id',
                max_children_per_parent=2,
            ),
        ),
    ]
    mappings = transformations._compute_row_root_mappings(tables, hierarchy)
    self.assertEqual(mappings['persons'].tolist(), [0, None, None])

  def test_compute_row_root_mappings_empty_tables(self):
    households = pd.DataFrame({'hh_id': pd.Series(dtype=str)})
    persons = pd.DataFrame(
        {'person_id': pd.Series(dtype=str), 'hh_id': pd.Series(dtype=str)}
    )
    tables = {'households': households, 'persons': persons}
    hierarchy = [
        (0, 'households', None),
        (
            1,
            'persons',
            rel_domain.ForeignKeyRelation(
                child_table='persons',
                child_foreign_key='hh_id',
                parent_table='households',
                parent_primary_key='hh_id',
                max_children_per_parent=2,
            ),
        ),
    ]
    mappings = transformations._compute_row_root_mappings(tables, hierarchy)
    self.assertEmpty(mappings['households'])
    self.assertEmpty(mappings['persons'])

  def test_compute_row_root_mappings_schema_validation_error(self):
    households = pd.DataFrame({'wrong_id': ['H0']})
    persons = pd.DataFrame({'person_id': ['P0'], 'hh_id': ['H0']})
    tables = {'households': households, 'persons': persons}
    hierarchy = [
        (0, 'households', None),
        (
            1,
            'persons',
            rel_domain.ForeignKeyRelation(
                child_table='persons',
                child_foreign_key='hh_id',
                parent_table='households',
                parent_primary_key='hh_id',
                max_children_per_parent=2,
            ),
        ),
    ]
    with self.assertRaisesRegex(ValueError, 'Parent primary key column'):
      transformations._compute_row_root_mappings(tables, hierarchy)

  def test_compute_hierarchical_weights_running_example(self):
    # 3-tier running example: Household -> Person (s1=2) -> Activity (s2=2)
    # H0: P0, P1 active (w = 1/2 = 0.5 each), P2 truncated (w = 0.0)
    # H1: P3 active (w = 1/1 = 1.0)
    # A0(P0), A1(P1), A2(P2-truncated), A3(P3), A4(orphan)
    # Root H0 active activities: A0, A1 -> k_eff = 2 -> w = 1/2 = 0.5 each
    # Root H1 active activities: A3 -> k_eff = 1 -> w = 1.0
    # Inactive activities (A2, A4) -> w = 0.0
    households = pd.DataFrame({'hh_id': ['H0', 'H1']})
    persons = pd.DataFrame({
        'person_id': ['P0', 'P1', 'P2', 'P3'],
        'hh_id': ['H0', 'H0', 'H0', 'H1'],
    })
    activities = pd.DataFrame({
        'activity_id': ['A0', 'A1', 'A2', 'A3', 'A4'],
        'person_id': ['P0', 'P1', 'P2', 'P3', 'P_orphan'],
    })
    tables = {
        'households': households,
        'persons': persons,
        'activities': activities,
    }

    hierarchy = [
        (0, 'households', None),
        (
            1,
            'persons',
            rel_domain.ForeignKeyRelation(
                child_table='persons',
                child_foreign_key='hh_id',
                parent_table='households',
                parent_primary_key='hh_id',
                max_children_per_parent=2,
            ),
        ),
        (
            2,
            'activities',
            rel_domain.ForeignKeyRelation(
                child_table='activities',
                child_foreign_key='person_id',
                parent_table='persons',
                parent_primary_key='person_id',
                max_children_per_parent=2,
            ),
        ),
    ]

    # Deterministic RNG ensuring P0, P1 are kept, P2 dropped
    rng = np.random.default_rng(42)
    weights = transformations.compute_hierarchical_weights(
        tables, hierarchy, rng=rng
    )

    # 1. Household weights: all 1.0
    np.testing.assert_allclose(weights['households'], [1.0, 1.0])

    # 2. Persons weights:
    #     H0 has 2 active -> 0.5 each; 1 truncated -> 0.0; H1 has 1 -> 1.0
    # P3 is at index 3 -> must be 1.0
    self.assertEqual(weights['persons'][3], 1.0)
    # Sum of weights for H0 must be exactly 1.0
    self.assertAlmostEqual(np.sum(weights['persons'][:3]), 1.0)
    self.assertEqual(np.count_nonzero(weights['persons'][:3]), 2)

    # 3. Activities weights:
    # A3 is at index 3 -> must be 1.0
    self.assertEqual(weights['activities'][3], 1.0)
    # Orphan A4 is at index 4 -> must be 0.0
    self.assertEqual(weights['activities'][4], 0.0)
    # Sum of weights for H0 activities must be exactly 1.0
    self.assertAlmostEqual(np.sum(weights['activities'][:3]), 1.0)

  def test_compute_hierarchical_weights_empty_and_orphans(self):
    households = pd.DataFrame({'hh_id': ['H0']})
    persons = pd.DataFrame({
        'person_id': ['P0', 'P1'],
        'hh_id': [None, 'H_orphan'],
    })
    tables = {'households': households, 'persons': persons}
    hierarchy = [
        (0, 'households', None),
        (
            1,
            'persons',
            rel_domain.ForeignKeyRelation(
                child_table='persons',
                child_foreign_key='hh_id',
                parent_table='households',
                parent_primary_key='hh_id',
                max_children_per_parent=2,
            ),
        ),
    ]
    weights = transformations.compute_hierarchical_weights(tables, hierarchy)
    np.testing.assert_allclose(weights['persons'], [0.0, 0.0])

  def test_get_slot_permutation_patterns(self):
    # k=0, o=2 (childless)
    p0, w0 = transformations._get_slot_permutation_patterns(
        k=0, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertEqual(p0, [(-1, -1)])
    self.assertEqual(w0, 1.0)

    p0_s, w0_s = transformations._get_slot_permutation_patterns(
        k=0, num_permutation_slots=2, strategy='size_sliced'
    )
    self.assertEqual(p0_s, [(0, 0)])
    self.assertEqual(w0_s, 1.0)

    # k=1, o=2 (single child)
    p1, w1 = transformations._get_slot_permutation_patterns(
        k=1, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertEqual(p1, [(0, -1), (-1, 0)])
    self.assertEqual(w1, 0.5)

    p1_s, w1_s = transformations._get_slot_permutation_patterns(
        k=1, num_permutation_slots=2, strategy='size_sliced'
    )
    self.assertEqual(p1_s, [(0, 0)])
    self.assertEqual(w1_s, 1.0)

    # k=2, o=2 (two children)
    p2, w2 = transformations._get_slot_permutation_patterns(
        k=2, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertEqual(p2, [(0, 1), (1, 0)])
    self.assertEqual(w2, 0.5)

    # k=3, o=2 (three children) -> 3 * 2 = 6 permutations, weight = 1/6
    p3, w3 = transformations._get_slot_permutation_patterns(
        k=3, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertLen(p3, 6)
    self.assertAlmostEqual(w3, 1.0 / 6.0)

  def test_build_permuted_exploration_dataset_running_example(self):
    # Running Example: Household -> Person
    # Household 0: (income=50000, region=0), 2 children: P0(age=35), P1(age=32)
    # Household 1: (income=75000, region=1), 0 children
    # Household 2: (income=30000, region=0), 1 child: P2(age=5)
    # s = 2, o = 2, strategy = 'empty_token'
    parent_dom = mbi.Domain.fromdict({'income': 100000, 'region': 2})
    parent_ds = mbi.Dataset(
        {
            'income': np.array([50000, 75000, 30000], dtype=np.int64),
            'region': np.array([0, 1, 0], dtype=np.int64),
        },
        parent_dom,
    )

    child_dom = mbi.Domain.fromdict({'age': 100})
    child_ds = mbi.Dataset(
        {'age': np.array([35, 32, 5], dtype=np.int64)}, child_dom
    )

    parent_pks = ['H0', 'H1', 'H2']
    child_fks = ['H0', 'H0', 'H2']

    expl_ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        max_group_size=2,
        num_permutation_slots=2,
        strategy='empty_token',
    )

    # Expected emitted rows:
    # H0 (k=2): 2 permutation rows [(0,1), (1,0)], weight=0.5 each
    # H1 (k=0): 1 row [(-1, -1)], weight=1.0
    # H2 (k=1): 2 permutation rows [(0,-1), (-1,0)], weight=0.5 each
    # Total rows = 2 + 1 + 2 = 5 rows
    self.assertEqual(expl_ds.records, 5)

    # 1. Weight mass invariant: sum(weights) == N_parents = 3.0
    assert expl_ds.weights is not None
    self.assertAlmostEqual(float(np.sum(expl_ds.weights)), 3.0)

    # 2. Domain check
    self.assertEqual(
        expl_ds.domain.attributes,
        ('income', 'region', 'group_size', 'slot_1.age', 'slot_2.age'),
    )
    self.assertEqual(expl_ds.domain.shape, (100000, 2, 3, 101, 101))

    # 3. Check group_size column matches true k
    group_sizes = expl_ds.data['group_size'].tolist()
    self.assertEqual(group_sizes.count(2), 2)  # H0
    self.assertEqual(group_sizes.count(0), 1)  # H1
    self.assertEqual(group_sizes.count(1), 2)  # H2

    # 4. Check slot values and <EMPTY> token (age=100)
    # In H1 (k=0), slot_1.age and slot_2.age must be 100 (<EMPTY>)
    h1_mask = expl_ds.data['group_size'] == 0
    np.testing.assert_array_equal(expl_ds.data['slot_1.age'][h1_mask], [100])
    np.testing.assert_array_equal(expl_ds.data['slot_2.age'][h1_mask], [100])

    # In H2 (k=1), one slot has age=5, other has 100 (<EMPTY>)
    h2_mask = expl_ds.data['group_size'] == 1
    h2_slot1 = expl_ds.data['slot_1.age'][h2_mask].tolist()
    h2_slot2 = expl_ds.data['slot_2.age'][h2_mask].tolist()
    self.assertCountEqual(h2_slot1, [5, 100])
    self.assertCountEqual(h2_slot2, [100, 5])

  def test_build_permuted_exploration_dataset_size_sliced(self):
    parent_dom = mbi.Domain.fromdict({'income': 100})
    parent_ds = mbi.Dataset(
        {'income': np.array([50, 75], dtype=np.int64)}, parent_dom
    )
    child_dom = mbi.Domain.fromdict({'age': 100})
    child_ds = mbi.Dataset({'age': np.array([12], dtype=np.int64)}, child_dom)

    # H0 has 1 child (age=12), H1 has 0 children
    expl_ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=['H0', 'H1'],
        child_foreign_keys=['H0'],
        max_group_size=2,
        num_permutation_slots=2,
        strategy='size_sliced',
    )
    self.assertEqual(expl_ds.records, 2)
    assert expl_ds.weights is not None
    self.assertAlmostEqual(float(np.sum(expl_ds.weights)), 2.0)
    self.assertEqual(expl_ds.domain.shape, (100, 3, 100, 100))

  def test_build_permuted_exploration_dataset_empty_and_orphans(self):
    parent_dom = mbi.Domain.fromdict({'p': 5})
    child_dom = mbi.Domain.fromdict({'c': 5})

    # Empty parent dataset
    parent_empty = mbi.Dataset({'p': np.empty(0, dtype=np.int64)}, parent_dom)
    child_empty = mbi.Dataset({'c': np.empty(0, dtype=np.int64)}, child_dom)
    expl_empty = transformations.build_permuted_exploration_dataset(
        parent_empty, child_empty, [], [], max_group_size=2
    )
    self.assertEqual(expl_empty.records, 0)
    assert expl_empty.weights is not None
    self.assertAlmostEqual(float(np.sum(expl_empty.weights)), 0.0)

    # All orphan children
    parent_ds = mbi.Dataset({'p': np.array([1, 2], dtype=np.int64)}, parent_dom)
    child_ds = mbi.Dataset({'c': np.array([3, 4], dtype=np.int64)}, child_dom)
    expl_orphans = transformations.build_permuted_exploration_dataset(
        parent_ds,
        child_ds,
        ['H0', 'H1'],
        ['H_nonexistent1', 'H_nonexistent2'],
        max_group_size=2,
    )
    # Both households evaluate to k=0 -> 2 rows total
    self.assertEqual(expl_orphans.records, 2)
    self.assertEqual(expl_orphans.data['group_size'].tolist(), [0, 0])

  def test_build_permuted_exploration_dataset_validation_errors(self):
    p_dom = mbi.Domain.fromdict({'p': 2})
    c_dom = mbi.Domain.fromdict({'c': 2})
    p_ds = mbi.Dataset({'p': np.array([0])}, p_dom)
    c_ds = mbi.Dataset({'c': np.array([0])}, c_dom)

    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['H0'], ['H0'], max_group_size=0
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['H0'], ['H0'], max_group_size=2, num_permutation_slots=0
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['H0', 'extra'], ['H0'], max_group_size=2
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['H0'], ['H0', 'extra'], max_group_size=2
      )


class TransformationsFormalGuaranteesPropertyTest(absltest.TestCase):
  """Property-based tests verifying mathematical invariants and DP guarantees."""

  def test_property_exploration_domain_zero_privacy_loss_dimensions(self):
    """Verifies exploration domain shapes depend purely on public metadata."""
    rng = np.random.default_rng(12345)
    for _ in range(15):
      n_p = int(rng.integers(1, 4))
      n_c = int(rng.integers(1, 4))
      s = int(rng.integers(1, 8))
      o = int(rng.integers(1, 5))

      parent_attrs = tuple(f'p{i}' for i in range(n_p))
      parent_shapes = tuple(int(x) for x in rng.integers(2, 10, size=n_p))
      parent_dom = mbi.Domain(parent_attrs, parent_shapes)

      child_attrs = tuple(f'c{i}' for i in range(n_c))
      child_shapes = tuple(int(x) for x in rng.integers(2, 10, size=n_c))
      child_dom = mbi.Domain(child_attrs, child_shapes)

      # Strategy A: empty_token (+1 to child attributes)
      dom_a = transformations.build_exploration_domain(
          parent_dom,
          child_dom,
          max_group_size=s,
          num_permutation_slots=o,
          strategy='empty_token',
      )
      expected_shape_a = (
          parent_shapes + (s + 1,) + tuple(sz + 1 for sz in child_shapes) * o
      )
      self.assertEqual(dom_a.shape, expected_shape_a)
      self.assertEqual(dom_a.attributes[len(parent_attrs)], 'group_size')

      # Strategy B: size_sliced (unextended child attributes)
      dom_b = transformations.build_exploration_domain(
          parent_dom,
          child_dom,
          max_group_size=s,
          num_permutation_slots=o,
          strategy='size_sliced',
      )
      expected_shape_b = parent_shapes + (s + 1,) + child_shapes * o
      self.assertEqual(dom_b.shape, expected_shape_b)

  def test_property_bounded_lineage_and_cascading_limits(self):
    """Verifies that descendant counts never exceed prod(s_j) and cascading truncation holds."""
    rng = np.random.default_rng(12345)
    for _ in range(15):
      n_h = int(rng.integers(1, 20))
      s1 = int(rng.integers(1, 4))
      s2 = int(rng.integers(1, 4))

      hh_ids = [f'H{i}' for i in range(n_h)]
      households = pd.DataFrame({'hh_id': hh_ids})

      n_p = int(rng.integers(0, n_h * 5 + 1))
      p_hhs = rng.choice(hh_ids, size=n_p) if n_p > 0 else []
      persons = pd.DataFrame({
          'person_id': [f'P{i}' for i in range(n_p)],
          'hh_id': p_hhs,
      })

      n_a = int(rng.integers(0, n_p * 5 + 1)) if n_p > 0 else 0
      a_persons = (
          rng.choice(persons['person_id'].values, size=n_a) if n_a > 0 else []
      )
      activities = pd.DataFrame({
          'activity_id': [f'A{i}' for i in range(n_a)],
          'person_id': a_persons,
      })

      tables = {
          'households': households,
          'persons': persons,
          'activities': activities,
      }
      hierarchy = [
          (0, 'households', None),
          (
              1,
              'persons',
              rel_domain.ForeignKeyRelation(
                  parent_table='households',
                  parent_primary_key='hh_id',
                  child_table='persons',
                  child_foreign_key='hh_id',
                  max_children_per_parent=s1,
              ),
          ),
          (
              2,
              'activities',
              rel_domain.ForeignKeyRelation(
                  parent_table='persons',
                  parent_primary_key='person_id',
                  child_table='activities',
                  child_foreign_key='person_id',
                  max_children_per_parent=s2,
              ),
          ),
      ]

      mappings = transformations._compute_row_root_mappings(
          tables, hierarchy, rng=rng
      )

      # Invariant 1: Root mappings count <= s1 per household
      p_counts = mappings['persons'].value_counts()
      for h_idx in range(n_h):
        self.assertLessEqual(p_counts.get(h_idx, 0), s1)

      # Invariant 2: Transitive descendant count <= s1 * s2 per household
      a_counts = mappings['activities'].value_counts()
      for h_idx in range(n_h):
        self.assertLessEqual(a_counts.get(h_idx, 0), s1 * s2)

      # Invariant 3: Cascading truncation
      # If person root is None, all linked activities are None
      p_roots = mappings['persons']
      p_none_ids = set(persons.loc[p_roots.isna(), 'person_id'])
      if p_none_ids and not activities.empty:
        a_orphans = activities['person_id'].isin(p_none_ids)
        self.assertTrue(mappings['activities'][a_orphans].isna().all())

  def test_property_unit_sensitivity_hierarchical_weights(self):
    """Verifies that sum of weights for any root entity is <= 1.0 (global Delta = 1.0)."""
    rng = np.random.default_rng(12345)
    for _ in range(15):
      n_h = int(rng.integers(1, 15))
      s = int(rng.integers(1, 4))
      hh_ids = [f'H{i}' for i in range(n_h)]
      households = pd.DataFrame({'hh_id': hh_ids})

      n_p = int(rng.integers(1, 40))
      persons = pd.DataFrame({
          'person_id': [f'P{i}' for i in range(n_p)],
          'hh_id': rng.choice(hh_ids + ['orphan'], size=n_p),
      })
      tables = {'households': households, 'persons': persons}
      hierarchy = [
          (0, 'households', None),
          (
              1,
              'persons',
              rel_domain.ForeignKeyRelation(
                  parent_table='households',
                  parent_primary_key='hh_id',
                  child_table='persons',
                  child_foreign_key='hh_id',
                  max_children_per_parent=s,
              ),
          ),
      ]

      seed = int(rng.integers(0, 1000000))
      weights = transformations.compute_hierarchical_weights(
          tables, hierarchy, rng=np.random.default_rng(seed)
      )
      mappings = transformations._compute_row_root_mappings(
          tables, hierarchy, rng=np.random.default_rng(seed)
      )

      # Invariant: For each root household i,
      # sum of active weights == 1.0 (or 0.0 if childless)
      p_roots = mappings['persons']
      p_weights = weights['persons']

      for h_idx in range(n_h):
        h_mask = (p_roots == h_idx).values
        h_sum = float(np.sum(p_weights[h_mask]))
        if np.any(h_mask):
          self.assertAlmostEqual(
              h_sum,
              1.0,
              places=12,
              msg=f'Root weight sum != 1.0 for H{h_idx}: {h_sum}',
          )
        else:
          self.assertEqual(h_sum, 0.0)

      # Invariant: Unlinked/truncated rows have weight 0.0
      none_mask = p_roots.isna().values
      self.assertTrue(np.all(p_weights[none_mask] == 0.0))

  def test_property_permutation_weight_mass_conservation(self):
    """Verifies sum(weights) == 1.0 across all (k, o) parameter configurations."""
    strategies: tuple[Literal['empty_token', 'size_sliced'], ...] = (
        'empty_token',
        'size_sliced',
    )
    for k in range(16):
      for o in range(1, 7):
        for strategy in strategies:
          patterns, weight = transformations._get_slot_permutation_patterns(
              k=k,
              num_permutation_slots=o,
              strategy=strategy,
          )
          total_weight = len(patterns) * weight
          self.assertAlmostEqual(
              total_weight,
              1.0,
              places=12,
              msg=f'Weight mass violated for k={k}, o={o}, strategy={strategy}',
          )

  def test_property_permutation_slot_exchangeability(self):
    """Verifies uniform slot marginals across relative child ranks."""
    for k in range(1, 7):
      for o in range(1, 5):
        patterns, _ = transformations._get_slot_permutation_patterns(
            k=k, num_permutation_slots=o, strategy='empty_token'
        )
        pattern_matrix = np.array(patterns, dtype=np.int64)  # (P_k, o)
        # For each child rank r, frequency in slot i must equal slot j.
        for r in range(k):
          slot_counts = [
              np.sum(pattern_matrix[:, slot] == r) for slot in range(o)
          ]
          self.assertLen(
              set(slot_counts),
              1,
              msg=(
                  f'Slot marginal imbalance for k={k}, o={o}, child={r}:'
                  f' {slot_counts}'
              ),
          )


if __name__ == '__main__':
  absltest.main()
