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

import itertools
import math
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
    domain = transformations._build_exploration_domain(
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
    self.assertEqual(domain.attributes, expected_attrs)
    self.assertEqual(domain.shape, (4, 3, 4, 11, 3, 11, 3))

  def test_build_exploration_domain_size_sliced(self):
    parent_domain = mbi.Domain.fromdict({'income': 4})
    child_domain = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    domain = transformations._build_exploration_domain(
        parent_domain=parent_domain,
        child_domain=child_domain,
        max_group_size=2,
        num_permutation_slots=2,
        strategy='size_sliced',
    )
    self.assertEqual(
        domain.attributes,
        (
            'income',
            'group_size',
            'slot_1.age',
            'slot_1.gender',
            'slot_2.age',
            'slot_2.gender',
        ),
    )
    self.assertEqual(domain.shape, (4, 3, 10, 2, 10, 2))

  def test_build_exploration_domain_single_group_size_and_3slots(self):
    parent_domain = mbi.Domain.fromdict({'p1': 2})
    child_domain = mbi.Domain.fromdict({'c1': 5})
    domain = transformations._build_exploration_domain(
        parent_domain=parent_domain,
        child_domain=child_domain,
        max_group_size=1,
        num_permutation_slots=3,
        strategy='empty_token',
    )
    self.assertEqual(
        domain.attributes,
        ('p1', 'group_size', 'slot_1.c1', 'slot_2.c1', 'slot_3.c1'),
    )
    self.assertEqual(domain.shape, (2, 2, 6, 6, 6))

  def test_get_slot_permutation_patterns_k0(self):
    patterns_empty, w_empty = transformations._get_slot_permutation_patterns(
        k=0, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertEqual(patterns_empty, [(-1, -1)])
    self.assertEqual(w_empty, 1.0)

    patterns_sliced, w_sliced = transformations._get_slot_permutation_patterns(
        k=0, num_permutation_slots=2, strategy='size_sliced'
    )
    self.assertEqual(patterns_sliced, [(0, 0)])
    self.assertEqual(w_sliced, 1.0)

  def test_get_slot_permutation_patterns_k1_o2(self):
    patterns_empty, w_empty = transformations._get_slot_permutation_patterns(
        k=1, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertCountEqual(patterns_empty, [(0, -1), (-1, 0)])
    self.assertEqual(w_empty, 0.5)

    patterns_sliced, w_sliced = transformations._get_slot_permutation_patterns(
        k=1, num_permutation_slots=2, strategy='size_sliced'
    )
    self.assertEqual(patterns_sliced, [(0, 0)])
    self.assertEqual(w_sliced, 1.0)

  def test_get_slot_permutation_patterns_k2_o2(self):
    patterns_empty, w_empty = transformations._get_slot_permutation_patterns(
        k=2, num_permutation_slots=2, strategy='empty_token'
    )
    self.assertCountEqual(patterns_empty, [(0, 1), (1, 0)])
    self.assertEqual(w_empty, 0.5)

    patterns_sliced, w_sliced = transformations._get_slot_permutation_patterns(
        k=2, num_permutation_slots=2, strategy='size_sliced'
    )
    self.assertCountEqual(patterns_sliced, [(0, 1), (1, 0)])
    self.assertEqual(w_sliced, 0.5)

  def test_get_slot_permutation_patterns_k3_o2(self):
    patterns, w = transformations._get_slot_permutation_patterns(
        k=3, num_permutation_slots=2, strategy='empty_token'
    )
    expected = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    self.assertCountEqual(patterns, expected)
    self.assertAlmostEqual(w, 1.0 / 6.0)

  def test_get_slot_permutation_patterns_k1_and_k2_o3(self):
    # o = 3, k = 1: 3 permutations
    patterns_k1, w_k1 = transformations._get_slot_permutation_patterns(
        k=1, num_permutation_slots=3, strategy='empty_token'
    )
    self.assertCountEqual(patterns_k1, [(0, -1, -1), (-1, 0, -1), (-1, -1, 0)])
    self.assertAlmostEqual(w_k1, 1.0 / 3.0)

    # o = 3, k = 2: 6 permutations
    patterns_k2, w_k2 = transformations._get_slot_permutation_patterns(
        k=2, num_permutation_slots=3, strategy='empty_token'
    )
    self.assertLen(patterns_k2, 6)
    self.assertAlmostEqual(w_k2, 1.0 / 6.0)

  def test_get_slot_permutation_patterns_k10_o3(self):
    # P(10, 3) = 10 * 9 * 8 = 720 permutations
    patterns_empty, w_empty = transformations._get_slot_permutation_patterns(
        k=10, num_permutation_slots=3, strategy='empty_token'
    )
    self.assertLen(patterns_empty, 720)
    self.assertLen(set(patterns_empty), 720)
    self.assertAlmostEqual(w_empty, 1.0 / 720.0)
    for p in patterns_empty:
      self.assertLen(p, 3)
      self.assertTrue(all(0 <= idx < 10 for idx in p))
      self.assertLen(set(p), 3)

    patterns_sliced, w_sliced = transformations._get_slot_permutation_patterns(
        k=10, num_permutation_slots=3, strategy='size_sliced'
    )
    self.assertEqual(patterns_empty, patterns_sliced)
    self.assertEqual(w_empty, w_sliced)

  def test_get_slot_permutation_patterns_weight_invariant(self):
    for k in range(6):
      for o in range(1, 5):
        for strategy in ['empty_token', 'size_sliced']:
          patterns, weight = transformations._get_slot_permutation_patterns(
              k=k, num_permutation_slots=o, strategy=strategy
          )
          self.assertAlmostEqual(len(patterns) * weight, 1.0)

  def test_build_permuted_exploration_dataset_running_example(self):
    # 4 households: k=0, 1, 2, 3 children
    parent_dom = mbi.Domain.fromdict({'income': 4})
    parent_data = {'income': np.array([0, 1, 2, 3], dtype=np.int64)}
    parent_ds = mbi.Dataset(parent_data, parent_dom)
    parent_pks = ['h0', 'h1', 'h2', 'h3']

    child_dom = mbi.Domain.fromdict({'age': 10})
    # h1: c0(3); h2: c1(5), c2(8); h3: c3(2), c4(6), c5(9)
    child_data = {'age': np.array([3, 5, 8, 2, 6, 9], dtype=np.int64)}
    child_ds = mbi.Dataset(child_data, child_dom)
    child_fks = ['h1', 'h2', 'h2', 'h3', 'h3', 'h3']

    ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        max_group_size=3,
        num_permutation_slots=2,
        strategy='empty_token',
    )

    # 1. Domain verification: group_size cardinality=4, age cardinality=11
    self.assertEqual(
        ds.domain.attributes,
        ('income', 'group_size', 'slot_1.age', 'slot_2.age'),
    )
    self.assertEqual(ds.domain.shape, (4, 4, 11, 11))

    # 2. Row count and weight mass verification
    # Rows: h0->1, h1->2, h2->2, h3->6 = 11 total rows
    self.assertEqual(ds.records, 11)
    self.assertAlmostEqual(float(np.sum(ds.weights)), 4.0)

    # 3. Exact slot exchangeability (Hermitian marginal symmetry P(S1) == P(S2))
    s1_hist = np.bincount(
        ds.data['slot_1.age'], weights=ds.weights, minlength=11
    )
    s2_hist = np.bincount(
        ds.data['slot_2.age'], weights=ds.weights, minlength=11
    )
    np.testing.assert_allclose(s1_hist, s2_hist)

    # 4. Total <EMPTY> token mass: h0 has 1(both empty), h1 has 1(one empty)
    empty_mass_s1 = float(np.sum(ds.weights[ds.data['slot_1.age'] == 10]))
    empty_mass_s2 = float(np.sum(ds.weights[ds.data['slot_2.age'] == 10]))
    self.assertAlmostEqual(empty_mass_s1, 1.5)
    self.assertAlmostEqual(empty_mass_s2, 1.5)

  def test_build_permuted_exploration_dataset_size_sliced(self):
    parent_dom = mbi.Domain.fromdict({'income': 4})
    parent_data = {'income': np.array([0, 1, 2, 3], dtype=np.int64)}
    parent_ds = mbi.Dataset(parent_data, parent_dom)
    parent_pks = ['h0', 'h1', 'h2', 'h3']

    child_dom = mbi.Domain.fromdict({'age': 10})
    child_data = {'age': np.array([3, 5, 8, 2, 6, 9], dtype=np.int64)}
    child_ds = mbi.Dataset(child_data, child_dom)
    child_fks = ['h1', 'h2', 'h2', 'h3', 'h3', 'h3']

    ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=parent_pks,
        child_foreign_keys=child_fks,
        max_group_size=3,
        num_permutation_slots=2,
        strategy='size_sliced',
    )

    # Domain shape has unextended child shape (10)
    self.assertEqual(ds.domain.shape, (4, 4, 10, 10))
    # Rows: h0->1, h1->1 (clone tiled), h2->2, h3->6 = 10 total rows
    self.assertEqual(ds.records, 10)
    self.assertAlmostEqual(float(np.sum(ds.weights)), 4.0)

    # h1 (row index 1) has both slots clone tiled with child 0 (age 3)
    h1_mask = ds.data['income'] == 1
    self.assertEqual(int(ds.data['slot_1.age'][h1_mask][0]), 3)
    self.assertEqual(int(ds.data['slot_2.age'][h1_mask][0]), 3)

  def test_build_permuted_exploration_dataset_order1(self):
    parent_dom = mbi.Domain.fromdict({'income': 2})
    parent_data = {'income': np.array([0, 1], dtype=np.int64)}
    parent_ds = mbi.Dataset(parent_data, parent_dom)

    child_dom = mbi.Domain.fromdict({'age': 5})
    # h0 has 0 children; h1 has 2 children (ages 1, 4)
    child_data = {'age': np.array([1, 4], dtype=np.int64)}
    child_ds = mbi.Dataset(child_data, child_dom)

    ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=['h0', 'h1'],
        child_foreign_keys=['h1', 'h1'],
        max_group_size=2,
        num_permutation_slots=1,
        strategy='empty_token',
    )

    self.assertEqual(
        ds.domain.attributes, ('income', 'group_size', 'slot_1.age')
    )
    self.assertEqual(ds.domain.shape, (2, 3, 6))
    # Rows: h0->1 (<EMPTY>=5), h1->2 (age 1, age 4 with w=0.5) = 3 rows
    self.assertEqual(ds.records, 3)
    self.assertAlmostEqual(float(np.sum(ds.weights)), 2.0)

  def test_build_permuted_exploration_dataset_order3(self):
    parent_dom = mbi.Domain.fromdict({'income': 2})
    parent_data = {'income': np.array([0, 1], dtype=np.int64)}
    parent_ds = mbi.Dataset(parent_data, parent_dom)

    child_dom = mbi.Domain.fromdict({'age': 5})
    child_data = {'age': np.array([1, 2], dtype=np.int64)}
    child_ds = mbi.Dataset(child_data, child_dom)

    ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=['h0', 'h1'],
        child_foreign_keys=['h1', 'h1'],
        max_group_size=2,
        num_permutation_slots=3,
        strategy='empty_token',
    )

    self.assertEqual(
        ds.domain.attributes,
        ('income', 'group_size', 'slot_1.age', 'slot_2.age', 'slot_3.age'),
    )
    self.assertEqual(ds.domain.shape, (2, 3, 6, 6, 6))
    # Rows: h0->1 (<EMPTY>), h1 (k=2, o=3)->6 patterns with w=1/6 = 7 rows
    self.assertEqual(ds.records, 7)
    self.assertAlmostEqual(float(np.sum(ds.weights)), 2.0)

  def test_build_permuted_exploration_dataset_multi_column_child(self):
    parent_dom = mbi.Domain.fromdict({'income': 3})
    parent_ds = mbi.Dataset(
        {'income': np.array([0, 1], dtype=np.int64)}, parent_dom
    )
    child_dom = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    child_data = {
        'age': np.array([3, 7], dtype=np.int64),
        'gender': np.array([0, 1], dtype=np.int64),
    }
    child_ds = mbi.Dataset(child_data, child_dom)

    ds = transformations.build_permuted_exploration_dataset(
        parent_dataset=parent_ds,
        child_dataset=child_ds,
        parent_primary_keys=['h0', 'h1'],
        child_foreign_keys=['h1', 'h1'],
        max_group_size=2,
        num_permutation_slots=2,
        strategy='empty_token',
    )

    self.assertEqual(
        ds.domain.attributes,
        (
            'income',
            'group_size',
            'slot_1.age',
            'slot_1.gender',
            'slot_2.age',
            'slot_2.gender',
        ),
    )
    self.assertEqual(ds.domain.shape, (3, 3, 11, 3, 11, 3))
    # For h0 (childless), empty tokens are age=10, gender=2
    h0_mask = ds.data['income'] == 0
    self.assertEqual(int(ds.data['slot_1.age'][h0_mask][0]), 10)
    self.assertEqual(int(ds.data['slot_1.gender'][h0_mask][0]), 2)
    self.assertEqual(int(ds.data['slot_2.age'][h0_mask][0]), 10)
    self.assertEqual(int(ds.data['slot_2.gender'][h0_mask][0]), 2)

  def test_build_permuted_exploration_dataset_edge_cases_and_validation(self):
    parent_dom = mbi.Domain.fromdict({'p': 2})
    child_dom = mbi.Domain.fromdict({'c': 3})

    # Empty parent dataset (Np = 0)
    empty_parent = mbi.Dataset({'p': np.empty(0, dtype=np.int64)}, parent_dom)
    empty_child = mbi.Dataset({'c': np.empty(0, dtype=np.int64)}, child_dom)
    ds_empty = transformations.build_permuted_exploration_dataset(
        parent_dataset=empty_parent,
        child_dataset=empty_child,
        parent_primary_keys=[],
        child_foreign_keys=[],
        max_group_size=2,
        num_permutation_slots=2,
        strategy='empty_token',
    )
    self.assertEqual(ds_empty.records, 0)
    self.assertAlmostEqual(float(np.sum(ds_empty.weights)), 0.0)

    # Orphaned foreign keys (dropped cleanly)
    p_ds = mbi.Dataset({'p': np.array([0], dtype=np.int64)}, parent_dom)
    c_ds = mbi.Dataset({'c': np.array([1, 2], dtype=np.int64)}, child_dom)
    ds_orphans = transformations.build_permuted_exploration_dataset(
        parent_dataset=p_ds,
        child_dataset=c_ds,
        parent_primary_keys=['h0'],
        child_foreign_keys=['orphan_1', 'orphan_2'],
        max_group_size=2,
        num_permutation_slots=2,
        strategy='empty_token',
    )
    # Parent h0 has 0 valid children -> 1 row with <EMPTY>
    self.assertEqual(ds_orphans.records, 1)
    self.assertEqual(int(ds_orphans.data['slot_1.c'][0]), 3)

    # Validation errors
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['h0'], ['h0', 'h0'], max_group_size=0
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds,
          c_ds,
          ['h0'],
          ['h0', 'h0'],
          max_group_size=2,
          num_permutation_slots=0,
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds,
          c_ds,
          ['h0'],
          ['h0', 'h0'],
          max_group_size=2,
          strategy='invalid',
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['h0', 'extra'], ['h0', 'h0'], max_group_size=2
      )
    with self.assertRaises(ValueError):
      transformations.build_permuted_exploration_dataset(
          p_ds, c_ds, ['h0'], ['h0'], max_group_size=2
      )

  def test_transformations_import_and_callable(self):
    self.assertTrue(callable(transformations.compute_hierarchical_weights))
    self.assertTrue(
        callable(transformations.build_permuted_exploration_dataset)
    )
    self.assertTrue(
        callable(transformations.create_slot_linear_chain_constraints)
    )
    self.assertTrue(callable(transformations.symmetrize_to_wide_domain))
    self.assertTrue(callable(transformations.quantile_copula_coupling))
    self.assertTrue(callable(transformations.unstack_wide_family_records))

  def test_compute_row_root_mappings_single_table(self):
    households = pd.DataFrame({
        'household_id': ['h1', 'h2', 'h3'],
        'income': [50000.0, 75000.0, 100000.0],
    })
    hierarchy = [(0, 'households', None)]
    mapping = transformations._compute_row_root_mappings(
        {'households': households}, hierarchy
    )
    self.assertIsInstance(mapping['households'], pd.Series)
    self.assertEqual(
        mapping['households'].tolist(),
        [0, 1, 2],
    )

  def test_compute_row_root_mappings_2tier(self):
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h2'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=3,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy
    )
    self.assertIsInstance(mapping['persons'], pd.Series)
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, 0, 1],
    )

  def test_compute_row_root_mappings_truncation_and_cascading(self):
    households = pd.DataFrame({'hid': ['h1']})
    # h1 has 3 persons, but max_children_per_parent is 2 -> exactly 2 chosen
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h1'],
    })
    # activities for each person
    activities = pd.DataFrame({
        'aid': ['a1', 'a2', 'a3'],
        'pid': ['p1', 'p2', 'p3'],
    })
    fk1 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    fk2 = rel_domain.ForeignKeyRelation(
        parent_table='persons',
        parent_primary_key='pid',
        child_table='activities',
        child_foreign_key='pid',
        max_children_per_parent=2,
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk1),
        (2, 'activities', fk2),
    ]

    rng = np.random.default_rng(42)
    mapping = transformations._compute_row_root_mappings(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy,
        rng=rng,
    )
    # Exactly 2 persons are active (non-None), 1 is truncated (None)
    active_persons = mapping['persons'].dropna().tolist()
    self.assertLen(active_persons, 2)
    self.assertEqual(mapping['persons'].isna().sum(), 1)

    # Subchildren of active persons active, subchild of truncated person is None
    active_activities = mapping['activities'].dropna().tolist()
    self.assertLen(active_activities, 2)
    self.assertEqual(mapping['activities'].isna().sum(), 1)

  def test_compute_row_root_mappings_branching_tree(self):
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({'pid': ['p1', 'p2'], 'hid': ['h1', 'h2']})
    vehicles = pd.DataFrame(
        {'vid': ['v1', 'v2', 'v3'], 'hid': ['h1', 'h1', 'h2']}
    )

    fk_p = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 2
    )
    fk_v = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'vehicles', 'hid', 5
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk_p),
        (1, 'vehicles', fk_v),
    ]
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons, 'vehicles': vehicles},
        hierarchy,
    )
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, 1],
    )
    self.assertEqual(
        mapping['vehicles'].tolist(),
        [0, 0, 1],
    )

  def test_compute_row_root_mappings_multi_tree_forest(self):
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({'pid': ['p1'], 'hid': ['h1']})
    companies = pd.DataFrame({'cid': ['c1']})
    departments = pd.DataFrame({'did': ['d1'], 'cid': ['c1']})

    fk_h = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 2
    )
    fk_c = rel_domain.ForeignKeyRelation(
        'companies', 'cid', 'departments', 'cid', 5
    )
    hierarchy = [
        (0, 'households', None),
        (0, 'companies', None),
        (1, 'persons', fk_h),
        (1, 'departments', fk_c),
    ]
    mapping = transformations._compute_row_root_mappings(
        {
            'households': households,
            'persons': persons,
            'companies': companies,
            'departments': departments,
        },
        hierarchy,
    )
    self.assertEqual(mapping['households'].tolist(), [0])
    self.assertEqual(mapping['companies'].tolist(), [0])
    self.assertEqual(mapping['persons'].tolist(), [0])
    self.assertEqual(mapping['departments'].tolist(), [0])

  def test_compute_row_root_mappings_custom_index_alignment(self):
    # Non-standard indices (strings, custom obj) can't break positional mapping
    households = pd.DataFrame(
        {'hid': ['h1', 'h2']}, index=['custom_a', 'custom_b']
    )
    persons = pd.DataFrame(
        {'pid': ['p1', 'p2', 'p3'], 'hid': ['h1', 'h2', 'h1']},
        index=[100, 200, 300],
    )
    fk = rel_domain.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 5)
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    # Positions are strictly 0 and 1 in households DataFrame
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, 1, 0],
    )

  def test_compute_row_root_mappings_orphans_and_validation(self):
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2'],
        'hid': ['h1', 'orphan_h'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    self.assertEqual(mapping['persons'].tolist(), [0, None])

    # Missing parent primary key
    bad_fk1 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='missing_id',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    with self.assertRaisesRegex(ValueError, 'Parent primary key'):
      transformations._compute_row_root_mappings(
          {'households': households, 'persons': persons},
          [(0, 'households', None), (1, 'persons', bad_fk1)],
      )

    # Missing child foreign key
    bad_fk2 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='missing_hid',
        max_children_per_parent=2,
    )
    with self.assertRaisesRegex(ValueError, 'Child foreign key'):
      transformations._compute_row_root_mappings(
          {'households': households, 'persons': persons},
          [(0, 'households', None), (1, 'persons', bad_fk2)],
      )

  def test_compute_row_root_mappings_nan_and_corrupt_data(self):
    households = pd.DataFrame({'hid': ['h1', np.nan, 'h2', 'h1']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3', 'p4', 'p5'],
        'hid': ['h1', np.nan, 'h2', None, 'orphan'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    mapping = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons},
        hierarchy,
    )
    self.assertEqual(
        mapping['persons'].tolist(),
        [0, None, 2, None, None],
    )

  def test_compute_row_root_mappings_empty_tables(self):
    empty_h = pd.DataFrame({'hid': []})
    persons = pd.DataFrame({'pid': ['p1'], 'hid': ['h1']})
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    mapping = transformations._compute_row_root_mappings(
        {'households': empty_h, 'persons': persons},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    self.assertEqual(mapping['households'].tolist(), [])
    self.assertEqual(mapping['persons'].tolist(), [None])

    empty_p = pd.DataFrame({'pid': [], 'hid': []})
    mapping2 = transformations._compute_row_root_mappings(
        {'households': empty_h, 'persons': empty_p},
        [(0, 'households', None), (1, 'persons', fk)],
    )
    self.assertEqual(mapping2['households'].tolist(), [])
    self.assertEqual(mapping2['persons'].tolist(), [])

  def test_compute_row_root_mappings_random_subsampling_reproducibility(self):
    # A single household with 10 persons, capacity s = 3
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': [f'p{i}' for i in range(10)],
        'hid': ['h1'] * 10,
    })
    fk = rel_domain.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 3)
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]

    # Same seed must yield identical active row selections
    rng1 = np.random.default_rng(123)
    mapping1 = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy, rng=rng1
    )
    rng2 = np.random.default_rng(123)
    mapping2 = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy, rng=rng2
    )
    self.assertEqual(mapping1['persons'].tolist(), mapping2['persons'].tolist())
    self.assertEqual(mapping1['persons'].dropna().count(), 3)
    self.assertEqual(mapping1['persons'].isna().sum(), 7)

    # Different seeds must produce valid selections of size exactly 3
    rng3 = np.random.default_rng(999)
    mapping3 = transformations._compute_row_root_mappings(
        {'households': households, 'persons': persons}, hierarchy, rng=rng3
    )
    self.assertEqual(mapping3['persons'].dropna().count(), 3)
    self.assertEqual(mapping3['persons'].isna().sum(), 7)

  def test_compute_hierarchical_weights_single_table(self):
    households = pd.DataFrame({
        'household_id': ['h1', 'h2', 'h3'],
        'income': [50000.0, 75000.0, 100000.0],
    })
    hierarchy = [(0, 'households', None)]
    weights = transformations.compute_hierarchical_weights(
        {'households': households}, hierarchy=hierarchy
    )
    self.assertIn('households', weights)
    self.assertEqual(weights['households'].shape, (3,))
    np.testing.assert_allclose(weights['households'], np.array([1.0, 1.0, 1.0]))
    self.assertAlmostEqual(weights['households'].sum(), 3.0)

  def test_compute_hierarchical_weights_2tier(self):
    # H1 has 2 persons (P1, P2) -> w = 0.5 each
    # H2 has 1 person (P3) -> w = 1.0
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h2'],
    })
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=3,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    weights = transformations.compute_hierarchical_weights(
        {'households': households, 'persons': persons}, hierarchy=hierarchy
    )
    np.testing.assert_allclose(weights['households'], np.array([1.0, 1.0]))
    np.testing.assert_allclose(weights['persons'], np.array([0.5, 0.5, 1.0]))
    # Total sum of weights in every table matches number of households (2.0)
    self.assertAlmostEqual(weights['households'].sum(), 2.0)
    self.assertAlmostEqual(weights['persons'].sum(), 2.0)

  def test_compute_hierarchical_weights_3tier_with_truncation(self):
    # H1 has 3 persons, but s1 = 2
    # -> 2 active (w = 0.5 each), 1 truncated (w = 0.0)
    # H1's active persons have 2 activities each (4 total) -> w = 0.25 each
    # H1's truncated person has 2 activities -> both cascade to w = 0.0
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3'],
        'hid': ['h1', 'h1', 'h1'],
    })
    activities = pd.DataFrame({
        'aid': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6'],
        'pid': ['p1', 'p1', 'p2', 'p2', 'p3', 'p3'],
    })
    fk1 = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    fk2 = rel_domain.ForeignKeyRelation(
        parent_table='persons',
        parent_primary_key='pid',
        child_table='activities',
        child_foreign_key='pid',
        max_children_per_parent=2,
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk1),
        (2, 'activities', fk2),
    ]

    rng = np.random.default_rng(42)
    weights = transformations.compute_hierarchical_weights(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy=hierarchy,
        rng=rng,
    )

    # Household sum = 1.0
    self.assertAlmostEqual(weights['households'].sum(), 1.0)
    # Person sum = 1.0 (2 active persons with 0.5, 1 truncated with 0.0)
    self.assertAlmostEqual(weights['persons'].sum(), 1.0)
    self.assertEqual((weights['persons'] == 0.0).sum(), 1)
    self.assertEqual((weights['persons'] == 0.5).sum(), 2)

    # Activity sum = 1.0 (4 active activities with 0.25, 2 truncated with 0.0)
    self.assertAlmostEqual(weights['activities'].sum(), 1.0)
    self.assertEqual((weights['activities'] == 0.0).sum(), 2)
    self.assertEqual((weights['activities'] == 0.25).sum(), 4)

  def test_compute_hierarchical_weights_empty_table(self):
    empty_h = pd.DataFrame({'hid': []})
    empty_p = pd.DataFrame({'pid': [], 'hid': []})
    fk = rel_domain.ForeignKeyRelation(
        parent_table='households',
        parent_primary_key='hid',
        child_table='persons',
        child_foreign_key='hid',
        max_children_per_parent=2,
    )
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]
    weights = transformations.compute_hierarchical_weights(
        {'households': empty_h, 'persons': empty_p}, hierarchy=hierarchy
    )
    self.assertEqual(weights['households'].shape, (0,))
    self.assertEqual(weights['persons'].shape, (0,))

  def test_compute_hierarchical_weights_multi_tree_forest(self):
    households = pd.DataFrame({'hid': ['h1', 'h2']})
    persons = pd.DataFrame({
        'pid': ['p1', 'p2', 'p3', 'p4'],
        'hid': ['h1', 'h1', 'h2', 'h2'],
    })
    companies = pd.DataFrame({'cid': ['c1']})
    departments = pd.DataFrame({
        'did': ['d1', 'd2'],
        'cid': ['c1', 'c1'],
    })

    fk_h = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 5
    )
    fk_c = rel_domain.ForeignKeyRelation(
        'companies', 'cid', 'departments', 'cid', 5
    )
    hierarchy = [
        (0, 'households', None),
        (0, 'companies', None),
        (1, 'persons', fk_h),
        (1, 'departments', fk_c),
    ]
    weights = transformations.compute_hierarchical_weights(
        {
            'households': households,
            'persons': persons,
            'companies': companies,
            'departments': departments,
        },
        hierarchy=hierarchy,
    )
    self.assertAlmostEqual(weights['households'].sum(), 2.0)
    self.assertAlmostEqual(weights['persons'].sum(), 2.0)
    self.assertAlmostEqual(weights['companies'].sum(), 1.0)
    self.assertAlmostEqual(weights['departments'].sum(), 1.0)

  def test_dp_adversarial_data_dependent_robustness(self):
    """Stress tests DP safety: mechanism must never crash on adversarial data."""
    # Dataset with special values, duplicate keys, and missing references.
    households = pd.DataFrame({
        'hid': [
            'h1',
            np.nan,
            None,
            float('inf'),
            float('-inf'),
            'h1',  # duplicate
            'h12345',
            'h_true',
        ],
        'val': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })
    persons = pd.DataFrame({
        'pid': [f'p{i}' for i in range(12)],
        'hid': [
            'h1',  # matches first h1
            'orphan_key',  # orphan
            np.nan,  # NaN
            None,  # None
            pd.NA,  # pd.NA
            float('inf'),  # matches inf
            float('-inf'),  # matches -inf
            'h12345',  # matches h12345
            'h_true',  # matches h_true
            'orphan_2',  # another orphan
            'orphan_3',  # another orphan
            'h1',  # another match to h1
        ],
    })
    activities = pd.DataFrame({
        'aid': [f'a{i}' for i in range(5)],
        'pid': ['p0', 'p1', 'p9', 'missing_person', 'p11'],
    })

    fk1 = rel_domain.ForeignKeyRelation(
        'households', 'hid', 'persons', 'hid', 1
    )
    fk2 = rel_domain.ForeignKeyRelation(
        'persons', 'pid', 'activities', 'pid', 2
    )
    hierarchy = [
        (0, 'households', None),
        (1, 'persons', fk1),
        (2, 'activities', fk2),
    ]

    rng = np.random.default_rng(100)
    # Must execute cleanly (no exceptions) on this corrupted dataset
    weights = transformations.compute_hierarchical_weights(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy=hierarchy,
        rng=rng,
    )

    # Output shapes must be strictly aligned with input DataFrame row counts
    self.assertEqual(weights['households'].shape, (len(households),))
    self.assertEqual(weights['persons'].shape, (len(persons),))
    self.assertEqual(weights['activities'].shape, (len(activities),))

    # All weights must be finite non-negative floats
    self.assertTrue(np.all(np.isfinite(weights['households'])))
    self.assertTrue(np.all(weights['households'] >= 0.0))
    self.assertTrue(np.all(np.isfinite(weights['persons'])))
    self.assertTrue(np.all(weights['persons'] >= 0.0))
    self.assertTrue(np.all(np.isfinite(weights['activities'])))
    self.assertTrue(np.all(weights['activities'] >= 0.0))

    # Sensitivity invariant: sum of weights for any single household <= 1.0
    mapping = transformations._compute_row_root_mappings(
        {
            'households': households,
            'persons': persons,
            'activities': activities,
        },
        hierarchy=hierarchy,
        rng=rng,
    )
    for table_name in ['households', 'persons', 'activities']:
      t_roots = mapping[table_name]
      t_weights = weights[table_name]
      for root in t_roots.dropna().unique():
        root_mask = (t_roots == root).values
        root_weight_sum = t_weights[root_mask].sum()
        self.assertAlmostEqual(root_weight_sum, 1.0, places=5)

  def test_create_slot_linear_chain_constraints_single_or_empty_attribute(self):
    # Single attribute: no within-slot Frankenstein combinations possible
    child_dom_1 = mbi.Domain.fromdict({'age': 10})
    constraints_1 = transformations.create_slot_linear_chain_constraints(
        child_domain=child_dom_1, num_permutation_slots=2
    )
    self.assertEqual(constraints_1, [])

    # Empty domain: no attributes -> empty constraints
    child_dom_0 = mbi.Domain((), ())
    constraints_0 = transformations.create_slot_linear_chain_constraints(
        child_domain=child_dom_0, num_permutation_slots=2
    )
    self.assertEqual(constraints_0, [])

  def test_create_slot_linear_chain_constraints_2attrs_o2(self):
    child_dom = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    constraints = transformations.create_slot_linear_chain_constraints(
        child_domain=child_dom, num_permutation_slots=2
    )
    # o = 2 slots, D = 2 attributes -> 2 * (2 - 1) = 2 constraints
    self.assertLen(constraints, 2)

    c1, c2 = constraints[0], constraints[1]
    self.assertEqual(c1.domain.attributes, ('slot_1.age', 'slot_1.gender'))
    self.assertEqual(c1.domain.shape, (11, 3))
    self.assertEqual(c2.domain.attributes, ('slot_2.age', 'slot_2.gender'))
    self.assertEqual(c2.domain.shape, (11, 3))

    # Verify log-potential values on c1
    pot1 = c1.potential
    # 1. Real + Real combinations (age in [0, 9], gender in [0, 1]) -> 0.0
    for a in range(10):
      for g in range(2):
        self.assertEqual(float(pot1.values[a, g]), 0.0)

    # 2. <EMPTY> + <EMPTY> combination (age = 10, gender = 2) -> 0.0
    self.assertEqual(float(pot1.values[10, 2]), 0.0)

    # 3. Mixed combinations (<EMPTY> + Real or Real + <EMPTY>) -> -inf
    for a in range(10):
      self.assertEqual(float(pot1.values[a, 2]), -np.inf)
    for g in range(2):
      self.assertEqual(float(pot1.values[10, g]), -np.inf)

  def test_create_slot_linear_chain_constraints_3attrs_transitivity(self):
    # 3 attributes: age (4 bins), gender (2 bins), edu (3 bins)
    # Extended shape with <EMPTY>: (5, 3, 4)
    child_dom = mbi.Domain.fromdict({'age': 4, 'gender': 2, 'edu': 3})
    constraints = transformations.create_slot_linear_chain_constraints(
        child_domain=child_dom, num_permutation_slots=1
    )
    # o = 1 slot, D = 3 attributes -> 1 * (3 - 1) = 2 pairwise constraints
    self.assertLen(constraints, 2)
    c_age_gender, c_gender_edu = constraints[0], constraints[1]

    self.assertEqual(
        c_age_gender.domain.attributes, ('slot_1.age', 'slot_1.gender')
    )
    self.assertEqual(c_age_gender.domain.shape, (5, 3))
    self.assertEqual(
        c_gender_edu.domain.attributes, ('slot_1.gender', 'slot_1.edu')
    )
    self.assertEqual(c_gender_edu.domain.shape, (3, 4))

    # Test transitivity by summing log-potentials across full 3D slot domain
    full_slot_domain = mbi.Domain(
        ('slot_1.age', 'slot_1.gender', 'slot_1.edu'), (5, 3, 4)
    )
    combined_pot = c_age_gender.potential.expand(
        full_slot_domain
    ) + c_gender_edu.potential.expand(full_slot_domain)

    # Invariants for combined potential:
    # 1. 100% Real states (age < 4, gender < 2, edu < 3) -> 0.0
    for a in range(4):
      for g in range(2):
        for e in range(3):
          self.assertEqual(float(combined_pot.values[a, g, e]), 0.0)

    # 2. 100% <EMPTY> state (age = 4, gender = 2, edu = 3) -> 0.0
    self.assertEqual(float(combined_pot.values[4, 2, 3]), 0.0)

    # 3. Any mixed/partial state -> -inf
    for a in range(5):
      for g in range(3):
        for e in range(4):
          is_all_real = (a < 4) and (g < 2) and (e < 3)
          is_all_empty = (a == 4) and (g == 2) and (e == 3)
          if not is_all_real and not is_all_empty:
            self.assertEqual(float(combined_pot.values[a, g, e]), -np.inf)

  def test_create_slot_linear_chain_constraints_multi_slot_and_attributes(self):
    # 4 attributes, 3 permutation slots -> 3 * (4 - 1) = 9 constraints
    child_dom = mbi.Domain.fromdict({'a': 2, 'b': 3, 'c': 4, 'd': 5})
    constraints = transformations.create_slot_linear_chain_constraints(
        child_domain=child_dom, num_permutation_slots=3
    )
    self.assertLen(constraints, 9)
    expected_pairs = [
        ('slot_1.a', 'slot_1.b'),
        ('slot_1.b', 'slot_1.c'),
        ('slot_1.c', 'slot_1.d'),
        ('slot_2.a', 'slot_2.b'),
        ('slot_2.b', 'slot_2.c'),
        ('slot_2.c', 'slot_2.d'),
        ('slot_3.a', 'slot_3.b'),
        ('slot_3.b', 'slot_3.c'),
        ('slot_3.c', 'slot_3.d'),
    ]
    actual_pairs = [c.domain.attributes for c in constraints]
    self.assertEqual(actual_pairs, expected_pairs)

  def test_create_slot_linear_chain_constraints_validation_errors(self):
    child_dom = mbi.Domain.fromdict({'age': 10, 'gender': 2})
    with self.assertRaises(ValueError):
      transformations.create_slot_linear_chain_constraints(
          child_domain=child_dom, num_permutation_slots=0
      )
    with self.assertRaises(ValueError):
      transformations.create_slot_linear_chain_constraints(
          child_domain=child_dom, num_permutation_slots=-1
      )

  def test_extract_slot_indices(self):
    # Empty clique
    self.assertEqual(transformations._extract_slot_indices(()), [])

    # Parent attributes and group_size only
    self.assertEqual(
        transformations._extract_slot_indices(('income', 'region')), []
    )
    self.assertEqual(
        transformations._extract_slot_indices(('income', 'group_size')), []
    )

    # Single slot
    self.assertEqual(
        transformations._extract_slot_indices(('slot_1.age',)), [1]
    )
    self.assertEqual(
        transformations._extract_slot_indices(
            ('income', 'slot_1.age', 'slot_1.gender')
        ),
        [1],
    )

    # Multi-slot sibling pairs
    self.assertEqual(
        transformations._extract_slot_indices(('slot_1.age', 'slot_2.age')),
        [1, 2],
    )
    self.assertEqual(
        transformations._extract_slot_indices(
            ('group_size', 'slot_2.gender', 'slot_1.age')
        ),
        [1, 2],
    )

    # Higher slot numbers and sorting
    self.assertEqual(
        transformations._extract_slot_indices(
            ('slot_4.b', 'slot_1.a', 'slot_3.c', 'slot_2.d')
        ),
        [1, 2, 3, 4],
    )

    # Adversarial / malformed column names
    self.assertEqual(
        transformations._extract_slot_indices(
            ('slot_abc.x', 'slot_1', 'my_slot_1.x', 'slot_0.y')
        ),
        [0],
    )

  def test_remap_clique_slots(self):
    # Empty clique
    self.assertEqual(transformations._remap_clique_slots((), {1: 2}), ())

    # Parent attributes and group_size only (unchanged)
    self.assertEqual(
        transformations._remap_clique_slots(('income', 'region'), {1: 2}),
        ('income', 'region'),
    )
    self.assertEqual(
        transformations._remap_clique_slots(('group_size',), {1: 3}),
        ('group_size',),
    )

    # Single slot substitution
    self.assertEqual(
        transformations._remap_clique_slots(('slot_1.age',), {1: 3}),
        ('slot_3.age',),
    )
    self.assertEqual(
        transformations._remap_clique_slots(
            ('income', 'slot_1.age', 'slot_1.gender'), {1: 4}
        ),
        ('income', 'slot_4.age', 'slot_4.gender'),
    )

    # Multi-slot sibling substitution
    self.assertEqual(
        transformations._remap_clique_slots(
            ('slot_1.age', 'slot_2.gender'), {1: 2, 2: 5}
        ),
        ('slot_2.age', 'slot_5.gender'),
    )
    self.assertEqual(
        transformations._remap_clique_slots(
            ('group_size', 'slot_2.gender', 'slot_1.age'), {1: 3, 2: 1}
        ),
        ('group_size', 'slot_1.gender', 'slot_3.age'),
    )

    # Slot not in mapping remains unchanged
    self.assertEqual(
        transformations._remap_clique_slots(('slot_1.a', 'slot_2.b'), {1: 5}),
        ('slot_5.a', 'slot_2.b'),
    )

    # Non-standard / malformed names remain unchanged
    self.assertEqual(
        transformations._remap_clique_slots(
            ('slot_abc.x', 'slot_1', 'my_slot_1.x'), {1: 2}
        ),
        ('slot_abc.x', 'slot_1', 'my_slot_1.x'),
    )

  def test_symmetrize_to_wide_domain_running_example(self):
    # Running example: Household -> Person (s=3, o=2)
    m1 = mbi.LinearMeasurement(
        np.array([100.0, 50.0]), ('income', 'group_size'), stddev=1.0
    )
    m2 = mbi.LinearMeasurement(
        np.array([40.0, 60.0]), ('income', 'slot_1.age'), stddev=1.5
    )
    m3 = mbi.LinearMeasurement(
        np.array([30.0, 70.0]), ('slot_1.age', 'slot_1.gender'), stddev=1.5
    )
    m4 = mbi.LinearMeasurement(
        np.array([20.0, 80.0]), ('slot_1.age', 'slot_2.age'), stddev=2.0
    )

    expanded = transformations.symmetrize_to_wide_domain(
        [m1, m2, m3, m4], max_children_per_parent=3, num_permutation_slots=2
    )

    # Expected:
    # m1: 1 copy on ('income', 'group_size')
    # m2: 3 copies on ('income', 'slot_1.age'), ('income', 'slot_2.age'),
    #     ('income', 'slot_3.age')
    # m3: 3 copies on ('slot_1.age', 'slot_1.gender'),
    #     ('slot_2.age', 'slot_2.gender'), ('slot_3.age', 'slot_3.gender')
    # m4: 3 copies (comb(3, 2)) on (1,2), (1,3), (2,3)
    # Total = 1 + 3 + 3 + 3 = 10 measurements
    self.assertLen(expanded, 10)

    cliques = [m.clique for m in expanded]

    # m1 passthrough
    self.assertIn(('income', 'group_size'), cliques)

    # m2 single slot copies
    self.assertIn(('income', 'slot_1.age'), cliques)
    self.assertIn(('income', 'slot_2.age'), cliques)
    self.assertIn(('income', 'slot_3.age'), cliques)

    # m3 intra-child copies
    self.assertIn(('slot_1.age', 'slot_1.gender'), cliques)
    self.assertIn(('slot_2.age', 'slot_2.gender'), cliques)
    self.assertIn(('slot_3.age', 'slot_3.gender'), cliques)

    # m4 sibling pairwise copies
    self.assertIn(('slot_1.age', 'slot_2.age'), cliques)
    self.assertIn(('slot_1.age', 'slot_3.age'), cliques)
    self.assertIn(('slot_2.age', 'slot_3.age'), cliques)

    # Stddev and measurement values preserved
    for m in expanded:
      if m.clique == ('income', 'slot_3.age'):
        self.assertEqual(m.stddev, 1.5)
        np.testing.assert_allclose(m.noisy_measurement, [40.0, 60.0])
      elif m.clique == ('slot_2.age', 'slot_3.age'):
        self.assertEqual(m.stddev, 2.0)
        np.testing.assert_allclose(m.noisy_measurement, [20.0, 80.0])

  def test_symmetrize_to_wide_domain_person_activity(self):
    # Person -> Activity (s=2, o=2)
    m_root = mbi.LinearMeasurement(np.array([500.0]), (), stddev=0.5)
    m_single = mbi.LinearMeasurement(
        np.array([10.0, 20.0]), ('age', 'slot_1.amount'), stddev=1.0
    )
    m_pair = mbi.LinearMeasurement(
        np.array([5.0, 15.0]), ('slot_1.amount', 'slot_2.type'), stddev=1.2
    )

    expanded = transformations.symmetrize_to_wide_domain(
        [m_root, m_single, m_pair],
        max_children_per_parent=2,
        num_permutation_slots=2,
    )

    # m_root -> 1, m_single -> 2, m_pair -> 1 (comb(2, 2)) = 4 total
    self.assertLen(expanded, 4)
    cliques = [m.clique for m in expanded]
    self.assertIn((), cliques)
    self.assertIn(('age', 'slot_1.amount'), cliques)
    self.assertIn(('age', 'slot_2.amount'), cliques)
    self.assertIn(('slot_1.amount', 'slot_2.type'), cliques)

  def test_symmetrize_to_wide_domain_large_s5_and_tri_slot(self):
    # s = 5, pairwise sibling query (comb(5, 2) = 10 copies)
    m_pair = mbi.LinearMeasurement(
        np.array([1.0]), ('slot_1.a', 'slot_2.b'), stddev=1.0
    )
    expanded_pair = transformations.symmetrize_to_wide_domain(
        [m_pair], max_children_per_parent=5, num_permutation_slots=2
    )
    self.assertLen(expanded_pair, 10)
    expected_pairs = [
        (f'slot_{i}.a', f'slot_{j}.b')
        for i, j in itertools.combinations(range(1, 6), 2)
    ]

    self.assertCountEqual([m.clique for m in expanded_pair], expected_pairs)

    # s = 4, tri-slot sibling query (comb(4, 3) = 4 copies)
    m_triple = mbi.LinearMeasurement(
        np.array([1.0]), ('slot_1.a', 'slot_2.b', 'slot_3.c'), stddev=1.0
    )
    expanded_triple = transformations.symmetrize_to_wide_domain(
        [m_triple], max_children_per_parent=4, num_permutation_slots=3
    )
    self.assertLen(expanded_triple, 4)
    expected_triples = [
        ('slot_1.a', 'slot_2.b', 'slot_3.c'),
        ('slot_1.a', 'slot_2.b', 'slot_4.c'),
        ('slot_1.a', 'slot_3.b', 'slot_4.c'),
        ('slot_2.a', 'slot_3.b', 'slot_4.c'),
    ]
    self.assertCountEqual([m.clique for m in expanded_triple], expected_triples)

  def test_symmetrize_to_wide_domain_capacity_s1(self):
    # If max capacity s = 1, single slot replicates once, pairwise query is
    # dropped (r=2 > s=1)
    m_single = mbi.LinearMeasurement(np.array([1.0]), ('slot_1.a',), stddev=1.0)
    m_pair = mbi.LinearMeasurement(
        np.array([1.0]), ('slot_1.a', 'slot_2.b'), stddev=1.0
    )
    expanded = transformations.symmetrize_to_wide_domain(
        [m_single, m_pair], max_children_per_parent=1, num_permutation_slots=2
    )
    self.assertLen(expanded, 1)
    self.assertEqual(expanded[0].clique, ('slot_1.a',))

  def test_symmetrize_to_wide_domain_validation_errors(self):
    m = mbi.LinearMeasurement(np.array([1.0]), ('income',), stddev=1.0)
    with self.assertRaises(ValueError):
      transformations.symmetrize_to_wide_domain(
          [m], max_children_per_parent=0, num_permutation_slots=2
      )
    with self.assertRaises(ValueError):
      transformations.symmetrize_to_wide_domain(
          [m], max_children_per_parent=3, num_permutation_slots=0
      )


class TransformationsFormalGuaranteesPropertyTest(absltest.TestCase):
  """Property-based tests verifying mathematical invariants and DP guarantees."""

  def test_property_bounded_lineage_and_cascading_limits(self):
    """Verifies that descendant counts never exceed prod(s_j) and cascading truncation holds."""
    rng = np.random.default_rng(12345)
    for _ in range(15):
      n_h = int(rng.integers(5, 20))
      s1 = int(rng.integers(1, 5))
      s2 = int(rng.integers(1, 5))

      h_pks = [f'h_{i}' for i in range(n_h)]
      households = pd.DataFrame({'hid': h_pks})

      # Generate persons (with some overflowing capacity s1)
      p_records = []
      p_pks = []
      for hid in h_pks:
        k1 = int(rng.integers(0, s1 * 3 + 1))
        for _ in range(k1):
          pid = f'p_{len(p_pks)}'
          p_pks.append(pid)
          p_records.append({'pid': pid, 'hid': hid})
      # Also add some orphan persons
      for _ in range(int(rng.integers(0, 5))):
        pid = f'p_{len(p_pks)}'
        p_pks.append(pid)
        p_records.append({'pid': pid, 'hid': 'orphan_h'})
      persons = (
          pd.DataFrame(p_records)
          if p_records
          else pd.DataFrame({'pid': [], 'hid': []})
      )

      # Generate activities (with some overflowing capacity s2)
      a_records = []
      for pid in p_pks:
        k2 = int(rng.integers(0, s2 * 3 + 1))
        for _ in range(k2):
          aid = f'a_{len(a_records)}'
          a_records.append({'aid': aid, 'pid': pid})
      activities = (
          pd.DataFrame(a_records)
          if a_records
          else pd.DataFrame({'aid': [], 'pid': []})
      )

      fk1 = rel_domain.ForeignKeyRelation(
          'households', 'hid', 'persons', 'hid', s1
      )
      fk2 = rel_domain.ForeignKeyRelation(
          'persons', 'pid', 'activities', 'pid', s2
      )
      hierarchy = [
          (0, 'households', None),
          (1, 'persons', fk1),
          (2, 'activities', fk2),
      ]

      mapping = transformations._compute_row_root_mappings(
          {
              'households': households,
              'persons': persons,
              'activities': activities,
          },
          hierarchy,
          rng=rng,
      )

      # 1. Bounded lineage check:
      # For persons: each household index appears at most s1 times
      # For activities: each household index appears at most s1 * s2 times
      p_roots = mapping['persons'].dropna()
      a_roots = mapping['activities'].dropna()

      if not p_roots.empty:
        p_counts = p_roots.value_counts()
        self.assertTrue(
            np.all(p_counts <= s1),
            msg=f'Persons bound exceeded: {p_counts}',
        )
      if not a_roots.empty:
        a_counts = a_roots.value_counts()
        self.assertTrue(
            np.all(a_counts <= s1 * s2),
            msg=f'Activities bound exceeded: {a_counts}',
        )

      # 2. Cascading truncation invariant check:
      # If persons[i] has root None, then any activity belonging to persons[i]
      # MUST have root None
      if not persons.empty and not activities.empty:
        dropped_p_set = set(persons.loc[mapping['persons'].isna(), 'pid'])
        if dropped_p_set:
          for _, a_row in activities.iterrows():
            if a_row['pid'] in dropped_p_set:
              a_idx = activities[activities['aid'] == a_row['aid']].index[0]
              self.assertIsNone(
                  mapping['activities'].iloc[a_idx],
                  msg=(
                      'Cascading truncation violated for activity'
                      f' {a_row["aid"]}'
                  ),
              )

  def test_property_order_agnostic_uniform_subsampling(self):
    """Verifies that child record truncation is statistically uniform and order-agnostic."""
    # A single parent with K=5 children, capacity s=2
    households = pd.DataFrame({'hid': ['h1']})
    persons = pd.DataFrame({
        'pid': ['p0', 'p1', 'p2', 'p3', 'p4'],
        'hid': ['h1'] * 5,
    })
    fk = rel_domain.ForeignKeyRelation('households', 'hid', 'persons', 'hid', 2)
    hierarchy = [(0, 'households', None), (1, 'persons', fk)]

    n_trials = 2500
    retention_counts = np.zeros(5, dtype=np.int64)
    rng = np.random.default_rng(2026)

    for _ in range(n_trials):
      mapping = transformations._compute_row_root_mappings(
          {'households': households, 'persons': persons},
          hierarchy,
          rng=rng,
      )
      active_mask = mapping['persons'].notna().to_numpy()
      self.assertEqual(active_mask.sum(), 2)  # Exactly s=2 chosen every trial
      retention_counts += active_mask.astype(np.int64)

    # Theoretical probability for each child is s/K = 2/5 = 0.40
    empirical_probs = retention_counts / n_trials
    np.testing.assert_allclose(empirical_probs, np.full(5, 0.40), atol=0.035)

    # Invariance to row reversal: reversing DataFrame row order yields the same
    # marginals
    persons_reversed = persons.iloc[::-1].reset_index(drop=True)
    retention_counts_rev = np.zeros(5, dtype=np.int64)
    for _ in range(n_trials):
      mapping_rev = transformations._compute_row_root_mappings(
          {'households': households, 'persons': persons_reversed},
          hierarchy,
          rng=rng,
      )
      active_mask_rev = mapping_rev['persons'].notna().to_numpy()
      retention_counts_rev += active_mask_rev.astype(np.int64)

    empirical_probs_rev = retention_counts_rev / n_trials
    np.testing.assert_allclose(
        empirical_probs_rev, np.full(5, 0.40), atol=0.035
    )

  def test_property_unit_sensitivity_and_weight_mass_conservation(self):
    """Verifies that hierarchical weights satisfy sum_{r in H_i} w_r <= 1.0 for all roots."""
    rng = np.random.default_rng(54321)
    for _ in range(20):
      n_h = int(rng.integers(1, 30))
      s1 = int(rng.integers(1, 6))
      s2 = int(rng.integers(1, 6))

      h_pks = [f'h_{i}' for i in range(n_h)]
      households = pd.DataFrame(
          {'hid': h_pks, 'x': rng.integers(0, 5, size=n_h)}
      )

      p_records = []
      for hid in h_pks:
        k = int(rng.integers(0, s1 * 2 + 1))
        for _ in range(k):
          p_records.append({
              'pid': f'p_{len(p_records)}',
              'hid': hid,
              'y': rng.integers(0, 5),
          })
      # Add random orphans
      for _ in range(int(rng.integers(0, 5))):
        p_records.append({
            'pid': f'p_{len(p_records)}',
            'hid': 'orphan_hid',
            'y': rng.integers(0, 5),
        })
      persons = (
          pd.DataFrame(p_records)
          if p_records
          else pd.DataFrame({'pid': [], 'hid': [], 'y': []})
      )

      a_records = []
      if not persons.empty:
        for pid in persons['pid']:
          k = int(rng.integers(0, s2 * 2 + 1))
          for _ in range(k):
            a_records.append({
                'aid': f'a_{len(a_records)}',
                'pid': pid,
                'z': rng.integers(0, 5),
            })
      activities = (
          pd.DataFrame(a_records)
          if a_records
          else pd.DataFrame({'aid': [], 'pid': [], 'z': []})
      )

      fk1 = rel_domain.ForeignKeyRelation(
          'households', 'hid', 'persons', 'hid', s1
      )
      fk2 = rel_domain.ForeignKeyRelation(
          'persons', 'pid', 'activities', 'pid', s2
      )
      hierarchy = [
          (0, 'households', None),
          (1, 'persons', fk1),
          (2, 'activities', fk2),
      ]

      seed = int(rng.integers(0, 1000000))
      weights = transformations.compute_hierarchical_weights(
          {
              'households': households,
              'persons': persons,
              'activities': activities,
          },
          hierarchy,
          rng=np.random.default_rng(seed),
      )
      mappings = transformations._compute_row_root_mappings(
          {
              'households': households,
              'persons': persons,
              'activities': activities,
          },
          hierarchy,
          rng=np.random.default_rng(seed),
      )

      for t_name in ['households', 'persons', 'activities']:
        t_weights = weights[t_name]
        t_roots = mappings[t_name]

        # Inactive rows have weight 0.0
        inactive_mask = t_roots.isna()
        np.testing.assert_allclose(t_weights[inactive_mask], 0.0)

        # For every active root, sum of weights is exactly 1.0, and each active
        # row has 1/k_eff
        active_roots = t_roots.dropna()
        if not active_roots.empty:
          for root_idx in active_roots.unique():
            root_mask = (t_roots == root_idx).values
            k_eff = root_mask.sum()
            self.assertGreater(k_eff, 0)
            np.testing.assert_allclose(t_weights[root_mask], 1.0 / k_eff)
            self.assertAlmostEqual(t_weights[root_mask].sum(), 1.0, places=10)

          # Total table weight equals number of unique active roots
          num_active_roots = len(active_roots.unique())
          self.assertAlmostEqual(
              t_weights.sum(), float(num_active_roots), places=10
          )

  def test_property_global_l1_sensitivity_neighbor_bound(self):
    """Verifies that adding one household changes any weighted histogram query by L1 <= 1.0."""
    rng = np.random.default_rng(9999)
    for _ in range(15):
      n_h = int(rng.integers(5, 25))
      s1, s2 = 3, 2

      h_pks = [f'h_{i}' for i in range(n_h)]
      h_df = pd.DataFrame({'hid': h_pks, 'cat': rng.integers(0, 4, size=n_h)})

      p_records = []
      for hid in h_pks:
        k = int(rng.integers(0, s1 + 1))
        for _ in range(k):
          p_records.append({
              'pid': f'p_{len(p_records)}',
              'hid': hid,
              'cat': rng.integers(0, 4),
          })
      p_df = (
          pd.DataFrame(p_records)
          if p_records
          else pd.DataFrame({'pid': [], 'hid': [], 'cat': []})
      )

      a_records = []
      if not p_df.empty:
        for pid in p_df['pid']:
          k = int(rng.integers(0, s2 + 1))
          for _ in range(k):
            a_records.append({
                'aid': f'a_{len(a_records)}',
                'pid': pid,
                'cat': rng.integers(0, 4),
            })
      a_df = (
          pd.DataFrame(a_records)
          if a_records
          else pd.DataFrame({'aid': [], 'pid': [], 'cat': []})
      )

      fk1 = rel_domain.ForeignKeyRelation(
          'households', 'hid', 'persons', 'hid', s1
      )
      fk2 = rel_domain.ForeignKeyRelation(
          'persons', 'pid', 'activities', 'pid', s2
      )
      hierarchy = [
          (0, 'households', None),
          (1, 'persons', fk1),
          (2, 'activities', fk2),
      ]

      # Base weights
      w_base = transformations.compute_hierarchical_weights(
          {'households': h_df, 'persons': p_df, 'activities': a_df},
          hierarchy,
          rng=np.random.default_rng(42),
      )

      # Neighbor dataset D' with 1 new household and arbitrary children
      new_hid = 'h_new'
      h_df_prime = pd.concat(
          [
              h_df,
              pd.DataFrame(
                  {'hid': [new_hid], 'cat': [int(rng.integers(0, 4))]}
              ),
          ],
          ignore_index=True,
      )
      k1_new = int(rng.integers(1, s1 + 3))
      new_p = [
          {
              'pid': f'p_new_{i}',
              'hid': new_hid,
              'cat': int(rng.integers(0, 4)),
          }
          for i in range(k1_new)
      ]
      p_df_prime = pd.concat([p_df, pd.DataFrame(new_p)], ignore_index=True)

      new_a = []
      for p_entry in new_p:
        k2_new = int(rng.integers(1, s2 + 3))
        for j in range(k2_new):
          new_a.append({
              'aid': f'a_new_{p_entry["pid"]}_{j}',
              'pid': p_entry['pid'],
              'cat': int(rng.integers(0, 4)),
          })
      a_df_prime = (
          pd.concat([a_df, pd.DataFrame(new_a)], ignore_index=True)
          if new_a
          else a_df.copy()
      )

      w_prime = transformations.compute_hierarchical_weights(
          {
              'households': h_df_prime,
              'persons': p_df_prime,
              'activities': a_df_prime,
          },
          hierarchy,
          rng=np.random.default_rng(42),
      )

      # Evaluate histogram queries on each table
      for t_name, df_base, df_prime in [
          ('households', h_df, h_df_prime),
          ('persons', p_df, p_df_prime),
          ('activities', a_df, a_df_prime),
      ]:
        hist_base = (
            np.bincount(df_base['cat'], weights=w_base[t_name], minlength=4)
            if not df_base.empty
            else np.zeros(4)
        )
        hist_prime = np.bincount(
            df_prime['cat'], weights=w_prime[t_name], minlength=4
        )
        l1_change = float(np.sum(np.abs(hist_prime - hist_base)))
        # Sensitivity Delta <= 1.0
        self.assertLessEqual(
            l1_change,
            1.0 + 1e-6,
            msg=f'L1 sensitivity violated on table {t_name}: {l1_change}',
        )

  def test_property_domain_shape_and_metadata_invariance(self):
    """Verifies that exploration domain shape is pure public metadata (eps=0)."""
    rng = np.random.default_rng(42)
    for _ in range(20):
      n_parent_attrs = rng.integers(1, 4)
      parent_shapes = tuple(rng.integers(2, 20, size=n_parent_attrs))
      parent_attrs = tuple(f'p_{i}' for i in range(n_parent_attrs))
      parent_dom = mbi.Domain(parent_attrs, parent_shapes)

      n_child_attrs = rng.integers(1, 4)
      child_shapes = tuple(rng.integers(2, 20, size=n_child_attrs))
      child_attrs = tuple(f'c_{i}' for i in range(n_child_attrs))
      child_dom = mbi.Domain(child_attrs, child_shapes)

      s = int(rng.integers(1, 10))
      o = int(rng.integers(1, 5))

      # Strategy A: empty_token (+1 to child attributes)
      dom_a = transformations._build_exploration_domain(
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
      dom_b = transformations._build_exploration_domain(
          parent_dom,
          child_dom,
          max_group_size=s,
          num_permutation_slots=o,
          strategy='size_sliced',
      )
      expected_shape_b = parent_shapes + (s + 1,) + child_shapes * o
      self.assertEqual(dom_b.shape, expected_shape_b)

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
              np.sum(pattern_matrix[:, col] == r) for col in range(o)
          ]
          self.assertTrue(
              all(c == slot_counts[0] for c in slot_counts),
              msg=(
                  f'Slot exchangeability violated for k={k}, o={o}, rank={r}:'
                  f' {slot_counts}'
              ),
          )

  def test_property_global_dataset_mass_conservation(self):
    """Verifies sum(weights) == N_parents on random relational datasets."""
    rng = np.random.default_rng(123)
    for _ in range(15):
      n_parents = int(rng.integers(1, 100))
      parent_dom = mbi.Domain.fromdict({'p1': 3})
      parent_ds = mbi.Dataset(
          {'p1': rng.integers(0, 3, size=n_parents)}, parent_dom
      )
      parent_pks = [f'p_{i}' for i in range(n_parents)]

      s = int(rng.integers(1, 6))
      o = int(rng.integers(1, 4))
      strategy = rng.choice(['empty_token', 'size_sliced'])

      # Random child distribution (some 0, some multi-child)
      group_sizes = rng.integers(0, s + 1, size=n_parents)
      child_fks = []
      for p_idx, k in enumerate(group_sizes):
        child_fks.extend([parent_pks[p_idx]] * k)

      n_children = len(child_fks)
      child_dom = mbi.Domain.fromdict({'c1': 5})
      child_ds = mbi.Dataset(
          {'c1': rng.integers(0, 5, size=n_children)}, child_dom
      )

      ds = transformations.build_permuted_exploration_dataset(
          parent_dataset=parent_ds,
          child_dataset=child_ds,
          parent_primary_keys=parent_pks,
          child_foreign_keys=child_fks,
          max_group_size=s,
          num_permutation_slots=o,
          strategy=strategy,
      )
      self.assertAlmostEqual(
          float(np.sum(ds.weights)),
          float(n_parents),
          places=10,
          msg=(
              'Global mass conservation violated: expected'
              f' {n_parents}, got {np.sum(ds.weights)}'
          ),
      )

  def test_property_global_l1_sensitivity_bound(self):
    """Verifies that adding one household changes histogram by L1 norm <= 1.0."""
    rng = np.random.default_rng(999)
    parent_dom = mbi.Domain.fromdict({'p': 2})
    child_dom = mbi.Domain.fromdict({'c': 3})
    s = 4
    o = 2

    for _ in range(10):
      # Base dataset D
      n_p = int(rng.integers(5, 30))
      p_pks = [f'h_{i}' for i in range(n_p)]
      p_data = {'p': rng.integers(0, 2, size=n_p)}
      p_ds = mbi.Dataset(p_data, parent_dom)

      c_fks = []
      for pk in p_pks:
        k = rng.integers(0, s + 1)
        c_fks.extend([pk] * k)
      c_data = {'c': rng.integers(0, 3, size=len(c_fks))}
      c_ds = mbi.Dataset(c_data, child_dom)

      ds_base = transformations.build_permuted_exploration_dataset(
          p_ds,
          c_ds,
          p_pks,
          c_fks,
          max_group_size=s,
          num_permutation_slots=o,
          strategy='empty_token',
      )

      # Neighboring dataset D' = D + {new_household with k_new children}
      k_new = int(rng.integers(0, s + 1))
      p_pks_prime = p_pks + ['h_new']
      p_data_prime = {'p': np.append(p_data['p'], rng.integers(0, 2))}
      p_ds_prime = mbi.Dataset(p_data_prime, parent_dom)

      c_fks_prime = c_fks + ['h_new'] * k_new
      c_data_prime = {
          'c': np.append(c_data['c'], rng.integers(0, 3, size=k_new))
      }
      c_ds_prime = mbi.Dataset(c_data_prime, child_dom)

      ds_prime = transformations.build_permuted_exploration_dataset(
          p_ds_prime,
          c_ds_prime,
          p_pks_prime,
          c_fks_prime,
          max_group_size=s,
          num_permutation_slots=o,
          strategy='empty_token',
      )

      # L1 sensitivity on measurement query (e.g. projection on (p, slot_1.c))
      proj_base = ds_base.project(('p', 'slot_1.c')).datavector()
      proj_prime = ds_prime.project(('p', 'slot_1.c')).datavector()
      l1_diff = float(np.sum(np.abs(proj_prime - proj_base)))

      self.assertAlmostEqual(l1_diff, 1.0, places=5)
      self.assertLessEqual(l1_diff, 1.0 + 1e-5)

  def test_property_slot_marginal_symmetry(self):
    """Verifies Hermitian marginal symmetry P(Slot_i) == P(Slot_j)."""
    rng = np.random.default_rng(777)
    for _ in range(10):
      n_p = int(rng.integers(10, 50))
      p_dom = mbi.Domain.fromdict({'p': 3})
      p_ds = mbi.Dataset({'p': rng.integers(0, 3, size=n_p)}, p_dom)
      p_pks = [f'h_{i}' for i in range(n_p)]

      c_dom = mbi.Domain.fromdict({'c': 6})
      c_fks = []
      for pk in p_pks:
        c_fks.extend([pk] * rng.integers(0, 5))
      c_ds = mbi.Dataset({'c': rng.integers(0, 6, size=len(c_fks))}, c_dom)

      o = int(rng.integers(2, 5))
      ds = transformations.build_permuted_exploration_dataset(
          p_ds,
          c_ds,
          p_pks,
          c_fks,
          max_group_size=4,
          num_permutation_slots=o,
          strategy='empty_token',
      )

      # Check 1-way marginals of all slots match
      s1_hist = np.bincount(
          ds.data['slot_1.c'], weights=ds.weights, minlength=7
      )
      for slot_idx in range(2, o + 1):
        s_hist = np.bincount(
            ds.data[f'slot_{slot_idx}.c'], weights=ds.weights, minlength=7
        )
        np.testing.assert_allclose(s1_hist, s_hist, atol=1e-10)

  def test_property_orphan_foreign_key_isolation(self):
    """Verifies that injecting orphan child records leaves mass invariant."""
    parent_dom = mbi.Domain.fromdict({'p': 2})
    child_dom = mbi.Domain.fromdict({'c': 3})
    p_ds = mbi.Dataset({'p': np.array([0, 1], dtype=np.int64)}, parent_dom)
    p_pks = ['h0', 'h1']

    # 1 child for h1, plus 5 orphan records
    c_fks_with_orphans = [
        'h1',
        'orphan_a',
        'orphan_b',
        'orphan_c',
        'orphan_d',
        'orphan_e',
    ]
    c_ds_with_orphans = mbi.Dataset(
        {'c': np.array([1, 0, 1, 2, 0, 1], dtype=np.int64)}, child_dom
    )

    ds_orphans = transformations.build_permuted_exploration_dataset(
        p_ds,
        c_ds_with_orphans,
        p_pks,
        c_fks_with_orphans,
        max_group_size=3,
        num_permutation_slots=2,
        strategy='empty_token',
    )
    # Total mass must still strictly equal 2.0 (number of parents)
    self.assertAlmostEqual(float(np.sum(ds_orphans.weights)), 2.0)
    # h0 is treated as childless (group_size=0)
    self.assertEqual(
        int(ds_orphans.data['group_size'][ds_orphans.data['p'] == 0][0]), 0
    )
    # h1 has group_size=1
    self.assertEqual(
        int(ds_orphans.data['group_size'][ds_orphans.data['p'] == 1][0]), 1
    )

  def test_property_slot_linear_chain_constraints_transitive_locking(self):
    """Verifies that linear chain constraints guarantee 100% Real or 100% <EMPTY>."""
    rng = np.random.default_rng(4242)
    for _ in range(10):
      n_attrs = int(rng.integers(2, 6))
      shapes = tuple(int(x) for x in rng.integers(2, 6, size=n_attrs))
      attrs = tuple(f'c_{i}' for i in range(n_attrs))
      child_dom = mbi.Domain(attrs, shapes)

      o = int(rng.integers(1, 4))
      constraints = transformations.create_slot_linear_chain_constraints(
          child_domain=child_dom, num_permutation_slots=o
      )

      # 1. Constraint count and pairwise structure: o * (D - 1)
      self.assertLen(constraints, o * (n_attrs - 1))

      # 2. Maximum clique size is exactly 2 (treewidth <= 2)
      for c in constraints:
        self.assertLen(c.domain.attributes, 2)
        self.assertLen(c.clique, 2)

      # 3. Test transitivity for Slot 1
      slot_1_constraints = [
          c for c in constraints if c.domain.attributes[0].startswith('slot_1.')
      ]
      self.assertLen(slot_1_constraints, n_attrs - 1)

      slot_1_attrs = tuple(f'slot_1.{a}' for a in attrs)
      slot_1_shapes = tuple(sz + 1 for sz in shapes)
      slot_1_full_domain = mbi.Domain(slot_1_attrs, slot_1_shapes)

      # Combine potentials
      combined_pot = slot_1_constraints[0].potential.expand(slot_1_full_domain)
      for c in slot_1_constraints[1:]:
        combined_pot = combined_pot + c.potential.expand(slot_1_full_domain)

      # Sample random configurations to verify all-or-nothing locking
      for _ in range(30):
        # Generate random state
        state = [int(rng.integers(0, sz + 1)) for sz in shapes]
        is_all_real = all(val < sz for val, sz in zip(state, shapes))
        is_all_empty = all(val == sz for val, sz in zip(state, shapes))

        pot_val = float(combined_pot.values[tuple(state)])
        if is_all_real or is_all_empty:
          self.assertEqual(
              pot_val,
              0.0,
              msg=f'Allowed state penalized: state={state}, shapes={shapes}',
          )
        else:
          self.assertEqual(
              pot_val,
              -np.inf,
              msg=f'Mixed state not zeroed: state={state}, shapes={shapes}',
          )

  def test_property_symmetrize_to_wide_domain_exchangeability_and_count(self):
    """Verifies that symmetrization expands cliques to exactly comb(s, r) copies and preserves properties."""
    rng = np.random.default_rng(789)
    for _ in range(15):
      s = int(rng.integers(1, 7))
      o = int(rng.integers(1, 4))

      # Random measurement mix
      m_root = mbi.LinearMeasurement(np.array([10.0]), (), stddev=0.5)
      m_parent = mbi.LinearMeasurement(
          np.array([5.0, 5.0]), ('p1', 'p2'), stddev=1.0
      )
      m_single = mbi.LinearMeasurement(
          np.array([1.0, 2.0]), ('p1', 'slot_1.c1'), stddev=1.2
      )
      m_pair = mbi.LinearMeasurement(
          np.array([3.0, 4.0]), ('slot_1.c1', 'slot_2.c2'), stddev=1.5
      )

      measurements = [m_root, m_parent, m_single, m_pair]
      expanded = transformations.symmetrize_to_wide_domain(
          measurements, max_children_per_parent=s, num_permutation_slots=o
      )

      # 1. Non-slot measurements: exactly 1 copy each
      self.assertEqual(sum(1 for m in expanded if not m.clique), 1)
      self.assertEqual(sum(1 for m in expanded if m.clique == ('p1', 'p2')), 1)

      # 2. Single slot measurement: exactly s copies
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

      # 3. Pairwise slot measurement: exactly comb(s, 2) copies (or 0 if s=1)
      pair_copies = [
          m
          for m in expanded
          if any(a.startswith('slot_') for a in m.clique)
          and not any(a.startswith('p') for a in m.clique)
      ]
      expected_pair_count = math.comb(s, 2) if s >= 2 else 0
      self.assertLen(pair_copies, expected_pair_count)
      if s >= 2:
        expected_pair_cliques = [
            (f'slot_{i}.c1', f'slot_{j}.c2')
            for i, j in itertools.combinations(range(1, s + 1), 2)
        ]
        self.assertCountEqual(
            [m.clique for m in pair_copies], expected_pair_cliques
        )

      # 4. Invariant: Measurement vector and stddev are identical across
      # all copies
      for m in single_copies:
        self.assertEqual(m.stddev, 1.2)
        np.testing.assert_allclose(m.noisy_measurement, [1.0, 2.0])
      for m in pair_copies:
        self.assertEqual(m.stddev, 1.5)
        np.testing.assert_allclose(m.noisy_measurement, [3.0, 4.0])


if __name__ == '__main__':
  absltest.main()
