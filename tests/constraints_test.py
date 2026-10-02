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

from absl.testing import absltest
from absl.testing import parameterized
from dpsynth import constraints
from dpsynth import domain
import mbi
import numpy as np


class ConstraintValidationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.software = domain.CategoricalAttribute(
        ['GameSuite', 'OfficePro', 'DevTool']
    )
    self.os = domain.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])

  def test_raises_on_unequal_attribute_lengths(self):
    with self.assertRaisesRegex(ValueError, 'must have the same length'):
      constraints.Constraint(
          attribute_names=('Software',),
          attribute_domains=(self.software, self.os),
          possible_combinations=[],
      )

  def test_raises_on_bad_combination_length(self):
    with self.assertRaisesRegex(ValueError, 'must have length equal'):
      constraints.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
          possible_combinations=[('GameSuite',)],
      )

  def test_raises_on_no_mode(self):
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      constraints.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
      )

  def test_raises_on_multiple_modes(self):
    with self.assertRaisesRegex(ValueError, 'exactly one'):
      constraints.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
          possible_combinations=[('GameSuite', 'Windows')],
          impossible_combinations=[('DevTool', 'Windows')],
      )

  def test_functional_dependency_requires_two_attributes(self):
    a = domain.CategoricalAttribute(['x', 'y'])
    b = domain.CategoricalAttribute(['p', 'q'])
    c = domain.CategoricalAttribute(['r', 's'])
    with self.assertRaisesRegex(ValueError, 'exactly 2 attributes'):
      constraints.Constraint(
          attribute_names=('a', 'b', 'c'),
          attribute_domains=(a, b, c),
          functional_dependency={'x': 'p'},
      )


class ConstraintToMbiTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.software = domain.CategoricalAttribute(
        ['GameSuite', 'OfficePro', 'DevTool']
    )
    self.os = domain.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])

  def test_possible_combinations(self):
    c = constraints.Constraint(
        attribute_names=('Software', 'OS'),
        attribute_domains=(self.software, self.os),
        possible_combinations=[
            ('GameSuite', 'Windows'),
            ('OfficePro', 'Windows'),
            ('OfficePro', 'MacOS'),
            ('DevTool', 'Linux'),
            ('DevTool', 'MacOS'),
        ],
    )
    mbi_c = c.to_mbi()
    self.assertIsInstance(mbi_c, mbi.Constraint)
    expected = np.full((3, 3), -np.inf)
    expected[0, 0] = 0  # GameSuite, Windows
    expected[1, 0] = 0  # OfficePro, Windows
    expected[1, 2] = 0  # OfficePro, MacOS
    expected[2, 1] = 0  # DevTool, Linux
    expected[2, 2] = 0  # DevTool, MacOS
    np.testing.assert_array_equal(mbi_c.potential.values, expected)

  def test_impossible_combinations(self):
    c = constraints.Constraint(
        attribute_names=('Software', 'OS'),
        attribute_domains=(self.software, self.os),
        impossible_combinations=[
            ('GameSuite', 'Linux'),
            ('GameSuite', 'MacOS'),
        ],
    )
    vals = np.asarray(c.to_mbi().potential.values)
    self.assertEqual(vals[0, 1], -np.inf)  # GameSuite, Linux
    self.assertEqual(vals[0, 2], -np.inf)  # GameSuite, MacOS
    self.assertEqual(vals[0, 0], 0.0)
    self.assertEqual(vals[1, 0], 0.0)
    self.assertEqual(vals[2, 2], 0.0)

  def test_functional_dependency(self):
    fine = domain.CategoricalAttribute(['a', 'b', 'c', 'd'])
    coarse = domain.CategoricalAttribute(['X', 'Y'])
    c = constraints.Constraint(
        attribute_names=('fine', 'coarse'),
        attribute_domains=(fine, coarse),
        functional_dependency={'a': 'X', 'b': 'X', 'c': 'Y', 'd': 'Y'},
    )
    mbi_c = c.to_mbi()
    self.assertTrue(mbi_c.is_deterministic)
    vals = np.asarray(mbi_c.potential.values)
    self.assertEqual(vals[0, 0], 0.0)  # a -> X
    self.assertEqual(vals[1, 0], 0.0)  # b -> X
    self.assertEqual(vals[2, 1], 0.0)  # c -> Y
    self.assertEqual(vals[3, 1], 0.0)  # d -> Y
    self.assertEqual(vals[0, 1], -np.inf)
    self.assertEqual(vals[2, 0], -np.inf)

  def test_numerical_categorical_possible_combinations(self):
    age = domain.NumericalAttribute(min_value=0, max_value=100, dtype='int')
    status = domain.CategoricalAttribute(['minor', 'adult'])
    c = constraints.Constraint(
        attribute_names=('age', 'status'),
        attribute_domains=(age, status),
        possible_combinations=[((0, 17), 'minor'), ((17, 100), 'adult')],
    )
    self.assertEqual(c.pinned_edges, {'age': (17.0,)})
    mbi_c = c.to_mbi({'age': [10.0, 17.0, 50.0]})
    vals = np.asarray(mbi_c.potential.values)
    # Bins 0 ([0, 10]) and 1 ((10, 17]) allow 'minor' (0) and forbid 'adult' (1)
    self.assertEqual(vals[0, 0], 0.0)
    self.assertEqual(vals[0, 1], -np.inf)
    self.assertEqual(vals[1, 0], 0.0)
    self.assertEqual(vals[1, 1], -np.inf)
    # Bins 2 ((17, 50]) and 3 ((50, 100]) allow 'adult' (1)
    self.assertEqual(vals[2, 1], 0.0)
    self.assertEqual(vals[2, 0], -np.inf)
    self.assertEqual(vals[3, 1], 0.0)
    self.assertEqual(vals[3, 0], -np.inf)

  def test_numerical_none_possible_combinations(self):
    status = domain.CategoricalAttribute(['missing', 'valid'])
    score = domain.NumericalAttribute(
        min_value=0.0, max_value=100.0, clip_to_range=False
    )
    c = constraints.Constraint(
        attribute_names=('status', 'score'),
        attribute_domains=(status, score),
        possible_combinations=[
            ('missing', None),
            ('valid', (0.0, 100.0)),
        ],
    )
    self.assertEqual(c.pinned_edges, {})
    mbi_c = c.to_mbi({'score': [50.0]})
    vals = np.asarray(mbi_c.potential.values)
    # shape is (2, 3): score has OOD at 0, [0, 50] at 1, (50, 100] at 2.
    expected = np.full((2, 3), -np.inf)
    expected[0, 0] = 0.0
    expected[1, 1] = 0.0
    expected[1, 2] = 0.0
    np.testing.assert_array_equal(vals, expected)

  def test_numerical_validation_errors(self):
    num_clipped = domain.NumericalAttribute(
        min_value=0.0, max_value=10.0, clip_to_range=True
    )
    cat = domain.CategoricalAttribute(['a', 'b'])
    with self.assertRaisesRegex(ValueError, 'clip_to_range=False'):
      constraints.Constraint(
          attribute_names=('num', 'cat'),
          attribute_domains=(num_clipped, cat),
          possible_combinations=[(None, 'a')],
      )
    with self.assertRaisesRegex(ValueError, 'must satisfy'):
      constraints.Constraint(
          attribute_names=('num', 'cat'),
          attribute_domains=(num_clipped, cat),
          possible_combinations=[((-1.0, 5.0), 'a')],
      )
    with self.assertRaisesRegex(ValueError, 'must satisfy'):
      constraints.Constraint(
          attribute_names=('num', 'cat'),
          attribute_domains=(num_clipped, cat),
          possible_combinations=[((5.0, 5.0), 'a')],
      )
    with self.assertRaisesRegex(ValueError, 'does not support'):
      constraints.Constraint(
          attribute_names=('num', 'cat'),
          attribute_domains=(num_clipped, cat),
          functional_dependency={(0.0, 5.0): 'a', (5.0, 10.0): 'b'},
      )
    num_int = domain.NumericalAttribute(min_value=0, max_value=10, dtype='int')
    with self.assertRaisesRegex(ValueError, 'integer bounds'):
      constraints.Constraint(
          attribute_names=('num', 'cat'),
          attribute_domains=(num_int, cat),
          possible_combinations=[((0, 5.5), 'a')],
      )


if __name__ == '__main__':
  absltest.main()
