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


class ConstraintsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.software = domain.CategoricalAttribute(
        ['GameSuite', 'OfficePro', 'DevTool']
    )
    self.os = domain.CategoricalAttribute(['Windows', 'Linux', 'MacOS'])
    self.constraint = constraints.Constraint(
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

  def test_init_raises_error_on_unequal_attribute_lengths(self):
    with self.assertRaisesRegex(ValueError, 'must have the same length'):
      constraints.Constraint(
          attribute_names=('Software',),
          attribute_domains=(self.software, self.os),
          possible_combinations=[],
      )

  def test_init_raises_error_on_bad_combination_length(self):
    with self.assertRaisesRegex(ValueError, 'must have length equal'):
      constraints.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
          possible_combinations=[('GameSuite',)],
      )

  def test_init_raises_error_on_value_not_in_domain(self):
    with self.assertRaisesRegex(ValueError, 'not found in possible values'):
      constraints.Constraint(
          attribute_names=('Software', 'OS'),
          attribute_domains=(self.software, self.os),
          possible_combinations=[('GameSuite', 'Solaris')],
      )

  def test_valid_constraint(self):
    self.assertLen(self.constraint.possible_combinations, 5)

  def test_possible_indices(self):
    expected = ((0, 0), (1, 0), (1, 2), (2, 1), (2, 2))
    self.assertEqual(constraints._possible_indices(self.constraint), expected)

  def test_potential(self):
    potential = constraints._mbi_potential(self.constraint)
    self.assertIsInstance(potential, mbi.Factor)
    self.assertEqual(potential.domain.shape, (3, 3))
    self.assertEqual(potential.domain.attrs, ('Software', 'OS'))
    expected_values = np.full((3, 3), -np.inf)
    expected_values[0, 0] = 0
    expected_values[1, 0] = 0
    expected_values[1, 2] = 0
    expected_values[2, 1] = 0
    expected_values[2, 2] = 0
    np.testing.assert_array_equal(potential.values, expected_values)


if __name__ == '__main__':
  absltest.main()
