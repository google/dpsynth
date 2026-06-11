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
from dpsynth import domain
from bin import derive_domain
import numpy as np
import pandas as pd


class DeriveDomainTest(absltest.TestCase):

  def test_derive_domain_from_data(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [0, 1, 2, 3, 4, 5],
        'float': [3.14, 2.72, 1.61, 1.41, 1.23, 1.05],
    })
    expected_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
        'int': domain.NumericalAttribute(
            min_value=0, max_value=5, clip_to_range=True, dtype='int'
        ),
        'float': domain.NumericalAttribute(
            min_value=1.05, max_value=3.14, dtype='float'
        ),
    }
    self.assertEqual(derive_domain.derive_domain_from_data(df), expected_domain)

  def test_derive_domain_from_data_single_value(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [0, 1, 2, 3, 4, 5],
        'int_single_value': [5, 5, 5, 5, 5, 5],
        'float': [3.14, 2.72, 1.61, 1.41, 1.23, 1.05],
        'float_single_value': [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
    })
    expected_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
        'int': domain.NumericalAttribute(
            min_value=0, max_value=5, clip_to_range=True, dtype='int'
        ),
        'int_single_value': domain.CategoricalAttribute(
            possible_values=[5],
        ),
        'float': domain.NumericalAttribute(
            min_value=1.05, max_value=3.14, dtype='float'
        ),
        'float_single_value': domain.CategoricalAttribute(
            possible_values=[3.14],
        ),
    }
    self.assertEqual(derive_domain.derive_domain_from_data(df), expected_domain)

  def test_derive_domain_from_data_nan(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [0, 1, 2, 3, 4, 5],
        'float': [np.nan, 2.72, 1.61, 1.41, 1.23, 1.05],
    })
    expected_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
        'int': domain.NumericalAttribute(
            min_value=0, max_value=5, clip_to_range=True, dtype='int'
        ),
        'float': domain.NumericalAttribute(
            min_value=1.05, max_value=2.72, dtype='float'
        ),
    }
    self.assertEqual(derive_domain.derive_domain_from_data(df), expected_domain)

  def test_derive_domain_from_data_only_nan(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'float': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    })

    # This is a hack to get around the fact that we can't compare
    # np.nan for equality. np.nan == np.nan is always False.

    self.assertEqual(
        derive_domain.derive_domain_from_data(df),
        {
            'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
            'int': domain.CategoricalAttribute(
                possible_values=[None],
            ),
            'float': domain.CategoricalAttribute(
                possible_values=[None],
            ),
        },
    )

  def test_derive_domain_from_single_value_and_nan(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [0, 1, 2, 3, 4, 5],
        'int_single_value': [5, 5, 5, np.nan, np.nan, np.nan],
        'float': [1.0, 2.72, 1.61, 1.41, 1.23, 1.05],
        'float_single_value': [np.nan, 3.14, np.nan, 3.14, np.nan, 3.14],
    })
    expected_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
        'int': domain.NumericalAttribute(
            min_value=0, max_value=5, clip_to_range=True, dtype='int'
        ),
        'int_single_value': domain.CategoricalAttribute(
            possible_values=[5],
        ),
        'float': domain.NumericalAttribute(
            min_value=1.0, max_value=2.72, dtype='float'
        ),
        'float_single_value': domain.CategoricalAttribute(
            possible_values=[3.14],
        ),
    }
    self.assertEqual(derive_domain.derive_domain_from_data(df), expected_domain)

  def test_derive_domain_from_data_sentinel_value(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [0, 1, 2, 3, 4, 5],
        'int_sentinel_value': [-1, 1, 2, 3, 4, -1],
        'float': [3.14, 2.72, 1.61, 1.41, 1.23, 1.05],
        'float_sentinel_value': [-1, 2.72, 1.61, 1.41, 1.23, -1],
    })
    expected_domain = {
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B', 'C']),
        'int': domain.NumericalAttribute(
            min_value=0, max_value=5, clip_to_range=True, dtype='int'
        ),
        'int_sentinel_value': domain.NumericalAttribute(
            min_value=1, max_value=4, clip_to_range=False, dtype='int'
        ),
        'float': domain.NumericalAttribute(
            min_value=1.05, max_value=3.14, dtype='float'
        ),
        'float_sentinel_value': domain.NumericalAttribute(
            min_value=1.23, max_value=2.72, clip_to_range=False, dtype='float'
        ),
    }
    self.assertEqual(
        derive_domain.derive_domain_from_data(df, -1), expected_domain
    )

  def test_derive_domain_from_data_sentinel_value_invalid(self):
    df = pd.DataFrame({
        'cat': ['A', 'B', 'C', 'A', 'B', 'C'],
        'int': [0, 1, 2, 3, 4, 5],
        'int_sentinel_value': [-1, 1, 2, 3, 4, -1],
        'float': [3.14, 2.72, 1.61, 1.41, 1.23, 1.05],
        'float_sentinel_value': [-1, 2.72, 1.61, 1.41, 1.23, -1],
    })
    with self.assertRaises(ValueError):
      derive_domain.derive_domain_from_data(df, 0)

  def test_derive_domain_categorical_attribute_with_ints(self):
    df = pd.DataFrame({'cat': ['A', 'B', 'C', -1, 'D', 1]})
    expected_domain = {
        'cat': domain.CategoricalAttribute(
            possible_values=[-1, 1, 'A', 'B', 'C', 'D']
        ),
    }
    self.assertEqual(derive_domain.derive_domain_from_data(df), expected_domain)


if __name__ == '__main__':
  absltest.main()
