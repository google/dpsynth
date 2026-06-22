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

from __future__ import annotations

from absl.testing import absltest
import dp_accounting
from dpsynth import data_generation_v3
from dpsynth import domain
import numpy as np
import pandas as pd

DataGenerationV3 = data_generation_v3.DataGenerationV3


class DataGenerationV3Test(absltest.TestCase):

  def test_end_to_end_categorical(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y', 'z'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
    rng = np.random.default_rng(0)
    calibrated = DataGenerationV3(domains=domains).calibrate(zcdp_rho=100.0)
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_end_to_end_numerical(self):
    domains = {
        'A': domain.NumericalAttribute(min_value=0, max_value=10),
        'B': domain.NumericalAttribute(min_value=-10, max_value=10),
    }
    df = pd.DataFrame({'A': [5, 5, 0], 'B': [5, -10, -5]}, dtype=float)
    rng = np.random.default_rng(0)
    calibrated = DataGenerationV3(domains=domains).calibrate(zcdp_rho=100.0)
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])
    for col, attr in domains.items():
      self.assertTrue(
          synthetic_df[col].between(attr.min_value, attr.max_value).all()
      )

  def test_end_to_end_mixed_domain(self):
    domains = {
        'A': domain.OpenSetCategoricalAttribute(),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    rng = np.random.default_rng(0)
    calibrated = DataGenerationV3(domains=domains).calibrate(
        zcdp_rho=100.0, delta=1e-5
    )
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_end_to_end_with_epsilon_delta(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y', 'z'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
    rng = np.random.default_rng(0)
    calibrated = DataGenerationV3(domains=domains).calibrate(
        epsilon=100, delta=0.1
    )
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_raises_on_freeform_text_attribute(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b']),
        'text': domain.FreeFormTextAttribute(max_tokens=128),
    }
    v3 = DataGenerationV3(domains=domains)
    with self.assertRaises(ValueError):
      v3.calibrate(zcdp_rho=1.0)

  def test_raises_when_not_calibrated(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    rng = np.random.default_rng(0)
    v3 = DataGenerationV3(domains=domains)
    with self.assertRaises(ValueError):
      v3(rng, df)

  def test_dp_event_returns_composed_event(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    calibrated = DataGenerationV3(domains=domains).calibrate(zcdp_rho=100.0)
    self.assertIsInstance(calibrated.dp_event, dp_accounting.ComposedDpEvent)

  def test_calibrate_raises_on_conflicting_params(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    v3 = DataGenerationV3(domains=domains)
    with self.assertRaises(ValueError):
      v3.calibrate(zcdp_rho=1.0, epsilon=1.0, delta=1e-5)

  def test_calibrate_small_epsilon(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.CategoricalAttribute(
            possible_values=['x', 'y', 'z'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
    rng = np.random.default_rng(0)
    calibrated = DataGenerationV3(domains=domains).calibrate(
        epsilon=0.2, delta=1e-5
    )
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])


if __name__ == '__main__':
  absltest.main()
