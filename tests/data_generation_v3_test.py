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
from absl.testing import parameterized
import dp_accounting
from dpsynth import data_generation_v3
from dpsynth import domain
import numpy as np
import pandas as pd

TabularSynthesizer = data_generation_v3.TabularSynthesizer


class DataGenerationV3Test(parameterized.TestCase):

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
    calibrated = TabularSynthesizer(domains=domains).configure(zcdp_rho=100.0)
    synthetic_df = calibrated(rng, df).synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_end_to_end_numerical(self):
    domains = {
        'A': domain.NumericalAttribute(min_value=0, max_value=10),
        'B': domain.NumericalAttribute(min_value=-10, max_value=10),
    }
    df = pd.DataFrame({'A': [5, 5, 0], 'B': [5, -10, -5]}, dtype=float)
    rng = np.random.default_rng(0)
    calibrated = TabularSynthesizer(domains=domains).configure(zcdp_rho=100.0)
    synthetic_df = calibrated(rng, df).synthetic_data
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
    calibrated = TabularSynthesizer(domains=domains).configure(
        zcdp_rho=100.0, delta=1e-5
    )
    synthetic_df = calibrated(rng, df).synthetic_data
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
    calibrated = TabularSynthesizer(domains=domains).calibrate(
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
    v3 = TabularSynthesizer(domains=domains)
    with self.assertRaises(ValueError):
      v3.configure(zcdp_rho=1.0)

  def test_raises_when_not_calibrated(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    rng = np.random.default_rng(0)
    v3 = TabularSynthesizer(domains=domains)
    with self.assertRaises(ValueError):
      v3(rng, df)

  def test_dp_event_returns_composed_event(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    calibrated = TabularSynthesizer(domains=domains).configure(zcdp_rho=100.0)
    self.assertIsInstance(calibrated.dp_event, dp_accounting.ComposedDpEvent)

  def test_calibrate_raises_on_conflicting_params(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    v3 = TabularSynthesizer(domains=domains)
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
    calibrated = TabularSynthesizer(domains=domains).calibrate(
        epsilon=0.2, delta=1e-5
    )
    result = calibrated(rng, df)
    synthetic_df = result.synthetic_data
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_numerical_only_uses_dp_count(self):
    """Numerical-only domains should allocate a DPGaussianCount for total."""
    domains = {
        'A': domain.NumericalAttribute(min_value=0, max_value=10),
        'B': domain.NumericalAttribute(min_value=-10, max_value=10),
    }
    df = pd.DataFrame({'A': [5, 5, 0], 'B': [5, -10, -5]}, dtype=float)
    rng = np.random.default_rng(0)
    calibrated = TabularSynthesizer(domains=domains).configure(zcdp_rho=100.0)

    # total_count_mechanism should be set for numerical-only domains.
    self.assertIsNotNone(calibrated.total_count_mechanism)
    synthetic_df = calibrated(rng, df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_mixed_domain_always_has_dp_count(self):
    """Mixed domains also allocate a DPGaussianCount for total."""
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    rng = np.random.default_rng(0)
    calibrated = TabularSynthesizer(domains=domains).configure(zcdp_rho=100.0)

    # DPGaussianCount is always allocated.
    self.assertIsNotNone(calibrated.total_count_mechanism)
    synthetic_df = calibrated(rng, df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  @parameterized.product(
      sentinel=[np.nan, None],
      clip_to_range=[True, False],
      dtype=['float', 'int'],
  )
  def test_nan_numerical_column(self, sentinel, clip_to_range, dtype):
    """Regression test: NaN/None/pd.NA numerical columns should not crash."""
    max_value = 100 if dtype == 'int' else 10
    domains = {
        'A': domain.NumericalAttribute(
            min_value=0,
            max_value=max_value,
            clip_to_range=clip_to_range,
            dtype=dtype,
        ),
        'B': domain.CategoricalAttribute(possible_values=['x', 'y']),
    }
    df = pd.DataFrame({
        'A': [sentinel, sentinel, sentinel],
        'B': ['x', 'y', 'x'],
    })
    rng = np.random.default_rng(0)
    calibrated = TabularSynthesizer(domains=domains).configure(zcdp_rho=100.0)
    result = calibrated(rng, df)
    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])


class MaxRecordsPerUserTest(parameterized.TestCase):
  """Tests the experimental user-level DP knob end to end."""

  def _categorical_domains(self):
    return {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }

  def test_configure_propagates_k_to_submechanisms(self):
    k = 5
    calibrated = TabularSynthesizer(
        domains=self._categorical_domains(),
        experimental_max_records_per_user=k,
    ).configure(zcdp_rho=100.0)
    self.assertEqual(calibrated.experimental_max_records_per_user, k)
    self.assertEqual(calibrated.total_count_mechanism.max_records_per_user, k)
    self.assertEqual(calibrated.discrete_mechanism.contribution_scale, k)
    for init in calibrated.initializers.values():
      self.assertEqual(init.mechanism.max_records_per_user, k)

  def test_dp_event_invariant_to_k(self):
    base = TabularSynthesizer(domains=self._categorical_domains()).configure(
        zcdp_rho=100.0
    )
    scaled = TabularSynthesizer(
        domains=self._categorical_domains(), experimental_max_records_per_user=5
    ).configure(zcdp_rho=100.0)
    self.assertEqual(repr(scaled.dp_event), repr(base.dp_event))

  def test_end_to_end_with_k(self):
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    calibrated = TabularSynthesizer(
        domains=self._categorical_domains(), experimental_max_records_per_user=3
    ).configure(zcdp_rho=100.0)
    synthetic_df = calibrated(np.random.default_rng(0), df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_open_set_with_k_supported(self):
    df = pd.DataFrame({'A': ['a', 'b', 'c', 'a', 'b', 'a'] * 5})
    domains = {'A': domain.OpenSetCategoricalAttribute()}
    base = TabularSynthesizer(domains=domains).configure(
        zcdp_rho=100.0, delta=1e-5
    )
    scaled = TabularSynthesizer(
        domains=domains, experimental_max_records_per_user=2
    ).configure(zcdp_rho=100.0, delta=1e-5)
    # Accounting is byte-identical across k; only the injected noise scales.
    self.assertEqual(repr(scaled.dp_event), repr(base.dp_event))
    synthetic_df = scaled(np.random.default_rng(0), df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A'])

  def test_custom_initializers_inherit_k(self):
    domains = self._categorical_domains()
    inits = data_generation_v3._create_initializers(domains, 32, 0.0)
    calibrated = TabularSynthesizer(
        domains=domains, initializers=inits, experimental_max_records_per_user=2
    ).configure(zcdp_rho=100.0)
    for init in calibrated.initializers.values():
      self.assertEqual(init.mechanism.max_records_per_user, 2)

  @parameterized.named_parameters(('zero', 0), ('negative', -3))
  def test_invalid_k_raises(self, k):
    with self.assertRaises(ValueError):
      TabularSynthesizer(
          domains=self._categorical_domains(),
          experimental_max_records_per_user=k,
      )


if __name__ == '__main__':
  absltest.main()
