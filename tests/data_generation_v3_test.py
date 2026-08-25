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
from dpsynth import constraints
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import aim_gdp
from dpsynth.discrete_mechanisms.independent import IndependentConfig
import mbi
import numpy as np
import pandas as pd

TabularConfig = data_generation_v3.TabularConfig


def _make_discrete_data(rng, n=1000):
  domains = mbi.Domain(['a', 'b', 'c'], [3, 3, 3])
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = (a + b + rng.integers(0, 2, size=n)) % 3
  return mbi.Dataset({'a': a, 'b': b, 'c': c}, domains)


def _make_mixed_data(rng, n=1000):
  domains = {
      'a': domain.CategoricalAttribute(possible_values=[0, 1, 2]),
      'b': domain.CategoricalAttribute(possible_values=[0, 1, 2]),
      'c': domain.NumericalAttribute(
          min_value=0.0, max_value=1.0, dtype='float'
      ),
  }
  a = rng.integers(0, 3, size=n)
  b = np.where(rng.random(n) < 0.75, a, rng.integers(0, 3, size=n))
  c = np.clip((a + b) / 4.0 + rng.normal(0.0, 0.1, size=n), 0.0, 1.0)
  return pd.DataFrame({'a': a, 'b': b, 'c': c}), domains


def _normalized_l1(data, model, clique):
  expected = data.project(clique).datavector().astype(float)
  actual = model.project(clique).datavector().astype(float)
  expected /= expected.sum()
  actual /= actual.sum()
  return np.abs(expected - actual).sum() / 2.0


def _discrete_workload_mechanism_baseline_errors(
    config, baseline_config, workload, zcdp_rho=5.0
):
  rng = np.random.default_rng(0)
  data = _make_discrete_data(rng)

  mechanism_result = config.configure(zcdp_rho=zcdp_rho)(rng, data)
  baseline_result = baseline_config.configure(zcdp_rho=zcdp_rho)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(data, mechanism_result.model, clique)
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(data, baseline_result.model, clique) for clique in workload
  ])
  return mechanism_error, baseline_error


def _mixed_workload_mechanism_baseline_errors(
    config, baseline_config, workload, zcdp_rho=5.0, numerical_bins=16
):
  rng = np.random.default_rng(0)
  data, domains = _make_mixed_data(rng, n=1000)

  mechanism_synth = TabularConfig(
      domains=domains,
      discrete_mechanism=config,
      numerical_bins=numerical_bins,
  )
  baseline_synth = TabularConfig(
      domains=domains,
      discrete_mechanism=baseline_config,
      numerical_bins=numerical_bins,
  )

  mechanism_result = mechanism_synth.configure(zcdp_rho=zcdp_rho)(rng, data)
  baseline_result = baseline_synth.configure(zcdp_rho=zcdp_rho)(rng, data)

  mechanism_error = np.mean([
      _normalized_l1(
          mechanism_result.codec.encode(data),
          mechanism_result.discrete_mechanism_result.model,
          clique,
      )
      for clique in workload
  ])
  baseline_error = np.mean([
      _normalized_l1(
          baseline_result.codec.encode(data),
          baseline_result.discrete_mechanism_result.model,
          clique,
      )
      for clique in workload
  ])
  return mechanism_error, baseline_error


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
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)
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
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)
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
    calibrated = TabularConfig(domains=domains).configure(
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
    calibrated = TabularConfig(domains=domains).calibrate(
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
    v3 = TabularConfig(domains=domains)
    with self.assertRaises(Exception):
      v3.configure(zcdp_rho=1.0)

  def test_raises_when_not_calibrated(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    rng = np.random.default_rng(0)
    v3 = TabularConfig(domains=domains)
    with self.assertRaises(Exception):
      v3(rng, df)

  def test_dp_event_returns_composed_event(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)
    self.assertIsInstance(calibrated.dp_event, dp_accounting.ComposedDpEvent)

  def test_calibrate_raises_on_conflicting_params(self):
    domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    v3 = TabularConfig(domains=domains)
    with self.assertRaises(Exception):
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
    calibrated = TabularConfig(domains=domains).calibrate(
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
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)

    # total_count_sigma should be set for numerical-only domains.
    self.assertIsNotNone(calibrated.total_count_sigma)
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
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)

    # DPGaussianCount is always allocated.
    self.assertIsNotNone(calibrated.total_count_sigma)
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
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)
    result = calibrated(rng, df)
    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])

  def test_discrete_workload_regression_with_aim(self):
    workload = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = IndependentConfig()
    mechanism_error, baseline_error = (
        _discrete_workload_mechanism_baseline_errors(
            config, baseline_config, workload
        )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_discrete_workload_regression_with_aim_gdp(self):
    workload = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = aim_gdp.AIMGDPConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    baseline_config = IndependentConfig()
    mechanism_error, baseline_error = (
        _discrete_workload_mechanism_baseline_errors(
            config, baseline_config, workload
        )
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_mixed_workload_regression_with_aim(self):
    workload = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = aim.AIMConfig(workload=workload, max_rounds=4, pgm_iters=500)
    baseline_config = IndependentConfig()
    mechanism_error, baseline_error = _mixed_workload_mechanism_baseline_errors(
        config, baseline_config, workload
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_mixed_workload_regression_with_aim_gdp(self):
    workload = [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]
    config = aim_gdp.AIMGDPConfig(
        workload=workload, max_rounds=4, pgm_iters=500
    )
    baseline_config = IndependentConfig()
    mechanism_error, baseline_error = _mixed_workload_mechanism_baseline_errors(
        config, baseline_config, workload
    )
    self.assertLess(mechanism_error, 0.05 * baseline_error)

  def test_empty_dataset(self):
    """Tests that DPSynth works without crashing on empty datasets, and outputs noisy rows."""
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    }
    df = pd.DataFrame(columns=['A', 'B'])
    rng = np.random.default_rng(0)
    calibrated = TabularConfig(domains=domains).configure(zcdp_rho=100.0)
    result = calibrated(rng, df)

    self.assertIsInstance(result.synthetic_data, pd.DataFrame)
    self.assertListEqual(result.synthetic_data.columns.tolist(), ['A', 'B'])
    # The true count is 0, but DPSynth always outputs at least one row.
    self.assertLen(result.synthetic_data, 1)


class MaxRecordsPerUserTest(parameterized.TestCase):
  """Tests the experimental user-level DP knob end to end."""

  def _categorical_domains(self):
    return {
        'A': domain.CategoricalAttribute(['a', 'b', 'c']),
        'B': domain.CategoricalAttribute(['x', 'y', 'z']),
    }

  def test_configure_propagates_k_to_submechanisms(self):
    k = 5
    config = TabularConfig(domains=self._categorical_domains())
    calibrated = config.configure(zcdp_rho=100.0, max_records_per_user=k)
    self.assertEqual(calibrated.max_records_per_user, k)
    self.assertEqual(calibrated.base_mechanism.max_records_per_user, k)

  def test_dp_event_invariant_to_k(self):
    config = TabularConfig(domains=self._categorical_domains())
    calibrated1 = config.configure(zcdp_rho=100.0)
    calibrated2 = config.configure(zcdp_rho=100.0, max_records_per_user=5)
    self.assertEqual(repr(calibrated1.dp_event), repr(calibrated2.dp_event))

  def test_end_to_end_with_k(self):
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    config = TabularConfig(domains=self._categorical_domains())
    calibrated = config.configure(zcdp_rho=100.0, max_records_per_user=3)
    synthetic_df = calibrated(np.random.default_rng(0), df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A', 'B'])

  def test_open_set_with_k_supported(self):
    df = pd.DataFrame({'A': ['a', 'b', 'c', 'a', 'b', 'a'] * 5})
    domains = {'A': domain.OpenSetCategoricalAttribute()}
    base = TabularConfig(domains=domains).configure(zcdp_rho=100.0, delta=1e-5)
    config = TabularConfig(domains=domains)
    mech = config.configure(zcdp_rho=100.0, delta=1e-5, max_records_per_user=3)
    # Accounting is byte-identical across k; only the injected noise scales.
    self.assertEqual(repr(mech.dp_event), repr(base.dp_event))
    synthetic_df = mech(np.random.default_rng(0), df).synthetic_data
    self.assertListEqual(synthetic_df.columns.tolist(), ['A'])

  def test_initializers_inherit_k(self):
    domains = self._categorical_domains()
    config = TabularConfig(domains=domains)
    calibrated = config.configure(zcdp_rho=100.0, max_records_per_user=2)
    for init in calibrated.initializers.values():
      self.assertEqual(init.max_records_per_user, 2)

  def test_pure_preset_configure_and_calibrate_with_schema(self):
    schema = domain.Schema({
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
    })
    df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': [1.0, 5.0, 10.0]})
    rng = np.random.default_rng(0)

    # Pure preset with no schema in constructor
    preset = TabularConfig(numerical_bins=16)

    # 1. configure(schema=...)
    calibrated = preset.configure(schema=schema, zcdp_rho=100.0)
    result = calibrated(rng, df).synthetic_data
    self.assertListEqual(result.columns.tolist(), ['A', 'B'])

    # 2. calibrate(schema=...)
    calibrated2 = preset.calibrate(schema=schema, epsilon=1.0, delta=1e-5)
    result2 = calibrated2(rng, df).synthetic_data
    self.assertListEqual(result2.columns.tolist(), ['A', 'B'])

  def test_configure_with_schema_constraints(self):
    c = constraints.Constraint(
        attribute_names=('A', 'B'),
        possible_combinations=[('a', 'x'), ('b', 'y')],
    )
    schema = domain.Schema(
        attributes={
            'A': domain.CategoricalAttribute(possible_values=['a', 'b']),
            'B': domain.CategoricalAttribute(possible_values=['x', 'y']),
        },
        constraints=(c,),
    )
    df = pd.DataFrame({'A': ['a', 'b'], 'B': ['x', 'y']})
    rng = np.random.default_rng(0)

    preset = TabularConfig()
    calibrated = preset.configure(schema=schema, zcdp_rho=100.0)
    synthetic_df = calibrated(rng, df).synthetic_data
    for _, row in synthetic_df.iterrows():
      self.assertIn((row['A'], row['B']), [('a', 'x'), ('b', 'y')])

  @parameterized.named_parameters(('zero', 0), ('negative', -3))
  def test_invalid_k_raises(self, k):
    config = TabularConfig(domains=self._categorical_domains())
    with self.assertRaises(Exception):
      _ = config.configure(zcdp_rho=100.0, max_records_per_user=k)

  def test_poisson_calibrate_with_categorical_domains_and_gdp_mech(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.CategoricalAttribute(possible_values=['x', 'y', 'z']),
    }
    config = TabularConfig(
        domains=domains,
        discrete_mechanism=discrete_mechanisms.IndependentConfig(),
    )
    mechanism = config.calibrate(
        epsilon=1.0,
        delta=1e-6,
        poisson_sampling_prob=0.1,
    )
    self.assertIsNotNone(mechanism)

  def test_poisson_calibrate_with_mixed_domains(self):
    domains = {
        'A': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
        'B': domain.NumericalAttribute(min_value=0, max_value=10),
        'C': domain.OpenSetCategoricalAttribute(),
    }
    config = TabularConfig(domains=domains)
    with self.assertRaises(dp_accounting.UnsupportedEventError):
      _ = config.calibrate(
          epsilon=1.0,
          delta=1e-6,
          poisson_sampling_prob=0.1,
      )


if __name__ == '__main__':
  absltest.main()

  def test_tabular_synthesizer_deprecated(self):
    with self.assertWarnsRegex(
        DeprecationWarning,
        'TabularSynthesizer is deprecated. Use TabularConfig for configuration '
        'and TabularMechanism for the calibrated runnable mechanism.',
    ):
      data_generation_v3.TabularSynthesizer(domains={})
