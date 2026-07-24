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

import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
from dpsynth import constraints
from dpsynth import data_generation_v3
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth.local_mode import beam_initializers
import numpy as np


def _rows_fn(rows):
  """Returns a create_rows_fn that emits the given in-memory rows."""
  return lambda p: p | beam.Create(rows)


class BeamSynthesizerTest(parameterized.TestCase):

  def test_two_pass_categorical_only(self):
    """End-to-end test with categorical columns via run_two_pass."""
    domains = {
        'color': domain.CategoricalAttribute(possible_values=['r', 'g', 'b']),
        'size': domain.CategoricalAttribute(possible_values=['s', 'm', 'l']),
    }
    synth = data_generation_v3.TabularSynthesizer(domains=domains)
    calibrated = synth.calibrate(zcdp_rho=100.0)

    rows = [
        {'color': 'r', 'size': 's'},
        {'color': 'r', 'size': 'm'},
        {'color': 'g', 'size': 'l'},
        {'color': 'g', 'size': 's'},
        {'color': 'b', 'size': 'm'},
        {'color': 'b', 'size': 'l'},
    ] * 100  # 600 rows for statistical stability.

    result = beam_initializers.run_two_pass(
        calibrated, np.random.default_rng(42), _rows_fn(rows)
    )
    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    # MST uses a noisy total count, so row count is approximate.
    self.assertBetween(len(result.synthetic_data), 550, 650)
    self.assertCountEqual(result.synthetic_data.columns, ['color', 'size'])

  def test_two_pass_mixed_types(self):
    """End-to-end test with mixed numerical + categorical columns."""
    domains = {
        'age': domain.NumericalAttribute(min_value=0, max_value=100),
        'grade': domain.CategoricalAttribute(possible_values=['a', 'b', 'c']),
    }
    synth = data_generation_v3.TabularSynthesizer(domains=domains)
    calibrated = synth.calibrate(zcdp_rho=100.0)

    rng_data = np.random.default_rng(0)
    rows = [
        {
            'age': float(rng_data.integers(0, 100)),
            'grade': rng_data.choice(['a', 'b', 'c']),
        }
        for _ in range(500)
    ]

    result = beam_initializers.run_two_pass(
        calibrated, np.random.default_rng(42), _rows_fn(rows)
    )
    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertBetween(len(result.synthetic_data), 450, 550)
    self.assertCountEqual(result.synthetic_data.columns, ['age', 'grade'])

  @parameterized.named_parameters(
      ('mst', discrete_mechanisms.MSTMechanism(pgm_iters=250)),
      (
          'independent',
          discrete_mechanisms.IndependentMechanism(pgm_iters=250),
      ),
      (
          'direct',
          discrete_mechanisms.DirectMechanism(
              prespecified_marginal_queries=[('a',), ('b',), ('a', 'b')],
              pgm_iters=250,
          ),
      ),
  )
  def test_two_pass_runs_across_mechanisms(self, mechanism):
    """run_two_pass generalizes to any mechanism via supporting_cliques."""
    domains = {
        'a': domain.CategoricalAttribute(possible_values=['x', 'y']),
        'b': domain.CategoricalAttribute(possible_values=['p', 'q', 'r']),
    }
    synth = data_generation_v3.TabularSynthesizer(
        domains=domains, discrete_mechanism=mechanism
    )
    calibrated = synth.calibrate(zcdp_rho=100.0)
    rows = [
        {'a': 'x', 'b': 'p'},
        {'a': 'y', 'b': 'q'},
        {'a': 'x', 'b': 'r'},
    ] * 100

    result = beam_initializers.run_two_pass(
        calibrated, np.random.default_rng(0), _rows_fn(rows)
    )
    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertCountEqual(result.synthetic_data.columns, ['a', 'b'])
    self.assertNotEmpty(result.synthetic_data)

  def test_two_pass_total_count_matches_input_under_high_budget(self):
    """With negligible noise, synthetic row count matches the input (F2)."""
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    synth = data_generation_v3.TabularSynthesizer(domains=domains)
    calibrated = synth.calibrate(zcdp_rho=1e8)
    rows = [{'a': 'x'}, {'a': 'y'}] * 150  # 300 rows.

    result = beam_initializers.run_two_pass(
        calibrated, np.random.default_rng(0), _rows_fn(rows)
    )
    self.assertBetween(len(result.synthetic_data), 298, 302)

  def test_two_pass_respects_impossible_combinations(self):
    """Cross-attribute constraints reach the discrete mechanism (F4)."""
    a_attr = domain.CategoricalAttribute(possible_values=['a0', 'a1'])
    b_attr = domain.CategoricalAttribute(possible_values=['b0', 'b1'])
    domains = {'a': a_attr, 'b': b_attr}
    constraint = constraints.Constraint(
        attribute_names=('a', 'b'),
        attribute_domains=(a_attr, b_attr),
        impossible_combinations=[('a0', 'b1')],
    )
    synth = data_generation_v3.TabularSynthesizer(
        domains=domains, cross_attribute_constraints=(constraint,)
    )
    calibrated = synth.calibrate(zcdp_rho=100.0)
    # The data respects the constraint (no (a0, b1) rows). Forwarding the
    # constraint keeps the forbidden cell a structural zero in the output;
    # mbi's constrained sampling may leak a rare row, so allow a small margin.
    rows = [
        {'a': 'a0', 'b': 'b0'},
        {'a': 'a1', 'b': 'b1'},
    ] * 150

    result = beam_initializers.run_two_pass(
        calibrated, np.random.default_rng(0), _rows_fn(rows)
    )
    forbidden = (result.synthetic_data['a'] == 'a0') & (
        result.synthetic_data['b'] == 'b1'
    )
    self.assertLess(forbidden.mean(), 0.05)

  def test_two_pass_preserves_domain_column_order(self):
    """Output columns follow domain declaration order (F5)."""
    domains = {
        'z': domain.CategoricalAttribute(possible_values=['a', 'b']),
        'm': domain.CategoricalAttribute(possible_values=['c', 'd']),
        'a': domain.CategoricalAttribute(possible_values=['e', 'f']),
    }
    synth = data_generation_v3.TabularSynthesizer(domains=domains)
    calibrated = synth.calibrate(zcdp_rho=100.0)
    rows = [
        {'z': 'a', 'm': 'c', 'a': 'e'},
        {'z': 'b', 'm': 'd', 'a': 'f'},
    ] * 50

    result = beam_initializers.run_two_pass(
        calibrated, np.random.default_rng(0), _rows_fn(rows)
    )
    self.assertEqual(list(result.synthetic_data.columns), ['z', 'm', 'a'])

  def test_two_pass_honors_explicit_temp_location(self):
    """Singleton results are written under the provided temp_location (F3)."""
    domains = {'a': domain.CategoricalAttribute(possible_values=['x', 'y'])}
    synth = data_generation_v3.TabularSynthesizer(domains=domains)
    calibrated = synth.calibrate(zcdp_rho=100.0)
    rows = [{'a': 'x'}, {'a': 'y'}] * 50
    temp_dir = self.create_tempdir().full_path

    result = beam_initializers.run_two_pass(
        calibrated,
        np.random.default_rng(0),
        _rows_fn(rows),
        temp_location=temp_dir,
    )
    self.assertIsInstance(result, data_generation_v3.DataGenerationResult)
    self.assertTrue(
        os.path.exists(os.path.join(temp_dir, 'clique_vector.pickle'))
    )

  def test_uncalibrated_raises(self):
    domains = {
        'x': domain.CategoricalAttribute(possible_values=['a', 'b']),
    }
    synth = data_generation_v3.TabularSynthesizer(domains=domains)
    with self.assertRaises(ValueError):
      beam_initializers.run_two_pass(
          synth, np.random.default_rng(0), lambda p: p
      )


if __name__ == '__main__':
  absltest.main()
