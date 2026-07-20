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

import doctest

from absl.testing import absltest
import dp_accounting
from dpsynth import domain
from dpsynth.experimental import nested
import numpy as np
import pandas as pd


class NestedTabularSynthesizerTest(absltest.TestCase):

  def _make_synth(self):
    """Creates a simple two-type synthesizer for testing."""
    shared_schema = {
        'platform': domain.CategoricalAttribute(
            possible_values=['web', 'mobile']
        ),
    }
    per_type_schemas = {
        'click': {
            'element': domain.CategoricalAttribute(
                possible_values=['button', 'link', 'image']
            ),
        },
        'purchase': {
            'amount': domain.CategoricalAttribute(
                possible_values=['low', 'medium', 'high']
            ),
        },
    }
    return nested.NestedTabularSynthesizer(
        shared_schema=shared_schema,
        per_type_schemas=per_type_schemas,
    )

  def test_type_vocabulary(self):
    synth = self._make_synth()
    self.assertEqual(synth.type_vocabulary, ['click', 'purchase'])

  def test_calibrate_returns_new_instance(self):
    synth = self._make_synth()
    calibrated = synth.configure(zcdp_rho=10.0)
    self.assertIsNot(synth, calibrated)
    # Original should not be calibrated.
    self.assertIsNone(synth._shared_synth)
    # Calibrated should have internal state set.
    self.assertIsNotNone(calibrated._shared_synth)
    self.assertIsNotNone(calibrated._detail_synths)

  def test_dp_event_before_calibrate_raises(self):
    synth = self._make_synth()
    with self.assertRaises(ValueError):
      _ = synth.dp_event

  def test_dp_event_is_composed(self):
    synth = self._make_synth()
    calibrated = synth.configure(zcdp_rho=10.0)
    event = calibrated.dp_event
    self.assertIsInstance(event, dp_accounting.ComposedDpEvent)
    # Detail level should be a ZCDpEvent (conservative parallel composition).
    self.assertLen(event.events, 2)
    self.assertIsInstance(event.events[1], dp_accounting.ZCDpEvent)

  def test_call_before_calibrate_raises(self):
    synth = self._make_synth()
    rng = np.random.default_rng(0)
    with self.assertRaises(ValueError):
      synth(rng, {})

  def test_end_to_end(self):
    synth = self._make_synth()
    calibrated = synth.configure(zcdp_rho=100.0)
    rng = np.random.default_rng(42)
    data = {
        'click': pd.DataFrame({
            'platform': ['web', 'mobile', 'web'] * 10,
            'element': ['button', 'link', 'image'] * 10,
        }),
        'purchase': pd.DataFrame({
            'platform': ['web', 'mobile', 'mobile'] * 10,
            'amount': ['low', 'medium', 'high'] * 10,
        }),
    }
    result = calibrated(rng, data)
    self.assertIsInstance(result, nested.NestedSynthesisResult)
    # Should produce output for at least one type.
    self.assertTrue(result.synthetic_data)
    for type_name, df in result.synthetic_data.items():
      self.assertIn(type_name, ['click', 'purchase'])
      self.assertIsInstance(df, pd.DataFrame)
      # Each output should have shared + type-specific columns.
      self.assertIn('platform', df.columns)

  def test_empty_type_schema_skipped(self):
    """Types with no type-specific columns get no detail model."""
    shared_schema = {
        'color': domain.CategoricalAttribute(possible_values=['red', 'blue']),
    }
    per_type_schemas = {
        'with_detail': {
            'size': domain.CategoricalAttribute(
                possible_values=['S', 'M', 'L']
            ),
        },
        'no_detail': {},
    }
    synth = nested.NestedTabularSynthesizer(
        shared_schema=shared_schema,
        per_type_schemas=per_type_schemas,
    )
    calibrated = synth.configure(zcdp_rho=10.0)
    self.assertNotIn('no_detail', calibrated._detail_synths)
    self.assertIn('with_detail', calibrated._detail_synths)


def load_tests(loader, tests, ignore):
  del loader, ignore  # Unused.
  tests.addTests(doctest.DocTestSuite(nested))
  return tests


if __name__ == '__main__':
  absltest.main()
