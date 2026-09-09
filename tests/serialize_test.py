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

"""Unit tests for cattrs-based serialize.py in DPSynth."""

import os
from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import dp_accounting
import dpsynth
from dpsynth import data_generation_v3
from dpsynth import domain
from dpsynth import relational
from dpsynth import serialize
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import direct
from dpsynth.discrete_mechanisms import discrete
from dpsynth.discrete_mechanisms import independent
from dpsynth.discrete_mechanisms import mst
from dpsynth.discrete_mechanisms import swift
from dpsynth.local_mode import initialization
import yaml

try:
  import jax_privacy  # pylint: disable=g-import-not-at-top

  _execution_plan: Any = getattr(jax_privacy, 'execution_plan', None)
except ImportError:
  jax_privacy = None
  _execution_plan = None


class SerializeTest(parameterized.TestCase):

  def test_mst_config_roundtrip_and_defaults_omitted(self):
    config = mst.MSTConfig(
        pgm_iters=2500,
        select_budget_fraction=0.75,
    )
    yaml_str = serialize.to_yaml(config)
    # Verify default maximum_marginal_size is omitted from YAML.
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'MSTConfig')
    self.assertEqual(raw_dict['pgm_iters'], 2500)
    self.assertEqual(raw_dict['select_budget_fraction'], 0.75)
    self.assertNotIn('maximum_marginal_size', raw_dict)

    # Roundtrip check
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_aim_config_roundtrip(self):
    config = aim.AIMConfig(
        max_rounds=25,
        max_model_size=50,
        max_marginal_size=2e6,
        anneal_factor=3.0,
        select_budget_fraction=0.2,
        pgm_iters=500,
    )
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_swift_config_roundtrip(self):
    config = swift.SWIFTConfig(
        max_clique_size=5e6,
        max_marginal_size=2e6,
        pgm_iters=8000,
        select_budget_frac=0.15,
    )
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_independent_config_roundtrip(self):
    config = independent.IndependentConfig(pgm_iters=3000)
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_direct_config_roundtrip(self):
    config = direct.DirectConfig(
        pgm_iters=4000,
        prespecified_marginal_queries=[('a', 'b'), ('c',)],
    )
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_discrete_config_roundtrip(self):
    config = discrete.DiscreteConfig(
        mechanism=aim.AIMConfig(pgm_iters=400),
        compress_columns=['col1', 'col2'],
        one_way_budget_fraction=0.2,
    )
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_tabular_config_roundtrip_and_defaults_omitted(self):
    config = data_generation_v3.TabularConfig(
        discrete_mechanism=mst.MSTConfig(pgm_iters=1500),
        numerical_bins=64,
        init_budget_fraction=0.15,
    )
    yaml_str = serialize.to_yaml(config)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'TabularConfig')
    self.assertEqual(raw_dict['numerical_bins'], 64)
    # Default domains=None and cross_attribute_constraints=() are omitted:
    self.assertNotIn('domains', raw_dict)
    self.assertNotIn('cross_attribute_constraints', raw_dict)
    self.assertNotIn('compress_columns', raw_dict)

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_tabular_config_with_compress_columns(self):
    config = data_generation_v3.TabularConfig(
        compress_columns=True,
    )
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)
    self.assertTrue(loaded.compress_columns)

  def test_tabular_config_with_legacy_domains(self):
    config = data_generation_v3.TabularConfig(
        domains={
            'age': domain.NumericalAttribute(min_value=0, max_value=120),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
            'notes': domain.FreeFormTextAttribute(max_tokens=100),
            'state': domain.OpenSetCategoricalAttribute(),
        },
        discrete_mechanism=mst.MSTConfig(pgm_iters=1500),
        numerical_bins=64,
    )
    yaml_str = serialize.to_yaml(config)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertIn('domains', raw_dict)
    self.assertNotIn('clip_to_range', raw_dict['domains']['age'])
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_initializer_configs_roundtrip(self):
    num_init = initialization.NumericalInitializerConfig(
        num_partitions=16,
    )
    yaml_str = serialize.to_yaml(num_init)
    self.assertEqual(serialize.from_yaml(yaml_str), num_init)

    cat_init = initialization.CategoricalInitializerConfig()
    yaml_str = serialize.to_yaml(cat_init)
    self.assertEqual(serialize.from_yaml(yaml_str), cat_init)

    open_init = initialization.OpenSetInitializerConfig(min_count=5)
    yaml_str = serialize.to_yaml(open_init)
    self.assertEqual(serialize.from_yaml(yaml_str), open_init)

  def test_multitable_config_roundtrip(self):
    config = relational.MultiTableConfig(
        discrete_mechanism=mst.MSTConfig(pgm_iters=500),
    )
    yaml_str = serialize.to_yaml(config)
    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_schema_roundtrip(self):
    schema = domain.Schema({
        'age': domain.NumericalAttribute(min_value=0, max_value=120),
        'cat': domain.CategoricalAttribute(possible_values=['A', 'B']),
    })
    yaml_str = serialize.to_yaml(schema)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'Schema')
    # Default constraints=() should be omitted:
    self.assertNotIn('constraints', raw_dict)

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, schema)

  def test_file_io_roundtrip(self):
    config = data_generation_v3.TabularConfig(
        discrete_mechanism=mst.MSTConfig(pgm_iters=1000),
    )
    tmp_path = os.path.join(self.create_tempdir().full_path, 'config.yaml')
    serialize.to_yaml(config, tmp_path)
    loaded = serialize.from_yaml(tmp_path)
    self.assertEqual(loaded, config)

  def test_dpsynth_top_level_helpers(self):
    config = mst.MSTConfig(pgm_iters=1234)
    yaml_str = dpsynth.to_yaml(config)
    self.assertEqual(dpsynth.from_yaml(yaml_str), config)

    tmp_path = os.path.join(self.create_tempdir().full_path, 'top_level.yaml')
    dpsynth.to_yaml(config, tmp_path)
    self.assertEqual(dpsynth.from_yaml(tmp_path), config)

  def test_zcdp_event_roundtrip(self):
    event = dp_accounting.ZCDpEvent(rho=0.5)
    yaml_str = serialize.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'ZCDpEvent')
    self.assertEqual(raw_dict['rho'], 0.5)

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_gaussian_dp_event_roundtrip(self):
    event = dp_accounting.GaussianDpEvent(noise_multiplier=1.25)
    yaml_str = serialize.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'GaussianDpEvent')
    self.assertEqual(raw_dict['noise_multiplier'], 1.25)

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_composed_dp_event_roundtrip(self):
    event = dp_accounting.ComposedDpEvent([
        dp_accounting.ZCDpEvent(rho=0.1),
        dp_accounting.dp_event.EpsilonDeltaDpEvent(epsilon=0.0, delta=1e-5),
    ])
    yaml_str = serialize.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'ComposedDpEvent')
    self.assertEqual(raw_dict['events'][0]['type'], 'ZCDpEvent')
    self.assertEqual(raw_dict['events'][1]['type'], 'EpsilonDeltaDpEvent')

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_self_composed_dp_event_roundtrip(self):
    event = dp_accounting.SelfComposedDpEvent(
        event=dp_accounting.GaussianDpEvent(noise_multiplier=1.5), count=10
    )
    yaml_str = serialize.to_yaml(event)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'SelfComposedDpEvent')
    self.assertEqual(raw_dict['event']['type'], 'GaussianDpEvent')
    self.assertEqual(raw_dict['count'], 10)

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_mechanism_dp_event_roundtrip(self):
    dom = {'col': domain.CategoricalAttribute(possible_values=['a', 'b'])}
    config = data_generation_v3.TabularConfig(
        discrete_mechanism=mst.MSTConfig(pgm_iters=100)
    )
    mech = config.calibrate(dom, epsilon=1.0, delta=1e-5)
    event = mech.dp_event
    yaml_str = dpsynth.to_yaml(event)
    loaded = dpsynth.from_yaml(yaml_str)
    self.assertEqual(loaded, event)

  def test_execution_plan_roundtrip(self):
    if _execution_plan is None:
      self.skipTest('jax_privacy.execution_plan is not available')
    config = _execution_plan.BandMFConfig(
        iterations=100,
        expected_participations=4.0,
        strategy=[0.5, 0.3, 0.2],
        noise_multiplier=1.0,
    )
    yaml_str = serialize.to_yaml(config)
    raw_dict = yaml.safe_load(yaml_str)
    self.assertEqual(raw_dict['type'], 'BandMFConfig')
    self.assertEqual(raw_dict['iterations'], 100)
    self.assertEqual(raw_dict['expected_participations'], 4.0)
    self.assertEqual(raw_dict['noise_multiplier'], 1.0)
    self.assertLen(raw_dict['strategy'], 3)

    loaded = serialize.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

    loaded_typed = serialize.from_yaml(
        yaml_str, expected_type=_execution_plan.BandMFConfig
    )
    self.assertEqual(loaded_typed, config)

  def test_execution_plan_without_type_tag(self):
    if _execution_plan is None:
      self.skipTest('jax_privacy.execution_plan is not available')
    yaml_str = """
iterations: 100
expected_participations: 4.0
strategy: [1.0, 0.5, 0.2]
noise_multiplier: 1.0
"""
    config = serialize.from_yaml(
        yaml_str, expected_type=_execution_plan.BandMFConfig
    )
    self.assertIsInstance(config, _execution_plan.BandMFConfig)
    self.assertEqual(config.iterations, 100)
    self.assertEqual(config.expected_participations, 4.0)
    self.assertEqual(config.noise_multiplier, 1.0)
    self.assertEqual(config.num_bands, 3)

  def test_unknown_type_raises(self):
    with self.assertRaises(ValueError):
      serialize.from_yaml('type: NonExistentConfig\n')


if __name__ == '__main__':
  absltest.main()
