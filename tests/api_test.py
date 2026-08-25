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

"""Unit tests for MechanismConfig serialization in api.py."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import dpsynth
from dpsynth import api
from dpsynth import constraints
from dpsynth import data_generation_v3
from dpsynth import domain
from dpsynth import relational
from dpsynth.discrete_mechanisms import aim
from dpsynth.discrete_mechanisms import aim_gdp
from dpsynth.discrete_mechanisms import direct
from dpsynth.discrete_mechanisms import discrete
from dpsynth.discrete_mechanisms import independent
from dpsynth.discrete_mechanisms import mst
from dpsynth.discrete_mechanisms import swift


class ApiYamlSerializationTest(parameterized.TestCase):

  def test_mst_config_roundtrip(self):
    config = mst.MSTConfig(
        pgm_iters=2500,
        select_budget_fraction=0.75,
        maximum_marginal_size=5_000_000,
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)
    self.assertEqual(mst.MSTConfig.from_yaml(yaml_str), config)

  def test_aim_config_roundtrip(self):
    config = aim.AIMConfig(
        max_rounds=25,
        max_model_size=50,
        max_marginal_size=2e6,
        anneal_factor=3.0,
        select_budget_fraction=0.2,
        pgm_iters=500,
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_swift_config_roundtrip(self):
    config = swift.SWIFTConfig(
        max_clique_size=5e6,
        max_marginal_size=2e6,
        pgm_iters=8000,
        select_budget_frac=0.15,
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_independent_config_roundtrip(self):
    config = independent.IndependentConfig(pgm_iters=3000)
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_direct_config_roundtrip(self):
    config = direct.DirectConfig(
        pgm_iters=4000,
        prespecified_marginal_queries=[('a', 'b'), ('c',)],
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_aim_gdp_config_roundtrip(self):
    config = aim_gdp.AIMGDPConfig(
        pgm_iters=500,
        max_rounds=10,
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_discrete_config_roundtrip(self):
    config = discrete.DiscreteConfig(
        mechanism=aim.AIMConfig(pgm_iters=400),
        compress_columns=['col1', 'col2'],
        one_way_budget_fraction=0.2,
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_tabular_config_pure_preset_roundtrip(self):
    config = data_generation_v3.TabularConfig(
        discrete_mechanism=mst.MSTConfig(pgm_iters=1500),
        numerical_bins=64,
        init_budget_fraction=0.15,
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_tabular_config_with_schema_roundtrip(self):
    schema = domain.Schema(
        attributes={
            'age': domain.NumericalAttribute(min_value=0, max_value=120),
            'gender': domain.CategoricalAttribute(possible_values=['M', 'F']),
            'notes': domain.FreeFormTextAttribute(max_tokens=100),
            'state': domain.OpenSetCategoricalAttribute(),
        },
        constraints=(
            constraints.Constraint(
                attribute_names=('gender',),
                impossible_combinations=[('X',)],
            ),
        ),
    )
    config = data_generation_v3.TabularConfig(
        schema=schema,
        discrete_mechanism=mst.MSTConfig(pgm_iters=1000),
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_multitable_config_roundtrip(self):
    config = relational.MultiTableConfig(
        domains={
            'users': {
                'age': domain.NumericalAttribute(min_value=18, max_value=80),
            },
            'orders': {
                'amount': domain.NumericalAttribute(min_value=0, max_value=100),
            },
        },
        foreign_keys=[
            relational.ForeignKeyRelation(
                parent_table='users',
                parent_primary_key='user_id',
                child_table='orders',
                child_foreign_key='user_id',
                max_children_per_parent=5,
            ),
        ],
        discrete_mechanism=mst.MSTConfig(pgm_iters=500),
    )
    yaml_str = config.to_yaml()
    loaded = api.MechanismConfig.from_yaml(yaml_str)
    self.assertEqual(loaded, config)

  def test_file_io_roundtrip(self):
    config = data_generation_v3.TabularConfig(
        domains={'age': domain.NumericalAttribute(min_value=0, max_value=100)},
        discrete_mechanism=mst.MSTConfig(pgm_iters=1000),
    )
    tmp_path = os.path.join(self.create_tempdir().full_path, 'config.yaml')
    config.to_yaml_file(tmp_path)
    loaded = api.MechanismConfig.from_yaml_file(tmp_path)
    self.assertEqual(loaded, config)

  def test_module_level_helpers(self):
    config = mst.MSTConfig(pgm_iters=1234)
    yaml_str = api.to_yaml(config)
    self.assertEqual(api.from_yaml(yaml_str), config)
    self.assertEqual(dpsynth.to_yaml(config), yaml_str)
    self.assertEqual(dpsynth.from_yaml(yaml_str), config)

    tmp_path = os.path.join(self.create_tempdir().full_path, 'helper.yaml')
    dpsynth.to_yaml_file(config, tmp_path)
    self.assertEqual(dpsynth.from_yaml_file(tmp_path), config)

  def test_missing_type_raises(self):
    with self.assertRaises(ValueError):
      api.MechanismConfig.from_yaml('pgm_iters: 1000\n')

  def test_unknown_type_raises(self):
    with self.assertRaises(ValueError):
      api.MechanismConfig.from_yaml('type: NonExistentConfig\n')


if __name__ == '__main__':
  absltest.main()
