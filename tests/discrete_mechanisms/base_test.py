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

"""Unit tests for the shared ``DiscreteMechanism`` base-class machinery."""

import dataclasses
from unittest import mock

from absl.testing import absltest
import dp_accounting
from dpsynth.discrete_mechanisms import base
from dpsynth.discrete_mechanisms import common
import mbi
import mbi.estimation
import numpy as np


def _dataset(n: int = 200) -> mbi.Dataset:
  return mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=n)


@dataclasses.dataclass(frozen=True, kw_only=True)
class _NoOpMechanism(base.DiscreteMechanism):
  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    return dp_accounting.GaussianDpEvent(noise_multiplier=1.0)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return [(a,) for a in domain.attributes]

  def _select(self, rng, data, measurements, phase_times):
    return []


@dataclasses.dataclass(frozen=True, kw_only=True)
class _NoOpMechanismConfig(base.DiscreteMechanismConfig):

  def _create_mechanism(self, **kwargs):
    return _NoOpMechanism(**kwargs)

  def supporting_cliques(self, domain):
    return []


@dataclasses.dataclass(frozen=True, kw_only=True)
class _NoSelectMechanism(base.DiscreteMechanism):
  @property
  def dp_event(self) -> dp_accounting.DpEvent:
    return dp_accounting.GaussianDpEvent(noise_multiplier=1.0)

  def supporting_cliques(self, domain: mbi.Domain) -> list[mbi.Clique]:
    return [(a,) for a in domain.attributes]


@dataclasses.dataclass(frozen=True, kw_only=True)
class _NoSelectMechanismConfig(base.DiscreteMechanismConfig):

  def _create_mechanism(self, **kwargs):
    return _NoSelectMechanism(**kwargs)

  def supporting_cliques(self, domain):
    return []


class ConfigureTest(absltest.TestCase):

  def test_default_fraction_splits_one_way_budget(self):
    configured = _NoOpMechanismConfig(one_way_budget_fraction=0.25).configure(
        zcdp_rho=100.0
    )
    self.assertEqual(configured.one_way_rho, 25.0)

  def test_zero_one_way_budget_fraction_skips_one_way(self):
    configured = _NoOpMechanismConfig(one_way_budget_fraction=0.0).configure(
        zcdp_rho=100.0
    )
    self.assertIsNone(configured.one_way_rho)

  def test_default_allocate_budget_leaves_measurement_rho_unset(self):
    configured = _NoOpMechanismConfig().configure(zcdp_rho=100.0)
    self.assertIsNone(configured.measurement_rho)

  def test_initial_measurements_skip_one_way(self):
    configured = _NoOpMechanismConfig(one_way_budget_fraction=0.5).configure(
        zcdp_rho=100.0, initial_measurements=[mock.sentinel.measurement]
    )
    self.assertIsNone(configured.one_way_rho)


class RunMachineryTest(absltest.TestCase):

  def test_precompile_failure_is_non_fatal(self):
    mechanism = _NoOpMechanismConfig(pgm_iters=100).configure(zcdp_rho=1000.0)
    with mock.patch.object(
        mbi.estimation.MirrorDescent,
        'precompile',
        side_effect=RuntimeError('simulated precompile failure'),
    ) as mocked_precompile:
      result = mechanism(np.random.default_rng(0), _dataset())
    mocked_precompile.assert_called_once()
    self.assertIsInstance(result, common.DiscreteMechanismResult)

  def test_missing_select_raises_not_implemented(self):
    mechanism = _NoSelectMechanismConfig(pgm_iters=100).configure(
        zcdp_rho=1000.0
    )
    with self.assertRaises(NotImplementedError):
      mechanism(np.random.default_rng(0), _dataset())
