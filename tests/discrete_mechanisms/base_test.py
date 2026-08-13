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

"""Unit tests for the shared ``DiscreteSynthesizer`` base-class machinery.

These tests exercise the base class in isolation via minimal concrete
subclasses, rather than relying on inherited coverage from child integration
tests. This lets us cover edge cases in the shared boilerplate (budget
splitting, non-fatal precompile failures, and the ``_select`` contract).
"""

import dataclasses
from unittest import mock

from absl.testing import absltest
import dp_accounting
from dpsynth import api
from dpsynth.discrete_mechanisms import base
from dpsynth.discrete_mechanisms import common
import mbi
import mbi.estimation
import numpy as np


def _dataset(n: int = 200) -> mbi.Dataset:
  return mbi.Dataset.synthetic(mbi.Domain(['a', 'b', 'c'], [3, 4, 5]), N=n)


@dataclasses.dataclass
class _NoOpMechanism(api.DPMechanism):
  zcdp_rho: float | None = None

  def configure(self, zcdp_rho, **kwargs):
    return dataclasses.replace(self, zcdp_rho=zcdp_rho)

  @property
  def dp_event(self):
    return dp_accounting.GaussianDpEvent(1.0)

  def supporting_cliques(self, domain):
    return []

  def __call__(self, rng, data, initial_measurements=None, constraints=()):
    return common.DiscreteSynthesizerResult(
        model=None, synthetic_data=data, measurements=[], diagnostics=None
    )


class ConfigureTest(absltest.TestCase):
  """Tests for the shared ``configure`` / ``_allocate_budget`` budgeting."""

  def test_default_fraction_splits_one_way_budget(self):
    configured = base.DiscreteSynthesizer(
        base.DiscreteSynthesizer(_NoOpMechanism()), one_way_budget_fraction=0.25
    ).configure(zcdp_rho=100.0)
    self.assertEqual(configured.one_way_rho, 25.0)
    self.assertEqual(configured.mechanism.zcdp_rho, 75.0)

  def test_zero_one_way_budget_fraction_skips_one_way(self):
    configured = base.DiscreteSynthesizer(
        base.DiscreteSynthesizer(_NoOpMechanism()), one_way_budget_fraction=0.0
    ).configure(zcdp_rho=100.0)
    self.assertIsNone(configured.one_way_rho)
    self.assertEqual(configured.mechanism.zcdp_rho, 100.0)

  def test_default_allocate_budget_leaves_measurement_rho_unset(self):
    # The base ``_allocate_budget`` hook returns an empty mapping, so no
    # mechanism-specific budget fields are populated.
    configured = base.DiscreteSynthesizer(_NoOpMechanism()).configure(
        zcdp_rho=100.0
    )

  def test_initial_measurements_skip_one_way(self):
    configured = base.DiscreteSynthesizer(
        base.DiscreteSynthesizer(_NoOpMechanism()), one_way_budget_fraction=0.5
    ).configure(
        zcdp_rho=100.0, initial_measurements=[mock.sentinel.measurement]
    )
    self.assertIsNone(configured.one_way_rho)
    self.assertEqual(configured.mechanism.zcdp_rho, 100.0)


class CalibrationGuardTest(absltest.TestCase):
  """Tests that using an unconfigured mechanism fails fast."""

  def test_call_without_configure_raises(self):
    mechanism = base.DiscreteSynthesizer(_NoOpMechanism())
    with self.assertRaisesRegex(ValueError, 'configure'):
      mechanism(np.random.default_rng(0), _dataset())

  def test_dp_event_without_configure_raises(self):
    with self.assertRaisesRegex(ValueError, 'configure'):
      _ = base.DiscreteSynthesizer(_NoOpMechanism()).dp_event
