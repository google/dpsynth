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

"""Smoke test for the deprecated data_generation_v2 shim."""

import warnings

from absl.testing import absltest
from dpsynth import data_generation_v2
from dpsynth import domain
import pandas as pd


class DeprecationShimTest(absltest.TestCase):

  def test_generate_emits_deprecation_warning(self):
    attribute_domains = {
        'A': domain.CategoricalAttribute(
            possible_values=['a', 'b', 'c'], out_of_domain_index=0
        ),
    }
    df = pd.DataFrame({'A': ['a', 'b', 'c']})
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter('always')
      synthetic_df = data_generation_v2.generate(
          df, attribute_domains, epsilon=100, delta=0.1, skip_compression=True
      )
    self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))
    self.assertIsInstance(synthetic_df, pd.DataFrame)


if __name__ == '__main__':
  absltest.main()
