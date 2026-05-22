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

from absl.testing import absltest
from dpsynth import constraints
from dpsynth import data_generation_v2
from dpsynth import domain
import pandas as pd


class MechanismTest(absltest.TestCase):

  def test_end_to_end_categorical(self):
    attribute_domains = {
        "A": domain.CategoricalAttribute(
            possible_values=["a", "b", "c"], out_of_domain_index=0
        ),
        "B": domain.CategoricalAttribute(
            possible_values=["x", "y", "z"], out_of_domain_index=0
        ),
        "C": domain.OpenSetCategoricalAttribute(),
    }

    values = [
        ["a", "x", "4"],
        ["b", "y", "4"],
        ["c", "z", "4"],
    ]

    df = pd.DataFrame(data=values, columns=["A", "B", "C"])
    synthetic_df = data_generation_v2.generate(
        df,
        attribute_domains,
        epsilon=100,
        delta=0.1,
        skip_compression=True,
    )
    self.assertIsInstance(synthetic_df, pd.DataFrame)

  def test_end_to_end_numerical(self):
    attribute_domains = {
        "A": domain.NumericalAttribute(min_value=0, max_value=10),
        "B": domain.NumericalAttribute(min_value=-10, max_value=10),
    }

    values = [
        [5, 5],
        [5, -10],
        [0, -5],
    ]

    df = pd.DataFrame(data=values, columns=["A", "B"], dtype=float)
    synthetic_df = data_generation_v2.generate(
        df, attribute_domains, 1000, 0.1
    )
    self.assertListEqual(synthetic_df.columns.tolist(), ["A", "B"])
    for col in attribute_domains:
      dom = attribute_domains[col]
      left, right = dom.min_value, dom.max_value
      self.assertTrue(synthetic_df[col].between(left, right).all())

  def test_end_to_end_categorical_with_constraint(self):
    attribute_domains = {
        "A": domain.CategoricalAttribute(
            possible_values=["a", "b", "c"], out_of_domain_index=0
        ),
        "B": domain.CategoricalAttribute(
            possible_values=["x", "y", "z"], out_of_domain_index=0
        ),
    }

    constraint = constraints.Constraint(
        attribute_names=("A", "B"),
        attribute_domains=(
            attribute_domains["A"],
            attribute_domains["B"],
        ),
        possible_combinations=[
            ("a", "x"),
            ("b", "y"),
            ("c", "z"),
        ],
    )

    values = [
        ["a", "x"],
        ["b", "y"],
        ["c", "z"],
        ["a", "y"],
        ["b", "x"],
        ["c", "x"],
    ]

    df = pd.DataFrame(data=values, columns=["A", "B"])
    synthetic_df = data_generation_v2.generate(
        df,
        attribute_domains,
        epsilon=1.0,
        delta=1e-5,
        discrete_config=data_generation_v2.discrete_mechanisms.MSTConfig(),
        cross_attribute_constraints=[constraint],
        skip_compression=True,
    )
    def is_valid(row):
      return (row["A"], row["B"]) in constraint.possible_combinations

    self.assertTrue(synthetic_df.apply(is_valid, axis=1).all())

  def test_end_to_end_mixed_domain(self):
    attribute_domains = {
        "A": domain.OpenSetCategoricalAttribute(),
        "B": domain.NumericalAttribute(min_value=0, max_value=10),
    }

    values = [
        ["a", 1],
        ["b", 5],
        ["c", 10],
    ]

    df = pd.DataFrame(data=values, columns=["A", "B"])
    df["B"] = df["B"].astype(float)
    synthetic_df = data_generation_v2.generate(
        df,
        attribute_domains,
        epsilon=100,
        delta=0.1,
        skip_compression=True,
    )
    self.assertIsInstance(synthetic_df, pd.DataFrame)
    self.assertListEqual(synthetic_df.columns.tolist(), ["A", "B"])
    dom_b = attribute_domains["B"]
    self.assertTrue(
        synthetic_df["B"].between(dom_b.min_value, dom_b.max_value).all()
    )


if __name__ == "__main__":
  absltest.main()
