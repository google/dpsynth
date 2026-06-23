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

"""Implementation of an end-to-end DP synthetic data generation mechanism.

In this module there is implementation to run locally.
"""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from absl import logging
import dp_accounting
from dpsynth import constraints
from dpsynth import discrete_mechanisms
from dpsynth import domain
from dpsynth import transformations
from dpsynth.discrete_mechanisms import accounting
from dpsynth.discrete_mechanisms import common
from dpsynth.pipeline_transformations import categorical_values_derivation
from dpsynth.pipeline_transformations import dp_auto_discretizer
import mbi
import numpy as np
import pandas as pd
import pipeline_dp

Dataset: TypeAlias = pd.DataFrame


def _compress_data(data, one_way_measurements):
  """Compresses the domain and measurements if necessary."""
  compressed_domain, compressed_one_way_measurements, compress_transforms = (
      common.get_domain_compression_transformations(one_way_measurements)
  )

  total_measurement = common.convert_to_total_measurement(one_way_measurements)

  logging.info(
      '[SynthKit Tabular]: Estimated Total %d',
      total_measurement.noisy_measurement,
  )
  compressed_data = mbi.Dataset(
      transformations.apply(data.df, compress_transforms),
      compressed_domain,
  )
  logging.info('[SynthKit Tabular]: Original domain: %s', data.domain)
  logging.info('[SynthKit Tabular]: Compressed domain: %s', compressed_domain)

  measurements = [total_measurement] + list(compressed_one_way_measurements)
  return compressed_data, measurements, compress_transforms


def _compute_privacy_parameters(
    epsilon: float,
    delta: float,
    one_way_marginal_budget_fraction: float,
    discrete_config: discrete_mechanisms.DiscreteMechanism,
) -> tuple[float, float]:
  """Compute privacy parameters for one-way marginals and discrete mechanism."""

  one_way_marginal_sigma = dp_accounting.get_sigma_gaussian(
      epsilon=one_way_marginal_budget_fraction * epsilon,
      delta=one_way_marginal_budget_fraction * delta,
  )
  one_way_marginal_gdp_mu = 1.0 / one_way_marginal_sigma**2

  def make_event_from_param(zcdp_rho):
    event1 = dp_accounting.GaussianDpEvent(one_way_marginal_sigma)
    event2 = discrete_config.calibrate(zcdp_rho=zcdp_rho).dp_event
    return dp_accounting.ComposedDpEvent([event1, event2])

  if isinstance(
      discrete_config.calibrate(zcdp_rho=1.0).dp_event, dp_accounting.ZCDpEvent
  ):
    make_fresh_accountant = dp_accounting.rdp.RdpAccountant
  else:
    make_fresh_accountant = dp_accounting.pld.PLDAccountant

  discrete_mechanism_zcdp_rho = dp_accounting.calibrate_dp_mechanism(
      make_event_from_param=make_event_from_param,
      target_epsilon=0.9 * epsilon,
      target_delta=0.9 * delta,
      make_fresh_accountant=make_fresh_accountant,
      bracket_interval=dp_accounting.LowerEndpointAndGuess(1e-3, 1.0),
  )

  return one_way_marginal_gdp_mu, discrete_mechanism_zcdp_rho


def generate(
    data: pd.DataFrame,
    domains: Mapping[str, domain.AttributeType],
    epsilon: float,
    delta: float,
    *,
    discrete_config: discrete_mechanisms.DiscreteMechanism = discrete_mechanisms.MSTMechanism(),
    numerical_bins: int = 32,
    one_way_marginal_budget_fraction: float = 0.1,
    cross_attribute_constraints: Sequence[constraints.Constraint] = (),
    skip_compression: bool = False,
) -> pd.DataFrame:
  """Generate synthetic data with record-level differential privacy.

  Ths function encodes the input categorical and numerical data into a
  discrete domain, then runs the specified mechanism on the discretized data.
  Finally, it converts the synthetic data back to the original domain.

  Args:
    data: The dataset to generate synthetic data for.
    domains: A mapping from column names to attribute domains. Every key in this
      mapping must be a column of `data`.
    epsilon: Privacy parameter.
    delta: Privacy parameter.
    discrete_config: The mechanism configuration for the discretized and
      integer-encoded data.
    numerical_bins: The number of bins to use for discretization.
    one_way_marginal_budget_fraction: The fraction of the total privacy budget
      to use for one-way marginal queries.
    cross_attribute_constraints: Constraints to enforce on the generated data.
    skip_compression: Whether to skip the domain compression step.

  Returns:
    A synthetic dataset.
  """
  assert 0 <= one_way_marginal_budget_fraction <= 1
  if not skip_compression and cross_attribute_constraints:
    raise ValueError(
        'Compression is not supported when cross-attribute constraints are'
        ' provided.'
    )
  for col in domains:
    if col not in data.columns:
      raise ValueError(
          f'{col=} not found in the dataset. Available columns: {data.columns}'
      )
    if isinstance(domains[col], domain.FreeFormTextAttribute):
      raise ValueError(
          f'FreeFormTextAttribute is not supported for column {col!r}.'
          ' Free-form text attributes cannot be synthesized by this mechanism.'
      )

  backend = pipeline_dp.LocalBackend()

  # only for initialization (numerical + unknown domain categorical)
  accountant = pipeline_dp.NaiveBudgetAccountant(0.1 * epsilon, 0.1 * delta)
  engine = pipeline_dp.DPEngine(accountant, backend)
  # for remainder of mechanism, not going through pipeline_dp accounting

  one_way_marginal_gdp_mu, discrete_zcdp_rho = _compute_privacy_parameters(
      0.9 * epsilon,
      0.9 * delta,
      one_way_marginal_budget_fraction,
      discrete_config,
  )

  ##################################################
  # Map the data to a standardized discrete domain #
  ##################################################
  transform_fns = {}
  discrete_domains = {}

  numerical_attributes = {
      col: dom
      for col, dom in domains.items()
      if isinstance(dom, domain.NumericalAttribute)
  }
  open_set_categorical_attributes = [
      col
      for col, dom in domains.items()
      if isinstance(dom, domain.OpenSetCategoricalAttribute)
  ]
  if numerical_attributes:
    # dp_auto_discretizer does not currently handle empty dict here.
    output_numerical = (
        dp_auto_discretizer.create_transformations_via_dp_quantiles(
            pcol=(dict(s) for _, s, in data.iterrows()),
            engine=engine,
            backend=backend,
            field_name_to_attribute=numerical_attributes,
            num_quanitle_buckets=numerical_bins,
        )
    )
  else:
    output_numerical = None

  if open_set_categorical_attributes:
    output_categorical = (
        categorical_values_derivation.derive_categorical_values(
            input_data=(dict(s) for _, s, in data.iterrows()),
            backend=backend,
            dp_engine=engine,
            attribute_keys_to_derive=list(open_set_categorical_attributes),
        )
    )
    logging.info('output_categorical: %s', output_categorical)
  else:
    output_categorical = None

  accountant.compute_budgets()
  if output_numerical is not None:
    for field_name, cat_attr, to_categorical in output_numerical:
      logging.info('Discretizing numerical column: %s', field_name)
      to_standardized = transformations.discrete_encoder(cat_attr)
      transform_fns[field_name] = to_standardized @ to_categorical
      discrete_domains[field_name] = cat_attr.size

  if output_categorical is not None:
    for field_name, cat_attr in list(output_categorical)[0].items():
      logging.info('Deriving categorical column: %s', field_name)
      transform_fns[field_name] = transformations.discrete_encoder(cat_attr)
      discrete_domains[field_name] = cat_attr.size

  categorical_attributes = {
      col: dom
      for col, dom in domains.items()
      if isinstance(dom, domain.CategoricalAttribute)
  }
  for col, attr in categorical_attributes.items():
    logging.info('Encoding categorical column: %s', col)
    transform_fns[col] = transformations.discrete_encoder(attr)
    discrete_domains[col] = attr.size

  discrete = {}
  for col in discrete_domains:
    logging.info('Encoding categorical column: %s', col)
    dtype = np.min_scalar_type(discrete_domains[col])
    values = data[col].map(transform_fns[col].transform).values
    discrete[col] = values.astype(dtype)

  discrete = mbi.Dataset(discrete, mbi.Domain.fromdict(discrete_domains))

  logging.info('[SynthKit Tabular]: Finished encoding data.')

  #######################################################################
  # Measure 1-way marginals and compress domain by merging rare values. #
  #######################################################################
  one_way_marginal_queries = [(col,) for col in discrete.domain]
  gdp_sigma = accounting.gdp_gaussian_sigma(one_way_marginal_gdp_mu)
  rng = np.random.default_rng()
  one_way_measurements = common.measure_marginals_with_noise(
      rng, discrete, one_way_marginal_queries, gdp_sigma
  )
  logging.info('[SynthKit Tabular]: Measured one-way marginals.')

  if not skip_compression:
    discrete, one_way_measurements, compress_transforms = _compress_data(
        discrete, one_way_measurements
    )
    for col in compress_transforms:
      transform_fns[col] = compress_transforms[col] @ transform_fns[col]

  # Run the mechanism on the discretized data.
  initial_potentials = constraints.get_initial_parameters(
      cross_attribute_constraints, discrete.domain
  )

  result = discrete_config.calibrate(zcdp_rho=discrete_zcdp_rho)(
      rng,
      data=discrete,
      initial_measurements=one_way_measurements,
      initial_potentials=initial_potentials,
  )

  synthetic_data = result.model.synthetic_data()
  logging.info('[SynthKit Tabular]: Generated discrete synthetic data.')

  # Convert synthetic data back to the original domain.
  synthetic_columns = {}
  for col in transform_fns:
    synthetic_columns[col] = pd.Series(
        [transform_fns[col].inverse(x) for x in synthetic_data.df[col]],
        dtype=data[col].dtype,
    )
  logging.info('[SynthKit Tabular]: Converted data back to original domain.')

  column_order = [col for col in data.columns if col in domains]
  return pd.DataFrame(synthetic_columns)[column_order]
