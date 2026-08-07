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

r"""Precomputes all 2-way marginals over the 1940 census dataset in a single pass.

Computes exact (non-DP) ground-truth marginals used to evaluate synthetic data
against ground truth. Discretization matches the published ``domain.yaml``:

  * NUMERICAL columns -> 64 equal-width bins over ``[min_value, max_value]``
  plus
    one extra out-of-domain bin (size 65) for the IPUMS N/A sentinels.
  * CATEGORICAL columns -> one index per ``possible_value`` plus an OOD index.
  * OPEN_SET columns -> the top-``open_set_top_k`` most frequent codes plus an
    OOD index (keeps otherwise-huge open-set marginals tractable).

Survey weights (see ``column_types.WEIGHTS``) are excluded -- they are design
artifacts, not analysis attributes.

The dataset is loaded and encoded into compact integer arrays, wrapped as an
``mbi.Dataset``, and computed in a single pass via
``mbi.extensions.precompute_marginals`` (using JIT-compiled ``jnp.bincount``
with power-of-2 domain bucketing and overlapped shape compilation). Runs on GPU
in ~15-30 seconds, or on multi-core CPU in a few minutes.
"""

from collections.abc import Sequence
import itertools
from typing import Any

from absl import app
from absl import flags
from absl import logging
from dpsynth import domain as dpdomain
from dpsynth.research.datasets.census import column_types
from etils import epath
import mbi
from mbi.domain import Attribute
from mbi.extensions.precompute_marginals import precompute_marginals as mbi_precompute
import numpy as np
import pandas as pd
import yaml

_PARQUET_DIR = flags.DEFINE_string(
    'parquet_dir', './data/parquet', 'Input parquet directory.'
)
_DOMAIN_PATH = flags.DEFINE_string(
    'domain_path', './data/domain.yaml', 'Input domain YAML.'
)
_OUTPUT_PATH = flags.DEFINE_string(
    'output_path', './data/marginals.npz', 'Output marginals npz.'
)
_OPEN_SET_TOP_K = flags.DEFINE_integer(
    'open_set_top_k', 256, 'Number of top codes kept for open-set columns.'
)
_NUM_BINS = flags.DEFINE_integer('num_bins', 64, 'Bins for numerical columns.')
_MAX_SHARDS = flags.DEFINE_integer(
    'max_shards', None, 'If set, only read this many shards (for smoke tests).'
)


def _read_shard(path: epath.PathLike) -> pd.DataFrame:
  with epath.Path(path).open('rb') as f:
    return pd.read_parquet(f)


def _load_domain(path: epath.PathLike) -> dict[str, Any]:
  """Loads a dpsynth domain YAML from file."""
  with epath.Path(path).open('r') as f:
    data = yaml.safe_load(f)
  out = {}
  for name, d in data.items():
    if not d:
      out[name] = dpdomain.OpenSetCategoricalAttribute()
    elif 'possible_values' in d:
      out[name] = dpdomain.CategoricalAttribute(**d)
    elif 'min_value' in d:
      out[name] = dpdomain.NumericalAttribute(**d)
    else:
      raise ValueError(f'Invalid domain entry for {name}: {d}')
  return out


class _Encoder:
  """Maps a column's raw integer codes to contiguous marginal indices."""

  def __init__(self, name: str, attribute: Any, num_bins: int):
    self.name = name
    self.attribute = attribute
    if isinstance(attribute, dpdomain.NumericalAttribute):
      self.kind = 'numerical'
      self.min_val = float(attribute.min_value)
      self.max_val = float(attribute.max_value)
      # Inner edges; np.digitize -> 0..num_bins-1 for in-range values.
      self._edges = np.linspace(self.min_val, self.max_val, num_bins + 1)[1:-1]
      self._ood = num_bins  # extra out-of-domain bin
      self.size = num_bins + 1
    elif isinstance(attribute, dpdomain.CategoricalAttribute):
      self.kind = 'categorical'
      self.min_val = None
      self.max_val = None
      self._lookup = {
          int(v): i for i, v in enumerate(attribute.possible_values)
      }
      self._ood = len(self._lookup)
      self.size = self._ood + 1
    else:  # OpenSetCategoricalAttribute; codes filled in later via set_codes().
      self.kind = 'open_set'
      self.min_val = None
      self.max_val = None
      self._lookup = {}
      self._ood = 0
      self.size = 1

  def set_codes(self, codes: list[int]) -> None:
    """Sets the kept open-set codes (top-K), assigning contiguous indices."""
    self._lookup = {int(c): i for i, c in enumerate(codes)}
    self._ood = len(self._lookup)
    self.size = self._ood + 1

  def encode(self, values: np.ndarray) -> np.ndarray:
    """Encodes raw integer values into contiguous marginal indices."""
    if (
        self.kind == 'numerical'
        and self.min_val is not None
        and self.max_val is not None
    ):
      idx = np.digitize(values, self._edges)
      ood = (values < self.min_val) | (values > self.max_val)
      idx = np.where(ood, self._ood, idx)
      dtype = np.min_scalar_type(self.size - 1)
      return idx.astype(dtype)
    mapped = pd.Series(values).map(self._lookup)
    dtype = np.min_scalar_type(self.size - 1)
    return mapped.fillna(self._ood).astype(dtype).to_numpy()


def _build_encoders(
    domain: dict[str, Any], shards: Sequence[epath.PathLike]
) -> tuple[list[str], dict[str, _Encoder]]:
  """Builds a per-column _Encoder map keyed by column name."""
  cols = [c for c in domain if c not in column_types.WEIGHTS]
  encoders = {c: _Encoder(c, domain[c], _NUM_BINS.value) for c in cols}
  open_set_cols = [c for c in cols if encoders[c].kind == 'open_set']
  if open_set_cols:
    ref = _read_shard(shards[0])
    for c in open_set_cols:
      vals, cnts = np.unique(ref[c].to_numpy(), return_counts=True)
      order = np.argsort(cnts)[::-1]
      top = vals[order][: _OPEN_SET_TOP_K.value].tolist()
      encoders[c].set_codes(top)
  return cols, encoders


def main(argv: Sequence[str]) -> None:
  del argv

  domain = _load_domain(_DOMAIN_PATH.value)
  shards = sorted(epath.Path(_PARQUET_DIR.value).glob('census-*.parquet'))
  if not shards:
    raise FileNotFoundError(f'No parquet shards under {_PARQUET_DIR.value}')
  if _MAX_SHARDS.value is not None:
    shards = shards[: _MAX_SHARDS.value]
    logging.info('Capped to %d shards for smoke test.', len(shards))

  cols, encoders = _build_encoders(domain, shards)
  logging.info('Configured encoders for %d domain attributes.', len(cols))

  # Stream through shards and accumulate compact encoded column arrays.
  col_chunks: dict[Attribute, list[np.ndarray]] = {c: [] for c in cols}
  for i, shard in enumerate(shards):
    df = _read_shard(shard)
    for c in cols:
      col_chunks[c].append(encoders[c].encode(df[c].to_numpy()))
    if (i + 1) % 20 == 0 or (i + 1) == len(shards):
      logging.info('Loaded and encoded %d/%d shards...', i + 1, len(shards))

  col_data: dict[Attribute, np.ndarray] = {
      c: np.concatenate(col_chunks[c]) for c in cols
  }
  del col_chunks

  num_rows = next(iter(col_data.values())).shape[0]
  logging.info('Dataset assembled: %d rows x %d columns.', num_rows, len(cols))

  # Construct mbi Domain and Dataset.
  mbi_domain = mbi.Domain.fromdict({c: encoders[str(c)].size for c in cols})
  dataset = mbi.Dataset(col_data, mbi_domain)

  # Generate all 2-way cliques.
  pairs = list(itertools.combinations(cols, 2))
  logging.info(
      'Computing %d 2-way marginals via mbi.precompute_marginals...', len(pairs)
  )

  clique_vector = mbi_precompute(dataset, pairs)

  # Save directly as an mbi.CliqueVector JAX pytree.
  with epath.Path(_OUTPUT_PATH.value).open('wb') as f:
    # pyrefly: ignore[bad-argument-type]
    mbi.save(clique_vector, f)

  logging.info(
      'Saved CliqueVector (%d 2-way marginals) to %s',
      len(clique_vector.cliques),
      _OUTPUT_PATH.value,
  )


if __name__ == '__main__':
  app.run(main)
