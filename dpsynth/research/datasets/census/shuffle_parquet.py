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

r"""Randomly shuffles the 1940 census parquet shards across shards.

Performs a distributed uniform shuffle so that every output shard is an i.i.d.
uniform sample of the full dataset -- a user can load one shard for a fast,
representative subsample.

The shuffle is a two-phase scatter/gather:

Scatter: assigns each row independently to a uniformly random output bucket.
Gather: concatenates parts, permutes the rows, and writes the final shard.
"""

from collections.abc import Sequence
from typing import cast

from absl import app
from absl import flags
from absl import logging
from etils import epath
import numpy as np
import pandas as pd

_PHASE = flags.DEFINE_enum(
    'phase', None, ['scatter', 'gather'], 'Shuffle phase to run.'
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir', './data/parquet', 'Input (unshuffled) parquet directory.'
)
_TMP_DIR = flags.DEFINE_string(
    'tmp_dir', './data/shuffle_tmp', 'Scratch directory for bucketed parts.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', './data/shuffled_parquet', 'Output shuffled directory.'
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 1, 'Number of workers, and number of output shards/buckets.'
)
_WORKER_ID = flags.DEFINE_integer(
    'worker_id', 0, 'Index of this worker in [0, num_workers).'
)
_SEED = flags.DEFINE_integer('seed', 20240730, 'Base RNG seed.')


def _read_parquet(path: epath.PathLike) -> pd.DataFrame:
  with epath.Path(path).open('rb') as f:
    return pd.read_parquet(f)


def _write_parquet(df: pd.DataFrame, path: epath.PathLike) -> None:
  with epath.Path(path).open('wb') as f:
    df.to_parquet(f, index=False)


def _scatter(worker_id: int, num_workers: int) -> None:
  """Assigns each input row to a uniform random bucket and writes the parts."""
  all_shards = sorted(epath.Path(_INPUT_DIR.value).glob('census-*.parquet'))
  if not all_shards:
    raise FileNotFoundError(f'No parquet shards under {_INPUT_DIR.value}')
  shards = all_shards[worker_id::num_workers]
  rng = np.random.default_rng(_SEED.value + worker_id)

  buckets = [[] for _ in range(num_workers)]
  total = 0
  for i, shard in enumerate(shards):
    df = _read_parquet(shard)
    assignment = rng.integers(0, num_workers, size=len(df))
    for b in range(num_workers):
      part = df[assignment == b]
      if not part.empty:
        buckets[b].append(part)
    total += len(df)
    if (i + 1) % 10 == 0 or (i + 1) == len(shards):
      logging.info(
          '[STATUS] worker %d: partitioned %d/%d shards (%d rows)',
          worker_id,
          i + 1,
          len(shards),
          total,
      )

  out_dir = epath.Path(_TMP_DIR.value) / f'from{worker_id:03d}'
  out_dir.mkdir(parents=True, exist_ok=True)

  for b in range(num_workers):
    if buckets[b]:
      df_b = cast(pd.DataFrame, pd.concat(buckets[b], ignore_index=True))
      _write_parquet(df_b, out_dir / f'bucket{b:03d}.parquet')
  logging.info(
      '[DONE] scatter worker %d: %d rows split into %d bucket files',
      worker_id,
      total,
      num_workers,
  )


def _gather(bucket_id: int) -> None:
  """Concatenates parts for bucket_id, permutes, and writes the final shard."""
  paths = sorted(
      epath.Path(_TMP_DIR.value).glob(f'from*/bucket{bucket_id:03d}.parquet')
  )
  if not paths:
    raise FileNotFoundError(
        f'No bucket files found for bucket {bucket_id:03d} under'
        f' {_TMP_DIR.value}'
    )

  dfs = [_read_parquet(p) for p in paths]
  df = pd.concat(dfs, ignore_index=True)

  rng = np.random.default_rng(_SEED.value + 10000 + bucket_id)
  perm = rng.permutation(len(df))
  df = cast(pd.DataFrame, df.iloc[perm]).reset_index(drop=True)

  epath.Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)

  out_path = epath.Path(_OUTPUT_DIR.value) / f'census-{bucket_id:05d}.parquet'
  _write_parquet(df, out_path)
  logging.info(
      '[DONE] gather bucket %d: %d parts -> %d rows -> %s',
      bucket_id,
      len(paths),
      len(df),
      out_path,
  )


def main(argv: Sequence[str]) -> None:
  del argv
  num_workers = _NUM_WORKERS.value
  worker_id = _WORKER_ID.value
  if not 0 <= worker_id < num_workers:
    raise ValueError(f'worker_id {worker_id} not in [0, {num_workers}).')

  if _PHASE.value == 'scatter':
    _scatter(worker_id, num_workers)
  elif _PHASE.value == 'gather':
    _gather(worker_id)
  else:
    raise ValueError(f'Unknown phase: {_PHASE.value}')


if __name__ == '__main__':
  flags.mark_flag_as_required('phase')
  app.run(main)
