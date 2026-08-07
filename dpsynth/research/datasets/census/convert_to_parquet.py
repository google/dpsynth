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

r"""Converts the raw fixed-width 1940 census extract to sharded parquet.

Single-pass, streaming design (no upfront line count): the raw ``.dat`` is read
in fixed-size row chunks and each chunk is written as one parquet shard. Every
kept attribute column (see ``column_types``) is stored as its raw integer IPUMS
code -- lossless, compact, and independent of how the column is later modeled.
A running ``person_id`` is added as the row identifier.

Parsing is vectorized: an IPUMS rectangular extract is a grid of fixed-length
records, so each chunk of ``stride``-byte records is read as raw bytes, viewed
as
an ``(n, stride)`` uint8 array, and each column's ASCII digits are converted to
``int64`` with a single matrix multiply.
"""

from collections.abc import Sequence
import io
import json
import sys

from absl import app
from absl import flags
from absl import logging
from dpsynth.research.datasets.census import column_types
from etils import epath
import numpy as np
import pandas as pd

_INPUT_PATH = flags.DEFINE_string(
    'input_path', './data/census_full_1940.dat', 'Raw fixed-width extract.'
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', './data/parquet', 'Output parquet directory.'
)
_STATS_DIR = flags.DEFINE_string(
    'stats_dir',
    './data/stats',
    'Output directory for per-worker partial stats JSON.',
)
_CHUNK_SIZE = flags.DEFINE_integer(
    'chunk_size', 500_000, 'Rows per parquet shard.'
)
_MAX_ROWS = flags.DEFINE_integer(
    'max_rows', None, 'If set, stop after this many rows (for smoke tests).'
)
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 1, 'Total number of parallel byte-range workers.'
)
_WORKER_ID = flags.DEFINE_integer(
    'worker_id', 0, 'Index of this worker in [0, num_workers).'
)
_DISTINCT_CAP = 4096


class _ColumnStats:
  """Running min/max and capped distinct-value set for one column."""

  __slots__ = ('min', 'max', 'distinct', 'capped')

  def __init__(self):
    self.min = None
    self.max = None
    self.distinct = set()
    self.capped = False

  def update(self, values: np.ndarray) -> None:
    if values.size == 0:
      return
    cmin = int(values.min())
    cmax = int(values.max())
    self.min = cmin if self.min is None else min(self.min, cmin)
    self.max = cmax if self.max is None else max(self.max, cmax)
    if not self.capped:
      self.distinct.update(int(v) for v in np.unique(values))
      if len(self.distinct) > _DISTINCT_CAP:
        self.capped = True
        self.distinct = set()

  def to_dict(self) -> dict[str, object]:
    return {
        'min': self.min,
        'max': self.max,
        'nunique': None if self.capped else len(self.distinct),
        'capped': self.capped,
        'values': None if self.capped else sorted(self.distinct),
    }


def _parse_int_column(field: np.ndarray) -> np.ndarray:
  """Converts an ``(n, w)`` uint8 array of ASCII digits to ``int64`` codes."""
  w = field.shape[1]
  f = field.astype(np.int64)
  is_digit = (f >= 48) & (f <= 57)  # ord('0')..ord('9')
  digits = np.where(is_digit, f - 48, 0)
  weights = 10 ** np.arange(w - 1, -1, -1, dtype=np.int64)
  values = digits @ weights
  values[~is_digit.any(axis=1)] = -1
  return values


def main(argv: Sequence[str]) -> None:
  del argv
  logging.get_absl_handler().python_handler.stream = sys.stderr

  kept = column_types.kept_columns()
  name_to_spec = dict(zip(column_types.COLUMN_NAMES, column_types.COLUMN_SPECS))
  kept_specs = [(name, *name_to_spec[name]) for name in kept]
  stats = {name: _ColumnStats() for name in kept}

  numerical = set(column_types.NUMERICAL)
  sentinels: dict[str, set[int]] = {}
  for name, (start, end) in name_to_spec.items():
    if name in numerical:
      all_nines = 10 ** (end - start) - 1
      sentinels[name] = {all_nines, all_nines - 1}

  epath.Path(_OUTPUT_DIR.value).mkdir(parents=True, exist_ok=True)
  epath.Path(_STATS_DIR.value).mkdir(parents=True, exist_ok=True)
  file_size = epath.Path(_INPUT_PATH.value).stat().length

  num_workers = _NUM_WORKERS.value
  worker_id = _WORKER_ID.value
  if not 0 <= worker_id < num_workers:
    raise ValueError(f'worker_id {worker_id} not in [0, {num_workers}).')

  with epath.Path(_INPUT_PATH.value).open('rb') as handle:
    header = handle.read(1 << 20)
    newline = header.find(b'\n')
    if newline < 0:
      raise ValueError(
          'No newline found in first 1 MiB; not a line-based file.'
      )
    stride = newline + 1
    if file_size % stride != 0:
      logging.warning(
          '[STATUS] file size %d not a multiple of stride %d; trailing '
          'partial record ignored.',
          file_size,
          stride,
      )
    total_records = file_size // stride

    per = total_records // num_workers
    extra = total_records % num_workers
    start_rec = worker_id * per + min(worker_id, extra)
    count = per + (1 if worker_id < extra else 0)
    remaining = (
        count if _MAX_ROWS.value is None else min(count, _MAX_ROWS.value)
    )
    logging.info(
        '[STATUS] worker %d/%d: stride=%d total_records=%d range=[%d, %d) '
        '(%d rows)',
        worker_id,
        num_workers,
        stride,
        total_records,
        start_rec,
        start_rec + count,
        remaining,
    )

    handle.seek(start_rec * stride)

    person_id = start_rec
    shard = 0
    leftover = b''
    eof = False
    while remaining > 0:
      needed = min(_CHUNK_SIZE.value, remaining) * stride
      parts = [leftover]
      have = len(leftover)
      while have < needed and not eof:
        more = handle.read(needed - have)
        if more:
          parts.append(more)
          have += len(more)
        else:
          eof = True
      block = b''.join(parts) if len(parts) > 1 else parts[0]
      n = min(len(block) // stride, remaining)
      if n == 0:
        break
      usable = n * stride
      arr = np.frombuffer(block, dtype=np.uint8, count=usable).reshape(
          n, stride
      )
      leftover = block[usable:]

      data = {'person_id': np.arange(person_id, person_id + n, dtype='int64')}
      for name, start, end in kept_specs:
        values = _parse_int_column(arr[:, start:end])
        data[name] = values
        if name in sentinels:
          excl = np.array(sorted(sentinels[name] | {-1}), dtype=np.int64)
          stats[name].update(values[~np.isin(values, excl)])
        else:
          stats[name].update(values)
      out = pd.DataFrame(data)

      buf = io.BytesIO()
      out.to_parquet(buf, index=False)
      shard_path = (
          f'{_OUTPUT_DIR.value}/census-w{worker_id:03d}-{shard:05d}.parquet'
      )
      with epath.Path(shard_path).open('wb') as f:
        f.write(buf.getvalue())

      person_id += n
      shard += 1
      remaining -= n
      logging.info(
          '[STATUS] worker %d wrote shard %d (%d rows done)',
          worker_id,
          shard,
          person_id - start_rec,
      )

  stats_out = {name: s.to_dict() for name, s in stats.items()}
  stats_out['__meta__'] = {
      'worker_id': worker_id,
      'num_workers': num_workers,
      'start_record': start_rec,
      'rows': person_id - start_rec,
      'num_shards': shard,
  }
  stats_path = f'{_STATS_DIR.value}/stats-w{worker_id:03d}.json'
  with epath.Path(stats_path).open('w') as f:
    json.dump(stats_out, f, indent=2)
  logging.info(
      '[DONE] worker %d: %d rows, %d shards, stats -> %s',
      worker_id,
      person_id - start_rec,
      shard,
      stats_path,
  )


if __name__ == '__main__':
  app.run(main)
