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

r"""Builds the data-independent ``domain.yaml`` and ``codebook.json``.

The domain is a *public* artifact: it is an input to the DP mechanism, so it
must not depend on the private microdata. Deriving numerical ranges or
categorical value sets from the observed data (e.g. per-column min/max in the
``stats.json`` emitted by ``convert_to_parquet.py``) would make the domain a
function of the private data and break the differential privacy guarantee.

Everything here is therefore derived from public sources only:

  * the static type classification in ``column_types`` (NUMERICAL,
    CATEGORICAL, OPEN_SET), derived from the public IPUMS codebook;
  * the IPUMS SAS value labels (``parse_sas``), a published codebook;
  * the public numerical ranges in ``column_types.NUMERICAL_RANGES``.

Attribute construction:

  * NUMERICAL -> a numerical attribute over the public range, with
    ``clip_to_range=False`` so out-of-range values (including the all-9s IPUMS
    N/A sentinels) fall into an explicit out-of-domain bin.
  * CATEGORICAL -> a categorical attribute whose ``possible_values`` are the
    codes listed for the column in the codebook (used verbatim regardless of
    cardinality). A categorical with no codebook entry -- nothing to enumerate
    -- degrades to an open-set attribute.
  * OPEN_SET -> an open-set attribute (empty YAML mapping; ``from_yaml_file``
    reconstructs an ``OpenSetCategoricalAttribute``).

Outputs:

  * ``domain.yaml`` -- the ``dpsynth.domain`` mapping.
  * ``codebook.json`` -- ``{column: {code: label}}`` so the human-readable IPUMS
    labels are preserved alongside the int-coded parquet.
"""

from collections.abc import Sequence
import json
import tempfile

from absl import app
from absl import flags
from absl import logging
from dpsynth import domain
from dpsynth.research.datasets.census import column_types
from dpsynth.research.datasets.census import parse_sas
from etils import epath
import yaml

_SAS_PATH = flags.DEFINE_string(
    'command_file_path',
    './data/command_file_sas.txt',
    'Input IPUMS SAS command file.',
)
_DOMAIN_PATH = flags.DEFINE_string(
    'domain_path', './data/domain.yaml', 'Output domain YAML.'
)
_CODEBOOK_PATH = flags.DEFINE_string(
    'codebook_path', './data/codebook.json', 'Output codebook JSON.'
)


def _numerical(name: str) -> dict[str, object]:
  """Builds a numerical attribute dict from the public range table."""
  lo, hi = column_types.NUMERICAL_RANGES[name]
  # clip_to_range=False models out-of-range values (including the IPUMS all-9s
  # N/A sentinels) as an explicit out-of-domain value rather than clipping.
  return {
      'min_value': lo,
      'max_value': hi,
      'dtype': 'int',
      'clip_to_range': False,
  }


def build_domain(
    labels: dict[str, dict[int, str]],
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
  """Returns ``(yaml_data, codebook)`` derived only from public inputs."""
  yaml_data: dict[str, object] = {}
  demotions: list[str] = []
  for name in column_types.domain_columns():
    ctype = column_types.column_type(name)

    if ctype is column_types.ColumnType.NUMERICAL:
      yaml_data[name] = _numerical(name)
      continue

    if ctype is column_types.ColumnType.CATEGORICAL:
      codes = sorted(labels[name]) if name in labels else []
      # A categorical needs a public, enumerable value set. If the codebook
      # lists values, they are used verbatim regardless of cardinality (they are
      # public, so there is no privacy reason to hide them). A column with no
      # codebook entry has nothing to enumerate, so it falls back to an open-set
      # attribute (DP partition selection discovers and prunes the tail).
      if codes:
        yaml_data[name] = {
            'possible_values': [int(v) for v in codes],
            'out_of_domain_index': 0,
        }
        continue
      demotions.append(name)

    # OPEN_SET (explicit, or a demoted categorical). domain.from_yaml_file only
    # reconstructs an OpenSetCategoricalAttribute from an empty mapping.
    yaml_data[name] = {}

  if demotions:
    logging.info(
        'Categoricals modeled as open-set (no codebook to enumerate): %s',
        ', '.join(sorted(demotions)),
    )

  codebook = {
      name: {str(code): lbl for code, lbl in labels[name].items()}
      for name in column_types.kept_columns()
      if name in labels
  }
  return yaml_data, codebook


def main(argv: Sequence[str]) -> None:
  del argv
  labels = parse_sas.parse_value_labels(_SAS_PATH.value)
  logging.info('Parsed value labels for %d columns.', len(labels))

  yaml_data, codebook = build_domain(labels)

  # Dump the mapping directly (rather than via domain.to_yaml_file) to preserve
  # the empty-mapping open-set encoding.
  with epath.Path(_DOMAIN_PATH.value).open('w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=True)
  with epath.Path(_CODEBOOK_PATH.value).open('w') as f:
    json.dump(codebook, f, indent=2)

  # Sanity check: the written domain must load back cleanly.
  with tempfile.NamedTemporaryFile('w', suffix='.yaml') as tmp:
    yaml.dump(yaml_data, tmp, default_flow_style=False, sort_keys=True)
    tmp.flush()
    reloaded = domain.from_yaml_file(tmp.name)
  logging.info(
      '[DONE] wrote domain (%d attributes) -> %s and codebook -> %s',
      len(reloaded),
      _DOMAIN_PATH.value,
      _CODEBOOK_PATH.value,
  )


if __name__ == '__main__':
  app.run(main)
