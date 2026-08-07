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

r"""Parses the IPUMS SAS command file into per-column value labels.

The raw extract ships with a SAS ``proc format`` command file that defines, for
each labeled variable, a ``value <NAME>_f`` block mapping integer codes to
human-readable labels, e.g.::

    value SEX_f
      1 = "Male"
      2 = "Female"
    ;

This module extracts those blocks into ``{column: {code: label}}``. It is the
shared source of value labels for both ``generate_proto.py`` (enum definitions)
and ``build_domain.py`` (the exported ``codebook.json``).

Note the SAS file is a generic multi-year IPUMS format file, so its enum blocks
are supersets of what appears in the 1940 100% extract; codes not present in the
data are simply unused.
"""

import re

from etils import epath

# A ``value <NAME>_f`` block followed by one or more ``<code> = "<label>"``
# lines terminated by a semicolon.
_BLOCK_RE = re.compile(
    r'value\s+([A-Za-z0-9_]+)_f\s*\n' r'((?:\s+[0-9\-]+\s*=\s*".*?"\s*\n)+);'
)
_LINE_RE = re.compile(r'\s*([0-9\-]+)\s*=\s*"(.*?)"\s*')


def parse_value_labels(
    command_file_path: epath.PathLike,
) -> dict[str, dict[int, str]]:
  """Returns ``{COLUMN: {code: label}}`` parsed from the SAS command file."""
  sas_code = epath.Path(command_file_path).read_text()

  labels: dict[str, dict[int, str]] = {}
  for match in _BLOCK_RE.finditer(sas_code):
    column = match.group(1).upper()
    code_to_label: dict[int, str] = {}
    for line in match.group(2).strip().split('\n'):
      line_match = _LINE_RE.fullmatch(line)
      if line_match:
        code_to_label[int(line_match.group(1))] = line_match.group(2)
      else:
        print(f'Warning: could not parse value line: {line!r}')
    labels[column] = code_to_label
  return labels


def enum_key(enum_name: str, label: str) -> str:
  """Sanitizes a value label into a proto3 enum key (prefixed by enum name)."""
  # Proto enum keys must be unique across the whole file, so we prefix with the
  # enum name. Kept byte-for-byte compatible with the original generator's
  # sanitization so regenerated protos are stable.
  key = (
      label.replace(' ', '_')
      .replace('/', '_')
      .replace('(', '')
      .replace(')', '')
      .replace('.', '')
      .replace('-', '_')
      .replace(',', '')
      .replace('%', 'p')
      .replace(':', '')
      .replace('?', '')
      .replace('$', '')
      .replace("'", '')
      .replace('+', '')
      .replace('___', '_')
      .replace('__', '_')
      .upper()
  )
  return f'{enum_name}_{key}'
