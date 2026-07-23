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

"""Smoke tests for public binary entrypoints."""

import pathlib
import subprocess
import sys
import tempfile

from absl.testing import absltest
from absl.testing import parameterized


_BINARIES = (
    'comparison.py',
    'main.py',
    'run_data_generation.py',
    'run_data_generation_from_model.py',
    'run_tabular_eval.py',
)


class BinEntrypointsTest(parameterized.TestCase):

  @parameterized.parameters(*_BINARIES)
  def test_help_succeeds(self, binary_name: str):
    repo_root = pathlib.Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / 'dpsynth' / 'bin' / binary_name),
            '--help',
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    self.assertNotIn('Traceback', output)
    self.assertIn('flags:', output)

  def test_run_tabular_eval_local_csv(self):
    repo_root = pathlib.Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as tmpdir:
      tmp_path = pathlib.Path(tmpdir)
      original_data = tmp_path / 'original.csv'
      synthetic_data = tmp_path / 'synthetic.csv'
      eval_report = tmp_path / 'eval_report.pb'
      original_data.write_text('cat\nA\nB\n', encoding='utf-8')
      synthetic_data.write_text('cat\nA\nC\n', encoding='utf-8')

      result = subprocess.run(
          [
              sys.executable,
              str(repo_root / 'dpsynth' / 'bin' / 'run_tabular_eval.py'),
              f'--original_data_path={original_data}',
              f'--synthetic_data_path={synthetic_data}',
              f'--eval_report_path={eval_report}',
              '--data_format=csv',
              '--use_beam=false',
          ],
          cwd=repo_root,
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(
          result.returncode,
          0,
          msg=f'run_tabular_eval.py failed:\n{result.stderr}',
      )
      self.assertTrue(eval_report.exists())
      self.assertNotEmpty(eval_report.read_text(encoding='utf-8'))


if __name__ == '__main__':
  absltest.main()
