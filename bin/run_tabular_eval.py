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

"""Run tabular evaluation."""

from collections.abc import Sequence
import time
from typing import Any

from absl import app
from absl import flags
import apache_beam as beam
from dpsynth.dataset_descriptors import dataset_descriptor
from dpsynth.dataset_descriptors import proto_descriptors
from dpsynth.eval import tabular_eval
from dpsynth.eval import types
from dpsynth.pipeline_transformations import diagnostic_info
from google.protobuf import text_format
import pandas as pd
import pipeline_dp

pipeline_runner = None
open_file = open

_ORIGINAL_DATA_PATH = flags.DEFINE_string(
    "original_data_path",
    None,
    "Path to original data file(s). Glob patterns can be used.",
    required=True,
)
_SYNTHETIC_DATA_PATH = flags.DEFINE_string(
    "synthetic_data_path",
    None,
    "Path to synthetic data file(s). Glob patterns can be used.",
    required=True,
)
_EVAL_REPORT_PATH = flags.DEFINE_string(
    "eval_report_path",
    None,
    "Path to save evaluation report file.",
    required=True,
)
_DATA_FORMAT_STR = flags.DEFINE_enum(
    "data_format",
    "csv",
    ["csv"],
    "Format of the dataset: csv.",
    required=False,
)

_ATTRIBUTES: flags.FlagHolder[list[str] | None] = flags.DEFINE_list(
    "attributes",
    None,
    "Comma separated attributes which should be used for generation. If None "
    "all attributes will be used.",
    required=False,
)
_USE_BEAM = flags.DEFINE_boolean(
    "use_beam",
    False,
    "Whether to use Beam for data generation. If False local data "
    "generation is used.",
    required=False,
)


def dataframe_to_list_of_tuples(df: pd.DataFrame) -> list[types.Record]:
  """Converts a pandas DataFrame to a list of tuples (records)."""
  return list(df.itertuples(index=False, name=None))


def read_csv(path: str) -> pd.DataFrame:
  """Reads a CSV file into a pandas DataFrame."""
  with open_file(path, "rt") as f:  # 'rt' for read text
    df = pd.read_csv(f)
    return df


def _read_csv_data():
  """Reads CSV data into a list of tuples."""
  synthetic_data_df = read_csv(_SYNTHETIC_DATA_PATH.value)
  original_data_df = read_csv(_ORIGINAL_DATA_PATH.value)
  # drop columns that are not in synthetic data
  original_data_df = original_data_df[synthetic_data_df.columns]

  synthetic_data = dataframe_to_list_of_tuples(synthetic_data_df)
  original_data = dataframe_to_list_of_tuples(original_data_df)

  attributes = _ATTRIBUTES.value
  if attributes is None:
    attributes = list(original_data_df.columns)
  attribute_types = []
  for col in attributes:
    dtype = original_data_df[col].dtype
    if pd.api.types.is_integer_dtype(dtype):
      attribute_types.append(types.DataType.INT_CATEGORICAL)
    elif pd.api.types.is_bool_dtype(dtype):
      attribute_types.append(types.DataType.BOOLEAN)
    else:
      attribute_types.append(types.DataType.STRING)

  config = diagnostic_info.TabularEvalConfig(
      attributes=attributes,
      attribute_types=[t.to_proto() for t in attribute_types],
  )
  return original_data, synthetic_data, config


def _proto_to_tuple(
    protos: beam.PCollection[Any],
    field_names: list[str],
    prefix_stage_name: str,
) -> tuple[Any, ...]:
  """Converts a proto to a tuple."""

  def to_tuple(proto):
    return tuple(getattr(proto, field_name) for field_name in field_names)

  return protos | f"{prefix_stage_name}_ToTuple" >> beam.Map(to_tuple)


def local_main():
  """Main function for local (in-process) execution."""
  assert (
      _DATA_FORMAT_STR.value == "CSV"
  ), "Unsupported data format for local execution."
  original_data, synthetic_data, config = _read_csv_data()

  eval_report_collection = tabular_eval.evaluate(
      original_data, synthetic_data, config
  )
  eval_report = list(eval_report_collection)[0]

  with open_file(_EVAL_REPORT_PATH.value, "wt") as f:
    f.write(text_format.MessageToString(eval_report))


def beam_main():
  """Main function for Beam (distributed) execution."""
  # Run the pipeline to collect results.
  with beam.Pipeline(runner=pipeline_runner) as p:
    backend = pipeline_dp.BeamBackend()
    original_data, synthetic_data, config = _read_csv_data()
    original_data = p | "ToCollectionOrig" >> beam.Create(original_data)
    synthetic_data = p | "ToCollectionSyn" >> beam.Create(synthetic_data)

    eval_report_collection = tabular_eval.evaluate(
        original_data, synthetic_data, config, backend
    )

    _ = (
        eval_report_collection
        | "ToTextProto" >> beam.Map(str)
        | "WriteReport" >> beam.io.WriteToText(_EVAL_REPORT_PATH.value)
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  if _USE_BEAM.value:
    beam_main()
  else:
    local_main()


if __name__ == "__main__":
  start = time.time()
  app.run(main)
  print(f"Elapsed time {time.time() - start}")
