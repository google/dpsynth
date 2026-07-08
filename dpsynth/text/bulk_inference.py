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

"""Bulk LLM inference for synthetic text generation."""

from collections.abc import Callable, Mapping, Sequence
import concurrent.futures
import dataclasses
import enum
import functools
import json
import random
import re
import time
from typing import Protocol, TypeVar

from absl import logging
from dpsynth import domain
from google import genai
from google.genai import types
import pandas as pd
import pydantic

# Schema accepted by annotate(): either a pydantic model or a dpsynth Domain.
AnnotationSchema = type[pydantic.BaseModel] | Mapping[str, domain.AttributeType]


class ModelName(enum.StrEnum):
  """Model names supported by the google.genai API."""

  GEMINI_2_5_FLASH_LITE = 'gemini-2.5-flash-lite'
  GEMINI_3_FLASH = 'gemini-3-flash-preview'
  GEMINI_3_5_FLASH = 'gemini-3.5-flash'
  GEMMA_4_27B = 'gemma-4-26b-a4b-it'
  GEMMA_4_31B = 'gemma-4-31b-it'


class TextGenerationBackend(Protocol):
  """Interface for bulk LLM inference operations.

  Implementations provide the two LLM inference capabilities needed by the
  synthetic text generation pipeline:

  1. **Annotation**: extracting structured categorical features from text,
     typically via constrained decoding with a pydantic response schema.
  2. **Generation**: producing free-form synthetic text conditioned on features.

  **Index-alignment guarantee**: both methods return output positionally
  aligned with the input.  ``len(output) == len(input)`` always holds.
  Annotation includes a ``_fields_decoded`` column (0 to ``len(schema)``).
  Generation represents failures as empty strings.
  """

  def annotate(
      self,
      texts: Sequence[str],
      schema: AnnotationSchema,
      system_prompt: str,
  ) -> pd.DataFrame:
    """Extract structured features from texts using an LLM.

    Args:
      texts: Input texts to annotate.
      schema: Pydantic ``BaseModel`` subclass or dpsynth ``Domain`` dict.
      system_prompt: System-level instructions for the LLM.

    Returns:
      DataFrame with ``len(texts)`` rows, one column per schema field,
      plus ``_fields_decoded`` (int, 0 to ``len(schema)``).
    """
    ...

  def generate(self, prompts: Sequence[str]) -> list[str]:
    """Generate free-form text from prompts.

    Args:
      prompts: Fully constructed prompts, each describing the desired output
        including the target features and any formatting requirements.

    Returns:
      A list of exactly ``len(prompts)`` strings.  Failed generations are
      represented as empty strings.
    """
    ...


@dataclasses.dataclass(frozen=True)
class GenAIBackend:
  """TextGenerationBackend using the google.genai API.

  Uses ``client.models.generate_content()`` for both annotation (with
  structured output via ``response_schema``) and free-form generation.

  Attributes:
    model: Model name string (e.g., ``'gemini-2.5-flash-lite'``).  Accepts any
      ``ModelName`` enum value or arbitrary string for unlisted models.
    api_key: API key for authentication.
    poll_interval_seconds: How often to poll for batch job completion.
    chunk_size: Number of texts per batch job.
    max_concurrent_jobs: Maximum number of active parallel batch jobs.
  """

  model: str = ModelName.GEMINI_2_5_FLASH_LITE
  api_key: str | None = None
  poll_interval_seconds: int = 60
  chunk_size: int = 100
  max_concurrent_jobs: int = 8

  @functools.cached_property
  def client(self) -> genai.Client:
    """Creates and caches a genai.Client."""
    kwargs = {'http_options': types.HttpOptions(api_version='v1alpha')}

    if self.api_key:
      kwargs['api_key'] = self.api_key  # pyrefly: ignore[bad-assignment]
    return genai.Client(**kwargs)

  def _parse_job_responses(
      self,
      batch_job: types.BatchJob,
      schema: AnnotationSchema,
  ) -> list[tuple[dict[str, object], int]]:
    """Parses responses; returns ``(row, fields_decoded)`` per response."""
    if batch_job.state != types.JobState.JOB_STATE_SUCCEEDED:
      error_msg = (
          f'Batch job {batch_job.name} ended with state={batch_job.state}.'
      )
      if batch_job.error:
        error_msg += f' Error: {batch_job.error}'
      raise RuntimeError(error_msg)

    inlined_responses = (
        batch_job.dest.inlined_responses if batch_job.dest else []
    ) or []

    field_names = _schema_field_names(schema)
    results: list[tuple[dict[str, object], int]] = []
    for inlined_resp in inlined_responses:
      try:
        if inlined_resp.error:
          raise ValueError('Item-level error from batch API.')
        response_text = inlined_resp.response and inlined_resp.response.text
        if not response_text:
          raise ValueError('Empty response text.')
        row, count = _parse_response(schema, response_text)
        results.append((row, count))
      except Exception:  # pylint: disable=broad-except
        if isinstance(schema, Mapping):
          results.append((_default_row(schema), 0))
        else:
          results.append(({f: None for f in field_names}, 0))
    return results

  def _submit_and_poll_chunk(
      self,
      chunk_texts: Sequence[str],
      config: types.GenerateContentConfig | None = None,
  ) -> types.BatchJob:
    """Submit a batch job for one chunk and poll until done."""

    inlined_requests = [
        types.InlinedRequest(contents=text, config=config)
        for text in chunk_texts
    ]

    job = _call_with_retry(
        lambda: self.client.batches.create(
            model=self.model, src=inlined_requests
        ),
        'create',
    )
    logging.info('Batch annotate: job %s created.', job.name)

    while not job.done:
      time.sleep(self.poll_interval_seconds)
      job = _call_with_retry(
          lambda: self.client.batches.get(name=job.name), 'get'  # pyrefly: ignore[bad-argument-type]
      )

    logging.info(
        'Batch annotate: job %s completed with state=%s',
        job.name,
        job.state,
    )
    return job

  def annotate(
      self,
      texts: Sequence[str],
      schema: AnnotationSchema,
      system_prompt: str,
  ) -> pd.DataFrame:
    """Extract structured features via the GenAI Batch API.

    Args:
      texts: Input texts to annotate.
      schema: Pydantic model or dpsynth Domain dict for constrained decoding.
      system_prompt: System-level instructions for the LLM.

    Returns:
      DataFrame with ``len(texts)`` rows, one column per schema field,
      plus ``_fields_decoded`` (int, 0 to ``len(schema)``).

    Raises:
      RuntimeError: If the batch job fails or is cancelled.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type='application/json',
        response_schema=_to_genai_schema(schema),
    )

    chunks = [
        texts[i : i + self.chunk_size]
        for i in range(0, len(texts), self.chunk_size)
    ]

    logging.info(
        'Batch annotate: processing %d chunks with concurrency limit %d...',
        len(chunks),
        self.max_concurrent_jobs,
    )

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self.max_concurrent_jobs
    ) as pool:
      completed_jobs = list(
          pool.map(
              functools.partial(self._submit_and_poll_chunk, config=config),
              chunks,
          )
      )

    logging.info('Batch annotate: all jobs completed. Parsing responses...')

    all_rows: list[dict[str, object]] = []
    fields_decoded: list[int] = []
    for batch_job, chunk_texts in zip(completed_jobs, chunks, strict=True):
      chunk_results = self._parse_job_responses(batch_job, schema)

      if len(chunk_results) != len(chunk_texts):
        raise ValueError(
            f'Batch annotate: job {batch_job.name} got {len(chunk_results)}'
            f' results for {len(chunk_texts)} inputs.'
        )

      for row, count in chunk_results:
        all_rows.append(row)
        fields_decoded.append(count)

    df = pd.DataFrame(all_rows)
    df['_fields_decoded'] = fields_decoded
    return df

  def generate(self, prompts: Sequence[str]) -> list[str]:
    """Generate free-form text via google.genai.

    Args:
      prompts: Fully constructed prompts.

    Returns:
      List of exactly ``len(prompts)`` strings.  Empty string on failure.
    """
    client = self.client
    results: list[str] = []
    for i, prompt in enumerate(prompts):
      try:
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        results.append(response.text or '')
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Generation failed for prompt %d. Error: %s', i, e, exc_info=True
        )
        results.append('')
    return results


def _strip_markdown_fences(text):
  """Strips markdown code fences from LLM output if present."""
  regex = r'^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$'
  m = re.compile(regex, re.DOTALL).match(text)
  return m.group(1).strip() if m else text.strip()


def _schema_field_names(schema: AnnotationSchema) -> list[str]:
  """Returns field names for either schema type."""
  if isinstance(schema, Mapping):
    return list(schema.keys())
  return list(schema.model_fields.keys())


def _to_genai_schema(schema: AnnotationSchema):
  """Converts an AnnotationSchema to the format GenAI expects."""
  if isinstance(schema, Mapping):
    return domain_to_json_schema(schema)
  return schema


def _default_value(attr: domain.AttributeType) -> object:
  """Returns the default sentinel value for a domain attribute type."""
  if isinstance(attr, domain.CategoricalAttribute):
    return attr.possible_values[attr.out_of_domain_index]
  if isinstance(attr, domain.OpenSetCategoricalAttribute):
    return attr.default_value
  if isinstance(attr, domain.NumericalAttribute):
    return attr.min_value
  if isinstance(attr, domain.FreeFormTextAttribute):
    return ''
  raise ValueError(f'Unsupported attribute type: {type(attr)}')


def _default_row(
    schema: Mapping[str, domain.AttributeType],
) -> dict[str, object]:
  """Returns a row with all fields set to their default sentinel values."""
  return {name: _default_value(attr) for name, attr in schema.items()}


def _parse_response(
    schema: AnnotationSchema,
    response_text: str,
) -> tuple[dict[str, object], int]:
  """Parses a JSON response, returning ``(row, fields_decoded)``."""
  cleaned = _strip_markdown_fences(response_text)
  if isinstance(schema, Mapping):
    try:
      parsed = json.loads(cleaned)
    except json.JSONDecodeError:
      return _default_row(schema), 0
    if not isinstance(parsed, dict):
      parsed = {}
    row = {}
    fields_decoded = 0
    for name, attr in schema.items():
      if name in parsed:
        row[name] = parsed[name]
        fields_decoded += 1
      else:
        row[name] = _default_value(attr)
    return row, fields_decoded
  result = schema.model_validate_json(cleaned).model_dump()
  return result, len(result)


_PYTHON_TO_JSON_TYPE: dict[type[object], str] = {
    str: 'string',
    int: 'integer',
    float: 'number',
    bool: 'boolean',
}


def _categorical_json_type(
    possible_values: Sequence[domain.CategoricalValue],
) -> str:
  """Infers the JSON Schema type from homogeneous possible_values."""
  if not possible_values:
    return 'string'
  value_type = type(possible_values[0])
  return _PYTHON_TO_JSON_TYPE.get(value_type, 'string')


def domain_to_json_schema(
    domain_spec: Mapping[str, domain.AttributeType],
) -> dict[str, object]:
  """Converts a dpsynth Domain to a JSON schema dict for GenAI."""
  properties = {}
  for name, attr in domain_spec.items():
    if isinstance(attr, domain.CategoricalAttribute):
      json_type = _categorical_json_type(attr.possible_values)
      prop = {'type': json_type, 'enum': attr.possible_values}
    elif isinstance(attr, domain.OpenSetCategoricalAttribute):
      prop = {'type': 'string'}
    elif isinstance(attr, domain.NumericalAttribute):
      prop = {
          'type': 'number',
          'minimum': attr.min_value,
          'maximum': attr.max_value,
      }
    elif isinstance(attr, domain.FreeFormTextAttribute):
      prop = {'type': 'string'}
    else:
      raise ValueError(f'Unsupported attribute type for {name!r}: {type(attr)}')
    if attr.description:
      prop['description'] = attr.description
    properties[name] = prop

  return {
      'type': 'object',
      'properties': properties,
      'required': list(properties.keys()),
  }


T = TypeVar('T')


def _call_with_retry(
    func: Callable[[], T],
    op_name: str,
    max_retries: int = 10,
    initial_delay: float = 5.0,
) -> T:  # pyrefly: ignore[bad-return]
  """Calls `func` with exponential backoff on exceptions."""
  delay = initial_delay
  for attempt in range(1, max_retries + 1):
    try:
      return func()
    except Exception as e:  # pylint: disable=broad-except
      if attempt == max_retries:
        logging.error(
            'Batch %s failed after %d attempts.', op_name, max_retries
        )
        raise

      sleep_time = delay + random.uniform(0, 5)
      logging.warning(
          'Batch %s failed (attempt %d/%d): %s. Retrying in %.1f sec...',
          op_name,
          attempt,
          max_retries,
          e,
          sleep_time,
      )
      time.sleep(sleep_time)
      delay *= 2
