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

from collections.abc import Callable, Sequence
import concurrent.futures
import dataclasses
import enum
import functools
import random
import re
import time
from typing import Protocol, TypeVar

from absl import logging
from google import genai
from google.genai import types
import pandas as pd
import pydantic


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

  **Index-alignment guarantee**: both methods return output that is
  positionally aligned with the input sequence — ``len(output) == len(input)``
  always holds.  Failed items are represented as null rows (annotation) or
  empty strings (generation).
  """

  def annotate(
      self,
      texts: Sequence[str],
      schema: type[pydantic.BaseModel],
      system_prompt: str,
  ) -> pd.DataFrame:
    """Extract structured features from texts using an LLM.

    Args:
      texts: Input texts to annotate.
      schema: Pydantic model class defining the features to extract.  Fields may
        use ``Literal`` type annotations for constrained decoding or plain types
        such as ``str`` for open-ended annotation.  Field names and descriptions
        guide the LLM.
      system_prompt: System-level instructions for the LLM describing how to
        annotate the texts.

    Returns:
      A DataFrame with exactly ``len(texts)`` rows and one column per field
      in ``schema``.  Rows where annotation failed contain ``None`` values.
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
      schema: type[pydantic.BaseModel],
  ) -> list[dict[str, str | None]]:
    """Parses responses from a completed BatchJob."""
    if batch_job.state != types.JobState.JOB_STATE_SUCCEEDED:
      error_msg = (
          f'Batch job {batch_job.name} ended with state={batch_job.state}.'
      )
      if batch_job.error:
        error_msg += f' Error: {batch_job.error}'
      raise RuntimeError(error_msg)

    null_row = {f: None for f in schema.model_fields.keys()}

    inlined_responses = (
        batch_job.dest.inlined_responses if batch_job.dest else []
    ) or []

    chunk_rows = []
    for i, inlined_resp in enumerate(inlined_responses):
      row = dict(null_row)  # Default to null row
      try:
        if inlined_resp.error:
          logging.warning(
              'Batch result %d in job %s had error: %s',
              i,
              batch_job.name,
              inlined_resp.error,
          )
        else:
          response_text = inlined_resp.response and inlined_resp.response.text
          if response_text:
            row = schema.model_validate_json(
                _strip_markdown_fences(response_text)
            ).model_dump()
          else:
            logging.warning(
                'Empty batch response in job %s for text %d.',
                batch_job.name,
                i,
            )
      except Exception as e:  # pylint: disable=broad-except
        logging.warning(
            'Failed to parse batch result %d in job %s: %s',
            i,
            batch_job.name,
            e,
        )
      chunk_rows.append(row)
    return chunk_rows

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
      schema: type[pydantic.BaseModel],
      system_prompt: str,
  ) -> pd.DataFrame:
    """Extract structured features via the GenAI Batch API.

    Submits texts as inlined requests to the batch prediction endpoint,
    polls for completion, and parses the inlined responses.

    Args:
      texts: Input texts to annotate.
      schema: Pydantic model used as the ``response_schema``.
      system_prompt: System-level instructions for the LLM.

    Returns:
      DataFrame with exactly ``len(texts)`` rows.  Failed rows have ``None``.

    Raises:
      RuntimeError: If the batch job fails or is cancelled.
    """
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type='application/json',
        response_schema=schema,
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

    all_rows = []
    for batch_job, chunk_texts in zip(completed_jobs, chunks, strict=True):
      chunk_rows = self._parse_job_responses(batch_job, schema)

      if len(chunk_rows) != len(chunk_texts):
        raise ValueError(
            f'Batch annotate: job {batch_job.name} got {len(chunk_rows)}'
            f' results for {len(chunk_texts)} inputs.'
        )

      all_rows.extend(chunk_rows)

    return pd.DataFrame(all_rows)

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
