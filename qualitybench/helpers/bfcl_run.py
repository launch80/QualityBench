#!/usr/bin/env python3
"""BFCL against any OpenAI-compatible server, scoring the server's own tool-call parsing (FC mode).

  OPENAI_BASE_URL=http://host:port/v1 OPENAI_API_KEY=local BFCL_PROJECT_ROOT=<dir> \
      python bfcl_run.py <model-id> generate|evaluate [extra bfcl args]

BFCL only knows models in its registry. This registers `<model-id>-FC` with the stock
OpenAICompletionsHandler (chat completions + `tools`), so the request carries <model-id> and the server's
tool parser turns the reply into structured tool_calls, exactly as an agent sees them. Requests get a one-hour
timeout and at most three attempts.
"""
import dataclasses, sys
import openai

_OpenAI = openai.OpenAI
openai.OpenAI = lambda *a, **k: _OpenAI(*a, **{**k, "timeout": 3600, "max_retries": 2})

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING  # noqa: E402
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler  # noqa: E402

model, cmd, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
template = next(c for c in MODEL_CONFIG_MAPPING.values() if c.model_handler is OpenAICompletionsHandler and c.is_fc_model)
key = f"{model}-FC"
MODEL_CONFIG_MAPPING[key] = dataclasses.replace(
    template, model_name=model, display_name=f"{model} (FC, OpenAI-compatible)", url="", org="local",
    license="", input_price=None, output_price=None, is_fc_model=True)

from bfcl_eval.__main__ import cli  # noqa: E402

sys.argv = ["bfcl", cmd, "--model", key, *rest]
cli()
