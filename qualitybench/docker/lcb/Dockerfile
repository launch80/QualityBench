# LiveCodeBench (commit 28fef95e) for generation + sandboxed scoring. Scoring runs with --network none.
FROM python:3.11-slim
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "datasets==3.6.0" "numpy<2.3" pebble tqdm \
      # client SDKs at the minimums in lcb's pyproject (newer anthropic dropped HUMAN_PROMPT, which lcb imports)
      "openai==1.59.6" "anthropic==0.42.0" "cohere==5.13.6" "google-genai==0.6.0" "mistralai==0.4.2" "together==1.3.14"
COPY src /lcb
# lcb_runner opens its few-shot json files by relative path: run from the repo root
WORKDIR /lcb
RUN pip install --no-cache-dir --no-deps -e /lcb && python -c "import lcb_runner.runner.custom_evaluator; print('lcb import ok')"
