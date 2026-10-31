# Adding models

Every model runs under the same frozen protocol: the same images, prompts, zoom tool
and scorer. The model roster is deliberately left out of the protocol hash
(`docs/protocol.md`), so adding a model doesn't reopen the protocol.

## Two adapters cover almost everything

| `provider` | Speaks | Use it for |
|---|---|---|
| `anthropic` | Anthropic Messages API (official SDK) | Claude models |
| `openai_compat` | OpenAI Chat Completions over HTTP (`httpx`) | OpenAI; Google Gemini (OpenAI-compatible endpoint); open-weight models on a hosted router (OpenRouter, Together, …) or your own vLLM / Ollama server |

Both adapters send the overview image and task text. In the zoom condition they offer
the same `zoom(start_s, end_s)` tool. Chat Completions tool results are text-only for
most providers, so `openai_compat` sends each zoomed image as a user message right
after the tool results. This is the only difference in what the model sees, and it
is recorded here so it can be reported.

## Steps to add one

1. **Get a key**, and add it to `.env` under the name you will put in `api_key_env`
   (for example `OPENAI_API_KEY=...`). A local server needs no key (`api_key_env: null`).
   Give the endpoint as `base_url`, or as `base_url_env` naming a `.env` variable that
   holds it (the router entries in the config use `ROUTER_BASE_URL`).
2. **Copy a template** from the commented block under `models:` in
   `config/benchmark.yaml`. Fill in the exact model id from the provider's model list.
   Don't guess ids. The model must accept image input.
3. **Add its price** under `pricing_usd_per_mtok` (USD per million input/output
   tokens, taken from the provider's pricing page on the day you run it). A local
   server gets `{input: 0, output: 0}`. Without a price the runner refuses to start,
   because the budget cap could not count its spend.
4. **Smoke test on dev**, 2 records:
   `python -m ecgbench.cli run --model <id> --partition dev --limit 2 --run-name smoke-<short>`
   Then read `results/dev/runs/smoke-<short>/attempts.jsonl`. Check that the answers
   parse and the zoom calls work.
5. **Dev run, then held-out** with a new run name, then `report --partition test
   --confirm-heldout --runs <all held-out attempts.jsonl files>`.

## Settings that differ by provider

| Key | Meaning |
|---|---|
| `max_tokens_param` | `max_completion_tokens` for current OpenAI models; `max_tokens` (default) elsewhere |
| `structured_output` | sends `response_format: json_schema`. Turn off for servers that reject it. Answers are validated strictly either way, so a model without it may score more invalid responses. That is part of the result. |
| `strict_tools` | strict JSON schema on the zoom tool, where supported |
| `extra_body` | provider-specific fields sent as-is, e.g. `reasoning_effort` |

## Model types worth covering

| Type | Examples to consider (verify exact ids and image support first) |
|---|---|
| Closed frontier (non-Anthropic) | OpenAI's current flagship vision model; Google's current Gemini Pro |
| Closed, small and cheap | OpenAI mini tier; Gemini Flash tier |
| Open weights, general vision-language | Qwen VL family; Llama vision; Gemma |
| Open weights, medical | MedGemma (multimodal) |

Open-weight models are also the ones you can train with reinforcement learning (RL)
later. The scorer already gives a per-case reward, but **never train on held-out
records** (`docs/datasets_next.md`).

## Fairness notes for the report

- Use the same `max_turns`, zoom limit, images and prompt for every model. Record any
  provider-specific setting (effort, structured output) in the results table.
- Some providers resize images internally. A 2400×600 image may be downscaled, and
  that is part of what the benchmark measures.
- Infrastructure failures are retried and logged the same way for every provider.
