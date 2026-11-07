# Running Amika in Docker (fork addition)

Linux + NVIDIA GPU + `nvidia-container-toolkit`. The container is a Python 3.11 home;
`launch.py` inside it spawns the services the browser app expects on loopback
(web 53000, TTS 8002, embed 8011, vision 8014), so the compose file uses host networking
and the `nvidia` runtime.

## Setup

1. Clone the fork and create the per-service venvs under `envs/` + `vendor/tts-engine`
   the main README describes (they are bind-mounted through the repo checkout, so build
   them in the checkout).
2. `cd docker && docker compose up -d`
3. `docker exec -d openamika python3 -u launch.py`
4. Open `http://<host>:53000`, run the setup wizard — the first sign-in creates the login
   account (fork feature: scrypt hash, stored in `config/auth.json`, never in the repo).

## Brain: direct OpenAI dialect (recommended)

Any server that speaks the OpenAI `/v1` API works directly — vLLM, llama.cpp server,
Strata, OpenRouter itself. No proxy needed.

- Setup wizard → **Ollama panel** → enter the base URL (`http://host:port`), pick
  **OpenAI-compatible** in the Dialect select, paste the server's API key if it needs one
  (fork feature), Detect models, choose one, let the wizard fill the roles grid.
- What the dialect changes under the hood (`config/ollama.json`, `"dialect": "openai"`):
  model discovery reads `GET /v1/models` instead of `/api/tags` + `/api/show`, request
  bodies are standard OpenAI (`max_tokens` top level, `stream_options`, JSON mode for the
  memory writer, standard `data:`-prefixed SSE with `[DONE]`), and the key is sent as a
  Bearer header on every call. Costs stay zero (local accounting — nothing is billed).
- Per-role thinking: the roles grid keeps a thinking switch per role; the fork sends
  `reasoning: {enabled: true|false}` on OpenAI-dialect servers when the model row is
  thinking-capable, and the native ollama `think` field otherwise.

## Brain: through the Ollama shim (optional)

`docker compose --profile proxy up -d` starts `eyalrot2/ollama-openai-proxy` on :11434
(patched `proxy-patch/models.py` for servers that omit `created` in their model list, and
`main_patched.py` adds an inbound-key gate — stock accepts none; set `INBOUND_API_KEYS` in
`docker/.env`). Amika's Ollama panel then points at `http://127.0.0.1:11434` with the key.
Only needed if something must see an Ollama-API surface; otherwise prefer the direct path.
