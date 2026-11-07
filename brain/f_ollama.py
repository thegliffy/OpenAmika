"""Ollama (local) transport for the four Build F roles — the fully-local option.

Discovery: ``GET {base}/api/version``, ``GET {base}/api/tags`` and ``POST {base}/api/show`` per model
(capabilities "thinking" / "vision", context length) → ``config/ollama.json``. f_roles generates one
option per detected model and role ("<role>:ollama:<model>"), zero-priced in the ledger.

Calls use Ollama's native ``/api/chat`` (NDJSON stream) rather than its OpenAI-compatible endpoint:
only the native API lets one request set the context window (``num_ctx`` — Ollama's default of 4k
would silently truncate her 8k-token prompts), switch thinking off per request (``think``) and keep
the model loaded between turns (``keep_alive``). Receipts, metering and the drain machinery are the
OpenRouter client's; this module only changes the request body, the line parser and the (zero) price.
"""
import copy
import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from f_memory import _write
from f_openrouter import MAX_OUTPUT_TOKENS, OpenRouterClient, OpenRouterError, ROLES, make_chat_body, normalize_usage

OLLAMA = "ollama"
OPENAI_DIALECT = "openai"         # config/ollama.json "dialect": speak OpenAI's /v1 API instead of Ollama's NDJSON
DEFAULT_URL = "http://127.0.0.1:11434"
CONFIG_NAME = "ollama.json"
DEFAULT_NUM_CTX = 16384        # her prompts are ~8k tokens; leave room for history growth and the answer
MIN_NUM_CTX = 8192
KEEP_ALIVE = "30m"             # keep the model loaded between turns (Ollama's default 5 min is too short)
ANSWER_TOKENS = {"surface": 400, "deep": 1500, "reflection": 3000, "memory_write": 6000}
THINKING_ROLES = ("reflection",)   # thinking (when the model can) only where latency does not matter

_URL_RE = re.compile(r"^https?://[A-Za-z0-9._\-\[\]]+(?::\d{1,5})?$")


def ollama_base_url(url):
    """A plain http(s)://host[:port] base, no path; empty → the local default."""
    url = str(url or "").strip().rstrip("/")
    if not url:
        return DEFAULT_URL
    if not _URL_RE.match(url):
        raise ValueError("The Ollama address must look like http://127.0.0.1:11434")
    return url


def _model_row(row):
    if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not row["name"].strip():
        return None
    ctx = row.get("context_length")
    size = row.get("size_gb")
    return {"name": row["name"].strip(), "family": str(row.get("family") or ""),
            "parameter_size": str(row.get("parameter_size") or ""), "quantization": str(row.get("quantization") or ""),
            "size_gb": float(size) if isinstance(size, (int, float)) and not isinstance(size, bool) else None,
            "context_length": ctx if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0 else None,
            "thinking": bool(row.get("thinking")), "vision": bool(row.get("vision"))}


def load_ollama_config(root):
    """config/ollama.json as the role manager uses it (defaults when the file is missing or odd)."""
    path = Path(root) / "config" / CONFIG_NAME
    doc = {}
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            doc = {}
    if not isinstance(doc, dict):
        doc = {}
    try:
        base = ollama_base_url(doc.get("base_url"))
    except ValueError:
        base = DEFAULT_URL
    num_ctx = doc.get("num_ctx")
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < MIN_NUM_CTX:
        num_ctx = DEFAULT_NUM_CTX
    models = [m for m in (_model_row(r) for r in (doc.get("models") or [])) if m]
    dialect = str(doc.get("dialect") or "ollama").strip().lower()
    if dialect not in ("ollama", OPENAI_DIALECT):
        dialect = "ollama"
    return {"base_url": base, "num_ctx": num_ctx, "keep_alive": str(doc.get("keep_alive") or KEEP_ALIVE),
            "api_key": str(doc.get("api_key") or ""), "dialect": dialect,
            "models": models, "detected_at": doc.get("detected_at"), "version": doc.get("version")}


def save_ollama_config(root, doc):
    _write(Path(root) / "config" / CONFIG_NAME, doc)


def _check(response):
    status = getattr(response, "status_code", 200)
    if status != 200:
        raise OpenRouterError(f"Ollama answered HTTP {status}")
    try:
        return response.json()
    except ValueError:
        raise OpenRouterError("Ollama answered with malformed JSON") from None
    finally:
        try:
            response.close()
        except Exception:
            pass


def _auth(api_key):
    """Bearer header for an Ollama server that asks for an API key (plain Ollama never does; ours can)."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def discover(http, base_url, *, timeout=(3, 10), show=True, api_key="", dialect="ollama", keep=()):
    """What this Ollama serves: {version, models:[{name, family, parameter_size, quantization, size_gb,
    context_length, thinking, vision}]}. Read-only; raises when the server does not answer.
    fork: dialect "openai" reads GET /v1/models instead of /api/tags + /api/show; the OpenAI list
    reports n_ctx and input modalities but no thinking flag, so thinking stays as saved (keep)."""
    base = ollama_base_url(base_url)
    hdr = _auth(api_key)
    if str(dialect or "ollama") == OPENAI_DIALECT:
        saved = {str(m.get("name") or ""): m for m in (keep or []) if isinstance(m, dict)}
        data = _check(http.get(base + "/v1/models", headers=hdr, timeout=timeout)) or {}
        models = []
        for row in data.get("data") or []:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                continue
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            arch = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
            modalities = arch.get("input_modalities") if isinstance(arch.get("input_modalities"), list) else []
            old = saved.get(row["id"].strip(), {})
            ctx = meta.get("n_ctx")
            models.append(_model_row({"name": row["id"].strip(), "family": str(meta.get("name") or ""),
                                      "parameter_size": "", "quantization": "", "size_gb": None,
                                      "context_length": ctx if isinstance(ctx, int) and not isinstance(ctx, bool) and ctx > 0 else old.get("context_length"),
                                      "thinking": bool(old.get("thinking")),
                                      "vision": ("image" in modalities) if modalities else bool(old.get("vision"))}))
        models = [m for m in models if m]
        models.sort(key=lambda m: m["name"])
        return {"version": "", "models": models}
    version = str((_check(http.get(base + "/api/version", headers=hdr, timeout=timeout)) or {}).get("version") or "")
    rows = (_check(http.get(base + "/api/tags", headers=hdr, timeout=timeout)) or {}).get("models") or []
    models = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or row.get("model")
        if not isinstance(name, str) or not name.strip():
            continue
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        size = row.get("size")
        item = {"name": name.strip(), "family": details.get("family") or "",
                "parameter_size": details.get("parameter_size") or "", "quantization": details.get("quantization_level") or "",
                "size_gb": round(size / 1e9, 1) if isinstance(size, (int, float)) and not isinstance(size, bool) else None,
                "context_length": None, "thinking": False, "vision": False}
        if show:
            try:
                info = _check(http.post(base + "/api/show", headers=hdr, json={"model": item["name"]}, timeout=timeout)) or {}
                caps = info.get("capabilities") if isinstance(info.get("capabilities"), list) else []
                item["thinking"] = "thinking" in caps
                item["vision"] = "vision" in caps
                model_info = info.get("model_info") if isinstance(info.get("model_info"), dict) else {}
                for key, value in model_info.items():
                    if str(key).endswith(".context_length") and isinstance(value, int) and not isinstance(value, bool):
                        item["context_length"] = value
                        break
            except Exception:
                pass      # the model is still listed; thinking/vision stay unknown (off)
        models.append(_model_row(item))
    models.sort(key=lambda m: m["name"])
    return {"version": version, "models": models}


def ollama_role_config(ollama, model_name, role, *, reasoning_enabled=None, max_tokens=None):
    """Per-role parameters for a detected Ollama model (the shape the role manager and the client read)."""
    role = "memory_write" if role == "memory_extraction" else role
    rows = {m["name"]: m for m in (ollama.get("models") or [])}
    if role not in ROLES or not isinstance(model_name, str) or model_name not in rows:
        raise ValueError("Unknown role or Ollama model not detected (detect models first)")
    row = rows[model_name]
    if reasoning_enabled is not None and not isinstance(reasoning_enabled, bool):
        raise ValueError("reasoning_enabled must be boolean")
    thinking = bool(reasoning_enabled) and bool(row.get("thinking"))
    answer = ANSWER_TOKENS[role] if max_tokens is None else max_tokens
    if isinstance(answer, bool) or not isinstance(answer, int) or not 1 <= answer <= MAX_OUTPUT_TOKENS:
        raise ValueError("max_tokens must be an integer between 1 and 8192")
    num_ctx = int(ollama.get("num_ctx") or DEFAULT_NUM_CTX)
    if row.get("context_length"):
        num_ctx = max(MIN_NUM_CTX, min(num_ctx, int(row["context_length"])))
    return {"transport": OLLAMA, "model_id": model_name, "api_model": model_name, "canonical_slug": model_name,
            "provider_tag": OLLAMA, "provider_name": "Ollama (local)", "quantization": row.get("quantization") or None,
            "role": role, "max_tokens": answer,
            "reasoning": {"enabled": thinking} if row.get("thinking") else None,
            "reasoning_enabled": thinking, "mandatory_reasoning": False, "think_supported": bool(row.get("thinking")),
            "max_price": {"prompt": 0.0, "completion": 0.0, "request": 0.0}, "reservation_input_rate": 0.0,
            "base_url": ollama.get("base_url") or DEFAULT_URL, "api_key": str(ollama.get("api_key") or ""),
            "num_ctx": num_ctx,
            "keep_alive": str(ollama.get("keep_alive") or KEEP_ALIVE), "json_mode": "json_object",
            "endpoint_context_tokens": row.get("context_length"), "endpoint_max_completion_tokens": None,
            "supported_parameters": [],
            "structured_outputs_advertised": (str(ollama.get("dialect") or "ollama") == OPENAI_DIALECT),
            "dialect": str(ollama.get("dialect") or "ollama"),
            "family": row.get("family") or "", "parameter_size": row.get("parameter_size") or ""}


def make_ollama_body(config, messages, *, stream=True, json_schema=None):
    """Native /api/chat request: context window, answer cap, thinking switch, JSON mode for the writer."""
    if config.get("transport") != OLLAMA or config.get("role") not in ROLES or not config.get("api_model"):
        raise ValueError("Use a validated Ollama role config")
    if not isinstance(messages, list) or not messages:
        raise ValueError("A nonempty message list is required")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("Only system, user, and assistant text messages are supported")
        if not isinstance(message.get("content"), str):
            raise ValueError("This transport accepts text-only messages")
    if str(config.get("dialect") or "ollama") == OPENAI_DIALECT:
        # Standard OpenAI chat body: max_tokens top-level (a plain server never reads the ollama
        # options block); usage comes back on the final SSE event.
        body = {"model": config["api_model"], "messages": copy.deepcopy(messages), "stream": bool(stream),
                "max_tokens": int(config["max_tokens"]), "stream_options": {"include_usage": True}}
        wants_json = config["role"] == "memory_write" or json_schema is not None
        if wants_json:
            body["response_format"] = {"type": "json_object"}
        return body
    body = {"model": config["api_model"], "messages": copy.deepcopy(messages), "stream": bool(stream),
            "keep_alive": config.get("keep_alive") or KEEP_ALIVE,
            "options": {"num_ctx": int(config["num_ctx"]), "num_predict": int(config["max_tokens"])}}
    if config.get("think_supported"):
        body["think"] = bool(config.get("reasoning_enabled"))
    wants_json = config["role"] == "memory_write" or json_schema is not None
    if wants_json and any("json" in message["content"].lower() for message in messages):
        body["format"] = "json"
    return body


class OllamaClient(OpenRouterClient):
    """Native Ollama chat with the OpenRouter client's receipts, metering and drain. Zero cost."""
    transport = OLLAMA

    def __init__(self, http, before_call, after_call, *, base_url, timeout=(10, 600), event_sink=None, drainer=None,
                 api_key="", dialect="ollama"):
        super().__init__(http, "local", before_call, after_call, timeout=timeout, event_sink=event_sink, drainer=drainer)
        self.dialect = str(dialect or "ollama")        # fork: "openai" = OpenAI-dialect server (juno's Strata)
        self.chat_url = ollama_base_url(base_url) + ("/v1/chat/completions" if self.dialect == OPENAI_DIALECT else "/api/chat")
        self.api_key = str(api_key or "")

    def _chat_body(self, config, messages, *, stream, json_schema=None):
        if self.dialect == OPENAI_DIALECT:
            return make_chat_body(config, messages, stream=stream, json_schema=json_schema)
        return make_ollama_body(config, messages, stream=stream, json_schema=json_schema)

    def _reservation(self, config, body):
        return 0.00001        # a positive reservation keeps the budget gate happy; the price is zero

    def _prepare(self, config, body, metadata, url):
        handle, receipt = super()._prepare(config, body, metadata, url)
        receipt["request_id"] = "ollama-" + uuid.uuid4().hex[:12]     # telemetry keys its timings by id
        receipt["actual_provider"] = OLLAMA
        return handle, receipt

    def _usage(self, receipt, embedding):
        usage = normalize_usage(dict(receipt.get("raw_usage") or {}, cost=0.0), embedding=embedding)
        usage["cost_source"] = "local"
        return usage

    def _post(self, url, body, stream):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = self.http.post(url, headers=headers, json=body, stream=stream,
                                  timeout=self.timeout, allow_redirects=False)
        status = getattr(response, "status_code", 200)
        if status >= 400:
            # Ollama explains itself in the body ("model requires more system memory", "model not found"…):
            # say so instead of a bare HTTP status.
            detail = ""
            try:
                payload = response.json()
                detail = str((payload or {}).get("error") or "")[:240]
            except Exception:
                pass
            try:
                response.close()
            except Exception:
                pass
            raise OpenRouterError(f"Ollama answered HTTP {status}" + (f": {detail}" if detail else ""))
        return response

    def _apply_chunk(self, config, receipt, data, started):
        """Shared by the stream and the non-stream path: errors, model, and the final usage."""
        if not isinstance(data, dict):
            raise OpenRouterError("Ollama returned a non-object line")
        if data.get("error"):
            raise OpenRouterError("Ollama error: " + str(data["error"])[:200])
        if receipt["timing"]["first_event_s"] is None:
            receipt["timing"]["first_event_s"] = time.perf_counter() - started
        if data.get("model"):
            receipt["actual_model"] = str(data["model"])
        message = data.get("message") if isinstance(data.get("message"), dict) else {}
        text = message.get("content")
        if text is not None and not isinstance(text, str):
            raise OpenRouterError("Ollama returned non-text content")
        if data.get("done"):
            reason = str(data.get("done_reason") or "stop")
            receipt["finish_reason"] = reason
            receipt["truncated"] = reason == "length"
            receipt["raw_usage"].update({"prompt_tokens": int(data.get("prompt_eval_count") or 0),
                                         "completion_tokens": int(data.get("eval_count") or 0), "cost": 0.0})
        return text or ""

    def _stream_line(self, config, receipt, line, started):
        """One NDJSON line: None = skip, True = done, else the content delta."""
        if self.dialect == OPENAI_DIALECT:
            return OpenRouterClient._stream_line(self, config, receipt, line, started)
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        line = (line or "").strip()
        if not line:
            return None
        data = json.loads(line)
        text = self._apply_chunk(config, receipt, data, started)
        if text:
            if receipt["timing"]["first_content_s"] is None:
                receipt["timing"]["first_content_s"] = time.perf_counter() - started
            receipt["content"] += text
            if self.event_sink:
                self.event_sink({"id": receipt["request_id"], "model": receipt["actual_model"],
                                 "choices": [{"delta": {"content": text}}]})
        if data.get("done"):
            return True
        return text or ""

    def complete(self, config, messages, *, metadata=None, json_schema=None):
        if self.dialect == OPENAI_DIALECT:
            return OpenRouterClient.complete(self, config, messages, metadata=metadata, json_schema=json_schema)
        body = self._chat_body(config, messages, stream=False, json_schema=json_schema)
        handle, receipt = self._prepare(config, body, metadata, self.chat_url)
        started = time.perf_counter()
        response, error, done, data = None, None, False, None
        try:
            response = self._post(self.chat_url, body, False)
            response.raise_for_status()
            data = response.json()
            text = self._apply_chunk(config, receipt, data, started)
            if not data.get("done"):
                raise OpenRouterError("Ollama returned an unfinished completion")
            receipt["timing"]["first_content_s"] = time.perf_counter() - started
            receipt["content"] = text
            done = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            self._settle(handle, receipt, started, done, False, error)
        if receipt["status"] != "complete":
            raise OpenRouterError("Incomplete local accounting; inspect the request receipt before retrying")
        return {"text": receipt["content"], "raw": data, "receipt": receipt}

    def embed(self, *args, **kwargs):
        raise ValueError("The Ollama transport serves chat roles only (embeddings run on the local bge-m3 server)")


# ------------------------------------------------------------------ brain routes
def install_ollama(g):
    """GET /api/build_f/ollama (what is detected, is the server up), POST …/ollama/detect (probe the
    server, save config/ollama.json, refresh the role options), POST …/ollama/use (one model for all
    four roles, or a role → model map). Installed by f_bridge after the role manager."""
    app, request, jsonify = g["app"], g["request"], g["jsonify"]
    root = Path(g["__file__"]).resolve().parent.parent

    def action_request(action):
        if request.headers.get("Origin") or request.headers.get("X-Build-F-Action") != action:
            raise PermissionError("Use the OpenAmika page for this action.")
        value = request.get_json(silent=True) if request.is_json else None
        if not isinstance(value, dict):
            raise ValueError("A JSON object is required.")
        return value

    def in_use(manager):
        if manager is None:
            return {}
        with manager.lock:
            return {role: manager.options[role][manager.selections[role]]["config"]["model_id"]
                    for role in ROLES if manager.options[role][manager.selections[role]].get("transport") == OLLAMA}

    def status(probe=True):
        manager = g.get("F_ROLES")
        ollama = manager.ollama if manager is not None else load_ollama_config(root)
        out = {"ok": True, "base_url": ollama["base_url"], "num_ctx": ollama["num_ctx"], "keep_alive": ollama["keep_alive"],
               "detected_at": ollama.get("detected_at"), "version": ollama.get("version"), "models": ollama["models"],
               "api_key_set": bool(ollama.get("api_key")), "dialect": str(ollama.get("dialect") or "ollama"),
               "reachable": None, "live_version": None, "error": None, "in_use": in_use(manager)}
        if probe:
            try:
                if str(ollama.get("dialect") or "ollama") == OPENAI_DIALECT:
                    data = _check(g["http_requests"].get(ollama["base_url"] + "/v1/models", headers=_auth(ollama.get("api_key")), timeout=(2, 3))) or {}
                    out["reachable"], out["live_version"] = True, "openai" if (data.get("data") or data.get("models")) else "?"
                else:
                    data = _check(g["http_requests"].get(ollama["base_url"] + "/api/version", headers=_auth(ollama.get("api_key")), timeout=(2, 3))) or {}
                    out["reachable"], out["live_version"] = True, str(data.get("version") or "")
            except Exception as error:
                out["reachable"], out["error"] = False, str(error)[:160]
        return out

    @app.route("/api/build_f/ollama", methods=["GET"])
    def ollama_get():
        return jsonify(status())

    @app.route("/api/build_f/ollama/detect", methods=["POST"])
    def ollama_detect():
        try:
            value = action_request("ollama-detect")
            base = ollama_base_url(value.get("base_url"))
            num_ctx = value.get("num_ctx")
            if num_ctx is not None and (isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or not MIN_NUM_CTX <= num_ctx <= 131072):
                raise ValueError(f"num_ctx must be an integer between {MIN_NUM_CTX} and 131072.")
            current = load_ollama_config(root)
            key = str(value.get("api_key") or "").strip() or current["api_key"]
            dialect = str(value.get("dialect") or current["dialect"]).strip().lower()
            if dialect not in ("ollama", OPENAI_DIALECT):
                raise ValueError("dialect must be ollama or openai.")
            try:
                found = discover(g["http_requests"], base, api_key=key, dialect=dialect, keep=current["models"])
            except Exception as error:
                return jsonify({"ok": False, "error": f"Ollama is not answering at {base} — is it running? "
                                                      f"(install it from ollama.com, then start it) · {str(error)[:120]}"}), 503
            doc = {"base_url": base, "num_ctx": num_ctx or current["num_ctx"], "keep_alive": current["keep_alive"],
                   "api_key": key, "dialect": dialect,
                   "models": found["models"], "version": found["version"],
                   "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            save_ollama_config(root, doc)
            manager = g.get("F_ROLES")
            if manager is not None:
                manager.reload_ollama()
            out = status(probe=False)
            out.update(reachable=True, live_version=found["version"])
            return jsonify(out)
        except PermissionError as error:
            return jsonify({"ok": False, "error": str(error)}), 403
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400

    @app.route("/api/build_f/ollama/use", methods=["POST"])
    def ollama_use():
        manager = g.get("F_ROLES")
        if manager is None:
            return jsonify({"ok": False, "error": "Model choices are not installed yet."}), 503
        try:
            value = action_request("ollama-use")
            model = value.get("model")
            roles = value.get("roles") if isinstance(value.get("roles"), dict) else {}
            selections = {}
            for role in ROLES:
                name = roles.get(role, model)
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Pick a detected Ollama model (one for all four roles, or one per role).")
                key = f"{role}:ollama:{name.strip()}"
                if key not in manager.options[role]:
                    raise ValueError(f"{name} is not a detected Ollama model for the {role} role — detect models first.")
                selections[role] = key
            result = manager.set(selections)
            return jsonify({"ok": True, "selections": result["selections"], "transport": result["transport"],
                            "in_use": in_use(manager)})
        except PermissionError as error:
            return jsonify({"ok": False, "error": str(error)}), 403
        except ValueError as error:
            return jsonify({"ok": False, "error": str(error)}), 400
        except RuntimeError:
            return jsonify({"ok": False, "error": "Model selection is busy. Wait for all speech and memory work to finish."}), 409

    print("[ollama] local model endpoints installed")
