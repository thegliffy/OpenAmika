"""Validated model-role selections for the isolated Build F service.

The deployment owns config/llm_roles.json and the reviewed catalogs
(config/openrouter_models.json; config/alibaba_models.json since 2026-10-09).
Website requests select existing option keys or presets only. They cannot supply
API URLs, models, providers, prices, prompt changes, or credential values.

Providers (owner decision 2026-10-09). An option row is an OpenRouter endpoint unless
it says ``"provider": "alibaba"`` (Alibaba Cloud Model Studio, Singapore workspace,
key ``DASHSCOPE_API_KEY``; list prices from pricing.json ``alibaba/<model>`` because
Model Studio reports no cost). Both go through the same metering hooks, receipts,
drain and ``F_OR_AFTER_CALL`` observer. ``disabled_providers`` names native providers
that nothing may call (Anthropic and xAI since 2026-10-09; f_bridge refuses them before
any request). Native rows of those providers, and any row marked ``"disabled": true``,
are listed with their reason but can never be selected; a saved selection that names
one falls back, role by role, to the fallback preset (``fallback_preset`` or the first
preset whose choices are all enabled). Presets may name a ``provider``;
``apply_preset`` (POST /api/build_f/preset) switches all four roles at once, which is
the OpenRouter <-> Alibaba provider switch.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import threading

from f_extract import SYSTEM, FACT_INSTRUCTION, parse_isolated_facts
from f_memory import _write
from f_openrouter import (ALIBABA, AlibabaClient, OpenRouterClient, alibaba_base_url, alibaba_role_config,
                          embedding_config, load_alibaba_catalog, load_catalog, role_config)
from f_ollama import OLLAMA, THINKING_ROLES, OllamaClient, load_ollama_config, ollama_role_config

ROLES = ("surface", "deep", "reflection", "memory_write")
NATIVE_KEYS = {"surface": "grok-4.3-fast", "deep": "claude-5-sonnet",
               "reflection": "claude-5-sonnet-thinking", "memory_write": "claude-sonnet-5"}
DEFAULT_NAMES = {"surface": "DEFAULT_SURFACE_MODEL", "deep": "DEFAULT_DEEP_MODEL",
                 "reflection": "DEFAULT_THINKING_MODEL"}
LIST_NAMES = {"surface": "SURFACE_MODEL_KEYS", "deep": "DEEP_MODEL_KEYS", "reflection": "THINKING_MODEL_KEYS"}
# Role transports (option "provider"); a row without one is OpenRouter. Ollama options are not rows of
# llm_roles.json: they are generated from config/ollama.json (whatever the local server has pulled).
PROVIDERS = ("openrouter", ALIBABA, OLLAMA)
# The native memory writer is not an AVAILABLE_MODELS entry; its provider is fixed.
NATIVE_PROVIDERS = {"claude-sonnet-5": "anthropic"}
# webui's legacy per-layer model routes (Flask endpoint names) and its model list.
MODEL_ROUTES = ("api_surface_model", "api_deep_model", "api_thinking_model")
MODEL_LIST_ROUTE = "get_models"


class RoleManager:
    def __init__(self, g, root, memory, meter, pricing=None):
        self.g, self.root, self.memory, self.meter = g, Path(root), memory, meter
        self.path = self.root / "config" / "llm_roles.json"
        self.document = json.loads(self.path.read_text(encoding="utf-8-sig"))
        self.pricing = pricing if isinstance(pricing, dict) else {}
        or_catalog = self.root / "config" / "openrouter_models.json"
        self.catalog = load_catalog(or_catalog) if or_catalog.is_file() else {}
        # Loaded on the first Alibaba option only: a deployment without Alibaba options never
        # depends on config/alibaba_models.json.
        self.alibaba = None
        # The local server's detected models (config/ollama.json; empty until the Setup page detects them).
        self.ollama = load_ollama_config(self.root)
        self.lock = threading.RLock()
        self.options = {}
        self.native_extract = memory.extract_fn
        self.native_embed = memory.embed_fn
        if self.document.get("version") != 1 or set(self.document.get("options", {})) != set(ROLES):
            raise ValueError("Invalid model role configuration version/options")
        self.disabled_providers = self._disabled_providers(self.document.get("disabled_providers"))
        # Role options join webui's AVAILABLE_MODELS only after the whole document validated,
        # so a refused configuration leaves the native model table untouched.
        self._registrations = {}
        used_keys = set()
        for role in ROLES:
            rows = self.document["options"][role]
            if not isinstance(rows, list) or not rows:
                raise ValueError("Each role requires at least one reviewed option")
            choices = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Model options must be objects")
                key = row.get("key")
                if not isinstance(key, str) or not key or key in used_keys:
                    raise ValueError("Model option keys must be nonempty and globally unique")
                used_keys.add(key)
                choices[key] = self._option(role, key, row)
            if all(choice.get("disabled") for choice in choices.values()):
                raise ValueError("Each role requires at least one enabled option")
            self.options[role] = choices
        self._ollama_options()
        self.selection_fallback = {}
        self.selections = self._initial(self.document.get("selections"))
        self.embedding = None
        if self.document.get("embedding_openrouter", False):
            self.embedding = embedding_config(self.document.get("embedding_endpoint") or {})
            self._add_price(self.embedding)
        self.g["AVAILABLE_MODELS"].update(self._registrations)
        self._apply(self.selections)

    # ------------------------------------------------------------- options
    @staticmethod
    def _disabled_providers(value):
        if value is None:
            return {}
        if isinstance(value, list):
            value = {name: "Disabled by the owner" for name in value}
        if not isinstance(value, dict) or any(not isinstance(k, str) or not k or not isinstance(v, str)
                                               for k, v in value.items()):
            raise ValueError("disabled_providers must map provider names to reasons")
        return {k.lower(): v for k, v in value.items()}

    def _native_provider(self, native):
        entry = self.g.get("AVAILABLE_MODELS", {}).get(native) or {}
        return str(entry.get("provider") or NATIVE_PROVIDERS.get(native) or "native").lower()

    def _option(self, role, key, row):
        item = {"key": key, "name": str(row.get("name") or key), "description": str(row.get("description") or "")}
        disabled = row.get("disabled", False)
        if not isinstance(disabled, bool):
            raise ValueError("Option 'disabled' must be true or false")
        reason = row.get("disabled_reason")
        if "native_key" in row:
            native = row["native_key"]
            if native != NATIVE_KEYS[role]:
                raise ValueError("Unapproved native model option")
            provider = self._native_provider(native)
            if provider in self.disabled_providers:
                disabled, reason = True, reason or self.disabled_providers[provider]
            if not disabled and role != "memory_write" and native not in self.g["AVAILABLE_MODELS"]:
                raise ValueError("Unapproved native model option")
            item.update(native_key=native, transport="native", native_provider=provider)
        else:
            transport = row.get("provider", "openrouter")
            if transport not in PROVIDERS:
                raise ValueError("Unknown model provider")
            item["transport"] = transport
            if not disabled:
                # A disabled row is listed only; its endpoint is never configured or called.
                config = self._config(role, transport, row)
                # Receipts carry the option key so two options of one model (e.g. reasoning on/off) stay distinguishable.
                config["option_key"] = key
                item["config"] = config
                if role != "memory_write":
                    if key in self.g["AVAILABLE_MODELS"]:
                        raise ValueError("Role option key collides with a native model key")
                    self._registrations[key] = {"name": item["name"], "description": item["description"],
                        "provider": transport, "model": config["model_id"], "_or_config": config, "_role_config": config}
                self._add_price(config)
        if disabled:
            item.update(disabled=True, disabled_reason=str(reason or "Disabled by the owner"))
        return item

    def _ollama_options(self):
        """One option per detected Ollama model and role ("<role>:ollama:<model>"), zero-priced."""
        for role in ROLES:
            for model in self.ollama.get("models") or []:
                key = f"{role}:ollama:{model['name']}"
                if key in self.options[role]:
                    continue
                bits = [b for b in (model.get("parameter_size"), model.get("quantization")) if b]
                if model.get("size_gb"):
                    bits.append(f"{model['size_gb']} GB")
                caps = [c for c, on in (("thinking", model.get("thinking")), ("vision", model.get("vision"))) if on]
                description = ("Runs on your own GPU through Ollama: no API key, no cost; speed depends on the card."
                               + (" " + " · ".join(bits) + "." if bits else "") + (" Can: " + ", ".join(caps) + "." if caps else ""))
                row = {"key": key, "name": f"{model['name']} (Ollama · local)", "description": description,
                       "provider": OLLAMA, "model_id": model["name"],
                       "reasoning_enabled": bool(model.get("thinking")) and role in THINKING_ROLES}
                try:
                    self.options[role][key] = self._option(role, key, row)
                except ValueError:
                    continue        # a model that cannot serve this role is simply not offered

    def reload_ollama(self):
        """Re-read config/ollama.json (after a detect) and swap the Ollama options; a selected model that
        vanished falls back to the fallback preset's choice for that role."""
        with self.lock:
            self.ollama = load_ollama_config(self.root)
            for role in ROLES:
                for key in [k for k, v in self.options[role].items() if v.get("transport") == OLLAMA]:
                    del self.options[role][key]
                    self._registrations.pop(key, None)
                    self.g["AVAILABLE_MODELS"].pop(key, None)
            self._ollama_options()
            self.g["AVAILABLE_MODELS"].update(self._registrations)
            selections = copy.deepcopy(self.selections)
            moved = False
            for role in ROLES:
                if selections[role] in self.options[role]:
                    continue
                preset = self._fallback_preset()
                selections[role] = (preset["selections"][role] if preset else
                                    next(k for k, item in self.options[role].items() if not item.get("disabled")))
                moved = True
            if moved:
                document = copy.deepcopy(self.document)
                document["selections"] = selections
                _write(self.path, document)
                self.document = document
            self._apply(selections)
            return self.get()

    def _config(self, role, transport, row):
        if transport == OLLAMA:
            return ollama_role_config(self.ollama, row.get("model_id"), role,
                                      reasoning_enabled=row.get("reasoning_enabled"), max_tokens=row.get("max_tokens"))
        if transport == ALIBABA:
            if self.alibaba is None:
                path = self.root / "config" / "alibaba_models.json"
                if not path.is_file():
                    raise ValueError("Alibaba options need the reviewed config/alibaba_models.json")
                self.alibaba = load_alibaba_catalog(path)
            return alibaba_role_config(self.alibaba, row.get("model_id"), role, pricing=self.pricing,
                                       max_tokens=row.get("max_tokens"), reasoning_enabled=row.get("reasoning_enabled"),
                                       thinking_budget=row.get("thinking_budget"),
                                       reasoning_effort=row.get("reasoning_effort"))
        if not self.catalog:
            raise ValueError("OpenRouter options need the reviewed config/openrouter_models.json")
        return role_config(self.catalog, row.get("model_id"), role,
                           max_tokens=row.get("max_tokens"), reasoning_enabled=row.get("reasoning_enabled"))

    def _add_price(self, config):
        if config.get("transport") == OLLAMA:
            # Local: nothing is charged. A priced (zero) entry keeps the meter's budget gate satisfied.
            self.meter.pricing["models"]["ollama:" + config["model_id"]] = {"input": 0.0, "output": 0.0, "cache_read": 0.0}
            return
        if config.get("transport") == ALIBABA:
            # Model Studio calls are priced by the transport from the reviewed list price and
            # recorded as reported cost. These snapshot rates (first tier, peak hours) are what
            # the meter falls back to, e.g. for f_settle's estimate of a call without usage.
            base = config["price"]["base"]
            self.meter.pricing["models"]["alibaba:" + config["model_id"]] = {
                "input": base["input"], "output": max(base["output"], base.get("output_thinking", 0)),
                "cache_read": base.get("cache_read", base["input"])}
            return
        # Maximum published rates are used only for conservative local estimates.
        # OR reported usage.cost is the independently recorded actual charge.
        price = config["max_price"]
        self.meter.pricing["models"]["openrouter:"+config["model_id"]] = {
            "input": config["reservation_input_rate"], "output": price["completion"],
            "cache_read": config["reservation_input_rate"],
            "cache_write_5m": config["reservation_input_rate"],
            "cache_write_1h": config["reservation_input_rate"]}

    # ---------------------------------------------------------- selections
    def _known(self, selections):
        if not isinstance(selections, dict) or set(selections) != set(ROLES):
            raise ValueError("Select all four model roles")
        if any(not isinstance(key, str) or key not in self.options[role] for role, key in selections.items()):
            raise ValueError("Select only the tested model options shown for each role")

    def _validate(self, selections):
        self._known(selections)
        for role, key in selections.items():
            choice = self.options[role][key]
            if choice.get("disabled"):
                raise ValueError(choice["name"] + " is disabled: " + choice["disabled_reason"])
        return copy.deepcopy(selections)

    def _preset_ok(self, preset):
        if not isinstance(preset, dict) or not isinstance(preset.get("name"), str):
            return False
        try:
            self._validate(preset.get("selections"))
        except ValueError:
            return False
        return True

    def _fallback_preset(self):
        presets = [p for p in self.document.get("presets", []) if self._preset_ok(p)]
        wanted = self.document.get("fallback_preset")
        return next((p for p in presets if p["name"] == wanted), None) or (presets[0] if presets else None)

    def _initial(self, selections):
        """The saved selection, with each disabled choice replaced by the fallback preset's."""
        self._known(selections)
        result = copy.deepcopy(selections)
        preset = None
        for role in ROLES:
            choice = self.options[role][result[role]]
            if not choice.get("disabled"):
                continue
            preset = preset or self._fallback_preset()
            target = preset["selections"][role] if preset else next(
                key for key, item in self.options[role].items() if not item.get("disabled"))
            self.selection_fallback[role] = {"from": result[role], "to": target, "reason": choice["disabled_reason"],
                                             "preset": preset["name"] if preset else None}
            result[role] = target
        return result

    def _apply(self, selections):
        for role, name in DEFAULT_NAMES.items():
            choice = self.options[role][selections[role]]
            self.g[name] = choice.get("native_key", choice["key"])
            self.g[LIST_NAMES[role]] = [row.get("native_key", row["key"]) for row in self.options[role].values()
                                        if not row.get("disabled")]
        self.selections = copy.deepcopy(selections)

    def transport_of(self, selections=None):
        """'openrouter' / 'alibaba' / 'native' when every role uses it, else 'mixed'."""
        selections = self.selections if selections is None else selections
        kinds = {self.options[role][selections[role]].get("transport") for role in ROLES}
        return kinds.pop() if len(kinds) == 1 else "mixed"

    def disabled_reason(self, key):
        """Why a model key (role option or webui AVAILABLE_MODELS key) may not be used, else None."""
        for choices in self.options.values():
            item = choices.get(key)
            if item is not None and item.get("disabled"):
                return item["disabled_reason"]
        entry = self.g.get("AVAILABLE_MODELS", {}).get(key) if isinstance(key, str) else None
        provider = str((entry or {}).get("provider") or "").lower()
        return self.disabled_providers.get(provider)

    def presets(self):
        result = []
        for preset in self.document.get("presets", []):
            if isinstance(preset, dict) and isinstance(preset.get("name"), str):
                item = copy.deepcopy(preset)
                item["available"] = self._preset_ok(preset)
                result.append(item)
        return result

    def get(self):
        with self.lock:
            options = {}
            for role, choices in self.options.items():
                options[role] = []
                for row in choices.values():
                    item = {k: row[k] for k in ("key", "name", "description")}
                    item["transport"] = row["transport"]
                    if "config" in row:
                        config = row["config"]
                        item.update(model=config["model_id"], provider=config["provider_tag"],
                                    max_tokens=config["max_tokens"], reasoning=config["reasoning"],
                                    price_ceiling_usd_per_million=config["max_price"])
                        if row["transport"] == ALIBABA:
                            item.update(price_usd_per_million=config["price"]["base"],
                                        implicit_cache=config["implicit_cache"])
                    elif "native_key" in row:
                        item.update(model=row["native_key"], provider="native")
                    else:
                        item.update(provider=row["transport"])
                    if row.get("disabled"):
                        item.update(disabled=True, disabled_reason=row["disabled_reason"])
                    options[role].append(item)
            return {"ok": True, "selections": copy.deepcopy(self.selections), "options": options,
                    "presets": self.presets(), "transport": self.transport_of(),
                    "selection_fallback": copy.deepcopy(self.selection_fallback),
                    "disabled_providers": dict(self.disabled_providers),
                    "ollama": {"base_url": self.ollama["base_url"], "num_ctx": self.ollama["num_ctx"],
                               "detected_at": self.ollama.get("detected_at"), "version": self.ollama.get("version"),
                               "models": [m["name"] for m in self.ollama["models"]]},
                    "embedding": "OpenRouter Gemini embedding 2" if self.embedding else "Gemini embedding 2"}

    def set(self, selections):
        selections = self._validate(selections)
        # Native dispatch takes _ws_active_lock while registering new work. Hold
        # it across persistence+selection so a partially changed config cannot
        # become observable. Memory locks prevent switching during ingestion.
        with self.lock, self.g["_ws_active_lock"]:
            if self.g.get("_ws_active") or self.g.get("_thinking_inflight", 0) or self.meter.summary()["active_requests"]:
                raise RuntimeError("Wait for active conversation/model work")
            with contextlib.ExitStack() as stack:
                for name in ("_worker_lock", "_query_lock"):
                    lock = getattr(self.memory, name)
                    if not lock.acquire(blocking=False):
                        raise RuntimeError("Wait for active memory work")
                    stack.callback(lock.release)
                with self.memory._lock:
                    if self.memory._active_retrievals or self.memory.status().get("pending"):
                        raise RuntimeError("Wait for queued memory work")
                    document = copy.deepcopy(self.document)
                    document["selections"] = selections
                    _write(self.path, document)
                    self.document = document
                    self._apply(selections)
                    self.selection_fallback = {}
        return self.get()

    def apply_preset(self, name=None, provider=None):
        """Select one validated preset for all four roles: by name, or the provider switch
        (the first available preset of that provider). Same locks and refusals as set()."""
        if (name is None) == (provider is None):
            raise ValueError("Choose a preset name or a provider")
        presets = [p for p in self.document.get("presets", []) if isinstance(p, dict)]
        if name is not None:
            if not isinstance(name, str):
                raise ValueError("Unknown preset")
            preset = next((p for p in presets if p.get("name") == name), None)
            if preset is None:
                raise ValueError("Unknown preset")
            if not self._preset_ok(preset):
                raise ValueError("This preset names an option that is disabled or no longer offered")
        else:
            if provider not in PROVIDERS:
                raise ValueError("Unknown provider")
            preset = next((p for p in presets if p.get("provider") == provider and self._preset_ok(p)), None)
            if preset is None:
                raise ValueError("No validated preset is available for this provider yet")
        result = self.set(preset["selections"])
        result["applied_preset"] = preset["name"]
        return result

    def selected(self, role):
        with self.lock:
            return self.options[role][self.selections[role]]


def install_roles(g, root, memory, meter, scope, local, pricing):
    """Called at the end of f_bridge.install, after its native meter wrappers."""
    manager = RoleManager(g, root, memory, meter, pricing)
    # f_bridge.start() refuses these native providers before any reservation or request.
    g["F_DISABLED_PROVIDERS"] = dict(manager.disabled_providers)
    receipts_dirs = {"openrouter": Path(root) / "memory_f" / "openrouter_receipts",
                     ALIBABA: Path(root) / "memory_f" / "alibaba_receipts",
                     OLLAMA: Path(root) / "memory_f" / "ollama_receipts"}

    def before(request):
        metadata = request["metadata"]
        observer = g.get("F_OR_BEFORE_CALL")
        if observer:
            observer(request)
        config = request["config"]
        return meter.begin(config.get("transport") or "openrouter", config["model_id"],
                           metadata.get("phase", config["role"]),
                           turn_id=metadata.get("turn_id", ""), estimated_max_usd=request["estimated_max_usd"])

    def after(handle, receipt):
        # Settle billing first. An archive disk error must not erase a paid call.
        transport = receipt.get("transport") or "openrouter"
        # Model Studio reports no cost: its receipt carries the price-table amount. Ollama costs nothing.
        extra = ({"reported_cost_usd": receipt.get("cost_usd")} if transport == ALIBABA
                 else {"reported_cost_usd": 0.0} if transport == OLLAMA else {})
        meter.finish(handle, usage=receipt["raw_usage"] or None, status=receipt["status"],
                     actual_model=receipt.get("actual_model"), actual_provider=receipt.get("actual_provider"),
                     provider_request_id=receipt.get("request_id"), error=receipt.get("error_type"), **extra)
        _write(receipts_dirs.get(transport, receipts_dirs["openrouter"]) / (handle + ".json"), receipt)
        observer = g.get("F_OR_AFTER_CALL")
        if observer:
            observer(handle, receipt)

    def event_sink(event):
        observer = g.get("F_OR_EVENT")
        if observer:
            observer(event)

    def client(config=None):
        if config is not None and config.get("transport") == OLLAMA:
            return OllamaClient(g["http_requests"], before, after, base_url=config["base_url"], event_sink=event_sink,
                                api_key=config.get("api_key") or "")
        if config is not None and config.get("transport") == ALIBABA:
            key = g.get("DASHSCOPE_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if not key:
                raise RuntimeError("Alibaba Model Studio key unavailable (DASHSCOPE_API_KEY)")
            base = manager.alibaba["endpoint"]["base_url"]
            configured = os.getenv("DASHSCOPE_BASE_URL")
            if configured:
                try:
                    configured = alibaba_base_url(configured)
                except ValueError:
                    raise RuntimeError("DASHSCOPE_BASE_URL is not an allowed Model Studio endpoint") from None
                if configured != base:
                    raise RuntimeError("DASHSCOPE_BASE_URL differs from the reviewed Alibaba endpoint")
            return AlibabaClient(g["http_requests"], key, before, after, base_url=base, event_sink=event_sink)
        key = g.get("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
        return OpenRouterClient(g["http_requests"], key, before, after, event_sink=event_sink)

    def metadata(phase, parts=None):
        return {"phase": phase, "turn_id": (parts or {}).get("_f_turn_id", getattr(local, "turn_id", ""))}

    original_pick = g["_pick_stream"]
    def pick(parts, config, api_key=None):
        provider = str(config.get("provider") or "").lower()
        if provider in manager.disabled_providers:
            raise RuntimeError(provider + " calls are disabled: " + manager.disabled_providers[provider])
        if provider not in PROVIDERS:
            return original_pick(parts, config, api_key=api_key)
        chosen = config.get("_role_config") or config["_or_config"]
        phase = parts.get("_f_phase", chosen["role"])
        def speech_probe(raw):
            if phase == "surface":
                # C14: every non-empty reply line is spoken, including a think-mode
                # filler; status is informational only (C18). Wait for the reply
                # label so a half-streamed field line is not taken for speech.
                if not re.search(r"(?m)^reply[:：][ \t]*\S", raw):
                    return False
                parsed = g["parse_surface_response"](raw) if "parse_surface_response" in g else []
            elif phase == "deep":
                parsed = g["parse_response"](raw)
            else:
                return False
            return bool(parsed and parsed[0].get("speech", "").strip())
        return client(chosen).stream(chosen, g["_openai_messages"](parts, no_think=False, split_system=False),
                                     metadata=metadata(phase, parts), cancel=getattr(local, "cancel", None),
                                     speech_probe=speech_probe)
    g["_pick_stream"] = pick

    def extract(events, chunk_id):
        choice = manager.selected("memory_write")
        if "native_key" in choice:
            return manager.native_extract(events, chunk_id)
        rows = [{"source_id": e["id"], "date": e["time"], "source_kind": e.get("source_kind", "mixed"),
                 "speakers": {"user": e.get("user", ""), "assistant": e.get("assistant", "")}} for e in events]
        chunk = {"id": "chunk-"+chunk_id, "events": rows}
        messages = [{"role": "system", "content": __import__("f_extract").writer_prompt()},
                    {"role": "user", "content": json.dumps(rows, ensure_ascii=False, separators=(",", ":"))}]
        with scope("memory_write", getattr(local, "turn_id", "")):
            result = client(choice["config"]).complete(choice["config"], messages, metadata=metadata("memory_write"))
        facts, rejected = parse_isolated_facts(result["text"], chunk)
        if rejected:
            path = memory.folder / "quarantine" / (hashlib.sha256(chunk_id.encode()).hexdigest()+".json")
            _write(path, {"chunk_id": chunk_id, "source_ids": [e["id"] for e in events], "rejected": rejected})
        times = {e["id"]: e["time"] for e in events}
        for fact in facts:
            fact["conversation_date"] = fact["date"]
            fact["date"] = max(times[i] for i in fact["source_ids"])
        return {"facts": facts, "rejected": [{k: r.get(k) for k in ("index", "id", "reason")} for r in rejected]}
    memory.extract_fn = extract

    if manager.embedding:
        def embed(texts, task_type):
            if not texts:
                return []
            phase = "query_embedding" if task_type == "query" else "memory_embedding"
            output = []
            prefix = ("task: search result | query: " if task_type == "query" else
                      "title: Amika historical conversation | text: ")
            batches, batch, size = [], [], 0
            for text in texts:
                item = prefix + text
                encoded = len(json.dumps(item, ensure_ascii=False).encode("utf-8")) + 2
                if batch and (len(batch) == 16 or size + encoded > 100000):
                    batches.append(batch)
                    batch, size = [], 0
                batch.append(item)
                size += encoded
            if batch:
                batches.append(batch)
            for batch in batches:
                with scope(phase, getattr(local, "turn_id", "")):
                    result = client().embed(manager.embedding, batch, metadata=metadata(phase))
                output.extend(result["vectors"])
            return output
        memory.embed_fn = embed

    def role_complete(role, messages, *, phase=None, turn_id=""):
        """Metered non-streaming completion on the selected model of `role`
        (f_tools uses this for the tool phase; `phase` defaults to the role name)."""
        choice = manager.selected(role)
        if "config" not in choice:
            raise RuntimeError(f"role {role!r} is on a native provider; the tool phase needs an OpenRouter/Alibaba model")
        cfg = choice["config"]
        with scope(phase or role, turn_id or getattr(local, "turn_id", "")):
            return client(cfg).complete(cfg, messages, metadata=metadata(phase or role))
    g["F_ROLE_COMPLETE"] = role_complete

    _guard_model_routes(g, manager)
    g["F_ROLES"] = manager
    return manager


def _guard_model_routes(g, manager):
    """webui's legacy /api/{surface,deep,thinking}_model POSTs and /api/models list: never
    select or offer a disabled provider's model (Flask endpoint names; absent in tests)."""
    views = getattr(g.get("app"), "view_functions", None)
    if not isinstance(views, dict) or not manager.disabled_providers and not any(
            item.get("disabled") for choices in manager.options.values() for item in choices.values()):
        return
    for endpoint in MODEL_ROUTES:
        original = views.get(endpoint)
        if original is None:
            continue
        def guarded(*args, _original=original, **kwargs):
            request = g.get("request")
            if request is not None and getattr(request, "method", "GET") == "POST":
                value = request.get_json(silent=True) if hasattr(request, "get_json") else None
                model = value.get("model") if isinstance(value, dict) else None
                reason = manager.disabled_reason(model.strip() if isinstance(model, str) else model)
                if reason:
                    return g["jsonify"]({"ok": False, "error": str(model) + " is disabled: " + reason}), 400
            return _original(*args, **kwargs)
        views[endpoint] = guarded
    listing = views.get(MODEL_LIST_ROUTE)
    if listing is not None:
        def models():
            rows = [{"key": key, "name": entry.get("name", key), "description": entry.get("description", "")}
                    for key, entry in g["AVAILABLE_MODELS"].items() if not manager.disabled_reason(key)]
            return g["jsonify"]({"models": rows})
        views[MODEL_LIST_ROUTE] = models
