"""The fully-local option: Ollama as a role transport (f_ollama + the role manager's generated options)."""
import contextlib
import json
import re
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain"))
from f_memory import MemoryRuntime                                   # noqa: E402
from f_ollama import (OLLAMA, OllamaClient, discover, load_ollama_config, make_ollama_body,   # noqa: E402
                      ollama_role_config, save_ollama_config)
from f_roles import ROLES, NATIVE_KEYS, install_roles              # noqa: E402
from f_usage import UsageMeter                                      # noqa: E402

MODELS = [{"name": "qwen3:14b", "family": "qwen3", "parameter_size": "14.8B", "quantization": "Q4_K_M", "size_gb": 9.3,
           "context_length": 40960, "thinking": True, "vision": False},
          {"name": "qwen2.5:7b", "family": "qwen2", "parameter_size": "7.6B", "quantization": "Q4_K_M", "size_gb": 4.7,
           "context_length": 32768, "thinking": False, "vision": False}]


class Response:
    def __init__(self, lines=None, data=None, status=200):
        self.status_code, self._lines, self._data, self.encoding = status, lines or [], data, None
        self.closed = False
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http " + str(self.status_code))
    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)
    def json(self):
        return self._data
    def close(self):
        self.closed = True


class HTTP:
    def __init__(self):
        self.responses, self.calls = [], []
    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def ndjson(*rows):
    return [json.dumps(r) for r in rows]


class BodyTests(unittest.TestCase):
    def setUp(self):
        self.ollama = {"base_url": "http://127.0.0.1:11434", "num_ctx": 16384, "keep_alive": "30m", "models": MODELS}

    def test_role_config_and_body(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "deep", reasoning_enabled=False)
        self.assertEqual((cfg["transport"], cfg["provider_tag"], cfg["max_price"]["completion"]), (OLLAMA, OLLAMA, 0.0))
        self.assertTrue(cfg["think_supported"])
        self.assertFalse(cfg["reasoning_enabled"])
        body = make_ollama_body(cfg, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        self.assertEqual(body["options"], {"num_ctx": 16384, "num_predict": cfg["max_tokens"]})
        self.assertIs(body["think"], False)                  # a thinking model, switched off for the deep role
        self.assertNotIn("format", body)
        # a model without the thinking capability never gets the switch; the writer asks for JSON mode
        cfg2 = ollama_role_config(self.ollama, "qwen2.5:7b", "memory_write", reasoning_enabled=True)
        self.assertFalse(cfg2["reasoning_enabled"])
        body2 = make_ollama_body(cfg2, [{"role": "system", "content": "answer in JSON"}, {"role": "user", "content": "u"}])
        self.assertNotIn("think", body2)
        self.assertEqual(body2["format"], "json")
        # the context window never exceeds what the model supports, and never drops below the floor
        small = dict(self.ollama, models=[dict(MODELS[1], context_length=4096)])
        self.assertEqual(ollama_role_config(small, "qwen2.5:7b", "surface")["num_ctx"], 8192)
        with self.assertRaises(ValueError):
            ollama_role_config(self.ollama, "nope:1b", "deep")

    def test_config_file_roundtrip_and_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            self.assertEqual(load_ollama_config(root)["models"], [])
            self.assertEqual(load_ollama_config(root)["num_ctx"], 16384)
            save_ollama_config(root, {"base_url": "http://box:11434/", "num_ctx": 32768, "models": MODELS, "detected_at": "t"})
            doc = load_ollama_config(root)
            self.assertEqual((doc["base_url"], doc["num_ctx"], [m["name"] for m in doc["models"]]),
                             ("http://box:11434", 32768, ["qwen3:14b", "qwen2.5:7b"]))

    def test_discover_reads_tags_and_capabilities(self):
        http = HTTP()
        http.responses = [Response(data={"version": "0.14.1"}),
                          Response(data={"models": [{"name": "qwen3:14b", "size": 9_300_000_000,
                                                     "details": {"family": "qwen3", "parameter_size": "14.8B", "quantization_level": "Q4_K_M"}}]}),
                          Response(data={"capabilities": ["completion", "tools", "thinking"], "model_info": {"qwen3.context_length": 40960}})]
        found = discover(http, "http://127.0.0.1:11434")
        self.assertEqual(found["version"], "0.14.1")
        self.assertEqual(found["models"][0]["name"], "qwen3:14b")
        self.assertTrue(found["models"][0]["thinking"])
        self.assertEqual(found["models"][0]["context_length"], 40960)
        self.assertEqual(found["models"][0]["size_gb"], 9.3)


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.ollama = {"base_url": "http://127.0.0.1:11434", "num_ctx": 16384, "keep_alive": "30m", "models": MODELS}
        self.http = HTTP()
        self.handles, self.receipts = [], []
        self.client = OllamaClient(self.http, lambda request: self.handles.append(request) or "h1",
                                   lambda handle, receipt: self.receipts.append((handle, receipt)),
                                   base_url="http://127.0.0.1:11434")

    def test_stream_parses_ndjson_and_settles_zero_cost(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "deep", reasoning_enabled=False)
        self.http.responses = [Response(lines=ndjson(
            {"model": "qwen3:14b", "message": {"role": "assistant", "content": "[casual_happy]\"Hel"}, "done": False},
            {"model": "qwen3:14b", "message": {"role": "assistant", "content": "lo.\""}, "done": False},
            {"model": "qwen3:14b", "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
             "prompt_eval_count": 812, "eval_count": 9}))]
        text = "".join(self.client.stream(cfg, [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                                          metadata={"phase": "deep", "turn_id": "t1"}))
        self.assertEqual(text, '[casual_happy]"Hello."')
        url, kwargs = self.http.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/api/chat")
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertEqual(kwargs["json"]["options"]["num_ctx"], 16384)
        handle, receipt = self.receipts[0]
        self.assertEqual((handle, receipt["status"], receipt["cost_usd"]), ("h1", "complete", 0.0))
        self.assertEqual(receipt["usage"]["tokens"]["total_input_tokens"], 812)
        self.assertEqual(receipt["usage"]["tokens"]["output_tokens"], 9)
        self.assertEqual(receipt["actual_provider"], OLLAMA)
        self.assertTrue(receipt["request_id"].startswith("ollama-"))
        self.assertEqual(self.handles[0]["estimated_max_usd"], 0.00001)

    def test_error_line_is_an_error(self):
        cfg = ollama_role_config(self.ollama, "qwen2.5:7b", "surface")
        self.http.responses = [Response(lines=ndjson({"error": "model not found"}))]
        with self.assertRaises(Exception):
            list(self.client.stream(cfg, [{"role": "user", "content": "u"}], metadata={"phase": "surface"}))
        self.assertEqual(self.receipts[0][1]["status"], "unknown")

    def test_complete_reads_the_single_message(self):
        cfg = ollama_role_config(self.ollama, "qwen2.5:7b", "memory_write")
        self.http.responses = [Response(data={"model": "qwen2.5:7b", "message": {"role": "assistant", "content": "{\"facts\": []}"},
                                              "done": True, "done_reason": "stop", "prompt_eval_count": 50, "eval_count": 5})]
        result = self.client.complete(cfg, [{"role": "system", "content": "reply in JSON"}, {"role": "user", "content": "u"}],
                                      metadata={"phase": "memory_write"})
        self.assertEqual(result["text"], "{\"facts\": []}")
        self.assertEqual(self.http.calls[0][1]["json"]["format"], "json")
        self.assertEqual(self.receipts[0][1]["status"], "complete")


class RoleManagerTests(unittest.TestCase):
    """The role manager offers one option per detected model and role, and survives a re-detect."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        shutil.copyfile(ROOT / "config/openrouter_models.json", self.root / "config/openrouter_models.json")
        self.document = {"version": 1, "options": {}, "selections": {}, "presets": [
            {"name": "cloud", "provider": "openrouter", "selections": {role: "or:" + role for role in ROLES}}], "fallback_preset": "cloud"}
        for role in ROLES:
            self.document["options"][role] = [
                {"key": "native:" + NATIVE_KEYS[role], "native_key": NATIVE_KEYS[role], "name": "Native"},
                {"key": "or:" + role, "model_id": "deepseek/deepseek-v4-flash-0731", "name": "Reviewed " + role}]
            self.document["selections"][role] = "or:" + role
        (self.root / "config/llm_roles.json").write_text(json.dumps(self.document), encoding="utf-8")
        save_ollama_config(self.root, {"base_url": "http://127.0.0.1:11434", "num_ctx": 16384, "models": MODELS, "detected_at": "t"})
        self.memory = MemoryRuntime(self.root / "memory_f/live", lambda events, chunk: {"native": True},
                                    lambda texts, task: [[1.] + [0.] * 767 for t in texts])
        self.meter = UsageMeter(self.root / "memory_f/usage.sqlite", {"models": {}}, budget_usd=1)
        self.local = threading.local()
        self.g = {"AVAILABLE_MODELS": {key: {"name": key, "provider": "native", "model": key} for key in NATIVE_KEYS.values()},
                  "_ws_active_lock": threading.RLock(), "_ws_active": {}, "_thinking_inflight": 0,
                  "http_requests": HTTP(), "OPENROUTER_KEY": "fake-key",
                  "_pick_stream": lambda p, c, **kw: iter(["native"]),
                  "_openai_messages": lambda p, **kw: [{"role": "system", "content": p.get("system", "")},
                                                      {"role": "user", "content": p["user_content"]}]}
        @contextlib.contextmanager
        def scope(phase, turn_id="", **kwargs):
            yield
        self.manager = install_roles(self.g, self.root, self.memory, self.meter, scope, self.local, {})

    def tearDown(self):
        self.memory.close()
        self.tmp.cleanup()

    def test_a_surface_turn_on_ollama_reaches_its_transport(self):
        """Fully Local: her surface turn goes through _pick_stream to the Ollama transport, as deep does. webui used to
        hand an Ollama surface to _stream_openai, which the metered bridge refuses ("Build F supports its configured
        Anthropic/xAI response models."): no Fully Local turn ever got a reply."""
        http = self.g["http_requests"]
        http.responses = [Response(lines=ndjson(
            {"model": "qwen2.5:7b", "message": {"role": "assistant", "content": "reply: hi"}, "done": False},
            {"model": "qwen2.5:7b", "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop",
             "prompt_eval_count": 12, "eval_count": 3}))]
        parts = {"system": "s", "user_content": "u", "_f_phase": "surface"}
        text = "".join(self.g["_pick_stream"](parts, self.g["AVAILABLE_MODELS"]["surface:ollama:qwen2.5:7b"]))
        self.assertEqual(text, "reply: hi")
        self.assertEqual(http.calls[-1][0], "http://127.0.0.1:11434/api/chat")
        # and her surface turn uses that dispatch, whatever the provider
        source = (ROOT / "brain" / "webui.py").read_text(encoding="utf-8")
        run = source[source.index("def _ws_run_generation("):]
        run = run[:run.index("\ndef ", 1)]
        self.assertEqual(re.findall(r"_surface_stream = (.+)", run), ["_pick_stream(parts, model_config)"])

    def test_options_selection_pricing_and_reload(self):
        for role in ROLES:
            self.assertIn(f"{role}:ollama:qwen3:14b", self.manager.options[role])
            self.assertEqual(self.manager.options[role][f"{role}:ollama:qwen2.5:7b"]["transport"], OLLAMA)
        # thinking only where latency does not matter, and only on a model that can
        self.assertFalse(self.manager.options["deep"]["deep:ollama:qwen3:14b"]["config"]["reasoning_enabled"])
        self.assertTrue(self.manager.options["reflection"]["reflection:ollama:qwen3:14b"]["config"]["reasoning_enabled"])
        self.assertEqual(self.meter.pricing["models"]["ollama:qwen3:14b"]["output"], 0.0)
        # the legacy console lists them too (non-writer roles), tagged with the transport
        self.assertEqual(self.g["AVAILABLE_MODELS"]["deep:ollama:qwen3:14b"]["provider"], OLLAMA)
        self.assertNotIn("memory_write:ollama:qwen3:14b", self.g["AVAILABLE_MODELS"])
        result = self.manager.set({role: f"{role}:ollama:qwen3:14b" for role in ROLES})
        self.assertEqual(result["transport"], OLLAMA)
        self.assertEqual(result["ollama"]["models"], ["qwen3:14b", "qwen2.5:7b"])
        self.assertEqual(json.loads((self.root / "config/llm_roles.json").read_text())["selections"]["deep"], "deep:ollama:qwen3:14b")
        # the model is removed from the server: the next detect drops the option and the role falls back to the preset
        save_ollama_config(self.root, {"base_url": "http://127.0.0.1:11434", "num_ctx": 16384, "models": MODELS[1:], "detected_at": "t2"})
        result = self.manager.reload_ollama()
        self.assertNotIn("deep:ollama:qwen3:14b", self.manager.options["deep"])
        self.assertNotIn("deep:ollama:qwen3:14b", self.g["AVAILABLE_MODELS"])
        self.assertEqual(result["selections"]["deep"], "or:deep")
        self.assertEqual(json.loads((self.root / "config/llm_roles.json").read_text())["selections"]["deep"], "or:deep")
        self.assertEqual(self.g["DEFAULT_DEEP_MODEL"], "or:deep")


if __name__ == "__main__":
    unittest.main()


# appended to tests/test_f_ollama.py — the OpenAI dialect on the Ollama panel (fork)
class OpenAIDialectTests(unittest.TestCase):
    """An OpenAI-dialect server wired through the Ollama panel: same rows and roles, wire is OpenAI."""

    def setUp(self):
        self.ollama = {"base_url": "http://127.0.0.1:11434", "num_ctx": 16384, "keep_alive": "30m",
                       "dialect": "openai", "api_key": "k", "models": MODELS}
        self.http = HTTP()
        self.handles, self.receipts = [], []
        self.client = OllamaClient(self.http, lambda request: self.handles.append(request) or "h1",
                                   lambda handle, receipt: self.receipts.append((handle, receipt)),
                                   base_url="http://127.0.0.1:11434", api_key="k", dialect="openai")

    def test_config_roundtrip_keeps_dialect_and_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config").mkdir()
            save_ollama_config(root, dict(self.ollama, detected_at="t"))
            doc = load_ollama_config(root)
            self.assertEqual((doc["dialect"], doc["api_key"]), ("openai", "k"))

    def test_role_config_advertises_structured_outputs(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "deep", reasoning_enabled=False)
        self.assertTrue(cfg["structured_outputs_advertised"])
        plain = ollama_role_config(dict(self.ollama, dialect="ollama"), "qwen3:14b", "deep")
        self.assertFalse(plain["structured_outputs_advertised"])

    def test_body_is_standard_openai(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "deep", reasoning_enabled=False)
        body = make_ollama_body(cfg, [{"role": "user", "content": "u"}])
        self.assertEqual(body["max_tokens"], cfg["max_tokens"])      # top level, not options.num_predict
        self.assertNotIn("options", body)
        self.assertNotIn("think", body)
        self.assertNotIn("response_format", body)                    # plain roles ask for nothing special
        wcfg = ollama_role_config(self.ollama, "qwen3:14b", "memory_write", reasoning_enabled=True)
        wbody = make_ollama_body(wcfg, [{"role": "user", "content": "answer in JSON"}, {"role": "user", "content": "u"}])
        self.assertEqual(wbody["response_format"], {"type": "json_object"})

    def test_discover_reads_the_openai_model_list(self):
        saved = [{"name": "glm-4.7-flash", "thinking": True, "context_length": 131072, "vision": False}]
        self.http.responses = [Response(data={"object": "list", "data": [
            {"id": "glm-4.7-flash",
             "meta": {"name": "strata", "n_ctx": 262144},
             "architecture": {"input_modalities": ["text", "image"]}}]})]
        found = discover(self.http, "http://127.0.0.1:11434", api_key="k", dialect="openai", keep=saved)
        self.assertEqual(found["models"][0]["name"], "glm-4.7-flash")
        self.assertEqual(found["models"][0]["context_length"], 262144)
        self.assertTrue(found["models"][0]["vision"])
        self.assertTrue(found["models"][0]["thinking"])              # the list has no thinking flag; kept from saved
        url, kwargs = self.http.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/v1/models")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")

    def test_stream_parses_sse_and_settles_zero_cost(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "deep", reasoning_enabled=False)
        self.http.responses = [Response(lines=[
            "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": "hi"}}]}),
            "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": " there"}}]}),
            "data: " + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}),
            "data: [DONE]"])]
        text = "".join(self.client.stream(cfg, [{"role": "user", "content": "u"}], metadata={"phase": "deep", "turn_id": "t1"}))
        self.assertEqual(text, "hi there")
        url, kwargs = self.http.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(kwargs["json"]["stream_options"], {"include_usage": True})
        handle, receipt = self.receipts[0]
        self.assertEqual((receipt["status"], receipt["cost_usd"]), ("complete", 0.0))

    def test_complete_reads_the_openai_message(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "deep", reasoning_enabled=False)
        self.http.responses = [Response(data={
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})]
        result = self.client.complete(cfg, [{"role": "user", "content": "ping"}], metadata={"phase": "deep", "turn_id": "t2"})
        self.assertEqual(result["text"], "pong")
        url, kwargs = self.http.calls[0]
        self.assertEqual(url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer k")
        self.assertEqual(self.receipts[0][1]["status"], "complete")


class OpenAIDialectReasoningTests(unittest.TestCase):
    """The per-role thinking switch travels on the OpenAI wire when the model can think."""

    def setUp(self):
        thinking_rows = [{"name": "qwen3:14b", "family": "", "parameter_size": "", "quantization": "",
                          "size_gb": None, "context_length": 262144, "thinking": True, "vision": True}]
        self.ollama = {"base_url": "http://127.0.0.1:11434", "num_ctx": 16384, "keep_alive": "30m",
                       "dialect": "openai", "api_key": "k", "models": thinking_rows}

    def test_fast_roles_ask_for_thinking_off(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "surface", reasoning_enabled=False)
        body = make_ollama_body(cfg, [{"role": "user", "content": "u"}])
        self.assertEqual(body["reasoning"], {"enabled": False})

    def test_reflection_role_asks_for_thinking_on(self):
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "reflection", reasoning_enabled=True)
        body = make_ollama_body(cfg, [{"role": "user", "content": "u"}])
        self.assertEqual(body["reasoning"], {"enabled": True})

    def test_no_switch_when_the_model_cannot_think(self):
        self.ollama["models"][0]["thinking"] = False
        cfg = ollama_role_config(self.ollama, "qwen3:14b", "surface", reasoning_enabled=True)
        body = make_ollama_body(cfg, [{"role": "user", "content": "u"}])
        self.assertNotIn("reasoning", body)

    def test_native_dialect_still_uses_the_think_field(self):
        cfg = ollama_role_config(dict(self.ollama, dialect="ollama"), "qwen3:14b", "surface", reasoning_enabled=False)
        body = make_ollama_body(cfg, [{"role": "user", "content": "u"}])
        self.assertEqual(body["think"], False)
        self.assertNotIn("reasoning", body)
