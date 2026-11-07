import os
import re
import time
import asyncio
import logging
import warnings
from io import BytesIO
from enum import Enum
from collections import deque
import base64
import uuid
from datetime import datetime
from typing import List, Optional, Tuple
from pathlib import Path

warnings.filterwarnings("ignore", message=".*ffmpeg.*")

import json
import aiohttp
import torch
import numpy as np
import soundfile as sf
import torchaudio
import socketio
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import Annotated

from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from sentence_splitter import StreamingSplitter
from hane_parser import SurfaceSpeechExtractor, DeepSpeechExtractor
import stt_carry
from tts_emotion import prefix_emotion
from tts_sfx import parse_sfx_delta, resolve_sfx, list_files as sfx_list_files
from tts_pause import pause_after
import voice_adapter
from silero_vad import load_silero_vad

import sys
logger = logging.getLogger("yumeha")

# ----------------------------- word replacer ------------------------------ #
# Standalone, clone-owned pronunciation and STT correction map.
_WORDMAP_PATH = str(Path(__file__).resolve().parent.parent / "config" / "word_replace.json")
_wordmap = {"mtime": None, "stt": [], "tts": []}


def _wordmap_compile(rows):
    out = []
    for r in rows or []:
        try:
            frm = str(r.get("from", ""))
            if not frm:
                continue
            flags = re.IGNORECASE if r.get("ci") else 0
            # Boundary the ASCII-alnum edges so a short romaji word (uni, osaka, hai) doesn't
            # fire INSIDE a longer one (university, osakana, hair). Patterns whose edge is
            # punctuation / kana still match as plain substrings (e.g. an em-dash rule).
            left = r"(?<![A-Za-z0-9])" if (frm[:1].isascii() and frm[:1].isalnum()) else ""
            right = r"(?![A-Za-z0-9])" if (frm[-1:].isascii() and frm[-1:].isalnum()) else ""
            out.append((len(frm), re.compile(left + re.escape(frm) + right, flags), str(r.get("to", ""))))
        except Exception:
            continue
    out.sort(key=lambda x: x[0], reverse=True)          # longest 'from' first
    return [(pat, to) for _, pat, to in out]


def _wordmap_refresh():
    try:
        mt = os.path.getmtime(_WORDMAP_PATH)
    except OSError:
        if _wordmap["stt"] or _wordmap["tts"]:
            _wordmap["stt"], _wordmap["tts"], _wordmap["mtime"] = [], [], None
        return
    if mt == _wordmap["mtime"]:
        return
    try:
        with open(_WORDMAP_PATH, encoding="utf-8") as f:
            data = json.load(f) or {}
        _wordmap["stt"] = _wordmap_compile(data.get("stt_to_llm"))
        _wordmap["tts"] = _wordmap_compile(data.get("llm_to_tts"))
        _wordmap["mtime"] = mt
        logger.info(f"[wordmap] {len(_wordmap['stt'])} stt + {len(_wordmap['tts'])} tts rules")
    except Exception as e:
        logger.warning(f"[wordmap] load failed: {e}")


def apply_wordmap(text, which):
    """which = 'stt' (transcript -> Hane) or 'tts' (reply -> Freya)."""
    if not text:
        return text
    _wordmap_refresh()
    for pat, to in _wordmap.get(which) or []:
        text = pat.sub(lambda m, _t=to: _t, text)      # literal replacement (no group refs)
    return text
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)
    logger.propagate = False

TARGET_FS = 16000

# TTS 播放队列里的"声效项"哨兵：队列项是 (key, value)；普通台词 key 是 emotion_key
# 字符串，声效项 key 用这个唯一对象，value=(声效名, 前一句的 emotion_key)。
_SFX_QUEUE = object()


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


DEVICE = os.getenv("YUMEHA_DEVICE", "cuda:0")
MODEL_DIR = os.getenv(
    "YUMEHA_MODEL",
    os.path.join(os.path.dirname(__file__), "models"),
)

logger.info(f"Loading Yumeha model from {MODEL_DIR} on {DEVICE} ...")
load_start = time.time()
m, kwargs = SenseVoiceSmall.from_pretrained(model=MODEL_DIR, device=DEVICE)
m.eval()
load_elapsed = time.time() - load_start
logger.info(f"Model loaded in {load_elapsed:.2f}s")

logger.info("Loading Silero VAD ...")
_vad_load_start = time.time()
vad_model = load_silero_vad()
logger.info(f"Silero VAD loaded in {time.time() - _vad_load_start:.2f}s")

regex = r"<\|.*?\|>"

# Clone identity (BUILD_SPEC A / CONTRACT §3.12): reported by /health and required from the
# brain's /api/build_f/status, so this speech layer never proxies the Build F control's brain.
APP_NAME = "OpenAmika"

app = FastAPI(
    title=f"{APP_NAME} Chat and Speech",
    description="梦羽 — 极速多语言语音识别服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# OpenAmika three-page UI (ui/) + user-supplied Live2D models (avatar/models/).
UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

import f_auth  # fork: browser login gate (speech/f_auth.py)
f_auth.install(app)


@app.middleware("http")
async def _revalidate_page_assets(request: Request, call_next):
    """The pages' scripts and styles must always be revalidated: without a Cache-Control header browsers apply
    heuristic caching and keep serving yesterday's ui/*.js after an update (a reload is cheap, everything is local)."""
    response = await call_next(request)
    if request.url.path.startswith(("/ui/", "/static/")) and "cache-control" not in response.headers:
        response.headers["Cache-Control"] = "no-cache"
    return response
AVATAR_MODELS_DIR = Path(__file__).resolve().parent.parent / "avatar" / "models"


_UI_ASSET_REF = re.compile(r'((?:src|href)=")(/ui/[^"?#]+)(")')


def _versioned_page(path: Path) -> HTMLResponse:
    """A page with every /ui/ script and stylesheet URL stamped with that file's modification time (?v=…).
    The page itself is never cached; a stamped asset URL changes the moment its file does, so the browser can never
    pair a new page with an old script — which is exactly what broke the Tools page on 2026-10-25 (a copy of
    tools.js the browser considered fresh was never re-requested, whatever its Cache-Control said)."""
    def stamp(match):
        asset = UI_DIR / match.group(2)[len("/ui/"):]
        try:
            version = format(asset.stat().st_mtime_ns, "x")
        except OSError:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}?v={version}{match.group(3)}"
    html = _UI_ASSET_REF.sub(stamp, path.read_text(encoding="utf-8"))
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


def _ui_page(name: str):
    path = UI_DIR / name
    if path.is_file():
        return _versioned_page(path)
    return None


@app.get("/", response_class=HTMLResponse)
async def root():
    if _ui_page("use.html") is not None:
        return RedirectResponse(url="/use", status_code=307)
    return HTMLResponse("<h1>OpenAmika</h1><p>ui/use.html not found</p>")


@app.get("/use", response_class=HTMLResponse)
async def ui_use():
    page = _ui_page("use.html")
    if page is None:
        raise HTTPException(status_code=404, detail="ui/use.html not found")
    return page


@app.get("/setup", response_class=HTMLResponse)
async def ui_setup():
    page = _ui_page("setup.html")
    if page is None:
        raise HTTPException(status_code=404, detail="ui/setup.html not found")
    return page


@app.get("/tools", response_class=HTMLResponse)
async def ui_tools():
    page = _ui_page("tools.html")
    if page is None:
        raise HTTPException(status_code=404, detail="ui/tools.html not found")
    return page


@app.get("/about", response_class=HTMLResponse)
async def ui_about():
    """About MekaHime, supporting the project (the donate card) and the licenses."""
    page = _ui_page("about.html")
    if page is None:
        raise HTTPException(status_code=404, detail="ui/about.html not found")
    return page



@app.get("/sfx_manifest")
async def sfx_manifest():
    """前端启动时拉一次，预加载并解码全部声效文件，避免首次播放卡顿/出缝。"""
    return {"dir": "/static/sfx/", "files": sfx_list_files()}


@app.get("/flow", response_class=HTMLResponse)
async def build_f_flow():
    path = STATIC_DIR / "flow.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Build F flow diagram unavailable")
    return _versioned_page(path)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "app": APP_NAME,
        "model": MODEL_DIR,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.post("/api/v1/asr")
async def asr(
    files: Annotated[List[UploadFile], File(description="音频文件")],
    keys: Annotated[Optional[str], Form(description="文件名，逗号分隔")] = None,
    lang: Annotated[Language, Form(description="语言")] = "auto",
    use_itn: Annotated[bool, Form(description="逆文本正则化")] = True,
):
    if not files:
        raise HTTPException(status_code=400, detail="未提供音频文件")

    request_start = time.time()
    audios = []
    audio_lengths = []

    for file in files:
        try:
            file_bytes = await file.read()
            file_io = BytesIO(file_bytes)
            data_np, audio_fs = sf.read(file_io, dtype="float32")
            audio_tensor = torch.from_numpy(data_np)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            else:
                audio_tensor = audio_tensor.T

            if audio_fs != TARGET_FS:
                resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
                audio_tensor = resampler(audio_tensor)

            audio_tensor = audio_tensor.mean(0)
            audios.append(audio_tensor)
            audio_lengths.append(len(audio_tensor) / TARGET_FS)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"处理音频文件 {file.filename} 失败: {str(e)}")

    if lang == "":
        lang = "auto"

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    inference_start = time.time()
    try:
        res = m.inference(
            data_in=audios,
            language=lang,
            use_itn=use_itn,
            ban_emo_unk=False,
            key=key,
            fs=TARGET_FS,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

    inference_elapsed = time.time() - inference_start
    total_elapsed = time.time() - request_start

    if len(res) == 0:
        return {"result": [], "meta": {"inference_time": inference_elapsed, "total_time": total_elapsed}}

    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])

    logger.info(
        f"ASR completed: {len(files)} file(s), "
        f"total_audio={sum(audio_lengths):.1f}s, "
        f"inference={inference_elapsed:.3f}s, "
        f"total={total_elapsed:.3f}s"
    )

    return {
        "result": res[0],
        "meta": {
            "inference_time": round(inference_elapsed, 3),
            "total_time": round(total_elapsed, 3),
            "audio_duration": round(sum(audio_lengths), 2),
        },
    }


# ============================================================
# 对话状态机：接收前端流式 PCM，做 VAD + ASR + 推测/提交
# ============================================================

def _now_ms() -> int:
    return int(time.time() * 1000)


# STT 误触过滤：部分无意义的超短识别结果（单字 / 两三个假名 / 单个英文短词）
# 经常由咳嗽、叹气、环境噪音触发。这类结果发给 Hane 会让 AI 乱猜意图，成本也不划算。
# 规则（把输入去掉所有非 alnum 字符，只看"有效字符"）：
#   - 纯汉字 <=1 个字                      -> trivial（"啊"、"嗯"）
#   - 纯日语假名 <=3 个字                  -> trivial（"あ"、"あい"、"どうも"）
#   - 含汉字+假名混合 <=3 个字             -> trivial
#   - 纯英文/数字 + 只有 1 个单词 + 字母 <=4 -> trivial（"OK"、"hi"、"yeah"）
#   - 英文 + 中文/日文混合                 -> 不 trivial（混说通常带真实意图）
_WORD_RE = re.compile(r"[A-Za-z']+")

def _is_trivial_utterance(text: str) -> bool:
    if not text:
        return True
    meaningful = [c for c in text if c.isalnum()]
    if not meaningful:
        return True
    cjk = sum(1 for c in meaningful if '\u4e00' <= c <= '\u9fff')
    kana = sum(1 for c in meaningful if (
        '\u3040' <= c <= '\u30ff'
        or '\u31f0' <= c <= '\u31ff'
        or '\uff65' <= c <= '\uff9f'
    ))
    other = len(meaningful) - cjk - kana
    if other > 0 and cjk == 0 and kana == 0:
        # 纯英文/数字：按"单个短词 <=4 字母"判 trivial
        words = _WORD_RE.findall(text)
        # 去撇号后计字母；例如 "I'm" 算 2 字母
        letters_in_first = len(words[0].replace("'", "")) if words else 0
        # 必须真是"只有 1 个单词"；有额外数字或其它单词就不算 trivial
        return len(words) == 1 and letters_in_first <= 4
    if other > 0:
        # 英文 + CJK/假名 混合（如 "OK 好的"、"はいOK"）—— 通常带实意，保留
        return False
    if kana == 0:
        return cjk <= 1
    if cjk == 0:
        return kana <= 3
    return (cjk + kana) <= 3


def _asr_sync(audio_np: np.ndarray) -> Tuple[str, str]:
    """同步运行 SenseVoice 推理。返回 (clean_text, raw_text)。
    raw_text 包含 <|HAPPY|><|zh|> 这样的 tags，供前端解析情感/语言/事件。
    """
    if audio_np.size == 0:
        return "", ""
    tensor = torch.from_numpy(audio_np)
    res = m.inference(
        data_in=[tensor],
        language="auto",
        use_itn=True,
        ban_emo_unk=False,
        key=["dlg"],
        fs=TARGET_FS,
        **kwargs,
    )
    if not res or not res[0]:
        return "", ""
    raw = res[0][0]["text"]
    clean = re.sub(regex, "", raw, 0, re.MULTILINE).strip()
    return clean, raw


async def _asr_async(audio_np: np.ndarray) -> Tuple[str, str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _asr_sync, audio_np)


# ---------------------------- Hane（AI Core）对接 ---------------------------- #

HANE_URL = os.getenv("HANE_URL", "http://127.0.0.1:5202")   # clone brain port (Build F control is 5201)
_dialog_sessions = set()
# Shared admission gate for memory wipe, model changes, and meter reset.
# Kept under the existing name so every input path uses the same gate. Only
# build_f_wipe retires sessions; other controls leave conversation state intact.
_wipe_in_progress = False


def _build_f_session_busy(session):
    if getattr(session, "_wipe_retired", False):
        return False
    pending = session._pending_partial
    return bool(
        getattr(session, "_connecting", False) or getattr(session, "_input_busy", False)
        or session._state in ("SPEAKING", "SHORT_PAUSE")
        or (pending is not None and not pending.done())
        or (session._committed_turn_id is not None and not session._ai_end_received)
        or session._tts_playing is not None or session._tts_inflight is not None
        or session._tts_sentence_queue or session._tts_pending_play
    )


async def _retire_wiped_sessions(backup_id=None):
    for session in list(_dialog_sessions):
        session._wipe_retired = True
        session._pending_context = ""
        session._context_revision += 1
        session._contexts_sent.clear()
        session._committed_turn_id = None
        session._current_sent_id = None
        session._pending_chunks.clear()
        session._buffered_proactive.clear()
        session._leftover.clear()
        session._carry_text = ""
        session._last_commit_text = ""
        session._carry_turn_ids.clear()
        session._carry_error_turns.clear()
        session._carry_snapshot = None
        session._protect_inflight = None
        session._reset_utterance()
        session._reset_tts_for_new_turn()
        await session._send({"type": "memory_wiped" if backup_id else "memory_reset_uncertain", "backup_id": backup_id})
        await asyncio.gather(session.hane.disconnect(), session.tts.disconnect(), return_exceptions=True)
        try:
            await session.ws.close(code=1000)
        except Exception:
            pass


@app.post("/api/build_f/wipe")
async def build_f_wipe(request: Request):
    """Confirmed same-origin memory reset under the shared service-control gate."""
    global _wipe_in_progress
    # Uvicorn applies forwarded scheme only for its configured trusted proxy
    # addresses (loopback by default). Never trust raw forwarded headers here.
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(status_code=403, detail="Memory reset requires the same-origin app page.")
    if request.headers.get("x-build-f-action") != "wipe-memory":
        raise HTTPException(status_code=403, detail="Memory reset confirmation header missing.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="Memory reset requires JSON.")
    if _wipe_in_progress or any(_build_f_session_busy(s) for s in list(_dialog_sessions)):
        raise HTTPException(status_code=409, detail="Amika is busy. Wait for conversation, speech, memory updates and service controls to finish.")
    # No await between the busy check and admission. New dialog inputs and
    # other controls cannot race the body read or the backend request.
    _wipe_in_progress = True
    dispatched = False
    try:
        body = await request.body()
        if len(body) > 1024:
            raise HTTPException(status_code=400, detail="Invalid memory reset request.")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise HTTPException(status_code=400, detail="Invalid memory reset request.") from None
        if not isinstance(payload, dict) or payload.get("confirmation") != "WIPE":
            raise HTTPException(status_code=400, detail="Type WIPE to confirm the memory reset.")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
            dispatched = True
            async with client.post(
                HANE_URL.rstrip("/") + "/api/build_f/wipe",
                json={"confirmation": "WIPE"},
                headers={"X-Build-F-Action": "wipe-memory"},
                allow_redirects=False,
            ) as response:
                if response.status in (400, 403, 409):
                    detail = "Amika is busy. Wait for conversation, speech and memory updates to finish, then try again." if response.status == 409 else "Memory reset was refused by the brain."
                    raise HTTPException(status_code=response.status, detail=detail)
                if response.status != 200:
                    raise ValueError("Brain reset outcome unconfirmed")
                result = await response.json()
        if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("backup_id"), str) or not result["backup_id"]:
            raise ValueError("Brain reset outcome unconfirmed")
        await _retire_wiped_sessions(result["backup_id"])
        return {key: result.get(key) for key in ("ok", "backup_id", "memory", "usage")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        if dispatched:
            # The brain might have committed a reset before the connection failed.
            # Keep the visible transcript, but never reuse old queued context.
            await _retire_wiped_sessions()
        raise HTTPException(status_code=503, detail="Memory reset result is unconfirmed. Chat was disconnected; check memory status before retrying.") from None
    finally:
        _wipe_in_progress = False


@app.post("/api/build_f/reset")
async def build_f_reset(request: Request):
    """Confirmed same-origin "reset everything": the brain wipes her memory (backup kept), puts the shipped
    persona files back and clears your names/city, so the first-run wizard plays again. Same gate and the
    same session retirement as the memory wipe (build_f_wipe stays self-contained for its tests)."""
    global _wipe_in_progress
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(status_code=403, detail="Reset requires the same-origin app page.")
    if request.headers.get("x-build-f-action") != "reset-all":
        raise HTTPException(status_code=403, detail="Reset confirmation header missing.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="Reset requires JSON.")
    if _wipe_in_progress or any(_build_f_session_busy(s) for s in list(_dialog_sessions)):
        raise HTTPException(status_code=409, detail="Amika is busy. Wait for conversation, speech, memory updates and service controls to finish.")
    _wipe_in_progress = True
    dispatched = False
    try:
        body = await request.body()
        if len(body) > 1024:
            raise HTTPException(status_code=400, detail="Invalid reset request.")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise HTTPException(status_code=400, detail="Invalid reset request.") from None
        if not isinstance(payload, dict) or payload.get("confirmation") != "RESET":
            raise HTTPException(status_code=400, detail="Type RESET to confirm the full reset.")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as client:
            dispatched = True
            async with client.post(
                HANE_URL.rstrip("/") + "/api/build_f/reset",
                json={"confirmation": "RESET"},
                headers={"X-Build-F-Action": "reset-all"},
                allow_redirects=False,
            ) as response:
                result = await response.json() if response.content_type == "application/json" else None
                if response.status in (400, 403, 409):
                    detail = (result.get("error") if isinstance(result, dict) and result.get("error")
                              else "Reset was refused by the brain.")
                    raise HTTPException(status_code=response.status, detail=detail)
                if response.status != 200:
                    raise ValueError("Brain reset outcome unconfirmed")
        if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("backup_id"), str) or not result["backup_id"]:
            raise ValueError("Brain reset outcome unconfirmed")
        await _retire_wiped_sessions(result["backup_id"])
        return {key: result.get(key) for key in ("ok", "backup_id", "restored", "user_name", "persona_name",
                                                 "timezone", "city", "onboarded", "memory", "usage")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        if dispatched:
            await _retire_wiped_sessions()
        raise HTTPException(status_code=503, detail="Reset result is unconfirmed. Chat was disconnected; check the Setup page before retrying.") from None
    finally:
        _wipe_in_progress = False


@app.get("/api/build_f/status")
async def build_f_status():
    """Read one fixed, local brain status route; never starts an AI request."""
    try:
        timeout = aiohttp.ClientTimeout(total=4)
        async with aiohttp.ClientSession(timeout=timeout) as client:
            async with client.get(
                HANE_URL.rstrip("/") + "/api/build_f/status",
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Build F status unavailable")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("app") != APP_NAME:
            raise HTTPException(status_code=502, detail="Unexpected Build F status response")
        # Do not expose arbitrary backend top-level fields or proxy arbitrary paths.
        # template_check: the brain's boot-time template self-check (CONTRACT §5).
        # "usage" is forwarded whole, so the cost-settlement fields (settled_extra_usd,
        # settled_calls, pending_settlement_calls, no_id_calls, estimated_unrecorded_usd,
        # total_estimated_usd, key_usage_*, drained_calls, drained_usd, settlement_worker,
        # drain_worker; also inside usage.display) reach the page unchanged.
        return {key: payload.get(key) for key in ("app", "usage", "memory", "pricing", "template_check")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Build F status unavailable") from None


@app.get("/api/build_f/models")
async def build_f_models():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/models", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Model choices are unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected model choices")
        # (the field list is spelled out here: tests load this function on its own, without module constants)
        return {key: payload.get(key) for key in ("ok", "selections", "options", "presets", "benchmark", "transport",
                                                  "ollama", "selection_fallback")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Model choices are unavailable.") from None


# ── the fully-local option: Ollama models, detected on the brain (config/ollama.json) ──
_OLLAMA_FIELDS = ("ok", "base_url", "num_ctx", "keep_alive", "detected_at", "version", "live_version", "models",
                  "reachable", "error", "in_use", "api_key_set", "dialect")


@app.get("/api/build_f/ollama")
async def build_f_ollama():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/ollama", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Ollama status is unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected Ollama status")
        return {key: payload.get(key) for key in _OLLAMA_FIELDS}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Ollama status is unavailable (is the brain running?).") from None


@app.post("/api/build_f/ollama/detect")
async def build_f_ollama_detect(request: Request):
    """Probe the Ollama server and record its models; touches no conversation state, so it may run while she talks."""
    return await _build_f_control(request, "/api/build_f/ollama/detect", "ollama-detect", _OLLAMA_FIELDS,
                                  check_busy=False, timeout=60)


@app.post("/api/build_f/ollama/use")
async def build_f_ollama_use(request: Request):
    return await _build_f_control(request, "/api/build_f/ollama/use", "ollama-use", ("ok", "selections", "transport", "in_use"))


@app.post("/api/build_f/preset")
async def build_f_preset(request: Request):
    """Apply a tested preset by name, or the first validated preset of a provider ({"provider": "openrouter"})."""
    return await _build_f_control(request, "/api/build_f/preset", "apply-preset",
                                  ("ok", "selections", "options", "presets", "benchmark", "transport", "ollama",
                                   "selection_fallback", "applied_preset"))


async def _build_f_control(request, route, action, allowed_fields, max_body=8192, check_busy=True, base=None,
                           timeout=10, precheck=None):
    """base: the service the control goes to (the brain unless given — the messengers service has its own API).
    timeout: seconds for the brain's answer (a model detection probes several models; it takes longer).
    precheck: called with the parsed JSON body before anything is forwarded; it refuses by raising HTTPException
    (the IoT safety statement gates use it). A refusal from the brain that carries a "code" reaches the page as
    detail {code, message}."""
    global _wipe_in_progress
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin or request.headers.get("x-build-f-action") != action:
        raise HTTPException(status_code=403, detail="Use the OpenAmika page for this action.")
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="JSON is required.")
    # check_busy=False: actions that touch nothing of hers (switching a plug) may run while she talks
    if _wipe_in_progress or (check_busy and any(_build_f_session_busy(s) for s in list(_dialog_sessions))):
        raise HTTPException(status_code=409, detail="Wait for Amika to finish speaking, updating memory and applying service controls.")
    _wipe_in_progress = True
    try:
        body = await request.body()
        if len(body) > max_body:
            raise HTTPException(status_code=400, detail="Control request is too large.")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise HTTPException(status_code=400, detail="Invalid JSON.") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="A JSON object is required.")
        if precheck is not None:
            precheck(payload)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as client:
            async with client.post((base or HANE_URL).rstrip("/") + route, json=payload,
                                   headers={"X-Build-F-Action": action}, allow_redirects=False) as response:
                result = await response.json()
                if response.status in (400, 403, 409, 503):
                    message = (result.get("error", "Action was refused.") if isinstance(result, dict)
                               else "Action was refused.")
                    code = result.get("code") if isinstance(result, dict) else None
                    raise HTTPException(status_code=response.status,
                                        detail={"code": code, "message": message} if isinstance(code, str) and code else message)
                if response.status != 200 or not isinstance(result, dict) or result.get("ok") is not True:
                    raise ValueError("Control result is unconfirmed")
        return {key: result.get(key) for key in allowed_fields}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="The action's result is unconfirmed. Refresh status before retrying.") from None
    finally:
        _wipe_in_progress = False


@app.post("/api/build_f/meter/reset")
async def build_f_meter_reset(request: Request):
    return await _build_f_control(request, "/api/build_f/meter/reset", "reset-meter", ("ok", "usage", "display_window"))


@app.post("/api/build_f/models")
async def build_f_select_models(request: Request):
    return await _build_f_control(request, "/api/build_f/models", "select-models",
                                  ("ok", "selections", "options", "presets", "benchmark", "transport", "ollama", "selection_fallback"))


# everything the Tools page renders: jobs/relay_queue (agent delegations), clerk_role, "config" (the raw
# tools.json, env/cwd included, so a save from the page never strips what the cards do not show) and
# "iot_disclaimer" (the IoT safety statement's version and acceptance)
_TOOLS_FIELDS = ("ok", "enabled", "clerk_role", "limits", "servers", "pending", "jobs", "relay_queue", "log", "config", "dialogue",
                 "iot_disclaimer")


# ── IoT safety statement: IoT turns on only after docs/IOT_DISCLAIMER.md was read and accepted on the Tools page ──
def _load_iot_disclaimer():
    """brain/iot_disclaimer.py (standard library only): the statement's version, what counts as IoT, and the
    acceptance in config/tools.json. The brain and the launcher apply the same file, so the three never disagree."""
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "brain" / "iot_disclaimer.py"
    spec = importlib.util.spec_from_file_location("openamika_iot_disclaimer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IOT_DISCLAIMER = _load_iot_disclaimer()


def _iot_refusal():
    """409 for a control that would turn IoT on, or use it, before the current statement was accepted; the Tools
    page answers detail.code "iot_disclaimer_required" by showing the statement."""
    return HTTPException(status_code=409, detail=IOT_DISCLAIMER.required_detail())


def _iot_config_gate(payload):
    """tools-config: a tools.json that switches a smart-home server on needs the accepted statement. The acceptance is
    read from config/tools.json on this machine, so the gate holds even when the brain does not answer."""
    config = payload.get("config")
    if isinstance(config, dict) and IOT_DISCLAIMER.turns_on_iot(config) \
            and not IOT_DISCLAIMER.accepted(IOT_DISCLAIMER.read_tools()):
        raise _iot_refusal()


def _iot_server_gate(payload):
    """tools-test / tools-start: connecting or starting a smart-home server needs the accepted statement."""
    name = payload.get("server")
    doc = IOT_DISCLAIMER.read_tools()
    servers = doc.get("mcpServers") if isinstance(doc.get("mcpServers"), dict) else {}
    if isinstance(name, str) and IOT_DISCLAIMER.is_smart_home(name, servers.get(name)) and not IOT_DISCLAIMER.accepted(doc):
        raise _iot_refusal()


def _iot_use_gate(payload):
    """iot-device / iot-pairing: the Smart home panel switches nothing before the statement was accepted."""
    if not IOT_DISCLAIMER.accepted(IOT_DISCLAIMER.read_tools()):
        raise _iot_refusal()


def _iot_accept_gate(payload):
    """iot-disclaimer: only an explicit acceptance of the current version goes on to the brain."""
    if payload.get("accept") is not True or payload.get("version") != IOT_DISCLAIMER.IOT_DISCLAIMER_VERSION:
        raise HTTPException(status_code=400, detail=(
            f"Read the current IoT safety statement (version {IOT_DISCLAIMER.IOT_DISCLAIMER_VERSION}) and tick "
            "“I have read and accept this” first."))


@app.get("/api/build_f/tools")
async def build_f_tools_status():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/tools", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Tool status is unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected tool status")
        return {key: payload.get(key) for key in _TOOLS_FIELDS}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Tool status is unavailable.") from None


@app.post("/api/build_f/tools/config")
async def build_f_tools_config(request: Request):
    """Save the Tools page's settings (the whole tools.json). A config that switches a smart-home server on is refused
    (409, iot_disclaimer_required) until the current IoT safety statement has been accepted."""
    return await _build_f_control(request, "/api/build_f/tools/config", "tools-config", _TOOLS_FIELDS, max_body=65536,
                                  precheck=_iot_config_gate)


@app.post("/api/build_f/tools/test")
async def build_f_tools_test(request: Request):
    return await _build_f_control(request, "/api/build_f/tools/test", "tools-test", ("ok", "server", "kind", "tools", "health"),
                                  precheck=_iot_server_gate)


@app.post("/api/build_f/tools/approve")
async def build_f_tools_approve(request: Request):
    return await _build_f_control(request, "/api/build_f/tools/approve", "tools-approve", ("ok", "result", "is_error", "job"))


@app.post("/api/build_f/tools/start")
async def build_f_tools_start(request: Request):
    """Start an agent's own process (config autostart) from the Tools page ({server})."""
    return await _build_f_control(request, "/api/build_f/tools/start", "tools-start", ("ok", "server", "health", "started"),
                                  precheck=_iot_server_gate)


@app.post("/api/build_f/tools/job")
async def build_f_tools_job(request: Request):
    """Stop a running agent job or answer its question from the Tools page ({server, job, action, text})."""
    return await _build_f_control(request, "/api/build_f/tools/job", "tools-job", ("ok", "result", "job"))


@app.post("/api/build_f/tools/dismiss")
async def build_f_tools_dismiss(request: Request):
    return await _build_f_control(request, "/api/build_f/tools/dismiss", "tools-dismiss", ("ok",))


@app.get("/api/build_f/persona")
async def build_f_persona():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/persona", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Personality data is unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected personality payload")
        return {key: payload.get(key) for key in ("ok", "user_name", "persona_name", "timezone", "city", "onboarded",
                                                  "clock", "timezones", "files", "template_check")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Personality data is unavailable.") from None


@app.post("/api/build_f/persona/save")
async def build_f_persona_save(request: Request):
    return await _build_f_control(request, "/api/build_f/persona/save", "persona-save",
                                  ("ok", "target", "problems", "note", "template_check"), max_body=300000)


@app.post("/api/build_f/persona/identity")
async def build_f_persona_identity(request: Request):
    return await _build_f_control(request, "/api/build_f/persona/identity", "persona-identity",
                                  ("ok", "user_name", "persona_name", "timezone", "city", "onboarded", "clock", "note"))


@app.get("/api/build_f/persona/simple")
async def build_f_persona_simple():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/persona/simple", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Persona builder data is unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected persona payload")
        return {key: payload.get(key) for key in ("ok", "user_name", "persona_name", "story", "days", "about")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Persona builder data is unavailable.") from None


@app.post("/api/build_f/persona/simple")
async def build_f_persona_simple_save(request: Request):
    return await _build_f_control(request, "/api/build_f/persona/simple", "persona-simple",
                                  ("ok", "touched", "problems"), max_body=200000)


@app.get("/api/build_f/openrouter_key")
async def build_f_openrouter_key():
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/openrouter_key", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Key status unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected key status")
        return {key: payload.get(key) for key in ("ok", "present", "hint")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Key status unavailable.") from None


@app.post("/api/build_f/openrouter_key")
async def build_f_openrouter_key_set(request: Request):
    return await _build_f_control(request, "/api/build_f/openrouter_key", "openrouter-key",
                                  ("ok", "present", "hint", "note"))


@app.get("/api/build_f/iot")
async def build_f_iot():
    """The Smart home panel's data (devices, readings, pairing state), straight from the brain. "blocked" says the
    smart-home server is switched on but held off until the IoT safety statement is accepted."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as client:
            async with client.get(HANE_URL.rstrip("/") + "/api/build_f/iot", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="Smart home data is unavailable.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected smart home payload")
        return {key: payload.get(key) for key in ("ok", "enabled", "connected", "server", "health", "devices", "discovery", "error",
                                                  "blocked", "iot_disclaimer")}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="Smart home data is unavailable.") from None


@app.post("/api/build_f/iot/device")
async def build_f_iot_device(request: Request):
    """Manual device action from the panel ({device, action} | {device, nickname} | {device, note})."""
    return await _build_f_control(request, "/api/build_f/iot/device", "iot-device", ("ok", "device", "result", "error"),
                                  check_busy=False, precheck=_iot_use_gate)


@app.post("/api/build_f/iot/pairing")
async def build_f_iot_pairing(request: Request):
    """Open/close zigbee2mqtt pairing from the panel ({enable, seconds})."""
    return await _build_f_control(request, "/api/build_f/iot/pairing", "iot-pairing", ("ok", "result", "discovery", "error"),
                                  check_busy=False, precheck=_iot_use_gate)


@app.get("/api/build_f/iot/disclaimer")
async def build_f_iot_disclaimer():
    """The IoT safety statement (docs/IOT_DISCLAIMER.md: English, Japanese, Simplified Chinese) and whether its current
    version was accepted (config/tools.json). The Tools page shows it before IoT can be turned on. Read on this
    machine, so it works while the brain is down; an unreadable or mismatched statement means IoT cannot be turned on."""
    try:
        statement = IOT_DISCLAIMER.statement()
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail=("The IoT safety statement (docs/IOT_DISCLAIMER.md) cannot be read, "
                                                     "so IoT cannot be turned on.")) from None
    return {"ok": True, **IOT_DISCLAIMER.status(IOT_DISCLAIMER.read_tools()), "statement": statement}


@app.post("/api/build_f/iot/disclaimer")
async def build_f_iot_disclaimer_accept(request: Request):
    """Record the person's acceptance of the current IoT safety statement ({accept: true, version}); the brain writes it
    into config/tools.json with the time. Only the statement's dialog on the Tools page sends this."""
    return await _build_f_control(request, "/api/build_f/iot/disclaimer", "iot-disclaimer", ("ok", "iot_disclaimer"),
                                  check_busy=False, precheck=_iot_accept_gate)


@app.get("/api/preflight")
async def preflight_report():
    """The launcher's last pre-flight report (logs/preflight.json) — the Setup page shows the same lights the
    console showed at start, so a red or yellow item is visible in the web UI too."""
    path = Path(__file__).resolve().parent.parent / "logs" / "preflight.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"at": None, "checks": [], "summary": {}}


MESSENGERS_URL = os.getenv("AMIKA_MESSENGERS_URL", "http://127.0.0.1:5215")
_MESSENGER_FIELDS = ("ok", "discord", "telegram", "last_chat", "live", "brain", "light", "discord_library", "config_path")


@app.get("/api/build_f/messengers")
async def build_f_messengers():
    """The Setup page's "Text her" card: bot settings (tokens masked) and live status, from messengers/service.py."""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as client:
            async with client.get(MESSENGERS_URL.rstrip("/") + "/api/status", allow_redirects=False) as response:
                if response.status != 200:
                    raise HTTPException(status_code=503, detail="The messengers service is not answering.")
                payload = await response.json()
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            raise ValueError("Unexpected messengers payload")
        return {key: payload.get(key) for key in _MESSENGER_FIELDS}
    except HTTPException:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
        raise HTTPException(status_code=503, detail="The messengers service is not running (launch.py starts it).") from None


@app.post("/api/build_f/messengers")
async def build_f_messengers_save(request: Request):
    """Save bot tokens / allow-lists; the service restarts the bots that changed. Not her state: allowed while she talks."""
    return await _build_f_control(request, "/api/config", "messengers", _MESSENGER_FIELDS, check_busy=False,
                                  base=MESSENGERS_URL)


@app.get("/api/tts_probe")
async def tts_probe():
    """Lightweight reachability check of the configured TTS server (Setup page health)."""
    info = {"url": TTS_URL, "character": TTS_CHARACTER_ID}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as client:
            async with client.get(TTS_URL.rstrip("/") + "/ping", allow_redirects=False) as response:
                return {**info, "ok": response.status == 200, "status": response.status}
    except (aiohttp.ClientError, asyncio.TimeoutError) as error:
        return {**info, "ok": False, "error": str(error)[:200]}


# ---------------------- one-shot speech: the first-run wizard, name previews ---------------------- #
_TTS_SAY_EMOTION_DEFAULT = "casual_happy"
_TTS_SAY_TIMEOUT_S = 25
_TTS_LANGUAGE_NAMES = {"CHINESE": "Chinese", "JAPANESE": "Japanese", "ENGLISH": "English"}


async def _tts_say_once(tts_text: str, language: Optional[str]) -> dict:
    """One synthesis over a transient Socket.IO connection to the TTS server (the same request the dialog
    sessions send). Returns the audio_ready payload; raises on connection failure, timeout or a TTS error.
    The balancer broadcasts every event, so only this task_id's events count."""
    sio = socketio.AsyncClient(reconnection=False, logger=False, engineio_logger=False)
    task_id = str(uuid.uuid4())
    done: asyncio.Future = asyncio.get_running_loop().create_future()

    @sio.on("audio_ready")
    async def on_audio(data=None):
        data = data or {}
        if str(data.get("task_id")) == task_id and not done.done():
            done.set_result(data)

    @sio.on("error")
    async def on_error(data=None):
        data = data or {}
        if str(data.get("task_id")) == task_id and not done.done():
            done.set_exception(RuntimeError(str(data.get("error") or "TTS error")))

    await sio.connect(TTS_URL, transports=["websocket"])
    try:
        payload: dict = {"task_id": task_id, "turn_id": f"say-{task_id[:8]}", "text": tts_text,
                         "character_id": TTS_CHARACTER_ID, "audio_format": TTS_AUDIO_FORMAT}
        if language:
            payload["language"] = language
        va_enabled, va_voice, va_map = voice_adapter.live_state()
        if va_enabled:
            payload["text"] = voice_adapter.remap_leading_tag(payload["text"], va_map)
            payload["character_id"] = va_voice
        await sio.emit("tts_request_gateway", payload)
        return await asyncio.wait_for(done, _TTS_SAY_TIMEOUT_S)
    finally:
        try:
            await sio.disconnect()
        except Exception:
            pass


@app.post("/api/tts_say")
async def tts_say(request: Request):
    """Speak one line in her voice outside a conversation (the first-run wizard, the name preview on the
    Setup page). Same-origin only. The LLM→TTS word map applies unless `raw` is set (a pronunciation guide
    typed as-is). Answers {audio (base64), audio_format, duration, spoken, emotion, language}."""
    import base64
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(status_code=403, detail="Use the OpenAmika page for this action.")
    body = await request.body()
    if len(body) > 4096:
        raise HTTPException(status_code=400, detail="The line is too long.")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="A JSON object is required.")
    text = " ".join(str(payload.get("text") or "").split())
    if not text or len(text) > 400:
        raise HTTPException(status_code=400, detail="text must be 1 to 400 characters.")
    emotion = str(payload.get("emotion") or _TTS_SAY_EMOTION_DEFAULT).strip()
    try:
        from po_contract import VALID_EMOTIONS
        if emotion not in VALID_EMOTIONS:
            emotion = _TTS_SAY_EMOTION_DEFAULT
    except Exception:
        pass
    spoken = text if payload.get("raw") else apply_wordmap(text, "tts")
    # `voice`: a V1 voice tag (happy1, caring2, …) to speak with directly — the adapter lets V1 ids through
    # untouched. The 69-key `emotion` still drives her face. Without it the adapter maps the emotion as usual.
    voice = str(payload.get("voice") or "").strip()
    voice_tag = voice if voice in voice_adapter.V1_ID_SET else emotion
    tts_text = prefix_emotion(voice_tag, "." + spoken)
    language = _TTS_LANGUAGE_NAMES.get(DialogSession._detect_lang(spoken))
    try:
        data = await _tts_say_once(tts_text, language)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="The TTS server did not answer in time.") from None
    except Exception as error:
        raise HTTPException(status_code=503, detail=f"Her voice is unavailable: {str(error)[:160]}") from None
    audio = data.get("audio")
    if isinstance(audio, (bytes, bytearray)):
        audio_b64 = base64.b64encode(bytes(audio)).decode("ascii")
    elif isinstance(audio, str) and audio:
        audio_b64 = audio
    else:
        raise HTTPException(status_code=502, detail="The TTS server returned no audio.")
    return {"ok": True, "audio": audio_b64, "audio_format": data.get("audio_format") or TTS_AUDIO_FORMAT,
            "duration": data.get("audio_duration"), "spoken": spoken, "emotion": emotion, "voice_tag": voice_tag,
            "language": language}


# How her voice says your name: one llm_to_tts rule in config/word_replace.json that the wizard and the
# Setup page own (role user_name), e.g. Amika → Ameeka. Everything she speaks goes through apply_wordmap,
# so the guide holds for the rest of the program; what she writes keeps the real spelling.
_NAME_RULE_ROLE = "user_name"


def _wordmap_doc() -> dict:
    try:
        with open(_WORDMAP_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    for key in ("stt_to_llm", "llm_to_tts"):
        if not isinstance(data.get(key), list):
            data[key] = []
    return data


def _name_rule(data: dict):
    for row in data.get("llm_to_tts") or []:
        if isinstance(row, dict) and row.get("role") == _NAME_RULE_ROLE:
            return row
    return None


@app.get("/api/name_pronunciation")
async def name_pronunciation_get():
    rule = _name_rule(_wordmap_doc()) or {}
    return {"ok": True, "name": str(rule.get("from") or ""), "say_as": str(rule.get("to") or "")}


@app.post("/api/name_pronunciation")
async def name_pronunciation_set(request: Request):
    """Set (or clear, with an empty say_as) how her voice pronounces your name."""
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin or request.headers.get("x-build-f-action") != "name-pronunciation":
        raise HTTPException(status_code=403, detail="Use the OpenAmika page for this action.")
    body = await request.body()
    if len(body) > 2048:
        raise HTTPException(status_code=400, detail="Too long.")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="A JSON object is required.")
    name = " ".join(str(payload.get("name") or "").split())
    say_as = " ".join(str(payload.get("say_as") or "").split())
    if not name or len(name) > 40 or len(say_as) > 80:
        raise HTTPException(status_code=400, detail="name (1-40) and say_as (0-80 characters) are required.")
    keep = bool(say_as) and say_as.lower() != name.lower()
    data = _wordmap_doc()
    rows = [r for r in data["llm_to_tts"] if not (isinstance(r, dict) and r.get("role") == _NAME_RULE_ROLE)]
    if keep:
        rows.insert(0, {"from": name, "to": say_as, "ci": True, "role": _NAME_RULE_ROLE})
    data["llm_to_tts"] = rows
    tmp = _WORDMAP_PATH + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, _WORDMAP_PATH)
    except OSError as error:
        raise HTTPException(status_code=503, detail=f"Could not write config/word_replace.json: {error}") from None
    _wordmap["mtime"] = None                      # reload on her next line
    return {"ok": True, "name": name, "say_as": say_as if keep else ""}


class HaneClient:
    """对接 brain（AI Core）的 Socket.IO 客户端。
    每个 DialogSession 独占一条连接，这样 Hane 端的 sid/历史按用户隔离。

    协议详见 brain/PROTOCOL.md。约定：
      - 所有 emit('message', {...}) 都带 turn_id（本端生成，递增）
      - Hane 推回的 begin/surface_chunk/surface_done/deep_chunk/deep_done/end/error
        payload 里都要带回同一个 turn_id —— 前端据此过滤 stale turn 的残余 chunk。
    """

    def __init__(self, session: "DialogSession"):
        self.session = session
        self.url = HANE_URL
        self.sio = socketio.AsyncClient(
            reconnection=True, reconnection_attempts=0,
            logger=False, engineio_logger=False,
        )
        self._connected = False
        self._register_handlers()

    async def connect(self):
        try:
            await self.sio.connect(self.url, transports=["websocket"])
            self._connected = True
            logger.info(f"[hane] connected to {self.url}")
        except Exception as e:
            logger.warning(f"[hane] connect failed ({self.url}): {e}")
            self._connected = False

    async def disconnect(self):
        try:
            if self._connected:
                await self.sio.disconnect()
        except Exception:
            pass
        self._connected = False

    async def send_message(
        self, text: str, turn_id: int, my_partial: Optional[str] = None, *, apply_stt_map: bool = True, deep_only: bool = False,
        speculative: bool = False,
    ):
        if not self._connected:
            logger.debug(f"[hane] skip emit message (not connected), turn={turn_id}")
            return
        if apply_stt_map:
            text = apply_wordmap(text, "stt")
        if not deep_only and text:
            # While she can actually see (camera on + fresh VLM description), the turn
            # carries what's on the video call — same contract as the original stack.
            vision_note_user_lang(self.session._detect_lang(text))
            suffix = vision_suffix_for(text)
            if suffix:
                text = text + suffix
        payload: dict = {"text": text, "turn_id": turn_id, "context": self.session._pending_context}
        if speculative:
            # P2: a micro-pause partial. The brain runs the surface at once and starts the deep only
            # after our "commit" for this id (or after speculative_deep_wait_s).
            payload["speculative"] = True
        self.session._contexts_sent[turn_id] = (self.session._context_revision, self.session._pending_context)
        if len(self.session._contexts_sent) > 100:
            self.session._contexts_sent.pop(next(iter(self.session._contexts_sent)))
        if my_partial:
            payload["my_partial"] = my_partial
        try:
            await self.sio.emit("inject" if deep_only else "message", payload)
            if turn_id == self.session._committed_turn_id:
                await self.session.consume_context(turn_id)
        except Exception as e:
            logger.warning(f"[hane] emit message failed: {e}")

    async def send_protect(self) -> bool:
        """Returns True only when the brain acknowledged the protect."""
        if not self._connected:
            return False
        try:
            ack = await self.sio.call("protect", {}, timeout=5)
        except Exception as e:
            logger.warning(f"[hane] emit protect failed: {e}")
            return False
        if isinstance(ack, dict) and ack.get("ok") is False:
            return False
        return True

    async def send_commit(self, turn_id) -> None:
        """P2: the utterance is final and reuses this speculative turn id; the brain may start the deep."""
        if not self._connected:
            return
        try:
            await self.sio.emit("commit", {"turn_id": turn_id})
        except Exception as e:
            logger.warning(f"[hane] emit commit failed: {e}")

    async def send_speech_done(self, turn_id: Optional[str]):
        """通知 Hane：这一轮 TTS 音频已彻底播完。Hane 用它在"换口气"后触发【接着说】续说轮。"""
        if not self._connected:
            return
        try:
            await self.sio.emit("speech_done", {"turn_id": turn_id})
        except Exception as e:
            logger.warning(f"[hane] emit speech_done failed: {e}")

    def _register_handlers(self):
        sio = self.sio

        @sio.event
        async def connect():
            self._connected = True

        @sio.event
        async def disconnect():
            self._connected = False
            logger.info("[hane] disconnected")

        for evt in ("begin", "surface_chunk", "surface_done",
                    "deep_chunk", "deep_done", "end", "error"):
            async def handler(data=None, _evt=evt):
                await self.session._on_ai_event(_evt, data or {})
            sio.on(evt, handler)

        # TELEMETRY_SPEC §1.1: one receipt per metered call. Kept out of _on_ai_event: receipts
        # are keyed by turn_id independently of the committed turn (a surface receipt often
        # arrives before the commit, the memory writer's after the round).
        async def receipt_handler(data=None):
            on_receipt = getattr(self.session, "_on_call_receipt", None)
            if on_receipt is not None:
                await on_receipt(data or {})
        sio.on("call_receipt", receipt_handler)


# ---------------------------- TTS 对接 ---------------------------- #

TTS_URL = os.getenv("AMIKA_TTS_URL", "http://127.0.0.1:8002")
# Shipping policy: V1 (old-vocabulary) voices only — the voice adapter is on by default
# and swaps character_id per request, so this value is the adapter-OFF fallback.
TTS_CHARACTER_ID = os.getenv("AMIKA_TTS_CHARACTER", "MekaHime_Amika")
# 浏览器 Web Audio decodeAudioData 只能解**带容器**的音频（WAV/MP3/Ogg/WebM…），
# 裸 Opus 帧解不了。本地局域网带宽充足，直接要 WAV 最省事。
TTS_AUDIO_FORMAT = "wav"


class TTSClient:
    """对接 TTS 服务（轮次模式）的 Socket.IO 客户端。
    协议详见 docs/TTS.md：
      - emit('tts_request_gateway', {task_id, turn_id, text, character_id})
      - on('audio_ready', {task_id, turn_id, audio, audio_format, audio_duration, text, ...})
      - on('error', {task_id, turn_id, error})
      - on('task_queued', {task_id, turn_id})
    注意：
      - 负载均衡器是广播回传，所有事件都要按 task_id / turn_id 自己过滤。
      - 一个请求只回一次 audio_ready（is_final=true）。
      - audio 是 bytes；python-socketio 会把二进制字段自动还原成 bytes。
    """

    def __init__(self, session: "DialogSession"):
        self.session = session
        self.url = TTS_URL
        self.sio = socketio.AsyncClient(
            reconnection=True, reconnection_attempts=0,
            logger=False, engineio_logger=False,
        )
        self._connected = False
        self._register_handlers()

    async def connect(self):
        try:
            await self.sio.connect(self.url, transports=["websocket"])
            self._connected = True
            logger.info(f"[tts] connected to {self.url}")
        except Exception as e:
            logger.warning(f"[tts] connect failed ({self.url}): {e}")
            self._connected = False

    async def disconnect(self):
        try:
            if self._connected:
                await self.sio.disconnect()
        except Exception:
            pass
        self._connected = False

    async def request(
        self, task_id: str, tts_turn_id: str, text: str,
        language: Optional[str] = None,
    ) -> bool:
        """发出 TTS 请求。`language` 为 'Chinese' / 'English' / 'Japanese'；
        None / 'MIXED' 时不带，让 TTS 按角色默认走。
        返回是否发送成功。
        """
        if not self._connected:
            logger.debug(f"[tts] skip emit (not connected), turn={tts_turn_id}")
            return False
        try:
            payload: dict = {
                "task_id": task_id,
                "turn_id": tts_turn_id,
                "text": text,
                "character_id": TTS_CHARACTER_ID,
                "audio_format": TTS_AUDIO_FORMAT,
            }
            if language:
                payload["language"] = language
            va_enabled, va_voice, va_map = voice_adapter.live_state()
            if va_enabled:
                # Old-voice path: translate the leading V2 tag into a V1 id and swap the
                # speaker. The V2 `emotion` reported to the browser/avatar is untouched
                # (one tag, two vocabularies, two consumers).
                payload["text"] = voice_adapter.remap_leading_tag(payload["text"], va_map)
                payload["character_id"] = va_voice
            await self.sio.emit("tts_request_gateway", payload)
            return True
        except Exception as e:
            logger.warning(f"[tts] emit failed: {e}")
            return False

    def _register_handlers(self):
        sio = self.sio

        @sio.event
        async def connect():
            self._connected = True

        @sio.event
        async def disconnect():
            self._connected = False
            logger.info("[tts] disconnected")

        @sio.on("task_queued")
        async def on_queued(data=None):
            await self.session._on_tts_event("task_queued", data or {})

        @sio.on("audio_ready")
        async def on_audio(data=None):
            await self.session._on_tts_event("audio_ready", data or {})

        @sio.on("error")
        async def on_err(data=None):
            await self.session._on_tts_event("error", data or {})


# ---------------------- voice adapter (V2 → V1 emotion remap) ---------------------- #
# Backward-compatible controller for old-vocabulary voices; see speech/voice_adapter.py.

@app.get("/api/voice_map")
async def voice_map_get():
    enabled, voice, full = voice_adapter.live_state()
    return {
        "ok": True, "enabled": enabled, "voice": voice, "map": full,
        "defaults": voice_adapter.DEFAULT_MAP, "v1_ids": voice_adapter.V1_IDS,
        "v1_voices": voice_adapter.V1_VOICES, "v2_grouped": voice_adapter.V2_GROUPED,
        "native_character": TTS_CHARACTER_ID,
    }


@app.post("/api/voice_map")
async def voice_map_set(request: Request):
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid voice map payload.")
    voice_adapter.save_state(payload.get("enabled"), payload.get("voice"), payload.get("map"))
    enabled, voice, full = voice_adapter.live_state()
    return {"ok": True, "enabled": enabled, "voice": voice, "map": full}


@app.post("/api/voice_map/reset")
async def voice_map_reset():
    voice_adapter.reset_state()
    enabled, voice, full = voice_adapter.live_state()
    return {"ok": True, "enabled": enabled, "voice": voice, "map": full}


# ---------------------- Live2D avatar: registry, banks, model files ---------------------- #
# Users drop Cubism models under avatar/models/<id>/ (see avatar/models/README.md).
# Banks map each of the 69 emotion keys to expressions/motions; absent entries fall back
# to the motion-file naming convention (casual_happy.motion3.json, casual_happy.2.…).

def _avatar_dir(model_id: str) -> Path:
    base = AVATAR_MODELS_DIR.resolve()
    try:
        path = (base / model_id).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid model id.") from None
    if base != path and base not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid model id.")
    if not path.is_dir() or path == base:
        raise HTTPException(status_code=404, detail="Unknown avatar model.")
    return path


def _avatar_model3(model_dir: Path) -> Optional[Path]:
    hits = sorted(model_dir.glob("*.model3.json")) or sorted(model_dir.rglob("*.model3.json"))
    return hits[0] if hits else None


# Agent-avatar placement hints in banks.json config (fractions of the stage): scale = height, nx/ny =
# position, head_y = where the head sits above the model centre (speech bubbles + glances aim there).
_AVATAR_STAGE_KEYS = ("scale", "nx", "ny", "head_y")
# config keys the Setup page edits; everything else in banks.json config survives a save untouched
_AVATAR_EDITABLE_KEYS = ("display_name", "mouth_open_param", "mouth_form_param", "scale")


def _avatar_stage_config(cfg: dict) -> dict:
    out = {}
    for key in _AVATAR_STAGE_KEYS:
        v = cfg.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and -2 <= v <= 3:
            out[key] = float(v)
    busy = cfg.get("busy_expression")
    if isinstance(busy, str) and busy.strip():
        out["busy_expression"] = busy.strip()
    return out


def _avatar_banks_doc(model_dir: Path) -> dict:
    doc = {"config": {}, "banks": {}}
    f = model_dir / "banks.json"
    if f.is_file():
        try:
            raw = json.loads(f.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                doc["config"] = raw.get("config") if isinstance(raw.get("config"), dict) else {}
                doc["banks"] = raw.get("banks") if isinstance(raw.get("banks"), dict) else {}
        except Exception:
            logger.warning(f"[avatar] unreadable banks.json in {model_dir.name}; using convention defaults")
    stems = sorted(p.name[:-len(".motion3.json")] for p in model_dir.rglob("*.motion3.json"))
    banks = doc["banks"]
    for key in voice_adapter.V2_KEYS:
        entry = banks.get(key)
        if not isinstance(entry, dict):
            moves = [s for s in stems if s == key or s.startswith(key + ".")]
            banks[key] = {"emotes": [], "movements": [{"name": s, "likelihood": 100} for s in moves]}
    if not isinstance(banks.get("__idle__"), dict):
        idles = [s for s in stems if s.lower().startswith("idle")]
        if idles:
            banks["__idle__"] = {"movements": [{"name": s, "likelihood": 100} for s in idles]}
    return doc


@app.get("/api/avatar/models")
async def avatar_models():
    models = []
    if AVATAR_MODELS_DIR.is_dir():
        for d in sorted(AVATAR_MODELS_DIR.iterdir()):
            if not d.is_dir():
                continue
            m3 = _avatar_model3(d)
            if not m3:
                continue
            rel = m3.relative_to(d).as_posix()
            doc = _avatar_banks_doc(d)
            cfg = doc["config"]
            models.append({
                "id": d.name,
                "name": cfg.get("display_name") or d.name,
                "model3_url": f"/avatar_models/{d.name}/{rel}",
                "motions_url": f"/api/avatar/models/{d.name}/motions",
                "banks_url": f"/api/avatar/models/{d.name}/banks",
                "assets_prefix": f"/avatar_models/{d.name}/",
                "mouth_open_param": cfg.get("mouth_open_param", "ParamMouthOpenY"),
                "mouth_form_param": cfg.get("mouth_form_param", "ParamMouthForm"),
                # "character" models are hers to wear. An "agent" model is the stand-in for her helpers
                # (Hermes, nanoMuse): the Use page seats it beside her and never lists it as her avatar.
                "role": "agent" if cfg.get("role") == "agent" else "character",
                "stage": _avatar_stage_config(cfg),
            })
    return {"ok": True, "models": models, "emotions": list(voice_adapter.V2_KEYS)}


@app.get("/api/avatar/models/{model_id}/motions")
async def avatar_motions(model_id: str):
    d = _avatar_dir(model_id)
    out = []
    for p in sorted(d.rglob("*.motion3.json")):
        rel = p.relative_to(d).as_posix()
        out.append({"type": "motion", "name": p.name[:-len(".motion3.json")],
                    "file": p.name, "path": f"/avatar_models/{model_id}/{rel}"})
    for p in sorted(d.rglob("*.exp3.json")):
        rel = p.relative_to(d).as_posix()
        out.append({"type": "expression", "name": p.name[:-len(".exp3.json")],
                    "file": p.name, "path": f"/avatar_models/{model_id}/{rel}"})
    return out


@app.get("/api/avatar/models/{model_id}/banks")
async def avatar_banks_get(model_id: str):
    return _avatar_banks_doc(_avatar_dir(model_id))


@app.post("/api/avatar/models/{model_id}/banks")
async def avatar_banks_set(model_id: str, request: Request):
    d = _avatar_dir(model_id)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid banks payload.")
    config_in = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    banks_in = payload.get("banks") if isinstance(payload.get("banks"), dict) else {}
    config = {}
    for k in ("display_name", "mouth_open_param", "mouth_form_param"):
        v = config_in.get(k)
        if isinstance(v, str) and v.strip():
            config[k] = v.strip()
    if isinstance(config_in.get("scale"), (int, float)) and 0.05 <= config_in["scale"] <= 3:
        config["scale"] = float(config_in["scale"])
    for k, v in _avatar_banks_doc(d)["config"].items():      # role / placement / busy_expression: not edited here
        if k not in _AVATAR_EDITABLE_KEYS and k not in config:
            config[k] = v
    valid_keys = set(voice_adapter.V2_KEYS) | {"__idle__"}

    def _clean_entries(items):
        out = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            try:
                lk = max(0, min(100, int(it.get("likelihood", 100))))
            except (TypeError, ValueError):
                lk = 100
            out.append({"name": name.strip(), "likelihood": lk})
        return out

    banks = {}
    for key, entry in banks_in.items():
        if key not in valid_keys or not isinstance(entry, dict):
            continue
        banks[key] = {"emotes": _clean_entries(entry.get("emotes")),
                      "movements": _clean_entries(entry.get("movements"))}
    doc = {"config": config, "banks": banks}
    tmp = d / "banks.json.tmp"
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, d / "banks.json")
    return {"ok": True}


@app.get("/avatar_models/{model_id}/{file_path:path}")
async def avatar_file(model_id: str, file_path: str):
    d = _avatar_dir(model_id)
    try:
        target = (d / file_path).resolve()
    except (OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid path.") from None
    if d.resolve() not in target.parents:
        raise HTTPException(status_code=403, detail="Path outside the model folder.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    if target.name.endswith(".model3.json"):
        # pixi-live2d-display builds an empty motion manager when the model3.json
        # declares no Motions — the sibling *.motion3.json files would be ignored.
        # Inject Motions AND Expressions groups on the fly when absent.
        try:
            doc = json.loads(target.read_text(encoding="utf-8-sig"))
            refs = doc.setdefault("FileReferences", {})

            # Cache-bust the heavy binaries: stamp each reference with its file mtime,
            # so an edited texture gets a URL the browser has never cached. (The old
            # un-versioned entries could sit "fresh" for weeks via heuristic caching.)
            def _ver(rel):
                if not isinstance(rel, str) or not rel:
                    return rel
                try:
                    stamp = int((target.parent / rel).stat().st_mtime)
                except OSError:
                    return rel
                return f"{rel}?v={stamp}"

            if isinstance(refs.get("Textures"), list):
                refs["Textures"] = [_ver(t) for t in refs["Textures"]]
            for key in ("Moc", "Physics", "DisplayInfo"):
                if refs.get(key):
                    refs[key] = _ver(refs[key])

            if not refs.get("Motions"):
                motions = [{"File": p.relative_to(target.parent).as_posix(),
                            "FadeInTime": 0.5, "FadeOutTime": 0.5}
                           for p in sorted(target.parent.rglob("*.motion3.json"))]
                if motions:
                    refs["Motions"] = {"Idle": motions}
            if not refs.get("Expressions"):
                exps = [{"Name": p.name, "File": p.relative_to(target.parent).as_posix()}
                        for p in sorted(target.parent.rglob("*.exp3.json"))]
                if exps:
                    refs["Expressions"] = exps
            return JSONResponse(doc, headers={"Cache-Control": "no-cache"})
        except Exception:
            logger.warning(f"[avatar] model3.json injection failed for {model_id}; serving raw file")
    # no-cache = always revalidate (cheap 304s): edited textures/banks show up on a
    # plain reload instead of hiding behind the browser's heuristic freshness window.
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


# ---------------------- stage scene: background photo + blur ---------------------- #
# User-uploaded stage backdrop for the Use page (avatar/stage/ is gitignored).

STAGE_DIR = AVATAR_MODELS_DIR.parent / "stage"
STAGE_CONF_PATH = STAGE_DIR / "stage.json"
STAGE_IMAGE_TYPES = {
    "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
    "image/gif": ".gif", "image/avif": ".avif",
}
STAGE_MAX_BYTES = 25 * 1024 * 1024


def _stage_require_same_origin(request: Request):
    expected_origin = f"{request.url.scheme}://{request.url.netloc}"
    if request.headers.get("origin") != expected_origin:
        raise HTTPException(status_code=403, detail="Stage changes require the OpenAmika page.")


def _stage_bg_path() -> Optional[Path]:
    if not STAGE_DIR.is_dir():
        return None
    hits = sorted(p for p in STAGE_DIR.glob("background.*") if p.suffix != ".tmp")
    return hits[0] if hits else None


def _stage_state() -> dict:
    try:
        conf = json.loads(STAGE_CONF_PATH.read_text(encoding="utf-8"))
        if not isinstance(conf, dict):
            conf = {}
    except Exception:
        conf = {}
    blur = conf.get("blur")
    if not isinstance(blur, (int, float)) or not (0 <= blur <= 40):
        blur = 12
    bg = _stage_bg_path()
    background = f"/api/stage/background?v={int(bg.stat().st_mtime)}" if bg else None
    return {"ok": True, "background": background, "blur": round(float(blur), 1)}


def _stage_save_conf(blur: float):
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STAGE_CONF_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"blur": round(float(blur), 1)}) + "\n", encoding="utf-8")
    os.replace(tmp, STAGE_CONF_PATH)


@app.get("/api/stage")
async def stage_get():
    return _stage_state()


@app.get("/api/stage/background")
async def stage_background():
    bg = _stage_bg_path()
    if not bg:
        raise HTTPException(status_code=404, detail="No stage background set.")
    return FileResponse(bg, headers={"Cache-Control": "no-cache"})


@app.post("/api/stage")
async def stage_set(request: Request):
    _stage_require_same_origin(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON.") from None
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid stage payload.")
    if "blur" in payload:
        blur = payload["blur"]
        if not isinstance(blur, (int, float)) or not (0 <= blur <= 40):
            raise HTTPException(status_code=400, detail="blur must be 0–40 (px).")
        _stage_save_conf(blur)
    if payload.get("clear_background"):
        for p in list(STAGE_DIR.glob("background.*")) if STAGE_DIR.is_dir() else []:
            try:
                p.unlink()
            except OSError:
                pass
    return _stage_state()


@app.post("/api/stage/background")
async def stage_background_upload(request: Request, file: UploadFile = File(...)):
    _stage_require_same_origin(request)
    ext = STAGE_IMAGE_TYPES.get((file.content_type or "").lower())
    if not ext:
        ext = Path(file.filename or "").suffix.lower()
        if ext == ".jpeg":
            ext = ".jpg"
        if ext not in STAGE_IMAGE_TYPES.values():
            raise HTTPException(status_code=400, detail="Use a PNG, JPEG, WebP, GIF or AVIF image.")
    data = await file.read()
    if len(data) > STAGE_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image is larger than 25 MB.")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STAGE_DIR / ("background" + ext + ".tmp")
    tmp.write_bytes(data)
    for p in STAGE_DIR.glob("background.*"):          # drop any previous photo (other ext)
        if p.suffix != ".tmp":
            try:
                p.unlink()
            except OSError:
                pass
    os.replace(tmp, STAGE_DIR / ("background" + ext))
    return _stage_state()


# ---------------------- vision: camera relay + local VLM ---------------------- #
# Mirrors the TravelProp roles locally: the browser streams JPEG frames to
# /ws/camera; the vision service (vision/vlm_service.py) subscribes on
# /ws/vlm_feed (binary frames + {"type":"prompt_update"} JSON with her latest
# spoken line as context); the gateway polls its /api/latest and, while the
# description is fresh, suffixes committed user turns with a language-matched
# "(you can see: …)" — the same consumption contract as the original stack.

VLM_URL = os.getenv("AMIKA_VLM_URL", "").rstrip("/")   # empty = vision disabled
VISION_EXPIRE_SEC = 3.0

_vision_desc: str = ""
_vision_desc_ts: float = 0.0
_vision_her_line: str = ""
_vision_lang: str = "CHINESE"
_vision_camera_clients: int = 0
_vlm_subscribers: list = []            # [(WebSocket, asyncio.Queue)]
_vision_poll_task = None

_VLM_PROMPT_TEMPLATES = {
    "ENGLISH": ('You are on a video call with someone. Describe what you see in two '
                'concise sentences. Output only the description itself — no preamble, '
                'no restating these instructions. For context, you just said: "{context}"'),
    "CHINESE": '你正在与别人进行视频通话。用两句简洁的话描述你看到的内容。只输出描述本身，不要开场白，不要复述这段指令。作为上下文参考，你刚才说了："{context}"',
    "JAPANESE": 'あなたは誰かとビデオ通話中です。見えているものを簡潔に2文で描写してください。描写だけを出力し、前置きやこの指示の復唱はしないでください。文脈として、あなたは先ほどこう言いました：「{context}」',
}
_VISION_SUFFIX_TEMPLATES = {
    "ENGLISH": "（on the video call you can see: {desc}）",
    "CHINESE": "（你在视频中看到：{desc}）",
    "JAPANESE": "（ビデオ通話で見えているのは：{desc}）",
}


def _vision_fresh_desc() -> str:
    if _vision_desc and (time.time() - _vision_desc_ts) <= VISION_EXPIRE_SEC:
        return _vision_desc
    return ""


def vision_suffix_for(text: str) -> str:
    """Suffix appended to a committed user turn while she can actually see."""
    if not VLM_URL or not text:
        return ""
    desc = _vision_fresh_desc()
    if not desc:
        return ""
    lang = _vision_lang if _vision_lang in _VISION_SUFFIX_TEMPLATES else "CHINESE"
    return _VISION_SUFFIX_TEMPLATES[lang].format(desc=desc)


def _build_vlm_prompt() -> str:
    lang = _vision_lang if _vision_lang in _VLM_PROMPT_TEMPLATES else "CHINESE"
    context = _vision_her_line or "(nothing yet)"
    return _VLM_PROMPT_TEMPLATES[lang].format(context=context)


def _vision_prompt_message() -> str:
    return json.dumps({"type": "prompt_update", "prompt": _build_vlm_prompt(),
                       "language": _vision_lang}, ensure_ascii=False)


def _vision_queue_put(q, item):
    """Drop-oldest put: subscribers always get the latest frame / prompt."""
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass


def _vision_broadcast_prompt():
    msg = _vision_prompt_message()
    for _ws, q in list(_vlm_subscribers):
        _vision_queue_put(q, msg)


def vision_note_her_speech(sentence: str):
    """Her latest spoken sentence becomes the VLM prompt context (TravelProp behavior)."""
    global _vision_her_line
    sentence = (sentence or "").strip()
    if not VLM_URL or not sentence or sentence == _vision_her_line:
        return
    _vision_her_line = sentence
    _vision_broadcast_prompt()


def vision_note_user_lang(lang: str):
    """Track the conversation language so prompt + suffix match it."""
    global _vision_lang
    if lang == "MIXED":
        lang = "CHINESE"
    if not VLM_URL or lang not in _VLM_PROMPT_TEMPLATES or lang == _vision_lang:
        return
    _vision_lang = lang
    _vision_broadcast_prompt()


async def _vision_poll_loop():
    """Poll the vision service for the latest description (like the orchestrator did)."""
    global _vision_desc, _vision_desc_ts
    timeout = aiohttp.ClientTimeout(total=2)
    while True:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as client:
                while True:
                    try:
                        async with client.get(f"{VLM_URL}/api/latest") as resp:
                            data = await resp.json()
                        desc = (data.get("description") or "").strip()
                        if desc and data.get("ok"):
                            _vision_desc = desc
                            _vision_desc_ts = time.time()
                    except Exception:
                        pass
                    await asyncio.sleep(0.7)
        except Exception:
            await asyncio.sleep(2.0)


def _ensure_vision_poller():
    global _vision_poll_task
    if VLM_URL and (_vision_poll_task is None or _vision_poll_task.done()):
        _vision_poll_task = asyncio.get_event_loop().create_task(_vision_poll_loop())


@app.websocket("/ws/camera")
async def ws_camera(ws: WebSocket):
    """Browser camera frames in (binary JPEG), fanned out to /ws/vlm_feed subscribers."""
    global _vision_camera_clients
    await ws.accept()
    _ensure_vision_poller()
    _vision_camera_clients += 1
    logger.info("[vision] camera client connected")
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            frame = msg.get("bytes")
            if frame:
                for _sub_ws, q in list(_vlm_subscribers):
                    _vision_queue_put(q, frame)
    except WebSocketDisconnect:
        pass
    except Exception as error:
        logger.debug(f"[vision] camera ws error: {error}")
    finally:
        _vision_camera_clients = max(0, _vision_camera_clients - 1)
        logger.info("[vision] camera client disconnected")


@app.websocket("/ws/vlm_feed")
async def ws_vlm_feed(ws: WebSocket):
    """VLM subscribers: receive camera frames + prompt_update messages (TravelProp contract)."""
    await ws.accept()
    _ensure_vision_poller()
    q: asyncio.Queue = asyncio.Queue(maxsize=3)
    entry = (ws, q)
    _vlm_subscribers.append(entry)
    logger.info(f"[vision] vlm subscriber connected ({len(_vlm_subscribers)})")
    try:
        await ws.send_text(_vision_prompt_message())    # context before the first frame
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                item = _vision_prompt_message()          # keepalive doubles as prompt refresh
            if isinstance(item, str):
                await ws.send_text(item)
            else:
                await ws.send_bytes(item)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as error:
        logger.debug(f"[vision] vlm feed error: {error}")
    finally:
        if entry in _vlm_subscribers:
            _vlm_subscribers.remove(entry)
        logger.info(f"[vision] vlm subscriber disconnected ({len(_vlm_subscribers)})")


@app.get("/api/vision")
async def vision_status():
    _ensure_vision_poller()
    service = None
    if VLM_URL:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2)) as client:
                async with client.get(f"{VLM_URL}/api/health") as resp:
                    service = await resp.json()
        except Exception:
            service = None
    age = (time.time() - _vision_desc_ts) if _vision_desc_ts else None
    return {
        "ok": True,
        "enabled": bool(VLM_URL),
        "url": VLM_URL or None,
        "service": service,
        "description": _vision_fresh_desc(),
        "last_description": _vision_desc,
        "description_age_s": round(age, 1) if age is not None else None,
        "camera_clients": _vision_camera_clients,
        "subscribers": len(_vlm_subscribers),
        "her_line": _vision_her_line,
        "language": _vision_lang,
    }


# ---------------------------- round telemetry (TELEMETRY_SPEC §1.2) ---------------------------- #

class TimelineBook:
    """Per-round latency/cost timelines for the browser's latency panel (TELEMETRY_SPEC §1.2).

    Pure Python and clock-free: every time is an epoch-ms value handed in by the caller
    (the speech server's `_now_ms()` or the brain's call_receipt), so tests can drive it
    without sockets. DialogSession feeds it with cheap dict updates on the hot path and
    sends `pop_messages()` from a debounced background task, never inline.

    One round = one committed user utterance (voice / typed / inject) or one proactive
    turn. Speculative partials only leave STT marks in `_spec_marks`; a round is opened
    on commit and inherits the marks of the partial id it reuses. Receipts are stored by
    turn_id independently of rounds, so a surface receipt that arrives before the commit
    and a writer receipt that arrives after the round are both attached.
    """

    MAX_ROUNDS = 200
    MAX_PARKED_TIDS = 400
    MAX_SPEC_MARKS = 64
    MARK_KEYS = ("speech_start_ms", "speech_end_ms", "transcript_ms", "commit_ms",
                 "first_audio_ms", "last_audio_end_ms", "interrupted_ms", "ai_end_ms")
    SEGMENT_ORDER = ("reflection_llm", "stt", "embedding", "surface_llm", "surface_tts",
                     "deep_llm", "deep_tts", "writer_llm")
    RECEIPT_FIELDS = ("turn_id", "phase", "status", "model", "provider", "started_ms",
                      "first_token_ms", "last_token_ms", "input_tokens", "cached_tokens",
                      "output_tokens", "cost_usd", "request_id", "open_turn_id")
    LLM_PHASES = ("surface", "deep", "memory_write")        # + the origin reflection

    def __init__(self, max_rounds: int = MAX_ROUNDS):
        self.max_rounds = max(1, int(max_rounds))
        self.reset()

    # ---------------- store ---------------- #

    def reset(self):
        """Browser reset button: forget every round and receipt (round numbers restart at 1)."""
        self.rounds = {}            # turn_id -> round (insertion order = age)
        self._round_seq = 0
        self._parked = {}           # turn_id -> [receipt] for turns with no round (yet)
        self._reflections = {}      # open_turn_id -> reflection receipt
        self._reflection_recent = []  # last reflection receipts without open_turn_id (origin fallback)
        self._spec_marks = getattr(self, "_spec_marks", {})  # survive a reset mid-utterance
        self._dirty = {}
        self._out = []              # void notices, in order

    @staticmethod
    def _cap(d: dict, n: int):
        while len(d) > n:
            d.pop(next(iter(d)))

    def has_round(self, tid) -> bool:
        return tid in self.rounds

    def has_pending(self) -> bool:
        return bool(self._dirty or self._out)

    def _touch(self, rd):
        if rd is not None:
            self._dirty[rd["turn_id"]] = True

    # ---------------- STT marks of speculative partials ---------------- #

    def note_speculative(self, tid, marks: dict):
        if not tid:
            return
        self._spec_marks[tid] = {k: v for k, v in (marks or {}).items() if v is not None}
        self._cap(self._spec_marks, self.MAX_SPEC_MARKS)

    def marks_for(self, tid) -> dict:
        return dict(self._spec_marks.get(tid) or {})

    # ---------------- rounds ---------------- #

    def open_round(self, tid, origin: str = "user", input_kind: str = "voice", marks=None,
                   origin_info=None, forced: bool = False, begin_ms=None):
        """Create (or return) the round for a committed / adopted turn id."""
        if not tid:
            return None
        rd = self.rounds.get(tid)
        if rd is not None:
            return rd
        self._round_seq += 1
        rd = {
            "turn_id": tid, "round": self._round_seq,
            "origin": "reflection" if origin == "reflection" else "user",
            "input": input_kind, "final": False, "status": "open", "forced": bool(forced),
            "marks": {k: None for k in self.MARK_KEYS},
            "receipts": [], "reflection": None, "phases": set(), "tts": {},
            "origin_info": dict(origin_info) if isinstance(origin_info, dict) else None,
            "begin_ms": begin_ms, "seq": 0, "emitted": False,
        }
        for k, v in (marks or {}).items():
            if k in rd["marks"] and v is not None:
                rd["marks"][k] = v
        self.rounds[tid] = rd
        while len(self.rounds) > self.max_rounds:
            old = next(iter(self.rounds))
            self.rounds.pop(old)
            self._dirty.pop(old, None)
        for r in self._parked.pop(tid, None) or []:
            self._attach(rd, r)
        refl = self._reflections.pop(tid, None)
        if refl is None and rd["origin_info"]:
            refl = self._match_origin_reflection(rd["origin_info"])
        if refl is not None and rd["origin"] == "reflection":
            rd["reflection"] = refl
        self._touch(rd)
        return rd

    def _match_origin_reflection(self, info):
        rid, rstart = info.get("receipt_turn_id"), info.get("receipt_started_ms")
        for r in reversed(self._reflection_recent):
            if rid is not None and r.get("turn_id") != rid:
                continue
            if rstart is not None and r.get("started_ms") != rstart:
                continue
            if rid is None and rstart is None:
                return None
            return r
        return None

    def void_round(self, tid) -> bool:
        """A committed turn the brain prunes (continued / replaced before it was protected)."""
        rd = self.rounds.pop(tid, None)
        if rd is None:
            return False
        self._dirty.pop(tid, None)
        if rd["round"] == self._round_seq:
            self._round_seq -= 1          # the continuation's round takes the same number
        if rd["emitted"]:
            self._out.append({"type": "ai_timeline_void", "turn_id": tid, "round": rd["round"]})
        return True

    def supersede(self, tid, now_ms, protected: bool, speaking: bool):
        """The user took over (voice or typed). Returns 'void' | 'interrupted' | 'final' | None."""
        rd = self.rounds.get(tid)
        if rd is None or rd["final"]:
            return None
        if not protected:
            self.void_round(tid)
            return "void"
        if speaking and rd["status"] != "error":
            rd["marks"]["interrupted_ms"] = now_ms
            rd["status"] = "interrupted"
            self.finalize(tid)
            return "interrupted"
        self.finalize(tid)
        return "final"

    def finalize(self, tid):
        rd = self.rounds.get(tid)
        if rd is None or rd["final"]:
            return
        rd["final"] = True
        if rd["status"] == "open":
            rd["status"] = "done"
        self._touch(rd)

    def mark(self, tid, name, value, only_if_unset: bool = False):
        rd = self.rounds.get(tid)
        if rd is None or name not in rd["marks"] or value is None:
            return
        if only_if_unset and rd["marks"][name] is not None:
            return
        rd["marks"][name] = value
        self._touch(rd)

    def note_phase(self, tid, phase):
        rd = self.rounds.get(tid)
        if rd is not None and phase not in rd["phases"]:
            rd["phases"].add(phase)
            self._touch(rd)

    def note_error(self, tid):
        rd = self.rounds.get(tid)
        if rd is not None:
            rd["status"] = "error"
            self._touch(rd)

    # ---------------- receipts (§1.1) ---------------- #

    @classmethod
    def clean_receipt(cls, data):
        if not isinstance(data, dict):
            return None
        r = {k: data.get(k) for k in cls.RECEIPT_FIELDS}
        if not r.get("phase"):
            return None
        return r

    def add_receipt(self, data):
        """Store one call_receipt; returns the cleaned receipt (None when malformed)."""
        r = self.clean_receipt(data)
        if r is None:
            return None
        if r["phase"] == "reflection":
            open_tid = r.get("open_turn_id")
            if open_tid:
                rd = self.rounds.get(open_tid)
                if rd is not None:
                    rd["reflection"] = r
                    self._touch(rd)
                else:
                    self._reflections[open_tid] = r
                    self._cap(self._reflections, self.MAX_PARKED_TIDS)
            else:
                self._reflection_recent.append(r)
                del self._reflection_recent[:-20]
            return r
        tid = r.get("turn_id")
        if not tid:
            return r
        rd = self.rounds.get(tid)
        if rd is not None:
            self._attach(rd, r)
            self._touch(rd)
        else:
            self._parked.setdefault(tid, []).append(r)
            self._cap(self._parked, self.MAX_PARKED_TIDS)
        return r

    @staticmethod
    def _receipt_key(r):
        return (r.get("phase"), r.get("request_id"), r.get("started_ms"), r.get("status"))

    def _attach(self, rd, r):
        key = self._receipt_key(r)
        if any(self._receipt_key(x) == key for x in rd["receipts"]):
            return
        rd["receipts"].append(r)

    # ---------------- TTS jobs ---------------- #

    def _tts_item(self, tid, idx):
        rd = self.rounds.get(tid)
        if rd is None or idx is None:
            return None, None
        it = rd["tts"].get(idx)
        if it is None:
            it = rd["tts"][idx] = {"kind": None, "sfx": False, "sent_ms": None, "received_ms": None,
                                   "duration_ms": None, "played_ms": None, "finished_ms": None,
                                   "failed": False}
        return rd, it

    def note_tts_request(self, tid, idx, kind, sent_ms):
        rd, it = self._tts_item(tid, idx)
        if it is None:
            return
        it["kind"] = kind if kind in ("surface", "deep") else "deep"
        it["sent_ms"] = sent_ms
        self._touch(rd)

    def note_tts_audio(self, tid, idx, received_ms, duration_s):
        rd, it = self._tts_item(tid, idx)
        if it is None:
            return
        it["received_ms"] = received_ms
        try:
            it["duration_ms"] = int(round(float(duration_s) * 1000)) if duration_s else None
        except (TypeError, ValueError):
            it["duration_ms"] = None
        self._touch(rd)

    def note_tts_failed(self, tid, idx):
        rd, it = self._tts_item(tid, idx)
        if it is None:
            return
        if it["received_ms"] is None:
            it["failed"] = True
            self._touch(rd)

    def note_sfx(self, tid, idx, kind, dispatched_ms):
        rd, it = self._tts_item(tid, idx)
        if it is None:
            return
        it.update(kind=kind, sfx=True, sent_ms=dispatched_ms, received_ms=dispatched_ms, duration_ms=None)
        self._touch(rd)

    def note_played(self, tid, idx, ms):
        rd = self.rounds.get(tid)
        it = rd["tts"].get(idx) if rd is not None else None
        if it is not None and it["played_ms"] is None:
            it["played_ms"] = ms
            self._touch(rd)

    def note_finished(self, tid, idx, ms):
        rd = self.rounds.get(tid)
        it = rd["tts"].get(idx) if rd is not None else None
        if it is not None and it["finished_ms"] is None:
            it["finished_ms"] = ms
            self._touch(rd)

    # ---------------- output ---------------- #

    def pop_messages(self) -> list:
        """Void notices and the full state of every round changed since the last call."""
        out, self._out = self._out, []
        dirty, self._dirty = list(self._dirty), {}
        for tid in dirty:
            if tid in self.rounds:
                out.append(self.snapshot(tid))
        return out

    def all_messages(self) -> list:
        """Every stored round, oldest first (re-sent when the browser reconnects)."""
        self._dirty = {}
        self._out = []
        return [self.snapshot(tid) for tid in list(self.rounds)]

    @staticmethod
    def _sum(values):
        vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return sum(vals) if vals else None

    @staticmethod
    def _num(v):
        return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    def _llm_segment(self, key, receipts):
        rs = sorted(receipts, key=lambda r: (self._num(r.get("started_ms")) is None, r.get("started_ms") or 0))
        start = self._num(rs[0].get("started_ms"))
        firsts = [self._num(r.get("first_token_ms")) for r in rs]
        firsts = [f for f in firsts if f is not None]
        lasts = [self._num(r.get("last_token_ms")) for r in rs]
        lasts = [x for x in lasts if x is not None]
        first = min(firsts) if firsts else None
        last = max(lasts) if lasts else None
        if start is not None and first is not None:
            solid = [start, first]
            stripe = [first, last] if last is not None and last >= first else None
        elif start is not None and last is not None:
            solid, stripe = [start, last], None      # no content: the whole call is solid
        else:
            solid, stripe = None, None
        return {"key": key, "solid": solid, "stripe": stripe,
                "tokens": self._tokens(rs), "cost_usd": self._cost(rs), "status": self._status(rs)}

    def _tokens(self, rs):
        t = {"input": self._sum(r.get("input_tokens") for r in rs),
             "cached": self._sum(r.get("cached_tokens") for r in rs),
             "output": self._sum(r.get("output_tokens") for r in rs)}
        if not any(v is not None for v in t.values()):
            return None
        # the input of the calls that reported a cache figure: the denominator of the cache share (Ollama reports
        # none, so its calls neither count as cached nor as missed)
        t["cache_input"] = self._cache_input(rs)
        return t

    def _cache_input(self, rs):
        return self._sum(r.get("input_tokens") for r in rs
                         if self._num(r.get("cached_tokens")) is not None and self._num(r.get("input_tokens")) is not None)

    def _cost(self, rs):
        c = self._sum(r.get("cost_usd") for r in rs)
        return round(c, 8) if c is not None else None

    @staticmethod
    def _status(rs):
        sts = [str(r.get("status") or "unknown") for r in rs]
        bad = [s for s in sts if s != "complete"]
        return bad[0] if bad else "complete"

    @staticmethod
    def _pending(key):
        return {"key": key, "solid": None, "stripe": None, "tokens": None, "cost_usd": None,
                "status": "pending"}

    def _audio(self, rd):
        """Per-kind TTS bars, first audio (played ack preferred) and the end of the last audio."""
        items = [rd["tts"][i] for i in sorted(rd["tts"])]
        played = [it["played_ms"] for it in items if it["played_ms"] is not None]
        received = [it["received_ms"] for it in items if it["received_ms"] is not None and not it["sfx"]]
        first_audio = min(played) if played else (min(received) if received else None)
        # End of the last audio: browser ack when present, else simulated serial playback of
        # what was received (start = max(previous end, received), end = start + duration).
        cursor = None
        last_end = None
        for it in items:
            if it["received_ms"] is None and it["played_ms"] is None:
                continue
            start = it["played_ms"]
            if start is None:
                start = it["received_ms"] if cursor is None else max(cursor, it["received_ms"])
            end = it["finished_ms"]
            if end is None:
                end = start + (it["duration_ms"] or 0)
            cursor = end
            last_end = end if last_end is None else max(last_end, end)
        interrupted = rd["marks"]["interrupted_ms"]
        if interrupted is not None and last_end is not None:
            last_end = min(last_end, interrupted)
        kinds = {}
        for kind in ("surface", "deep"):
            ks = [it for it in items if it["kind"] == kind and not it["sfx"] and it["sent_ms"] is not None]
            if not ks:
                continue
            first_req = min(it["sent_ms"] for it in ks)
            recv = [it for it in ks if it["received_ms"] is not None]
            first_recv = min(it["received_ms"] for it in recv) if recv else None
            last_it = max(recv, key=lambda it: it["received_ms"]) if recv else None
            outstanding = [it for it in ks if it["received_ms"] is None and not it["failed"]]
            failed = [it for it in ks if it["failed"]]
            if interrupted is not None and outstanding:
                status = "interrupted"
            elif outstanding and not rd["final"]:
                status = "pending"
            elif failed or outstanding:
                status = "partial" if recv else "failed"
            else:
                status = "complete"
            kinds[kind] = {
                "key": f"{kind}_tts",
                "solid": [first_req, first_recv] if first_recv is not None else None,
                "stripe": [first_recv, last_it["received_ms"]] if last_it is not None else None,
                "tokens": None, "cost_usd": None, "status": status,
                "sentences": len(ks),
                "audio_ms": self._sum(it["duration_ms"] for it in recv),
                "first_received_ms": first_recv,
                "last_end_ms": (last_it["received_ms"] + (last_it["duration_ms"] or 0)) if last_it else None,
            }
        return first_audio, last_end, kinds

    def snapshot(self, tid) -> dict:
        rd = self.rounds[tid]
        marks = dict(rd["marks"])
        first_audio, last_end, kinds = self._audio(rd)
        marks["first_audio_ms"] = first_audio
        marks["last_audio_end_ms"] = last_end
        by_phase = {}
        for r in rd["receipts"]:
            by_phase.setdefault(r.get("phase"), []).append(r)
        seg = {}
        if rd["origin"] == "reflection" and rd["reflection"] is not None:
            seg["reflection_llm"] = self._llm_segment("reflection_llm", [rd["reflection"]])
        elif rd["origin"] == "reflection" and not rd["final"] and rd["origin_info"]:
            seg["reflection_llm"] = self._pending("reflection_llm")
        if marks["speech_end_ms"] is not None and marks["transcript_ms"] is not None:
            seg["stt"] = {"key": "stt", "solid": [marks["speech_end_ms"], marks["transcript_ms"]], "stripe": None,
                          "tokens": None, "cost_usd": None, "status": "complete"}
        if by_phase.get("query_embedding"):
            rs = by_phase["query_embedding"]
            starts = [self._num(r.get("started_ms")) for r in rs]
            ends = [self._num(r.get("last_token_ms")) for r in rs]
            starts = [s for s in starts if s is not None]
            ends = [e for e in ends if e is not None]
            seg["embedding"] = {"key": "embedding",
                                "solid": [min(starts), max(ends)] if starts and ends else None,
                                "stripe": None, "tokens": self._tokens(rs), "cost_usd": self._cost(rs),
                                "status": self._status(rs)}
        for phase, key in (("surface", "surface_llm"), ("deep", "deep_llm"), ("memory_write", "writer_llm")):
            if by_phase.get(phase):
                seg[key] = self._llm_segment(key, by_phase[phase])
            elif phase in rd["phases"]:
                seg[key] = self._pending(key)      # the brain streamed it; receipt not in yet
        for kind in ("surface", "deep"):
            if kind in kinds:
                seg[f"{kind}_tts"] = kinds[kind]
        segments = []
        for key in self.SEGMENT_ORDER:
            if key in seg:
                s = dict(seg[key])
                s.pop("first_received_ms", None)
                s.pop("last_end_ms", None)
                segments.append(s)
        # ---- round_summary (§1.2); missing values are null, never 0 ---- #
        latency = None
        if marks["first_audio_ms"] is not None and marks["speech_end_ms"] is not None:
            latency = marks["first_audio_ms"] - marks["speech_end_ms"]
        gap = None
        if "surface" in kinds and "deep" in kinds:
            s_end, d_first = kinds["surface"]["last_end_ms"], kinds["deep"]["first_received_ms"]
            if s_end is not None and d_first is not None:
                gap = d_first - s_end
        llm = [r for r in rd["receipts"] if r.get("phase") in self.LLM_PHASES]
        if rd["origin"] == "reflection" and rd["reflection"] is not None:
            llm.append(rd["reflection"])
        total_in = self._sum(r.get("input_tokens") for r in llm)
        # Prompt cache share over the calls that reported a cache figure only: a provider that reports none
        # (Ollama) leaves cached_tokens null, and its input must not count as a miss. None = "n/a".
        reporting = [r for r in llm if self._num(r.get("input_tokens")) is not None
                     and self._num(r.get("cached_tokens")) is not None]
        cached = self._sum(r.get("cached_tokens") for r in reporting)
        cache_in = self._sum(r.get("input_tokens") for r in reporting)
        cached_pct = round(100.0 * cached / cache_in, 1) if cache_in else None
        every = list(rd["receipts"]) + ([rd["reflection"]] if rd["origin"] == "reflection" and rd["reflection"] else [])
        cost = self._cost(every) if every else None
        t0 = self._round_start(rd, marks)
        end = marks["interrupted_ms"] if marks["interrupted_ms"] is not None else marks["last_audio_end_ms"]
        # §1.2: speech_start -> last audio end (or interrupted_ms). A typed/inject round has no
        # speech_start, so its duration is null (t0_ms = commit still anchors the label and bar);
        # a proactive round starts at its reflection call (the round starts with that bar).
        d0 = marks["speech_start_ms"] if rd["origin"] != "reflection" else t0
        duration = end - d0 if end is not None and d0 is not None else None
        rd["seq"] += 1
        rd["emitted"] = True
        return {
            "type": "ai_timeline", "turn_id": tid, "round": rd["round"], "origin": rd["origin"],
            "final": rd["final"], "marks": marks, "segments": segments,
            "round_summary": {
                "latency_to_surface_ms": latency, "surface_to_deep_gap_ms": gap,
                "total_input_tokens": total_in, "cached_pct": cached_pct, "cost_usd": cost,
                "round_duration_ms": duration,
                # additive: the numbers behind cached_pct (null when no call reported a cache figure)
                "cached_tokens": cached if cache_in else None, "cache_input_tokens": cache_in,
            },
            # additive fields (not in §1.2's list; the browser may ignore them)
            "input": rd["input"], "status": rd["status"], "forced": rd["forced"],
            "t0_ms": t0, "seq": rd["seq"],
        }

    def _round_start(self, rd, marks):
        """Round origin t0_ms (label time, bar origin): speech_start for voice rounds; commit for
        typed/inject rounds (their round_duration_ms is null); the reflection call's start (or
        begin) for proactive rounds (spec gap: those have no speech_start)."""
        if marks["speech_start_ms"] is not None:
            return marks["speech_start_ms"]
        if rd["origin"] == "reflection":
            refl = rd["reflection"]
            if refl is not None and self._num(refl.get("started_ms")) is not None:
                return refl["started_ms"]
            info = rd["origin_info"] or {}
            if self._num(info.get("receipt_started_ms")) is not None:
                return info["receipt_started_ms"]
            return rd["begin_ms"]
        return marks["commit_ms"] if marks["commit_ms"] is not None else rd["begin_ms"]


class DialogSession:
    """
    单条 WebSocket 对话的状态机。

    协议：
      前端 -> 后端：
        - 二进制帧：16kHz 单声道 PCM16 LE（裸字节）
        - JSON 控制消息：
            {"type": "hello"}         连接后先发一次
            {"type": "reset"}         清空状态
            {"type": "stop"}          结束采集，立即 commit 剩余段
      后端 -> 前端：
        {"type": "ready"}                      WS 就绪
        {"type": "speech_start", "ts": ..}     VAD 检测到开始说话
        {"type": "partial", "text": .., "ts":..}   短停顿后的推测 ASR 结果
        {"type": "cancel", "ts": ..}           短停顿内又说话 -> 推测作废
        {"type": "commit", "text": .., "turn_id": "YYYYMMDD-HHMMSS-N", "ts":..}
                                               长停顿 -> 定稿；turn_id 用于匹配 AI 回复
        {"type": "ai_begin"|"ai_surface_chunk"|"ai_surface_done"|
                 "ai_deep_chunk"|"ai_deep_done"|"ai_end"|"ai_error",
         "turn_id": "YYYYMMDD-HHMMSS-N", ...}  Hane 回来的事件（仅转发匹配 committed_turn_id 的）
        {"type": "tts_audio", "turn_id": "..-0", "idx": 0, "text": "..",
         "audio_format": "opus"|"wav", "audio_duration": 1.23,
         "audio_b64": "base64 编码的音频"}   TTS 生成的一段音频，前端解码播放
        {"type": "tts_error", "turn_id": "..-0", "idx": 0, "error": "..."}
        tts_audio / tts_sfx / tts_error carry "kind" (surface|deep) and "surface_last" (this is the
        surface's final audio); ai_surface_done carries "surface_tts_done" (no surface audio still to
        come) — the Use page shows her "loading" emote only between the surface's last audio and the
        deep's first.
        {"type": "ai_call_receipt", ...}     brain call_receipt (TELEMETRY_SPEC §1.1), for tooltips
        {"type": "ai_timeline", ...}         per-round timeline, full state each time (§1.2)
        {"type": "ai_timeline_void", "turn_id", "round"}  a round the brain pruned (continued before protect)
      前端 -> 后端（另）：{"type": "timeline_reset"} 清空所有 round 时间线
    """

    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512            # silero-vad 在 16kHz 要求固定 512
    FRAME_BYTES = FRAME_SAMPLES * 2  # s16le
    FRAME_MS = FRAME_SAMPLES * 1000 / SAMPLE_RATE  # 32ms

    # VAD 阈值迟滞（hysteresis）：避免概率在 0.5 附近抖动
    # 从 IDLE 切到 SPEAKING 需要 >= POS，反之 < NEG 才算静音
    VAD_POS_THRESHOLD = 0.60
    VAD_NEG_THRESHOLD = 0.45
    # redemption 机制：连续 REDEMPTION_FRAMES 帧都判为静音才真正算短停顿
    # 单帧偶发的低概率（词尾轻辅音、短促换气）不会误触发 partial
    REDEMPTION_FRAMES = 1   # 1 帧 ≈ 32ms（配合迟滞阈值 NEG=0.45，已足够稳）
    LONG_PAUSE_MS = 500     # 在线：从开始静音起到此时长则定稿（commit 信号），音频分界交给 protect
    OFFLINE_PAUSE_MS = 900  # 离线：Hane/TTS 没连上时退化为纯 STT，900ms 直接 reset（原有行为）
    PRE_SPEECH_PAD_MS = 200 # IDLE 时保留最近 N ms 音频，开口时塞回前面，防止丢字头
    MAX_UTTERANCE_SEC = 30
    TTS_FIRST_AUDIO_TIMEOUT_MS = 2000  # 发出 TTS 请求后 2s 内没首音频 -> 视为已到达，直接 protect
    TTS_INFLIGHT_TIMEOUT_MS = 5000  # 任一句 TTS 5s 内没回 audio_ready/error -> 跳过排空队列，防卡死
    # 真打断"迟到确认"门控：仅在 AI 实际正在说话（_protected=True 且 TTS 还在活跃）时生效。
    # 从 VAD 首次检测到用户语音起：
    #   0  - INTERRUPT_GATE_LOCK_MS   锁定期，静默吃掉；帧入 probe 缓冲
    #   LOCK - UNLOCK_END            解锁窗口，若仍有语音帧 -> 真打断触发
    #   > UNLOCK_END                  窗口过期，重开新探针（VAD 继续监听）
    # 另外：探针期内遇到任一静音帧即重置 -> 需要"连续"语音才能穿透门控。
    INTERRUPT_GATE_LOCK_MS = 400
    INTERRUPT_GATE_UNLOCK_END_MS = 600
    # Round telemetry (TELEMETRY_SPEC §1.2): one TimelineBook per process (only one dialog is
    # open at a time) so the rounds survive a reconnect; sent from a debounced task.
    TIMELINE_EMIT_DELAY_S = 0.2
    _TIMELINE_BOOK = None

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.voice_output = True
        self._pending_context = ""
        self._context_revision = 0
        self._contexts_sent = {}
        self._peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"
        self._leftover = bytearray()
        self._speech_frames: List[np.ndarray] = []
        self._silence_frames = 0
        self._state = "IDLE"  # IDLE / SPEAKING / SHORT_PAUSE
        self._last_partial = ""
        self._last_partial_raw = ""
        self._asr_seq = 0
        self._active_asr_seq = 0
        self._pending_partial: Optional[asyncio.Task] = None
        # IDLE 时的滚动缓冲：开口那一刻把它塞到 speech_frames 前面
        pad_frames = max(1, int(self.PRE_SPEECH_PAD_MS / self.FRAME_MS))
        self._pre_buffer: deque = deque(maxlen=pad_frames)
        vad_model.reset_states()
        # AI Core 对接
        # turn_id 格式：YYYYMMDD-HHMMSS-N（N 是本连接累计的轮数）。
        # 方便 Hane 端历史记录按 id 回看时间，不精确到毫秒。
        self._turn_seq = 0
        # _current_sent_id: 最近一次发给 Hane 的 message 的 id（每次 partial 都会换）。
        # _committed_turn_id: 前端已 commit 的那轮 id。在 commit 前保持 None。
        # _pending_chunks: 针对 _current_sent_id 提前到达的事件，commit 时一次性 flush 给浏览器。
        # 关键：Hane 收到 partial 的 message 后就开始生成，回复的 chunk 可能在我们还没 commit
        # 时已经开始到达；我们先缓存，commit 时把它们当作"这一轮的正式回复"flush 出去。
        self._current_sent_id: Optional[str] = None
        self._committed_turn_id: Optional[str] = None
        self._pending_chunks: List[dict] = []
        self.hane = HaneClient(self)
        # TTS 对接
        # 串行发送：一句音频回来之后再发下一句。
        # _tts_inflight: 当前在途的 TTS 请求 {task_id, tts_turn_id, idx, text}
        # _tts_sentence_queue: 分句器已切出但还没轮到发的句子列表
        # _tts_next_idx: 该 committed 轮次下一个句子的序号（turn_id 后缀）
        # _tts_first_audio_sent: 这一轮是否已经收到过首段音频 —— 首段返回后
        #                       0.4s emit protect（给浏览器缓冲 + 开始播）
        self.tts = TTSClient(self)
        # Hane 原始输出 → speech 流式抽取。surface/deep 两路分别走，
        # 抽到的 speech 增量才送给前端显示 + 分句器 + TTS。详见 hane_parser.py。
        self._surface_extractor = SurfaceSpeechExtractor()
        self._deep_extractor = DeepSpeechExtractor()
        # 分句器：给 TTS 喂短句。首句快、后续顺。详见 sentence_splitter.py。
        self._sentence_splitter = StreamingSplitter()
        self._tts_sentence_queue: deque = deque()
        self._tts_inflight: Optional[dict] = None
        # 已派发但前端还没报"播完"的声效 idx 集合：用于完成判定（声效不走 inflight，
        # 没有这个集合的话最后一段是声效时会在 ai_end 误判成"已排空"提前发 speech_done）
        self._tts_pending_play: set = set()
        self._tts_next_idx = 0
        self._tts_first_audio_sent = False
        # 当前 _committed_turn_id 是否已经发过 protect。
        # 音频分界点 = protect 发出的那一刻（而不是 500ms 长停顿）。
        # 在 protect 之前，用户继续说话 = 上一段的延续，_speech_frames 保留;
        # 在 protect 之后，用户继续说话 = 新一轮，_speech_frames 清空。
        self._protected = False
        # TTS 播放追踪（前端上报 tts_playing / tts_finished）+ 打断锚点：
        # _tts_sentences_by_idx[idx] = {"text": 原始句子, "duration": 音频秒数, "turn_id": tts_turn_id}
        # _tts_playing:    当前正在播的 {idx, text, duration, start_ms}，不在播就是 None
        # _tts_last_finished: 最后一段刚播完的 {idx, text, duration}，给"protect 后静默期被打断"用
        # _my_partial_pending: 打断时算出的 ~4 字锚点；下次发 Hane message 时带上，
        #                     AI 成功 protect 后清空
        self._tts_sentences_by_idx: dict = {}
        self._tts_playing: Optional[dict] = None
        self._tts_last_finished: Optional[dict] = None
        self._my_partial_pending: Optional[str] = None
        # 打断那一刻 AI 正在说的语言（CHINESE / ENGLISH / JAPANESE / MIXED），
        # 用来选占位符；生命周期随 _my_partial_pending 一起。
        self._my_partial_lang: Optional[str] = None
        # AI 是否已经完整结束本轮（ai_end 事件到达）。配合 TTS 队列/inflight/playing
        # 一起判断"AI 当下是否还在说"——避免用户等 AI 说完再开口时被误判成打断。
        self._ai_end_received: bool = False
        # 本轮音频排空后是否已给 Hane 发过 speech_done（每轮一次）。
        self._speech_done_sent: bool = False
        # 「说话时点的念弹幕 / SC」缓存：主动轮 begin 到达时若音频还忙，不丢弃而是把
        # 这一轮的全部事件（begin…end）缓存在这里，等当前音频彻底排空后再放出来认领出声
        # （见 _maybe_fire_buffered_proactive）。{turn_id: [ {event_type, data}, ... ]}，
        # 按到达顺序排队。用户开口打断时（speech_start 两处）整片清空——过时的不会迟到乱念。
        self._buffered_proactive: dict = {}
        # 真打断门控探针状态（AI 说话期间抵御 0.4s 以内的瞬间噪声/咳嗽）：
        # _interrupt_probe_start_ms: 第一次探到语音的 wall-clock 时间；None = 无活动探针
        # _probe_frames: 探针期攒下的语音帧；穿透门控时一起拼到 speech_frames 前面保留字头
        self._interrupt_probe_start_ms: Optional[int] = None
        self._probe_frames: List[np.ndarray] = []
        # Build F (2026-09-26): text of an utterance that was force-committed at
        # MAX_UTTERANCE_SEC but never protected. It is prefixed to the next
        # messages sent to Hane so a monologue split by the cap keeps its first
        # half instead of losing it to the brain's prune of unprotected turns.
        # Cleared when a turn is protected or the memory is wiped.
        self._carry_text: str = ""
        self._last_commit_text: str = ""
        # Turn ids whose brain message contains the carry (or that are the
        # force-committed turn itself). Protecting one of them makes the carried
        # text durable, so only then is the carry dropped.
        self._carry_turn_ids: set = set()
        # Turns that ended with a brain error never became durable; their
        # protect must not drop the carry.
        self._carry_error_turns: set = set()
        # Turn whose protect round trip is in flight, and the (turn, text) that the
        # last acknowledged protect made durable (restored if that turn errors).
        self._protect_inflight: Optional[str] = None
        self._carry_snapshot: Optional[tuple] = None
        # PromptOptimized (CONTRACT §4.5 / §5): the brain's per-turn reply language
        # ('zh' / 'en' / 'ja' / None) from the begin / surface_done payloads. Used only as the
        # tie-break for MIXED / kanji-only TTS sentences and for observe-only [lang] logging.
        self._turn_reply_lang: Optional[str] = None
        self._turn_reply_lang_reason: Optional[str] = None
        # Whether this turn's surface extractor produced any spoken text (think-mode filler check).
        self._surface_spoke = False
        # Round telemetry: which LLM the sentences being queued belong to ('surface' until the
        # surface_done flush, then 'deep'); VAD start of the current utterance; the last
        # SHORT_PAUSE trigger (end of the user's last word) for the fresh-ASR commit path.
        self._tts_kind = "surface"
        self._utt_speech_start_ms: Optional[int] = None
        self._last_trigger_ms: Optional[int] = None
        self._tl = self._shared_timeline_book()
        self._tl_outbox: list = []
        self._tl_task = None

    def _make_turn_id(self) -> str:
        self._turn_seq += 1
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{ts}-{self._turn_seq}"

    def _reset_tts_for_new_turn(self):
        """新一轮 commit 开始时调用。清空上一轮 TTS 状态。
        之前 in-flight 的 TTS 请求我们不显式取消（也没法取消），让它 audio_ready
        回来时通过 _committed_turn_id 前缀比对被识别为 stale 丢弃即可。
        """
        self._surface_extractor.reset()
        self._deep_extractor.reset()
        self._sentence_splitter.reset()
        self._tts_sentence_queue.clear()
        self._tts_inflight = None
        self._tts_next_idx = 0
        self._tts_first_audio_sent = False
        self._protected = False
        self._tts_sentences_by_idx = {}
        self._tts_playing = None
        self._tts_last_finished = None
        self._tts_pending_play.clear()
        self._ai_end_received = False
        self._speech_done_sent = False
        # 新一轮的 reply_lang 由它自己的 begin / surface_done 带来
        self._turn_reply_lang = None
        self._turn_reply_lang_reason = None
        self._surface_spoke = False
        self._tts_kind = "surface"
        # 新一轮 -> 之前残留的门控探针无意义，清掉
        self._interrupt_probe_start_ms = None
        self._probe_frames = []
        # 注意：_my_partial_pending 在 protect 成功后才清，本轮 partial/commit 每次都带

    @property
    def _ai_pipeline_online(self) -> bool:
        """Hane 和 TTS 都连上才算 AI 管道在线。
        离线时整个 POST_COMMIT/protect 机制失效，退化为纯 STT（900ms 直接 reset）。
        """
        return self.hane._connected

    @property
    def long_pause_frames(self) -> int:
        ms = self.LONG_PAUSE_MS if self._ai_pipeline_online else self.OFFLINE_PAUSE_MS
        return max(1, int(ms / self.FRAME_MS))

    @property
    def utterance_frames(self) -> int:
        return len(self._speech_frames)

    def _reset_utterance(self):
        self._speech_frames = []
        self._silence_frames = 0
        self._state = "IDLE"
        self._last_partial = ""
        self._last_partial_raw = ""
        self._pending_partial = None
        self._pre_buffer.clear()
        vad_model.reset_states()
        # MAX_UTTERANCE 强制 reset / 离线兜底 / force_stop 等异常路径也清锚点，
        # 避免陈旧 my_partial 带到下一轮。protect 成功的正常路径不走这里，
        # my_partial 的清空由 _protect_after_first_audio 负责。
        self._my_partial_pending = None
        self._my_partial_lang = None
        self._interrupt_probe_start_ms = None
        self._probe_frames = []

    async def _send(self, obj: dict):
        try:
            await self.ws.send_json(obj)
        except Exception as e:
            logger.debug(f"[dlg {self._peer}] send failed: {e}")

    async def feed_pcm(self, data: bytes):
        """把一段裸 PCM16 追加进来，按 512 样本切帧处理。"""
        if not data:
            return
        self._leftover.extend(data)
        while len(self._leftover) >= self.FRAME_BYTES:
            raw = bytes(self._leftover[:self.FRAME_BYTES])
            del self._leftover[:self.FRAME_BYTES]
            pcm16 = np.frombuffer(raw, dtype="<i2")
            pcm_f32 = pcm16.astype(np.float32) / 32768.0
            await self._on_frame(pcm_f32)

    async def _on_frame(self, frame: np.ndarray):
        with torch.no_grad():
            prob = vad_model(torch.from_numpy(frame), self.SAMPLE_RATE).item()

        # 阈值迟滞：IDLE 时用高阈值，已在说话时用低阈值
        if self._state == "IDLE":
            is_speech = prob >= self.VAD_POS_THRESHOLD
        else:
            is_speech = prob >= self.VAD_NEG_THRESHOLD

        if is_speech:
            if self._state == "IDLE":
                # AI 正在说话时，真打断要经过"迟到确认"门控：
                # 先攒 0.4s 探针，期间短暂爆发（咳嗽/清嗓）直接吃掉；过了 0.4s 仍在
                # 说话 -> 真打断。细节在 _gate_handle_probe，返回 False 代表"还没穿透"。
                if self._protected and self._is_ai_speaking_now():
                    if not self._gate_handle_probe(frame):
                        return
                    # 穿透门控：算 my_partial（必须在 _reset_tts_for_new_turn 清
                    # _tts_playing 之前），接着走原打断流程
                    mp = self._compute_my_partial()
                    if mp:
                        self._my_partial_pending = mp
                        logger.info(f"[dlg {self._peer}] interrupt AI (gate passed), my_partial={mp!r}")
                # 门控穿透时 _probe_frames 已经是锁定期 + 当前前期的连续语音，
                # 需要在 _reset_tts_for_new_turn 把它清掉之前先 stash 起来
                stashed_probe = self._probe_frames
                self._probe_frames = []
                probe_start_ms = self._interrupt_probe_start_ms
                self._interrupt_probe_start_ms = None
                self._state = "SPEAKING"
                now_ms = _now_ms()
                # telemetry: the utterance began at the probe start when the gate held it back
                self._utt_speech_start_ms = probe_start_ms if probe_start_ms is not None else now_ms
                self._last_trigger_ms = None
                self._tl_supersede(now_ms, interrupted=self._protected and self._is_ai_speaking_now())
                # 用户开口 = 旧一轮作废：committed 立即清空，TTS 队列 / inflight
                # 重置。这样后续旧 turn 的 audio_ready 回来时 _on_tts_event 会比对
                # committed 失败直接 drop，queue 里还没发的句子也不再发出。
                # 前端同步在 speech_start 事件里 stopAllTTS()。
                self._committed_turn_id = None
                self._reset_tts_for_new_turn()
                # 用户开口接管 = 缓存的念弹幕/SC 已过时，整片丢弃，绝不迟到乱念
                self._buffered_proactive.clear()
                # pre-speech padding: 把 IDLE 期间攒的音频塞到 utterance 前面
                # 防止 VAD 判断滞后 / 锁定期吃帧导致字头被切掉
                if self._pre_buffer:
                    self._speech_frames.extend(self._pre_buffer)
                    self._pre_buffer.clear()
                if stashed_probe:
                    self._speech_frames.extend(stashed_probe)
                await self._send({"type": "speech_start", "ts": now_ms})
            elif self._state == "SHORT_PAUSE":
                # 用户在短停顿内又开口了，推测作废
                self._state = "SPEAKING"
                self._active_asr_seq = self._asr_seq + 1  # 让进行中的 partial 结果失效
                await self._send({"type": "cancel", "ts": _now_ms()})
            elif self._state == "POST_COMMIT":
                # 长停顿 commit 之后又开口。分两种情况：
                #   a) protect 已经发出（_protected=True）：AI 已经在说话，这次
                #      是真正的新一轮 / 打断 → 彻底 reset（和 IDLE → SPEAKING 一样）。
                #   b) protect 未发：AI 还没出声，这次是"上一段的延续"——
                #      旧 AI turn 作废（_committed_turn_id=None + 清 TTS 队列），
                #      但 _speech_frames 保留；下次 partial 的 ASR 对象就是
                #      "从上次 protect 后开口以来的完整录音"。
                was_protected = self._protected
                now_ms = _now_ms()
                # telemetry: an unprotected commit is pruned by the brain (void); a protected
                # one ends here (interrupted when she is still speaking)
                self._tl_supersede(now_ms, interrupted=was_protected and self._is_ai_speaking_now())
                if was_protected:
                    self._utt_speech_start_ms = now_ms       # a new utterance
                    self._last_trigger_ms = None
                if was_protected and self._is_ai_speaking_now():
                    # 新一轮 + 真打断：先算 my_partial，再清音频 / TTS 状态
                    mp = self._compute_my_partial()
                    if mp:
                        self._my_partial_pending = mp
                        logger.info(f"[dlg {self._peer}] interrupt AI, my_partial={mp!r}")
                    # 和 IDLE→SPEAKING 一样重置音频，并拼 pre_buffer
                    self._speech_frames = []
                    vad_model.reset_states()
                self._committed_turn_id = None
                self._reset_tts_for_new_turn()
                # 用户开口接管 = 缓存的念弹幕/SC 已过时，整片丢弃，绝不迟到乱念
                self._buffered_proactive.clear()
                # 旧轮 AI 的缓存 chunks / 进行中的投机 chunks 统统作废
                self._pending_chunks = []
                # 让下一次 partial 从零覆盖
                self._last_partial = ""
                self._last_partial_raw = ""
                if was_protected and self._pre_buffer:
                    self._speech_frames.extend(self._pre_buffer)
                self._pre_buffer.clear()
                self._state = "SPEAKING"
                await self._send({
                    "type": "speech_start",
                    "ts": _now_ms(),
                    "continuation": (not was_protected),
                })
            self._silence_frames = 0
            self._speech_frames.append(frame)
        else:
            if self._state == "IDLE":
                # IDLE 时滚动保留最近若干帧，供下次 speech_start 时做 pre-padding
                self._pre_buffer.append(frame)
                # 门控探针期间遇到静音 -> 重置（要求"连续"语音才能穿透门控，
                # 断断续续的短爆发不算）
                if self._interrupt_probe_start_ms is not None:
                    self._interrupt_probe_start_ms = None
                    self._probe_frames = []
            elif self._state == "SPEAKING":
                self._speech_frames.append(frame)
                self._silence_frames += 1
                # redemption: 连续 N 帧都静音才算 short_pause（防止词间抖动误触发）
                if self._silence_frames >= self.REDEMPTION_FRAMES:
                    self._state = "SHORT_PAUSE"
                    trigger_ms = _now_ms()
                    self._last_trigger_ms = trigger_ms
                    self._pending_partial = asyncio.create_task(self._do_partial(trigger_ms))
            elif self._state == "SHORT_PAUSE":
                # 继续累积音频（含中间静音）。用户若再开口，整段会是连续的录音，
                # ASR 能拿到完整上下文；当前 utterance 一直延伸，直到长停顿才 commit。
                self._speech_frames.append(frame)
                self._silence_frames += 1
                if self._silence_frames >= self.long_pause_frames:
                    online = self._ai_pipeline_online
                    await self._do_commit()
                    if online:
                        # 在线：长停顿 = commit 信号（告诉前端/TTS 本轮定稿），
                        # 但**不**切音频。音频分界点交给 protect。
                        self._state = "POST_COMMIT"
                    else:
                        # 离线纯 STT：没有 protect 可等，直接 reset 回到干净状态
                        self._reset_utterance()
            elif self._state == "POST_COMMIT":
                # commit 已发、等 protect 的空窗期。静音继续累积到 _speech_frames,
                # 保持 "从上次 protect 后开口到现在" 的完整录音；
                # 真正清空要么在 protect 发出时，要么是新一轮的 speech_start。
                self._speech_frames.append(frame)
                self._silence_frames += 1

        # 安全：单句太长，强行 commit。不管当前 state，强制 reset 避免音频无限涨
        # （例如 Hane/TTS 故障导致 protect 永远不来）。
        if self.utterance_frames * self.FRAME_MS / 1000 > self.MAX_UTTERANCE_SEC:
            logger.warning(f"[dlg {self._peer}] utterance too long, force commit")
            if self._state != "POST_COMMIT":
                if self._state == "SPEAKING":
                    # Build F: the last finished partial does not cover speech after the
                    # owner resumed; drop it so _do_commit runs fresh ASR over every frame.
                    self._last_partial = ""
                    self._last_partial_raw = ""
                # 还没 commit 过，补一刀
                await self._do_commit(cut_ms=_now_ms() if self._state == "SPEAKING" else None)
            # Build F: keep the committed text for the next message unless this
            # turn is protected or its protect is in flight (then it is durable in
            # history). Offline dictation and noise placeholders are never carried.
            committed_text = self._last_commit_text or self._last_partial
            substituted = bool(self._my_partial_pending) and _is_trivial_utterance(committed_text)
            durable = self._protected or (self._protect_inflight is not None and self._protect_inflight == self._committed_turn_id)
            if self._ai_pipeline_online and not substituted:
                before = self._carry_text
                self._carry_text = stt_carry.note_force_commit(self._carry_text, committed_text, durable)
                if self._carry_text != before:
                    if self._committed_turn_id:
                        self._carry_turn_ids.add(self._committed_turn_id)
                    logger.info(f"[dlg {self._peer}] carry force-committed text ({len(self._carry_text)} chars)")
            self._reset_utterance()

    async def _do_partial(self, trigger_ms: int):
        """trigger_ms 是 VAD 判定进入 SHORT_PAUSE 的时间戳（ms, epoch），
        用于前端显示端到端延迟。
        """
        if not self._speech_frames:
            return
        self._asr_seq += 1
        seq = self._asr_seq
        self._active_asr_seq = seq
        audio = np.concatenate(self._speech_frames)
        t0 = time.time()
        text, raw = await _asr_async(audio)
        dt = time.time() - t0
        asr_done_ms = _now_ms()
        if seq != self._active_asr_seq:
            logger.info(f"[dlg {self._peer}] partial #{seq} dropped (stale, took {dt:.2f}s)")
            return
        if self._state not in ("SHORT_PAUSE", "SPEAKING"):
            return
        # 误触过滤：太短（1 个汉字 / <=3 个假名）的识别，平时不更新 _last_partial
        # 也不发给 Hane。**但**：打断场景（_my_partial_pending 非空）下要放行，
        # 由 _send_to_ai 统一替换成噪音占位符——用户真的打断了 AI，即便只是咳嗽
        # 也得让 Hane 知道"被打断了，没有有效内容"。
        trivial = _is_trivial_utterance(text)
        is_interrupt = self._my_partial_pending is not None
        if trivial and not is_interrupt:
            logger.info(f"[dlg {self._peer}] partial #{seq} ({dt:.2f}s) trivial, skip AI: {text!r}")
            await self._send({
                "type": "partial",
                "text": text,
                "raw": raw,
                "trivial": True,
                "ts": _now_ms(),
                "trigger_ms": trigger_ms,
                "asr_ms": int(dt * 1000),
            })
            return
        self._last_partial = text
        self._last_partial_raw = raw
        if trivial:
            logger.info(f"[dlg {self._peer}] partial #{seq} ({dt:.2f}s) trivial UNDER INTERRUPT: {text!r} -> noise placeholder")
        else:
            logger.info(f"[dlg {self._peer}] partial #{seq} ({dt:.2f}s): {text!r}")
        await self._send({
            "type": "partial",
            "text": text,
            "raw": raw,
            "ts": _now_ms(),
            "trigger_ms": trigger_ms,
            "asr_ms": int(dt * 1000),
        })
        # 投机触发 AI 生成：不更新 committed_turn_id，所以后端推回的 chunk 都会被丢弃
        await self._send_to_ai(text, is_commit=False, marks={
            "speech_start_ms": self._utt_speech_start_ms,
            "speech_end_ms": trigger_ms,
            "transcript_ms": asr_done_ms,
        })

    async def _do_commit(self, cut_ms: Optional[int] = None):
        """定稿：直接**复用最近一次 partial 的 turn_id**——那轮 Hane 已经在生成，
        某些 chunk 可能甚至已经到达并被缓存在 self._pending_chunks 里。我们把
        committed_turn_id 指过去，flush 缓存，后续同 id 的 chunk 就直接转发。

        兜底：整段音频太短 / 静音 / 全无识别结果，没跑过 partial，也就没 _current_sent_id。
        这种情况下跑一次 fresh ASR，若有文字就新发一条 message 作为 commit。
        """
        if not self._speech_frames:
            return
        # 等待最近一次 partial 的 ASR 跑完（通常已完成，这里是保险）
        pending = self._pending_partial
        if pending is not None and not pending.done():
            try:
                await pending
            except Exception:
                pass

        text = self._last_partial
        raw = self._last_partial_raw
        reuse_id = self._current_sent_id
        self._last_commit_text = text or ""

        if text and reuse_id is not None:
            # 复用最近一次 partial 的 id；Hane 上那一轮正在跑（或已经在推 chunk）
            logger.info(
                f"[dlg {self._peer}] commit (reuse partial id={reuse_id}): {text!r}; "
                f"buffered chunks={len(self._pending_chunks)}"
            )
            self._committed_turn_id = reuse_id
            await self.consume_context(reuse_id)
            await self.hane.send_commit(reuse_id)   # P2: release the deep of the reused speculative turn
            self._reset_tts_for_new_turn()
            commit_ms = _now_ms()
            self._tl_open_voice_round(reuse_id, commit_ms, reuse=True, cut_ms=cut_ms)
            await self._send({
                "type": "commit", "text": text, "raw": raw,
                "turn_id": reuse_id, "ts": commit_ms,
            })
            # 按到达顺序走"清洗 → 分句 → 发送"管道（_handle_committed_event
            # 里已经在 end / 产句时 pump TTS；这里不用再额外 pump）
            pending = self._pending_chunks
            self._pending_chunks = []
            for entry in pending:
                await self._handle_committed_event(
                    entry["event_type"], reuse_id, entry["data"]
                )
            return

        # 兜底：没跑过 partial 或 partial 被 trivial 过滤掉了 —— 跑 fresh ASR
        fresh_transcript_ms = None
        if not text:
            audio = np.concatenate(self._speech_frames)
            t0 = time.time()
            text, raw = await _asr_async(audio)
            dt = time.time() - t0
            fresh_transcript_ms = _now_ms()
            logger.info(f"[dlg {self._peer}] commit (fresh ASR {dt:.2f}s): {text!r}")
            # fresh ASR 的结果也要过 trivial 过滤（和 partial 同一把尺）——
            # 但打断场景下不过滤，由下面的 _maybe_substitute_noise 替换成占位符。
            is_interrupt = self._my_partial_pending is not None
            if _is_trivial_utterance(text) and not is_interrupt:
                logger.info(f"[dlg {self._peer}] commit fresh trivial, skip AI: {text!r}")
                text = ""
            self._last_commit_text = text or ""

        if text:
            tid = self._make_turn_id()
            self._committed_turn_id = tid
            self._current_sent_id = tid
            self._pending_chunks = []
            self._reset_tts_for_new_turn()
            commit_ms = _now_ms()
            self._tl_open_voice_round(tid, commit_ms, reuse=False, cut_ms=cut_ms,
                                      transcript_ms=fresh_transcript_ms)
            await self._send({
                "type": "commit", "text": text, "raw": raw,
                "turn_id": tid, "ts": commit_ms,
            })
            text_for_hane = self._maybe_substitute_noise(text)
            if text_for_hane == text and self._carry_text:
                # Build F: a force-committed earlier segment rides along (not for noise placeholders).
                text_for_hane = stt_carry.with_carry(self._carry_text, text)
                self._carry_turn_ids.add(tid)
            await self.hane.send_message(text_for_hane, tid, my_partial=self._my_partial_pending)
        else:
            # 没识别到文字就不打扰 Hane
            await self._send({"type": "commit", "text": "", "raw": raw, "ts": _now_ms()})

    # 按"AI 被打断那一刻正在说的语言"挑占位符；MIXED / 未知都退到中文。
    NOISE_INTERRUPT_PLACEHOLDERS = {
        "CHINESE":  "（有一些噪音打断你了）",
        "ENGLISH":  "(Some noise interrupted you.)",
        "JAPANESE": "（雑音が入って遮ってしまいました）",
    }
    # 兼容旧名，外部若引用这个常量仍可用（= 中文版）
    NOISE_INTERRUPT_PLACEHOLDER = NOISE_INTERRUPT_PLACEHOLDERS["CHINESE"]

    def _maybe_substitute_noise(self, text: str) -> str:
        """打断场景（_my_partial_pending 非空）下，若 text 还是 trivial/乱码，
        替换成噪音占位符告诉 Hane"被打断但没有效内容"。
        占位符用 _my_partial_lang 区分中/英/日；平时（非打断）这种 text 在
        _do_partial / _do_commit 前就过滤掉了，不会走到这里。
        """
        if self._my_partial_pending and _is_trivial_utterance(text):
            lang = self._my_partial_lang or "CHINESE"
            return self.NOISE_INTERRUPT_PLACEHOLDERS.get(
                lang, self.NOISE_INTERRUPT_PLACEHOLDERS["CHINESE"]
            )
        return text

    async def _send_to_ai(self, text: str, is_commit: bool, marks: Optional[dict] = None):
        """把文本发给 Hane。每次发都是一个新 turn_id。
        partial 发送时：_current_sent_id 更新为新 id，_committed_turn_id 清空，
        之前缓存的 chunks 全部作废（Hane 也会 prune 掉旧 turn）。
        若上次用户打断过 AI，_my_partial_pending 存有 ~4 字锚点，
        本轮每次 message 都带上，直到 AI 新一轮 protect 成功后清除。"""
        tid = self._make_turn_id()
        self._current_sent_id = tid
        self._pending_chunks = []
        if marks:
            # telemetry: STT marks of this speculative id, inherited if the commit reuses it
            self._tl_hook("note_speculative", tid, marks)
        if is_commit:
            self._committed_turn_id = tid
        else:
            # 新一轮推测 —— 旧的已 commit 那轮（若存在）也随之作废
            self._committed_turn_id = None
        text_to_send = self._maybe_substitute_noise(text)
        if text_to_send == text and self._carry_text:
            # Build F: a force-committed earlier segment rides along (not for noise placeholders).
            text_to_send = stt_carry.with_carry(self._carry_text, text)
            self._carry_turn_ids.add(tid)
        await self.hane.send_message(text_to_send, tid, my_partial=self._my_partial_pending, speculative=not is_commit)

    async def _emit_speech_parts(
        self, kind: str, turn_id: str, parts,
    ) -> bool:
        """抽取器吐出的 [(emotion_key, delta), ...] 这批 speech 片段统一处理：
        每段 delta 单独转发给前端（带 emotion 字段）、依次喂给分句器；每切出
        一个句子就把它和"当前 tone"绑起来入 TTS 队列。

        特殊协议：**delta == "\\n" 是"段封闭哨兵"**（由 DeepSpeechExtractor
        在引号闭合时 emit），收到这种 delta 后 flush 一次分句器，把这段 tone
        下的残留强制归给当前 tone，避免下一段 tone 染进来。
        这样 deep 也能像 surface 一样"来字即吐"，不用等整段引号闭合。
        """
        produced = False
        for emotion_key, delta in parts:
            if not delta:
                continue
            # 声效哨兵：标签处强制分句（把声效前攒的半句以"前一句情绪"挤出来），
            # 再把声效作为独立项入队（不送 TTS、不进显示文本）。
            sfx_name = parse_sfx_delta(delta)
            if sfx_name is not None:
                for s in self._sentence_splitter.flush():
                    self._tts_sentence_queue.append((emotion_key, s, self._tts_kind))
                self._tts_sentence_queue.append((_SFX_QUEUE, (sfx_name, emotion_key), self._tts_kind))
                produced = True
                continue
            await self._send({
                "type": f"ai_{kind}_chunk",
                "turn_id": turn_id,
                "text": delta,
                "emotion": emotion_key,
            })
            if kind == "surface" and delta.strip():
                self._surface_spoke = True
            sentences = self._sentence_splitter.feed(delta)
            if sentences:
                for s in sentences:
                    self._tts_sentence_queue.append((emotion_key, s, self._tts_kind))
                produced = True
            if delta == "\n":
                # 段封闭哨兵：把本段未达 min_len 的残留强制归到当前 tone
                residue = self._sentence_splitter.flush()
                if residue:
                    for s in residue:
                        self._tts_sentence_queue.append((emotion_key, s, self._tts_kind))
                    produced = True
        return produced

    def _flush_splitter_with_tone(self, tone_key) -> bool:
        """无条件 flush 分句器，残留按给定 tone 入 TTS 队列。"""
        residue = self._sentence_splitter.flush()
        if not residue:
            return False
        for s in residue:
            self._tts_sentence_queue.append((tone_key, s, self._tts_kind))
        return True

    async def _handle_committed_event(self, event_type: str, turn_id: str, data: dict):
        """commit 后的这一轮 Hane 事件：
        surface/deep_chunk  -> 流式抽 speech delta、转发前端、喂分句器
        surface/deep_done   -> flush 对应 extractor 把残留挤出来
        end                 -> 保险再 flush 一次、flush 分句器、pump TTS
        其它（begin/error） -> 原样透传
        """
        if event_type == "error":
            # Build F: a turn that failed in the brain is discarded there, so its
            # protect (sent on the following 'end') must not drop the carried text;
            # if it was already protected, the text it carried is put back in front.
            self._carry_error_turns.add(turn_id)
            if self._carry_snapshot and self._carry_snapshot[0] == turn_id:
                self._carry_text = stt_carry.with_carry(self._carry_snapshot[1], self._carry_text)
                self._carry_snapshot = None
                logger.info(f"[dlg {self._peer}] brain error after protect; carried text restored")
        elif event_type == "end" and self._carry_snapshot and self._carry_snapshot[0] == turn_id and turn_id not in self._carry_error_turns:
            self._carry_snapshot = None
        self._tl_note_brain_event(event_type, turn_id, data)
        if event_type == "surface_chunk":
            parts = self._surface_extractor.feed(data.get("text", "") or "")
            if await self._emit_speech_parts("surface", turn_id, parts):
                await self._pump_tts_queue()
            return

        if event_type == "deep_chunk":
            self._tts_kind = "deep"
            parts = self._deep_extractor.feed(data.get("text", "") or "")
            if await self._emit_speech_parts("deep", turn_id, parts):
                await self._pump_tts_queue()
            return

        if event_type == "surface_done":
            tail = self._surface_extractor.flush()
            produced = await self._emit_speech_parts("surface", turn_id, tail)
            # surface 整体结束，无条件 flush 分句器把 surface 残留归 surface tone
            if self._flush_splitter_with_tone(
                self._surface_extractor.current_emotion_key
            ):
                produced = True
            # every sentence queued from here on is the deep's (surface and deep are sequential)
            self._tts_kind = "deep"
            if produced:
                await self._pump_tts_queue()
            self._note_reply_lang(data)
            self._arm_deep_blurt(data)
            # CONTRACT §5: surface_done carries no recall any more (dropped here too if an
            # older brain still sends it); status stays informational.
            payload = {k: v for k, v in data.items() if k != "recall"}
            # the Use page's "loading" emote: True when no surface audio is still to come (every surface
            # sentence has already gone out as tts_audio/tts_sfx — or there was none, or TTS is off)
            payload["surface_tts_done"] = not self._surface_tts_pending()
            await self._send({**payload, "type": "ai_surface_done"})
            return

        if event_type == "deep_done":
            self._tts_kind = "deep"
            tail = self._deep_extractor.flush()
            produced = await self._emit_speech_parts("deep", turn_id, tail)
            # 正常情况下每段的 `\n` 哨兵已 flush 过；兜底再来一次
            if self._flush_splitter_with_tone(
                self._deep_extractor.current_emotion_key
            ):
                produced = True
            if produced:
                await self._pump_tts_queue()
            await self._send({**data, "type": "ai_deep_done"})
            return

        if event_type == "end":
            # 保险：正常情况 done 时已 flush；这里兜底一次
            s_tail = self._surface_extractor.flush()
            if s_tail:
                await self._emit_speech_parts("surface", turn_id, s_tail)
            d_tail = self._deep_extractor.flush()
            if d_tail:
                await self._emit_speech_parts("deep", turn_id, d_tail)
            # 分句器残留：用当前已知 tone（surface 优先，否则 deep 最后一段）
            residue = self._sentence_splitter.flush()
            if residue:
                residue_key = (
                    self._surface_extractor.current_emotion_key
                    or self._deep_extractor.current_emotion_key
                )
                for s in residue:
                    self._tts_sentence_queue.append((residue_key, s, self._tts_kind))
            self._ai_end_received = True
            # Text displayed in the standalone UI must survive the next turn even
            # with voice disabled or unavailable. Await the brain's acknowledgement.
            if not self._protected:
                await self._protect_after_first_audio(turn_id, delay=0)
            await self._send({**data, "type": "ai_end"})
            await self._pump_tts_queue()
            # 若音频在 ai_end 前就已播完（深层很快、音频很短），这里补一次排空检测
            await self._maybe_send_speech_done()
            return

        # begin / error / 其它 —— 原样透传
        if event_type == "begin":
            self._note_reply_lang(data)
        await self._send({**data, "type": f"ai_{event_type}"})

    # ------------------- PromptOptimized: reply language / blurt ------------------- #

    def _note_reply_lang(self, data: dict) -> None:
        """begin / surface_done carry the brain's reply_lang for this turn (CONTRACT §5)."""
        if not isinstance(data, dict) or "reply_lang" not in data:
            return
        lang = data.get("reply_lang")
        self._turn_reply_lang = lang if lang in ("zh", "en", "ja") else None
        if "reply_lang_reason" in data:
            self._turn_reply_lang_reason = data.get("reply_lang_reason")

    def _arm_deep_blurt(self, data: dict) -> None:
        """surface_done: when the deep continues after a spoken surface reply (say+think, or a
        think-mode filler, C14), hand that reply to the deep extractor so the deep's first
        segment does not say it twice (CONTRACT §1.1 rule 6 / §4.7). The brain passes the
        same text to the deep as the blurt."""
        mode = str(data.get("mode") or "")
        reply = data.get("reply")
        if not isinstance(reply, str):
            reply = getattr(self._surface_extractor, "reply_text", "") or ""
        reply = reply.strip()
        if mode == "think" and reply:
            # think-mode filler is spoken like a say reply (CONTRACT §4.6)
            logger.info(f"[dlg {self._peer}] think-mode filler spoken={self._surface_spoke}: {reply[:40]!r}")
        if mode not in ("say+think", "think") or not reply:
            return
        set_blurt = getattr(self._deep_extractor, "set_blurt", None)
        if set_blurt is not None and set_blurt(reply) is False:
            logger.info(f"[dlg {self._peer}] deep already speaking; blurt dedupe skipped")

    def _tts_language(self, spoken: str) -> Optional[str]:
        """TTS `language` for one sentence (CONTRACT §4.5, B6/B8/R3). _detect_lang decides; when it
        says MIXED, or the sentence is kanji-only (Chinese and Japanese look alike), the turn's
        expected reply language breaks the tie. One observe-only [lang] log line per sentence;
        nothing is removed or gated."""
        # Imported here: tests exec DialogSession extracted from this file with stub globals.
        from po_contract import LANG_TO_TTS, is_kanji_only, lang_counts
        detected = self._detect_lang(spoken)
        kanji_only = is_kanji_only(spoken)
        expected = self._turn_reply_lang
        language = {"CHINESE": "Chinese", "JAPANESE": "Japanese", "ENGLISH": "English"}.get(detected)
        if (detected == "MIXED" or kanji_only) and expected in LANG_TO_TTS:
            language = LANG_TO_TTS[expected]
        text40 = " ".join((spoken or "")[:40].split())
        logger.info(f"[lang] expected={expected or 'none'} detected={detected} "
                    f"kanji_only={kanji_only} text={text40}")
        if (expected in LANG_TO_TTS and language and language != LANG_TO_TTS[expected] and not kanji_only
                and detected in ("CHINESE", "JAPANESE", "ENGLISH") and self._turn_reply_lang_reason != "switch"
                and len((spoken or "").strip()) >= 12):
            # A whole sentence in another language than the turn's — the one thing her language rule forbids.
            # Short interjections (オッケー) and a kanji-only tie are not flagged. Observe-only: nothing is gated.
            logger.warning(f"[lang] MISMATCH expected={expected} spoke={detected} "
                           f"reason={self._turn_reply_lang_reason or 'unknown'} text={text40}")
        if detected == "ENGLISH":
            kana = lang_counts(spoken)[1]
            if 0 < kana <= 4:
                logger.info(f"[lang] short-ja-interjection kana={kana} text={text40}")
        return language

    async def _on_ai_event(self, event_type: str, data: dict):
        """Hane 推回事件的分发：
        - 等于 _committed_turn_id     -> 走 _handle_committed_event（清洗后转发）
        - 等于 _current_sent_id       -> 还没 commit 这轮；暂存原始事件，commit 时统一走清洗
        - 其它                         -> 过期的 stale turn，丢弃
        """
        turn_id = data.get("turn_id") if isinstance(data, dict) else None
        # 协议允许 error 事件不带 turn_id（极早期错误）—— 这种全局错误直接透传
        if turn_id is None:
            if event_type == "error":
                await self._send({**(data or {}), "type": "ai_error"})
                logger.warning(f"[dlg {self._peer}] AI error (no turn_id): {data}")
            else:
                logger.warning(
                    f"[dlg {self._peer}] AI {event_type} 没带 turn_id，已丢弃。"
                )
            return

        if turn_id == self._committed_turn_id:
            await self._handle_committed_event(event_type, turn_id, data)
            return

        if turn_id == self._current_sent_id:
            # 投机 turn 的 chunk 提前到达：缓存原始事件，_do_commit 时统一走清洗管道
            self._pending_chunks.append({"event_type": event_type, "data": data})
            return

        # 已在缓存里的主动轮（说话时点的念弹幕 / SC，begin 时音频忙被缓存）：把它后续的
        # deep_chunk/deep_done/end 继续收进同一缓存，等排空后整片回放（见下方 begin 分支
        # 与 _maybe_fire_buffered_proactive）。否则这些事件会被当 stale 丢掉。
        if turn_id in self._buffered_proactive:
            self._buffered_proactive[turn_id].append(
                {"event_type": event_type, "data": data}
            )
            return

        # 服务端主动发起的轮次（思考层「开口」/「注入」）：我们这边从没发过对应
        # message，turn_id 既不等于 committed 也不等于 current。begin 带 proactive
        # 标记时主动「认领」它作为新的一轮 committed turn——否则后续 deep_chunk/
        # deep_done/end 会被当 stale 丢弃，深层虽然回复了却完全不出声。
        if event_type == "begin" and isinstance(data, dict) and data.get("proactive"):
            # 竞态保护：只在「真有 TTS 音频在播/在途/排队」时才让位，别被主动轮顶掉。
            # 注意不能用 _is_ai_speaking_now()——它在 _ai_end_received=False 时（含从未
            # 开始过对话的空闲态）也算"在说话"，会把正常该认领的主动轮误挡掉。
            if (self._tts_playing is not None or self._tts_inflight is not None
                    or self._tts_sentence_queue):
                # 音频忙：不丢弃，开一个缓存把这轮事件先攒着，等当前音频彻底排空再放出来
                # 认领出声（见 _maybe_fire_buffered_proactive）。这样「说话时点念弹幕/SC」
                # 会等她这句说完再念，绝不只进历史不出声。打断会清空缓存，不会迟到乱念。
                logger.info(
                    f"[dlg {self._peer}] buffer proactive begin (audio busy), tid={turn_id}"
                )
                self._buffered_proactive[turn_id] = [
                    {"event_type": "begin", "data": data}
                ]
                return
            logger.info(f"[dlg {self._peer}] adopt proactive turn tid={turn_id}")
            self._tl_hook("finalize", self._committed_turn_id)
            self._committed_turn_id = turn_id
            self._current_sent_id = None
            self._pending_chunks = []
            self._reset_tts_for_new_turn()
            await self._handle_committed_event("begin", turn_id, data)
            return

        # 过期的 stale turn（被更新的 partial prune 掉了）
        logger.debug(
            f"[dlg {self._peer}] drop stale {event_type} "
            f"(tid={turn_id}, current={self._current_sent_id}, "
            f"committed={self._committed_turn_id})"
        )

    # ---------------------------- TTS 流水 ---------------------------- #

    # ---------------------------- her "loading" emote ---------------------------- #
    # The Use page shows it only in the silent gap between the surface's last audio and the deep's first.
    # These two flags make that gap exact: the page cannot tell from timing alone whether another
    # surface sentence is still being synthesised.

    def _surface_tts_pending(self) -> bool:
        """A surface sentence that has not gone out to the page yet: in flight at the TTS, or queued."""
        if (self._tts_inflight or {}).get("kind") == "surface":
            return True
        return any(len(item) > 2 and item[2] == "surface" for item in self._tts_sentence_queue)

    def _surface_last_item(self, kind) -> bool:
        """This surface item is the surface's last audio: the surface stream is closed (everything queued
        from here on is the deep's) and no surface sentence waits behind it."""
        return (kind == "surface" and self._tts_kind == "deep"
                and not any(len(item) > 2 and item[2] == "surface" for item in self._tts_sentence_queue))

    async def _pump_tts_queue(self):
        """当前没有 in-flight TTS 请求时，取队首句发出去。严格串行：
        下一句要等 _on_tts_event('audio_ready') 或 _on_tts_event('error') 回来再发。
        """
        if self._tts_inflight is not None:
            return
        if not self._tts_sentence_queue:
            return
        if self._committed_turn_id is None:
            # 没 committed 的这些句子其实属于已作废的投机 turn，丢弃
            self._tts_sentence_queue.clear()
            return
        if not self.voice_output or not self.tts._connected:
            # TTS 不可用：把队列清空，只保留文字显示
            logger.warning(f"[dlg {self._peer}] tts offline, drop {len(self._tts_sentence_queue)} sentences")
            self._tts_sentence_queue.clear()
            return

        item = self._tts_sentence_queue.popleft()
        key, payload = item[0], item[1]
        kind = item[2] if len(item) > 2 else None
        # 声效项：不请求 TTS，直接派发 tts_sfx 事件让前端播本地音效，然后继续下一项。
        if key is _SFX_QUEUE:
            await self._dispatch_sfx(payload, kind)
            await self._pump_tts_queue()
            return
        emotion_key, sentence = key, payload
        # 跳过没有可读内容的退化片段（孤立引号 / 纯标点）：有些 TTS 语音永远不回
        # audio_ready，会把 _tts_inflight 永久卡死、"audio busy" 门一直为真，后续 inject 全被吞。
        if not re.search(r"[^\W_]", sentence or ""):   # 没有字母/数字/CJK/假名 = 没东西可念
            logger.info(f"[dlg {self._peer}] skip empty/punct-only fragment: {sentence!r}")
            await self._pump_tts_queue()
            return
        # 句首加一个英文 '.' 作为 TTS 的"启动锚点"，缓解孤立短请求场景下的首音频慢。
        # 加在 emotion prefix 之后，让 emotion tag 仍然是字符串开头被 TTS 解析；
        # 也只改发给 TTS 的文本，前端展示用的 inflight['sentence'] 保持原句不带点。
        # 词替换只改发给 Freya 的发音文本（spoken）；前端展示用的 sentence 保持原样。
        # 语言检测也按 spoken 走（替换可能改变脚本，如英文->假名）。
        spoken = apply_wordmap(sentence, "tts")
        tts_text = prefix_emotion(emotion_key, "." + spoken)
        # 按句检测语言：Chinese / Japanese / English 直接传给 TTS；MIXED 或纯汉字句
        # 用本轮大脑给的 reply_lang 定（没有 reply_lang 时 MIXED 仍不传，让 TTS 按角色默认走）。
        tts_language = self._tts_language(spoken)
        idx = self._tts_next_idx
        self._tts_next_idx += 1
        tts_turn_id = f"{self._committed_turn_id}-{idx}"
        task_id = str(uuid.uuid4())
        self._tts_inflight = {
            "task_id": task_id,
            "tts_turn_id": tts_turn_id,
            "idx": idx,
            "text": tts_text,
            "emotion": emotion_key,
            "sentence": sentence,
            "sent_ms": _now_ms(),
            "language": tts_language,
            "kind": kind,
        }
        self._tl_hook("note_tts_request", self._committed_turn_id, idx, kind, self._tts_inflight["sent_ms"])
        logger.info(
            f"[dlg {self._peer}] tts request #{idx} turn={tts_turn_id} "
            f"emotion={emotion_key} lang={tts_language or 'default'}: {sentence!r}"
        )
        ok = await self.tts.request(task_id, tts_turn_id, tts_text, language=tts_language)
        if not ok:
            # 发送失败：直接当作完成，推下一句
            self._tl_hook("note_tts_failed", tts_turn_id.rsplit("-", 1)[0], idx)
            self._tts_inflight = None
            await self._pump_tts_queue()
            return
        # 每句兜底：这一句若 TTS_INFLIGHT_TIMEOUT_MS 内既没 audio_ready 也没 error 回来，
        # 跳过它继续排空队列（否则一句卡死整轮永远 "audio busy"，后续 inject 全被吞）。
        asyncio.create_task(
            self._tts_inflight_watchdog(task_id, self._committed_turn_id)
        )
        # 超时兜底：1s 内没收到首段音频 → 视为已收到，走 protect + reset，避免卡死
        if not self._tts_first_audio_sent:
            asyncio.create_task(
                self._tts_first_audio_watchdog(task_id, self._committed_turn_id)
            )

    async def _dispatch_sfx(self, payload, kind: Optional[str] = None):
        """派发一个声效：按"前一句情绪的语气域 + 保声音含义"选片，发 tts_sfx 给前端
        从本地 static/sfx 取文件播。占一个 idx（参与播放/打断/完成判定），但不走 TTS。
        选不出文件（标签不认识/所有语气域都没有）就跳过、不占 idx。
        """
        name, emotion_key = payload
        filename = resolve_sfx(name, emotion_key)
        if not filename:
            logger.info(
                f"[dlg {self._peer}] sfx '{name}' unresolved (emotion={emotion_key}), skip"
            )
            return
        idx = self._tts_next_idx
        self._tts_next_idx += 1
        tts_turn_id = f"{self._committed_turn_id}-{idx}"
        self._tts_sentences_by_idx[idx] = {
            "text": "", "duration": 0.0, "turn_id": tts_turn_id, "is_sfx": True,
        }
        self._tts_pending_play.add(idx)
        self._tl_hook("note_sfx", self._committed_turn_id, idx, kind, _now_ms())
        logger.info(
            f"[dlg {self._peer}] sfx #{idx} turn={tts_turn_id} {name} -> {filename}"
        )
        await self._send({
            "type": "tts_sfx",
            "turn_id": tts_turn_id,
            "idx": idx,
            "file": filename,
            "kind": kind,                                   # surface | deep, like tts_audio
            "surface_last": self._surface_last_item(kind),
            "pause_after": 0.0,   # 声效不是"一句话"，后面不额外加停顿
        })
        # 声效作为首段音频也要触发 protect（保护该轮历史）
        if not self._tts_first_audio_sent:
            self._tts_first_audio_sent = True
            asyncio.create_task(self._protect_after_first_audio(self._committed_turn_id))

    async def _on_tts_event(self, event_type: str, data: dict):
        """TTS 服务回传的 task_queued / audio_ready / error 分发。
        广播回传，先按 task_id 认领——不是我们发的直接丢。然后用 committed_turn_id
        前缀比对：如果这段音频的 Hane 轮次已不再是当前 committed，说明用户打断了，
        丢弃不转发给浏览器。
        """
        task_id = data.get("task_id")
        tts_turn_id = data.get("turn_id") or ""

        # 认领：必须是当前 in-flight 的那个
        inflight = self._tts_inflight
        if inflight is None or task_id != inflight.get("task_id"):
            return

        if event_type == "task_queued":
            logger.debug(f"[dlg {self._peer}] tts queued: {tts_turn_id}")
            return

        # audio_ready / error：消耗 inflight
        self._tts_inflight = None

        # 前缀比对（committed_turn_id 是 tts_turn_id 去掉最后一段 -idx 的部分）
        hane_turn_id = tts_turn_id.rsplit("-", 1)[0] if "-" in tts_turn_id else tts_turn_id
        is_current = (hane_turn_id == self._committed_turn_id)

        if event_type == "error":
            err = data.get("error") or "unknown"
            logger.warning(f"[dlg {self._peer}] tts error turn={tts_turn_id}: {err}")
            if getattr(self, "_tl", None) is not None:
                self._tl_hook("note_tts_failed", hane_turn_id, inflight.get("idx"))
            if is_current:
                await self._send({
                    "type": "tts_error",
                    "turn_id": tts_turn_id,
                    "idx": inflight.get("idx"),
                    "error": err,
                    "kind": inflight.get("kind"),
                    # a failed final surface sentence still closes the surface for the "loading" emote
                    "surface_last": self._surface_last_item(inflight.get("kind")),
                })
            # 继续推下一句（坏段跳过）
            await self._pump_tts_queue()
            return

        # audio_ready
        if not is_current:
            logger.info(
                f"[dlg {self._peer}] drop stale tts audio: {tts_turn_id} "
                f"(committed={self._committed_turn_id})"
            )
            await self._pump_tts_queue()
            return

        audio = data.get("audio")
        if not isinstance(audio, (bytes, bytearray)):
            logger.warning(f"[dlg {self._peer}] audio_ready without bytes: {type(audio)}")
            await self._pump_tts_queue()
            return

        received_ms = _now_ms()
        duration = data.get("audio_duration", 0)
        if getattr(self, "_tl", None) is not None:
            # telemetry: bookkeeping only; the ai_timeline message goes out later
            self._tl_hook("note_tts_audio", hane_turn_id, inflight.get("idx"), received_ms, duration)
        audio_b64 = base64.b64encode(bytes(audio)).decode("ascii")
        audio_fmt = data.get("audio_format", "opus")
        latency_ms = received_ms - inflight.get("sent_ms", received_ms)
        logger.info(
            f"[dlg {self._peer}] tts audio #{inflight.get('idx')} turn={tts_turn_id} "
            f"fmt={audio_fmt} dur={duration:.2f}s bytes={len(audio)} latency={latency_ms}ms"
        )
        # Include ordinary audio while it is being delivered/decoded/queued in
        # the browser, not only after the browser reports that playback began.
        idx = inflight.get("idx")
        if idx is not None:
            self._tts_pending_play.add(idx)
            self._tts_sentences_by_idx[idx] = {
                "text": inflight.get("sentence") or "",
                "duration": float(duration) if duration else 0.0,
                "turn_id": tts_turn_id,
            }
        # her newest spoken line becomes the VLM prompt context (vision section)
        vision_note_her_speech(inflight.get("sentence") or "")
        await self._send({
            "type": "tts_audio",
            "turn_id": tts_turn_id,
            "idx": inflight.get("idx"),
            # 前端显示用这段话原文（不含【emotion】prefix），TTS 播哪句就显哪句
            "text": inflight.get("sentence") or data.get("text", ""),
            "emotion": inflight.get("emotion"),   # 实际播放时 UI 显示用
            "kind": inflight.get("kind"),         # surface | deep: the Use page drops her "loading" emote on the deep's first words
            "surface_last": self._surface_last_item(inflight.get("kind")),   # the surface's final audio: the emote may start once it has played
            # 这句说完后的停顿（按 本句情绪×语言 查 tts_pause 表），前端排进时间轴
            "pause_after": pause_after(
                inflight.get("emotion"),
                self._detect_lang(inflight.get("sentence") or ""),
            ),
            "audio_format": audio_fmt,
            "audio_duration": duration,
            "audio_b64": audio_b64,
        })
        # 存下这段 TTS 的原文（不带 emotion prefix）和时长，等前端上报
        # tts_playing / tts_finished 时查它 → 算打断锚点 my_partial。
        # 首段音频：0.4s 后给 Hane 发 protect（保护该轮对话历史）
        if not self._tts_first_audio_sent:
            self._tts_first_audio_sent = True
            asyncio.create_task(self._protect_after_first_audio(self._committed_turn_id))

        # 串行：一句回来再发下一句
        await self._pump_tts_queue()

    async def _protect_after_first_audio(
        self, turn_id_at_request: Optional[str], delay: float = 0.4,
    ):
        """首段 TTS 音频到达后等 delay 秒再 emit protect。
        正常路径 delay=0.4，是给浏览器解码 + 真的开始播的时间余量。
        超时兜底路径 delay=0（已经等过 1s 了，没必要再等）。

        如果这期间用户又说话了（_committed_turn_id 变成了 None 或别的），
        那我们就**不**保护：Hane 下一轮 message 会 prune 掉它。
        """
        if delay > 0:
            await asyncio.sleep(delay)
        if self._committed_turn_id != turn_id_at_request:
            logger.info(
                f"[dlg {self._peer}] skip protect (turn changed): "
                f"was={turn_id_at_request} now={self._committed_turn_id}"
            )
            return
        logger.info(f"[dlg {self._peer}] protect turn {turn_id_at_request}")
        # Build F: protecting a turn that holds the carried text (or the force-committed
        # turn itself) makes that text durable. Take the carry before the round trip so a
        # continuation partial racing it is not prefixed a second time; restore on failure.
        carried = ""
        if turn_id_at_request in self._carry_turn_ids and turn_id_at_request not in self._carry_error_turns:
            carried, self._carry_text = self._carry_text, ""
            self._carry_turn_ids.clear()
            self._carry_error_turns.clear()
        self._protect_inflight = turn_id_at_request
        try:
            acknowledged = await self.hane.send_protect()
        except Exception:
            acknowledged = False
        finally:
            self._protect_inflight = None
        if carried:
            if acknowledged:
                # Remember what became durable with this turn; a later brain error on
                # the same turn (which discards it) puts the text back in front.
                self._carry_snapshot = (turn_id_at_request, carried)
            else:
                logger.warning(f"[dlg {self._peer}] protect not acknowledged; carried text kept")
                self._carry_text = stt_carry.with_carry(carried, self._carry_text)
                self._carry_turn_ids.add(turn_id_at_request)
        # protect 发出 = 本轮真正的音频分界点。这一刻才把 _speech_frames 清空。
        # 注意：不清 _committed_turn_id —— AI 后续 chunks（deep_chunk / end）还会带这个 id 过来。
        self._protected = True
        # AI 本轮顺利出声，上一轮被打断的 my_partial 锚点不再相关
        self._my_partial_pending = None
        self._my_partial_lang = None
        if self._state == "POST_COMMIT":
            # 用户 commit 后一直没再说话，干净地回到 IDLE，pre_buffer 重新滚动
            self._speech_frames = []
            self._silence_frames = 0
            self._state = "IDLE"
            self._last_partial = ""
            self._last_partial_raw = ""
            self._pending_partial = None
            self._pre_buffer.clear()
            vad_model.reset_states()

    async def _tts_first_audio_watchdog(
        self, task_id: str, turn_id_at_request: Optional[str],
    ):
        """TTS 首段音频超时兜底：
        发出请求后 1s 内没收到任何首段音频 → 视为已到达，立即走 protect 流程，
        避免 TTS 回包丢失/慢时永远卡在 POST_COMMIT，音频不分界。
        """
        await asyncio.sleep(self.TTS_FIRST_AUDIO_TIMEOUT_MS / 1000)
        # 已经收到首段 / 已经被清空 / 这个 task 不再是 inflight → 不做事
        if self._tts_first_audio_sent:
            return
        inflight = self._tts_inflight
        if inflight is None or inflight.get("task_id") != task_id:
            return
        # turn 已经换了（用户打断 / 延续）—— protect 流程里自会判断并跳过
        logger.warning(
            f"[dlg {self._peer}] tts first-audio timeout ({self.TTS_FIRST_AUDIO_TIMEOUT_MS}ms), "
            f"turn={turn_id_at_request} — treat as arrived"
        )
        # 标记 + 清掉 inflight 和队列：TTS 看起来丢了，后面的句子也别指望
        self._tts_first_audio_sent = True
        self._tl_hook("note_tts_failed", turn_id_at_request, inflight.get("idx"))
        self._tts_inflight = None
        self._tts_sentence_queue.clear()
        # 直接 protect（delay=0）：已经等过 1s 了，不再叠加
        asyncio.create_task(
            self._protect_after_first_audio(turn_id_at_request, delay=0.0)
        )

    async def _tts_inflight_watchdog(
        self, task_id: str, turn_id_at_request: Optional[str],
    ):
        """每句 TTS 兜底：发出后 TTS_INFLIGHT_TIMEOUT_MS 内既没 audio_ready 也没 error
        回来 → 视为这句丢了，清 inflight 并继续排空队列。否则一句卡死会让 _tts_inflight
        永远非空、"audio busy" 门一直为真，后续 inject（念弹幕 / SC）全被吞。"""
        await asyncio.sleep(self.TTS_INFLIGHT_TIMEOUT_MS / 1000)
        inflight = self._tts_inflight
        if inflight is None or inflight.get("task_id") != task_id:
            return                       # 已被 audio_ready / error / reset 消耗
        logger.warning(
            f"[dlg {self._peer}] tts inflight timeout ({self.TTS_INFLIGHT_TIMEOUT_MS}ms) "
            f"idx={inflight.get('idx')} turn={turn_id_at_request} — skip, drain queue"
        )
        self._tl_hook("note_tts_failed", turn_id_at_request, inflight.get("idx"))
        self._tts_inflight = None
        await self._pump_tts_queue()

    # ---------------------------- 打断锚点 my_partial ---------------------------- #

    @staticmethod
    def _effective_chars(text: str) -> List[str]:
        """抽出"AI 嘴上真正会念出来的"字符：
        - 去掉开头的【emotion】prefix
        - 只保留字母/数字/CJK/日语假名，其它（标点、空格、括号）全过滤
        """
        if not text:
            return []
        text = re.sub(r"^\s*【[^】]*】\s*", "", text)
        out: List[str] = []
        for c in text:
            if c.isspace():
                continue
            is_cjk = "\u4e00" <= c <= "\u9fff"
            is_kana = (
                "\u3040" <= c <= "\u309f"
                or "\u30a0" <= c <= "\u30ff"
                or "\u31f0" <= c <= "\u31ff"
                or "\uff66" <= c <= "\uff9f"
            )
            if c.isalnum() or is_cjk or is_kana:
                out.append(c)
        return out

    @staticmethod
    def _english_words(text: str) -> List[str]:
        """抽出英文单词，用于英文句子的 my_partial 按单词对齐。
        - 去掉开头的【emotion】prefix
        - 支持 don't / it's / 数字 / 复合词中连字符（保留字面）
        """
        if not text:
            return []
        text = re.sub(r"^\s*【[^】]*】\s*", "", text)
        return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", text)

    @staticmethod
    def _detect_lang(text: str) -> str:
        """按 CJK / 拉丁字母 / 假名 字符比例粗判句子语言。
        参考 brain/webui.detect_language 的实现。
        返回 'CHINESE' | 'JAPANESE' | 'ENGLISH' | 'MIXED'
        """
        if not text:
            return "CHINESE"
        # 去掉【emotion】前缀；它里面的拉丁字母不该被当成"英文句子"的依据
        text = re.sub(r"^\s*【[^】]*】\s*", "", text)
        cn = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        en = sum(1 for c in text if c.isalpha() and ord(c) < 128)
        jp = sum(
            1 for c in text
            if ("\u3040" <= c <= "\u309f")
            or ("\u30a0" <= c <= "\u30ff")
            or ("\u31f0" <= c <= "\u31ff")
            or ("\uff66" <= c <= "\uff9f")
        )
        total = cn + en + jp
        if total == 0:
            return "CHINESE"
        ratios = {"cn": cn / total, "en": en / total, "jp": jp / total}
        top = max(ratios, key=ratios.get)
        if ratios[top] < 0.6:
            return "MIXED"
        # 日汉混写也按日语处理（假名 > 30%）
        if top == "jp" or ratios["jp"] > 0.3:
            return "JAPANESE"
        if top == "en":
            return "ENGLISH"
        return "CHINESE"

    @classmethod
    def _window_for_lang(cls, lang: str) -> int:
        """my_partial 的 window 大小——大约"4 个单词"的信息量。
        - 中文：1 字 ≈ 1 词 → 4
        - 日语：多假名组合成一个词 → 6
        - 英文：直接按单词 → 4
        - 混合：无主导 → 退回 4 字
        """
        if lang == "JAPANESE":
            return 6
        return 4

    def _is_ai_speaking_now(self) -> bool:
        """AI 当下是否还在说话。用来判定"用户开口"是真打断还是只是等她说完后回话。
        条件任一成立都算在说：
          - 前端正在播某段 TTS（_tts_playing 非空）
          - TTS 队列里还有待播的句子
          - 有 in-flight 的 TTS 请求
          - Hane 还没发 ai_end（后面还可能来新句子 / 新段）
        """
        if self._tts_playing is not None:
            return True
        if self._tts_inflight is not None:
            return True
        if self._tts_sentence_queue:
            return True
        if self._tts_pending_play:
            return True
        if not self._ai_end_received:
            return True
        return False

    def _gate_handle_probe(self, frame: np.ndarray) -> bool:
        """真打断门控：AI 说话期间，VAD 首次探到用户语音时启动探针计时。
        返回值 True = 当前语音帧**穿透**门控、可以作为真打断触发；
              False = 仍在锁定/过期/未到窗口，应该早退（调用方直接 return）。

        时间轴（相对探针起点）：
          0 ~ INTERRUPT_GATE_LOCK_MS      : 锁定期 -> 吞掉，frame 入 _probe_frames
          LOCK ~ UNLOCK_END               : 解锁窗口 -> 任一语音帧即穿透
          > UNLOCK_END                    : 过期 -> 丢弃旧探针，以当前帧为新起点

        另外：探针期遇到静音帧（在 _on_frame 的 else 分支里）会重置探针，
        要求"连续"语音才能穿透。
        """
        now = _now_ms()
        start = self._interrupt_probe_start_ms
        if start is None:
            # 第一次探针：攒起来，但先不触发
            self._interrupt_probe_start_ms = now
            self._probe_frames = [frame]
            return False
        elapsed = now - start
        if elapsed < self.INTERRUPT_GATE_LOCK_MS:
            self._probe_frames.append(frame)
            return False
        if elapsed <= self.INTERRUPT_GATE_UNLOCK_END_MS:
            # 解锁窗口内：穿透成立。当前帧不入 _probe_frames，
            # 由调用方统一 speech_frames.append(frame) 加一次，避免重复。
            return True
        # 过期：旧探针失效，以这帧为新探针起点
        self._interrupt_probe_start_ms = now
        self._probe_frames = [frame]
        return False

    @classmethod
    def _tokenize_for_my_partial(cls, text: str):
        """按句子语言选择 token 粒度。
        返回 (lang, tokens, joiner)。English tokens 是单词，其它都是单字。
        """
        lang = cls._detect_lang(text)
        if lang == "ENGLISH":
            return lang, cls._english_words(text), " "
        return lang, cls._effective_chars(text), ""

    def _compute_my_partial(self) -> Optional[str]:
        """用户打断 AI 的瞬间：按播放进度在当前句里定位一个"大致说到这里"的位置，
        取附近约 4 个单词的信息量作为锚点返回。具体窗口按句子语言决定：
          中文 4 字 / 日语 6 字 / 英文 4 单词 / 混合回退 4 字。
        没有正在播的段则退而求其次用刚播完那句末尾。
        同时副作用地把那句的语言写进 self._my_partial_lang，给占位符用。
        """
        playing = self._tts_playing
        if playing and playing.get("duration", 0) > 0:
            text = playing.get("text") or ""
            lang, tokens, joiner = self._tokenize_for_my_partial(text)
            if tokens:
                self._my_partial_lang = lang
                n_window = self._window_for_lang(lang)
                elapsed = (_now_ms() - playing["start_ms"]) / 1000.0
                ratio = max(0.0, min(1.0, elapsed / playing["duration"]))
                k = int(round(ratio * len(tokens)))
                start = max(0, k - n_window)
                snippet = tokens[start:k]
                if len(snippet) < n_window:
                    # 往前不够，往后补
                    end = min(len(tokens), start + n_window)
                    snippet = tokens[start:end]
                if len(snippet) < n_window:
                    # 整句都不够，整句返回
                    snippet = tokens
                return joiner.join(snippet)
        # 没在播：用上一段刚播完的末尾
        last = self._tts_last_finished
        if last:
            text = last.get("text") or ""
            lang, tokens, joiner = self._tokenize_for_my_partial(text)
            if tokens:
                self._my_partial_lang = lang
                n_window = self._window_for_lang(lang)
                tail = tokens[-n_window:] if len(tokens) >= n_window else tokens
                return joiner.join(tail)
        return None

    async def on_frontend_tts_playing(self, data: dict):
        """前端通知：某段 TTS 音频真的开始播了。
        按 idx 从 _tts_sentences_by_idx 查原文 + 时长，记下起播时间戳供打断时估位。"""
        turn_id = data.get("turn_id") or ""
        idx = data.get("idx")
        hane_tid = turn_id.rsplit("-", 1)[0] if "-" in turn_id else turn_id
        if hane_tid != self._committed_turn_id:
            return
        info = self._tts_sentences_by_idx.get(idx)
        if not info:
            return
        start_ms = _now_ms()
        self._tl_hook("note_played", hane_tid, idx, start_ms)
        self._tts_playing = {
            "idx": idx,
            "text": info.get("text") or "",
            "duration": info.get("duration", 0.0),
            "start_ms": start_ms,
            "is_sfx": info.get("is_sfx", False),
        }

    async def _maybe_send_speech_done(self):
        """本轮音频彻底排空（ai_end 已到 + 队列空 + 无 inflight + 没在播）时，
        给 Hane 发一次 speech_done。Hane 用它在"换口气"后触发【接着说】续说轮——
        等当前这句真说完再续，续说轮发起时音频不忙，正常走主动轮认领出声。
        每轮只发一次。"""
        if self._speech_done_sent:
            return
        if self._committed_turn_id is None or not self._ai_end_received:
            return
        if (self._tts_sentence_queue
                or self._tts_inflight is not None
                or self._tts_playing is not None
                or self._tts_pending_play):
            return
        self._speech_done_sent = True
        self._tl_hook("finalize", self._committed_turn_id)
        logger.info(f"[dlg {self._peer}] speech_done turn {self._committed_turn_id}")
        await self.hane.send_speech_done(self._committed_turn_id)
        # 本轮音频已排空：若有「说话时点的念弹幕/SC」缓存着，现在放出来认领出声。
        await self._maybe_fire_buffered_proactive()

    async def _maybe_fire_buffered_proactive(self):
        """当前音频彻底排空后，把队首那条被缓存的主动轮（说话时点的念弹幕/SC）放出来：
        认领为新 committed 轮、reset 干净状态、回放它缓存的全部事件（begin…end）正常出声。
        多条按到达顺序依次放（每条播完排空再放下一条）。打断会在 speech_start 处清空缓存，
        所以过时的绝不会迟到乱念。注意 _reset_tts_for_new_turn 不动 _buffered_proactive，
        故认领当前这条不会误清后面排队的。"""
        if not self._buffered_proactive:
            return
        # 仍在忙（极端竞态）就先不放，等下次排空再来
        if (self._tts_sentence_queue
                or self._tts_inflight is not None
                or self._tts_playing is not None
                or self._tts_pending_play):
            return
        turn_id = next(iter(self._buffered_proactive))
        events = self._buffered_proactive.pop(turn_id)
        logger.info(f"[dlg {self._peer}] fire buffered proactive turn tid={turn_id}")
        self._tl_hook("finalize", self._committed_turn_id)
        self._committed_turn_id = turn_id
        self._current_sent_id = None
        self._pending_chunks = []
        self._reset_tts_for_new_turn()
        for ev in events:
            await self._handle_committed_event(ev["event_type"], turn_id, ev["data"])

    async def on_frontend_tts_finished(self, data: dict):
        """前端通知：某段 TTS 音频播完。"""
        turn_id = data.get("turn_id") or ""
        idx = data.get("idx")
        hane_tid = turn_id.rsplit("-", 1)[0] if "-" in turn_id else turn_id
        if hane_tid != self._committed_turn_id:
            return
        self._tl_hook("note_finished", hane_tid, idx, _now_ms())
        self._tts_pending_play.discard(idx)
        if self._tts_playing and self._tts_playing.get("idx") == idx:
            # 声效段没文字，别覆盖打断锚点——保留声效前那句真实台词的末尾，
            # 这样在声效播放/刚播完时被打断，my_partial 仍落在真实台词上。
            if not self._tts_playing.get("is_sfx"):
                self._tts_last_finished = {
                    "idx": idx,
                    "text": self._tts_playing.get("text"),
                    "duration": self._tts_playing.get("duration"),
                }
            self._tts_playing = None
        # 这段播完后若整轮音频已排空 -> 通知 Hane（触发可能的【接着说】续说）
        await self._maybe_send_speech_done()

    async def on_text(self, data: dict):
        text = str(data.get("text") or "").strip()
        if not text:
            return
        if len(text) > 12000:
            await self._send({"type": "ai_error", "message": "Message exceeds 12,000 characters."})
            return
        if not self.hane._connected:
            await self._send({"type": "ai_error", "message": "Brain disconnected. Reconnect the chat."})
            return
        speaking = self._is_ai_speaking_now()
        partial = self._compute_my_partial() if speaking else None
        self._tl_supersede(_now_ms(), interrupted=speaking)
        if self._pending_partial and not self._pending_partial.done():
            self._pending_partial.cancel()
        self._asr_seq += 1
        self._active_asr_seq = self._asr_seq
        self._reset_utterance()
        self._leftover.clear()
        self._reset_tts_for_new_turn()
        self._buffered_proactive.clear()
        self._my_partial_pending = partial
        tid = self._make_turn_id()
        self._committed_turn_id = self._current_sent_id = tid
        self._pending_chunks = []
        commit_ms = _now_ms()
        self._tl_hook("open_round", tid, "user", "inject" if data.get("type") == "inject" else "text",
                      {"commit_ms": commit_ms})
        await self._send({"type": "commit", "text": text, "raw": text,
                          "turn_id": tid, "ts": commit_ms, "input": data.get("type", "text")})
        text_for_hane = text
        if data.get("type") != "inject" and self._carry_text:
            # Build F: a typed message is the next turn too; the carry is dictation, so it
            # gets the STT word map that send_message skips for typed text.
            text_for_hane = stt_carry.with_carry(apply_wordmap(self._carry_text, "stt"), text)
            self._carry_turn_ids.add(tid)
        await self.hane.send_message(text_for_hane, tid, my_partial=partial, apply_stt_map=False,
                                     deep_only=(data.get("type") == "inject"))

    async def consume_context(self, turn_id):
        sent = self._contexts_sent.pop(turn_id, None)
        if sent and sent[0] == self._context_revision and sent[1]:
            self._pending_context = ""
            await self._send({"type": "context_status", "queued": False, "turn_id": turn_id})

    async def queue_context(self, data):
        text = str(data.get("text") or "").strip()
        if len(text) > 12000:
            await self._send({"type": "ai_error", "message": "Context exceeds 12,000 characters."})
            return
        self._context_revision += 1
        self._pending_context = text
        await self._send({"type": "context_status", "queued": bool(text)})

    async def force_stop(self):
        """外部请求停止：如果还有积累的语音，当作 commit。"""
        if self._speech_frames:
            await self._do_commit(cut_ms=_now_ms() if self._state == "SPEAKING" else None)
        self._reset_utterance()

    # ------------------- round telemetry (TELEMETRY_SPEC §1.2) ------------------- #
    # Everything below is bookkeeping on TimelineBook (dict updates) plus one debounced
    # background sender per session. Nothing here is awaited on the STT / brain / TTS path.

    @classmethod
    def _shared_timeline_book(cls):
        """Process-wide TimelineBook, so the rounds survive a browser reconnect. None when the
        class runs without it (tests that AST-extract DialogSession alone)."""
        book = cls._TIMELINE_BOOK
        if book is None:
            book_cls = globals().get("TimelineBook")
            if book_cls is None:
                return None
            book = cls._TIMELINE_BOOK = book_cls()
        return book

    def _tl_hook(self, name, *args):
        """Call one TimelineBook method; never raises, never awaits."""
        tl = getattr(self, "_tl", None)
        if tl is None:
            return None
        try:
            result = getattr(tl, name)(*args)
        except Exception as e:
            logger.debug(f"[timeline] {name} failed: {e}")
            return None
        self._tl_schedule()
        return result

    def _tl_supersede(self, now_ms: int, interrupted: bool):
        """The user takes over (voice speech_start or typed text) while a round is committed.
        Unprotected -> the brain prunes that turn -> void; protected and still speaking ->
        interrupted_ms; otherwise the round simply ends."""
        tid = self._committed_turn_id
        if tid is None or getattr(self, "_tl", None) is None:
            return
        protected = bool(self._protected) or (self._protect_inflight is not None and self._protect_inflight == tid)
        self._tl_hook("supersede", tid, now_ms, protected, bool(interrupted))

    def _tl_open_voice_round(self, tid, commit_ms, reuse: bool, cut_ms=None, transcript_ms=None):
        """A voice commit. Reusing a partial id inherits that partial's STT marks (its speech_end is
        the utterance end); the fresh-ASR path uses the last pause trigger (or the cut time of a
        force commit) and the time its own ASR finished."""
        tl = getattr(self, "_tl", None)
        if tl is None or not self._ai_pipeline_online:
            return          # offline dictation: no brain turn, no round
        marks = tl.marks_for(tid) if reuse else {}
        if marks.get("speech_start_ms") is None:
            marks["speech_start_ms"] = self._utt_speech_start_ms
        if not reuse:
            marks["speech_end_ms"] = cut_ms if cut_ms is not None else self._last_trigger_ms
            marks["transcript_ms"] = transcript_ms
        marks["commit_ms"] = commit_ms
        self._tl_hook("open_round", tid, "user", "voice", marks, None, cut_ms is not None)

    def _tl_note_brain_event(self, event_type: str, turn_id: str, data: dict):
        tl = getattr(self, "_tl", None)
        if tl is None:
            return
        if event_type == "begin":
            if isinstance(data, dict) and data.get("proactive") and not tl.has_round(turn_id):
                origin = data.get("origin") if isinstance(data.get("origin"), dict) else None
                self._tl_hook("open_round", turn_id, "reflection", "proactive", None, origin, False, _now_ms())
        elif event_type in ("surface_chunk", "surface_done"):
            self._tl_hook("note_phase", turn_id, "surface")
        elif event_type in ("deep_chunk", "deep_done"):
            self._tl_hook("note_phase", turn_id, "deep")
        elif event_type == "end":
            self._tl_hook("mark", turn_id, "ai_end_ms", _now_ms())
        elif event_type == "error":
            self._tl_hook("note_error", turn_id)

    async def _on_call_receipt(self, data: dict):
        """Brain call_receipt (§1.1): store it by turn_id (attached to its round now or when the
        round appears) and forward a copy to the browser as ai_call_receipt."""
        receipt = self._tl_hook("add_receipt", data)
        if receipt is None:
            return
        self._tl_outbox.append({**receipt, "type": "ai_call_receipt"})
        self._tl_schedule()

    def _tl_schedule(self):
        tl = getattr(self, "_tl", None)
        if tl is None or not (self._tl_outbox or tl.has_pending()):
            return
        task = self._tl_task
        if task is not None and not task.done():
            return          # the running sender re-checks before it exits
        try:
            self._tl_task = asyncio.get_running_loop().create_task(self._tl_flush_later())
        except RuntimeError:
            pass

    async def _tl_flush_later(self):
        """Coalesce updates for TIMELINE_EMIT_DELAY_S, then send receipts and the full state of
        each changed round. One sender per session keeps the messages in order."""
        while True:
            await asyncio.sleep(self.TIMELINE_EMIT_DELAY_S)
            outbox, self._tl_outbox = self._tl_outbox, []
            try:
                messages = outbox + self._tl.pop_messages()
            except Exception as e:
                logger.debug(f"[timeline] build failed: {e}")
                messages = outbox
            for message in messages:
                await self._send(message)
            if not (self._tl_outbox or self._tl.has_pending()):
                return

    def _tl_resend_all(self):
        """Browser (re)connected: send every stored round, oldest first."""
        tl = getattr(self, "_tl", None)
        if tl is None:
            return
        try:
            self._tl_outbox.extend(tl.all_messages())
        except Exception as e:
            logger.debug(f"[timeline] resend failed: {e}")
        self._tl_schedule()

    def _tl_reset(self):
        """Browser reset button (timeline_reset): forget every round."""
        tl = getattr(self, "_tl", None)
        if tl is None:
            return
        tl.reset()
        self._tl_outbox = [m for m in self._tl_outbox if m.get("type") == "ai_call_receipt"]
        logger.info(f"[dlg {self._peer}] timeline reset")

    def _tl_on_disconnect(self):
        """The browser went away: the committed round will get no more audio; end it. The next
        connection re-sends it. Late receipts (writer, drain) for its turns reach the next brain
        connection: the brain forgets a disconnected sid and falls back to its current client."""
        tl = getattr(self, "_tl", None)
        if tl is None:
            return
        try:
            if self._committed_turn_id is not None:
                tl.finalize(self._committed_turn_id)
        except Exception:
            pass
        task = self._tl_task
        if task is not None and not task.done():
            task.cancel()


# Original Hane history and VAD are global: one companion session at a time.
_dialog_active = False


@app.websocket("/ws/dialog")
async def ws_dialog(ws: WebSocket):
    global _dialog_active
    await ws.accept()
    if _wipe_in_progress:
        await ws.send_json({"type": "ai_error", "message": "A service control is in progress. Reconnect when it finishes."})
        await ws.close(code=1008)
        return
    if _dialog_active:
        await ws.send_json({"type": "ai_error", "message": "OpenAmika is open in another tab. Disconnect it first."})
        await ws.close(code=1008)
        return
    _dialog_active = True
    session = None
    try:
        session = DialogSession(ws)
        session._connecting = True
        session._input_busy = False
        session._wipe_retired = False
        _dialog_sessions.add(session)
        await asyncio.gather(session.hane.connect(), session.tts.connect())
        session._connecting = False
        await ws.send_json({"type": "ready", "sample_rate": DialogSession.SAMPLE_RATE,
                            "frame_samples": DialogSession.FRAME_SAMPLES,
                            "format": "pcm_s16le_mono", "brain": session.hane._connected,
                            "tts": session.tts._connected})
        # TELEMETRY_SPEC §1.2: re-send the stored round timelines to a reconnected browser
        resend = getattr(session, "_tl_resend_all", None)
        if resend is not None:
            resend()
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if session._wipe_retired:
                break
            if _wipe_in_progress:
                await session._send({"type": "ai_error", "message": "A service control is in progress. This input was not sent; try again when it finishes."})
                continue
            session._input_busy = True
            if msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    session._input_busy = False
                    continue
                if not isinstance(data, dict):
                    session._input_busy = False
                    continue
                kind = data.get("type")
                if kind in ("text", "inject"):
                    await session.on_text(data)
                elif kind == "context_inject":
                    await session.queue_context(data)
                elif kind == "settings":
                    session.voice_output = bool(data.get("voice_output", True))
                elif kind == "reset":
                    session._reset_utterance()
                elif kind == "stop":
                    await session.force_stop()
                elif kind == "tts_playing":
                    await session.on_frontend_tts_playing(data)
                elif kind == "tts_finished":
                    await session.on_frontend_tts_finished(data)
                elif kind == "timeline_reset":
                    session._tl_reset()
            elif msg.get("bytes") is not None:
                await session.feed_pcm(msg["bytes"])
            session._input_busy = False
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("OpenAmika dialog failed")
    finally:
        if session is not None:
            _dialog_sessions.discard(session)
            if session._pending_partial and not session._pending_partial.done():
                session._pending_partial.cancel()
            on_close = getattr(session, "_tl_on_disconnect", None)
            if on_close is not None:
                on_close()
            session._committed_turn_id = None
            await session.hane.disconnect()
            await session.tts.disconnect()
        _dialog_active = False


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("YUMEHA_HOST", "127.0.0.1")
    port = int(os.getenv("YUMEHA_PORT", "53000"))   # clone speech port (Build F control is 52000)
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
