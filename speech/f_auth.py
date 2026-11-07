"""Fork addition (ours, not upstream): login gate for the browser entry (:53000).

Upstream ships no auth; this fork is ours to protect. While no account exists the gate
stays open so first-run can happen — the first sign-in (login page or the wizard's
account step) CREATES the account, once. After that the gate is closed.

Storage: config/auth.json — username, scrypt hash, salt, session secret (0600).
Sessions: signed cookies (HMAC), 30 days.

Gate rule:
  - allow: /login, /logout, /static, /favicon.ico, /health (the launcher's monitor)
  - no account yet            -> everything open (first sign-in creates it)
  - proxied request (browser) -> needs a valid session cookie
  - direct loopback (launcher, services, TTS, messengers) -> always allowed
WebSockets pass the same test; browsers send the cookie on the same-origin upgrade.
"""
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

ROOT = Path(__file__).resolve().parents[1]
AUTH_FILE = ROOT / "config" / "auth.json"
COOKIE = "amika_session"
MAX_AGE = 30 * 24 * 3600
ALLOW = {"/login", "/logout", "/favicon.ico", "/health"}


def _load():
    if not AUTH_FILE.is_file():
        return None
    try:
        doc = json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) and doc.get("username") and doc.get("secret") else None
    except (ValueError, OSError):
        return None


def _hash(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return salt.hex(), dk.hex()


def _sign(secret, payload):
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _token(doc, username):
    payload = "%s.%d" % (username, int(time.time()) + MAX_AGE)
    return payload + "." + _sign(doc["secret"], payload)


def _valid(doc, token):
    if not token:
        return False
    parts = str(token).split(".")
    if len(parts) != 3:
        return False
    user, exp, sig = parts
    try:
        if int(exp) < time.time():
            return False
    except ValueError:
        return False
    return user == doc["username"] and hmac.compare_digest(sig, _sign(doc["secret"], "%s.%s" % (user, exp)))


def _proxied(headers):
    return any(k in headers for k in (b"x-forwarded-for", b"x-real-ip", b"x-forwarded-proto"))


def _cookie(request, value, max_age=None):
    secure = request.headers.get("x-forwarded-proto") == "https"
    return ("%s=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=%d%s"
            % (COOKIE, urllib.parse.quote(value), MAX_AGE if max_age is None else max_age,
               "; Secure" if secure else ""))


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OpenAmika · sign in</title><style>
body{background:#14101f;color:#e8e2f5;font:15px/1.5 system-ui,sans-serif;display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
.card{background:#1d1730;border:1px solid #342a52;border-radius:14px;padding:30px 34px;width:min(92vw,360px);box-shadow:0 12px 40px #0009}
h1{font-size:19px;margin:0 0 4px}h1 b{color:#d36ef7;font-style:italic;font-weight:600}
p.sub{color:#9c90bd;margin:0 0 20px;font-size:13px}
label{display:block;font-size:12px;color:#9c90bd;margin:12px 0 4px}
input{width:100%;box-sizing:border-box;background:#14101f;border:1px solid #342a52;border-radius:8px;color:#e8e2f5;padding:9px 10px;font-size:14px}
button{margin-top:18px;width:100%;background:#8b46c8;border:0;border-radius:8px;color:#fff;padding:10px;font-size:14px;cursor:pointer}
button:hover{background:#a05ce0}.err{color:#ff7b9c;font-size:13px;margin-top:12px;min-height:16px}
</style></head><body><form class="card" id="f">
<h1 id="h">Sign in to <b>her</b></h1><p class="sub" id="sub">This is her computer. Only you hold the key.</p>
<label>Username</label><input id="u" autocomplete="username" autofocus>
<label>Password</label><input id="p" type="password" autocomplete="current-password">
<div class="err" id="e"></div><button>Enter</button></form><script>
fetch("/api/auth/status").then(r => r.json()).then((s) => {
  if (!s.configured) { document.getElementById("h").textContent = "Set her key";
    document.getElementById("sub").textContent = "First run — pick a username and a password (8+ characters). This sets the account, once."; }
});
document.getElementById("f").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const r = await fetch("/login", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username: document.getElementById("u").value.trim(),
                           password: document.getElementById("p").value }) });
  const j = await r.json().catch(() => ({}));
  if (r.ok) location.href = "/use"; else document.getElementById("e").textContent = j.error || "failed";
});
</script></body></html>"""


class LoginGate:
    """Pure ASGI middleware so HTTP and WebSocket both pass the gate."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        doc = _load()
        if doc is None:
            return await self.app(scope, receive, send)  # first sign-in creates the account
        headers = dict(scope.get("headers") or [])
        if not _proxied(headers):
            return await self.app(scope, receive, send)  # internal, loopback direct
        path = urllib.parse.unquote(scope.get("path", "/"))
        if path in ALLOW or path.startswith("/static"):
            return await self.app(scope, receive, send)
        cookie = (headers.get(b"cookie") or b"").decode()
        token = dict(p.strip().split("=", 1) for p in cookie.split(";") if "=" in p).get(COOKIE, "")
        if _valid(doc, token):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        if path.startswith("/api"):
            resp = JSONResponse({"error": "not logged in"}, status_code=401)
        else:
            resp = RedirectResponse(url="/login", status_code=302)
        await resp(scope, receive, send)


def install(app):
    @app.get("/login", include_in_schema=False)
    async def login_page(request: Request):
        return HTMLResponse(PAGE)

    @app.get("/api/auth/status", include_in_schema=False)
    async def auth_status(request: Request):
        doc = _load()
        return {"ok": True, "configured": doc is not None,
                "username": doc["username"] if doc else None}

    @app.post("/login", include_in_schema=False)
    async def login_post(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = dict(await request.form())
        user = str((body or {}).get("username") or "").strip()
        pw = str((body or {}).get("password") or "")
        doc = _load()
        created = False
        if doc is None:
            if not user or len(pw) < 8:
                return JSONResponse({"error": "pick a username and a password of 8 characters or more"}, status_code=400)
            salt, dk = _hash(pw)
            doc = {"username": user, "salt": salt, "hash": dk, "secret": secrets.token_hex(32),
                   "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
            AUTH_FILE.write_text(json.dumps(doc, indent=2), encoding="utf-8")
            os.chmod(AUTH_FILE, 0o600)
            created = True
        elif not user or not pw or not hmac.compare_digest(_hash(pw, doc["salt"])[1], doc["hash"]):
            return JSONResponse({"error": "wrong name or password"}, status_code=401)
        resp = JSONResponse({"ok": True, "created": created})
        resp.headers["set-cookie"] = _cookie(request, _token(doc, doc["username"]))
        return resp

    @app.post("/logout", include_in_schema=False)
    async def logout(request: Request):
        resp = JSONResponse({"ok": True})
        resp.headers["set-cookie"] = _cookie(request, "", max_age=0)
        return resp

    app.add_middleware(LoginGate)  # outermost layer — Starlette puts later-added middleware outermost
