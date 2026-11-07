"""
Main entry point for the Ollama to OpenAI proxy service.
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src._version import get_version, get_version_info
from src.config import get_settings
from src.middleware.logging_middleware import LoggingMiddleware
from src.middleware.metrics_middleware import MetricsMiddleware
from src.routers import chat, embeddings, metrics, models, version
from src.utils.exceptions import ProxyException, UpstreamError
from src.utils.http_client import close_global_client
from src.utils.logging import get_logger, setup_logging

# Initialize settings and logging
settings = get_settings()
setup_logging(level=settings.LOG_LEVEL)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.

    Handles startup and shutdown events.
    """
    # Startup
    # Log version information
    version_info = get_version_info()
    logger.info(
        "Ollama-OpenAI Proxy starting",
        extra={
            "extra_data": {
                "version": version_info["version"],
                "build_date": version_info["build_date"],
                "commit_sha": os.getenv("GITHUB_SHA", "development"),
                "api_version": version_info["api_version"],
                "project_name": version_info["project_name"],
            }
        },
    )

    # Log environment variables for debugging
    logger.info(
        "Environment variables for SSL",
        extra={
            "extra_data": {
                "env_DISABLE_SSL_VERIFICATION": os.getenv(
                    "DISABLE_SSL_VERIFICATION", "not_set"
                ),
                "env_DEBUG": os.getenv("DEBUG", "not_set"),
            }
        },
    )

    logger.info(
        "Starting Ollama-OpenAI Proxy",
        extra={
            "extra_data": {
                "proxy_port": settings.PROXY_PORT,
                "target_url": settings.OPENAI_API_BASE_URL,
                "log_level": settings.LOG_LEVEL,
                "version": get_version(),
                "disable_ssl_verification": settings.DISABLE_SSL_VERIFICATION,
                "ssl_verification_enabled": not settings.DISABLE_SSL_VERIFICATION,
                "debug": settings.DEBUG,
            }
        },
    )

    yield

    # Shutdown
    logger.info("Shutting down Ollama-OpenAI Proxy")

    # Close global HTTP client
    await close_global_client()


# Create FastAPI app
app = FastAPI(
    title="Ollama-OpenAI Proxy",
    description="Proxy service to translate Ollama API calls to OpenAI format",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# Add request ID middleware
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class InboundKeyMiddleware:
    """Fork patch (ours): optional API-key gate on the proxy's own endpoints — the old image ships
    with none, so any client could use the upstream it fronts. Ollama itself never sends a key;
    OpenAmika's fork does (config/ollama.json api_key), and it matches INBOUND_API_KEYS here.
    Unset or empty keeps the proxy wide open."""

    _OPEN = {"", "/", "/health", "/ready", "/docs", "/redoc", "/openapi.json"}

    def __init__(self, app):
        self.app = app

    @staticmethod
    def _keys():
        return {k.strip() for k in os.getenv("INBOUND_API_KEYS", "").split(",") if k.strip()}

    async def __call__(self, scope, receive, send):
        keys = self._keys()
        path = scope.get("path", "") if scope.get("type") == "http" else ""
        if keys and path and path not in self._OPEN:
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
            given = headers.get("authorization", "")
            given = given[7:].strip() if given.lower().startswith("bearer ") else (given or headers.get("x-api-key", ""))
            if given not in keys:
                from starlette.responses import JSONResponse as _JR
                await _JR(status_code=401, content={"error": {"message": "missing or wrong API key", "type": "unauthorized"}})(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Add metrics middleware (before logging to capture all requests)
app.add_middleware(MetricsMiddleware, track_request_body=True, track_response_body=True)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Fork patch (ours): inbound API-key gate — outermost, so it runs before metrics/logging
app.add_middleware(InboundKeyMiddleware)


# Error handlers
@app.exception_handler(ProxyException)
async def proxy_exception_handler(
    request: Request, exc: ProxyException
) -> JSONResponse:
    """Handle proxy-specific exceptions."""
    return JSONResponse(
        status_code=400,  # ProxyException doesn't have a status_code attribute
        content={
            "error": {
                "message": exc.message,
                "type": exc.__class__.__name__,
                "code": exc.error_code,
                "details": exc.details,
            }
        },
        headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
    )


@app.exception_handler(UpstreamError)
async def upstream_error_handler(request: Request, exc: UpstreamError) -> JSONResponse:
    """Handle upstream service errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": "upstream_error",
                "status_code": exc.status_code,
                "service": (
                    exc.details.get("service") if "service" in exc.details else None
                ),
                "details": exc.details,
            }
        },
        headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.error(
        f"Unexpected error: {str(exc)}",
        exc_info=exc,
        extra={
            "extra_data": {
                "request_id": getattr(request.state, "request_id", "unknown"),
                "path": request.url.path,
                "method": request.method,
            }
        },
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        },
        headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
    )


# Include routers
app.include_router(version.router, prefix="/v1", tags=["version"])
app.include_router(chat.router, prefix="/v1", tags=["chat"])
app.include_router(models.router, prefix="/v1", tags=["models"])
app.include_router(embeddings.router, prefix="/v1", tags=["embeddings"])
app.include_router(metrics.router, prefix="/v1", tags=["metrics"])

# Also include Ollama-style endpoints
app.include_router(chat.router, prefix="/api", tags=["ollama-chat"])
app.include_router(models.router, prefix="/api", tags=["ollama-mode"])
app.include_router(embeddings.router, prefix="/api", tags=["ollama-embeddings"])
app.include_router(metrics.router, prefix="/api", tags=["ollama-metrics"])


# Health check endpoints
@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "ollama-openai-proxy", "version": "1.0.0"}


@app.get("/ready", tags=["health"])
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check endpoint.

    Verifies the service is ready to handle requests.
    """
    # In the future, this could check:
    # - Database connections
    # - External service availability
    # - Model loading status

    return {
        "status": "ready",
        "service": "ollama-openai-proxy",
        "checks": {"config": "ok", "logging": "ok"},
    }


# Root endpoint
@app.get("/", tags=["info"])
async def root() -> Dict[str, Any]:
    """Root endpoint with service information."""
    return {
        "service": "Ollama-OpenAI Proxy",
        "version": "1.0.0",
        "description": "Proxy service to translate Ollama API calls to OpenAI format",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs" if settings.DEBUG else None,
            "openai": {
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "embeddings": "/v1/embeddings",
                "metrics": "/v1/metrics",
            },
            "ollama": {
                "generate": "/api/generate",
                "chat": "/api/chat",
                "models": "/api/tags",
                "embeddings": "/api/embeddings",
                "metrics": "/api/metrics",
            },
        },
    }


def main():
    """Main entry point for the CLI."""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.PROXY_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()