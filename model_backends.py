"""Lightweight optional model backend helpers.

This module intentionally keeps only the OpenAI-compatible HTTP backend.
The default Transformers backend remains in run.py to avoid a large refactor.
"""

from __future__ import annotations

import json
from typing import Any

import requests


OPENAI_COMPATIBLE_ALIASES = {"openai", "openai_compatible", "api", "llama_server", "ollama", "lmstudio"}


def is_openai_compatible_backend(name: str | None) -> bool:
    return (name or "").strip().lower() in OPENAI_COMPATIBLE_ALIASES


def normalize_api_base(api_base: str) -> str:
    api_base = (api_base or "").strip()
    if not api_base:
        return ""
    if api_base.endswith("/chat/completions"):
        return api_base
    if api_base.endswith("/v1"):
        return api_base.rstrip("/") + "/chat/completions"
    return api_base.rstrip("/") + "/v1/chat/completions"


def ping_openai_compatible(api_base: str, timeout: float = 3.0) -> dict[str, Any]:
    """Best-effort health check for a local OpenAI-compatible endpoint.

    Some servers don't expose /models, so this is advisory only.
    """
    endpoint = normalize_api_base(api_base)
    if not endpoint:
        return {"ok": False, "error": "LLM_API_BASE is empty"}

    models_url = endpoint.replace("/chat/completions", "/models")
    try:
        response = requests.get(models_url, timeout=timeout)
        if response.ok:
            return {"ok": True, "endpoint": endpoint, "models_endpoint": models_url}
        return {"ok": False, "endpoint": endpoint, "status_code": response.status_code, "warning": response.text[:200]}
    except Exception as exc:
        return {"ok": False, "endpoint": endpoint, "error": str(exc)[:200]}


def generate_openai_compatible(
    *,
    api_base: str,
    model_name: str,
    user_prompt: str,
    system_prompt: str = "",
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 120.0,
    api_key: str = "",
) -> str:
    endpoint = normalize_api_base(api_base)
    if not endpoint:
        raise RuntimeError("MODEL_BACKEND=openai_compatible 时必须配置 LLM_API_BASE。")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name or "local-model",
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }

    response = requests.post(
        endpoint,
        headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()

    choices = data.get("choices") or []
    if not choices:
        return ""

    first = choices[0]
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        content = first.get("text", "")
    return str(content or "").strip()
