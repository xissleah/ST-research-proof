"""
Lightweight direct-source resolver for the local AI search tool.

Purpose:
- If the user gives a concrete URL, fetch that URL directly before falling back to SearXNG.
- Keep platform-specific logic out of run.py.
- Add future sources by appending one adapter to DIRECT_SOURCE_ADAPTERS.

Each adapter returns result dictionaries compatible with SearXNG-style results:
{
    "title": "...",
    "url": "...",
    "content": "...",
    "_matched_query": "...",
    "_direct_source": "github_api" | "generic_url" | ...,
}
"""

from __future__ import annotations

import base64
import re
from html import unescape
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urlparse

import requests


DEFAULT_TIMEOUT = (3, 8)
DEFAULT_MAX_BYTES = 900_000
DEFAULT_TEXT_MAX_CHARS = 4_000


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta_description = ""
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = (tag or "").lower()
        attrs_dict = {k.lower(): (v or "") for k, v in attrs}

        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return

        if tag == "title":
            self._in_title = True
            return

        if tag == "meta":
            name = attrs_dict.get("name", "").lower()
            prop = attrs_dict.get("property", "").lower()
            content = attrs_dict.get("content", "").strip()
            if content and (name == "description" or prop == "og:description"):
                if not self.meta_description:
                    self.meta_description = content

    def handle_endtag(self, tag):
        tag = (tag or "").lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if not data or self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title_parts.append(text)
        else:
            self.text_parts.append(text)

    @property
    def title(self) -> str:
        return _clean_text(" ".join(self.title_parts), max_chars=220)

    @property
    def text(self) -> str:
        return _clean_text(" ".join(self.text_parts), max_chars=DEFAULT_TEXT_MAX_CHARS)


def _clean_text(text: str, max_chars: int = DEFAULT_TEXT_MAX_CHARS) -> str:
    text = unescape(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "..."
    return text


def _strip_url_punctuation(url: str) -> str:
    return (url or "").strip().rstrip(".,;:!?，。；：！？）)]}>'\"")


def extract_urls(text: str) -> list[str]:
    urls = []
    seen = set()
    for match in re.finditer(r"https?://[^\s<>'\"`]+", text or "", flags=re.IGNORECASE):
        url = _strip_url_punctuation(match.group(0))
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def normalize_domain(domain_or_url: str) -> str:
    value = (domain_or_url or "").strip().lower()
    if value.startswith("http://") or value.startswith("https://"):
        value = urlparse(value).netloc.lower()
    if value.startswith("www."):
        value = value[4:]
    return value


def get_direct_url_domains(text: str) -> set[str]:
    domains = set()
    for url in extract_urls(text):
        domain = normalize_domain(urlparse(url).netloc)
        if domain:
            domains.add(domain)
    return domains


def _headers(config: dict | None = None, *, accept: str | None = None) -> dict:
    config = config or {}
    headers = {
        "User-Agent": config.get("user_agent") or "local-ai-search/1.0",
    }
    if accept:
        headers["Accept"] = accept
    return headers


def match_github_repo(text: str, consumed_urls: set[str]) -> list[dict]:
    matches = []
    for url in extract_urls(text):
        parsed = urlparse(url)
        domain = normalize_domain(parsed.netloc)
        if domain != "github.com":
            continue

        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            continue

        owner = parts[0]
        repo = parts[1].removesuffix(".git")
        if not owner or not repo:
            continue

        consumed_urls.add(url)
        matches.append({
            "url": url,
            "owner": owner,
            "repo": repo,
            "matched_query": url,
        })
    return matches


def fetch_github_repo(match: dict, config: dict | None = None) -> list[dict]:
    config = config or {}
    owner = match["owner"]
    repo = match["repo"]
    html_url = f"https://github.com/{owner}/{repo}"
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    headers = _headers(config, accept="application/vnd.github+json")
    token = (config.get("github_token") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    timeout = config.get("timeout") or DEFAULT_TIMEOUT

    response = requests.get(api_url, headers=headers, timeout=timeout)

    if response.status_code == 404:
        return [{
            "title": f"GitHub 仓库不存在或不可访问：{owner}/{repo}",
            "url": html_url,
            "content": "GitHub API 返回 404。可能原因：仓库不存在、私有仓库、拼写错误，或当前 token 没有访问权限。",
            "_matched_query": match.get("matched_query") or html_url,
            "_direct_source": "github_api",
        }]

    response.raise_for_status()
    data = response.json()

    readme_text = ""
    readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
        readme_resp = requests.get(readme_url, headers=headers, timeout=timeout)
        if readme_resp.ok:
            readme_data = readme_resp.json()
            encoded = readme_data.get("content") or ""
            if encoded:
                readme_text = base64.b64decode(encoded).decode("utf-8", errors="ignore")
    except Exception:
        readme_text = ""

    topics = ", ".join(data.get("topics") or [])
    license_info = data.get("license") or {}
    license_name = license_info.get("spdx_id") or license_info.get("name") or "未知"

    content = "\n".join([
        f"仓库：{data.get('full_name') or f'{owner}/{repo}'}",
        f"简介：{data.get('description') or ''}",
        f"语言：{data.get('language') or '未知'}",
        f"Stars：{data.get('stargazers_count', 0)}",
        f"Forks：{data.get('forks_count', 0)}",
        f"License：{license_name}",
        f"Topics：{topics}",
        f"创建时间：{data.get('created_at') or ''}",
        f"最近推送：{data.get('pushed_at') or ''}",
        "",
        "README 摘录：",
        _clean_text(readme_text, max_chars=3000),
    ])

    return [{
        "title": f"{data.get('full_name') or f'{owner}/{repo}'} - GitHub repository",
        "url": data.get("html_url") or html_url,
        "content": _clean_text(content, max_chars=5000),
        "_matched_query": match.get("matched_query") or html_url,
        "_direct_source": "github_api",
    }]


def match_generic_url(text: str, consumed_urls: set[str]) -> list[dict]:
    matches = []
    for url in extract_urls(text):
        if url in consumed_urls:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        consumed_urls.add(url)
        matches.append({
            "url": url,
            "matched_query": url,
        })
    return matches


def fetch_generic_url(match: dict, config: dict | None = None) -> list[dict]:
    config = config or {}
    url = match["url"]
    timeout = config.get("timeout") or DEFAULT_TIMEOUT
    max_bytes = int(config.get("max_bytes") or DEFAULT_MAX_BYTES)

    response = requests.get(
        url,
        headers=_headers(config, accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    raw = bytearray()
    for chunk in response.iter_content(chunk_size=16384):
        if not chunk:
            continue
        raw.extend(chunk)
        if len(raw) >= max_bytes:
            break

    text = bytes(raw).decode(response.encoding or "utf-8", errors="ignore")

    if "html" in content_type.lower() or "<html" in text[:1000].lower():
        parser = _TextExtractor()
        parser.feed(text)
        title = parser.title or url
        description = _clean_text(parser.meta_description, max_chars=800)
        body = parser.text
        content = "\n".join(part for part in [description, body] if part).strip()
    else:
        title = url
        content = _clean_text(text, max_chars=DEFAULT_TEXT_MAX_CHARS)

    return [{
        "title": title,
        "url": response.url or url,
        "content": content or "已成功访问该 URL，但未能提取到足够正文。",
        "_matched_query": match.get("matched_query") or url,
        "_direct_source": "generic_url",
    }]


# Add future platforms here. Keep generic_url last as fallback.
DIRECT_SOURCE_ADAPTERS: list[dict[str, Callable]] = [
    {"name": "github", "match": match_github_repo, "fetch": fetch_github_repo},
    {"name": "generic_url", "match": match_generic_url, "fetch": fetch_generic_url},
]


def resolve_direct_sources(text: str, config: dict | None = None) -> list[dict]:
    """Resolve concrete URLs in user/query text into search-result-like records."""
    results: list[dict] = []
    consumed_urls: set[str] = set()
    seen_result_urls: set[str] = set()

    for adapter in DIRECT_SOURCE_ADAPTERS:
        name = adapter["name"]
        matches = adapter["match"](text, consumed_urls)
        for match in matches:
            try:
                fetched = adapter["fetch"](match, config)
            except Exception as exc:
                url = match.get("url", "")
                fetched = [{
                    "title": f"{name} 直连解析失败",
                    "url": url,
                    "content": f"已识别为直接 URL，但直连解析失败：{exc}",
                    "_matched_query": match.get("matched_query") or url,
                    "_direct_source": name,
                }]

            for item in fetched or []:
                url = (item.get("url") or "").strip()
                if url and url in seen_result_urls:
                    continue
                if url:
                    seen_result_urls.add(url)
                item.setdefault("_direct_source", name)
                item.setdefault("_matched_query", match.get("matched_query") or match.get("url") or text)
                results.append(item)

    return results
