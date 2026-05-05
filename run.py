import os
import re
import json
import sys
import time
import webbrowser
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
from html import unescape
from html.parser import HTMLParser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import requests

try:
    from direct_sources import resolve_direct_sources, get_direct_url_domains
except Exception:
    # direct_sources.py 是轻量直连解析模块；缺失时保留原有搜索流程。
    def resolve_direct_sources(text: str, config: dict | None = None) -> list:
        return []

    def get_direct_url_domains(text: str) -> set:
        return set()

try:
    from local_kb import (
        ensure_kb_layout,
        get_kb_status,
        rebuild_kb_index,
        search_local_kb,
        sync_kb_index,
    )
except Exception:
    # local_kb.py 缺失时，保留原有联网搜索能力。
    def ensure_kb_layout(*args, **kwargs):
        return None

    def get_kb_status(*args, **kwargs) -> dict:
        return {"ok": False, "error": "local_kb.py 未加载"}

    def rebuild_kb_index(*args, **kwargs) -> dict:
        return {"ok": False, "error": "local_kb.py 未加载"}

    def search_local_kb(*args, **kwargs) -> list:
        return []

    def sync_kb_index(*args, **kwargs) -> dict:
        return {"ok": False, "error": "local_kb.py 未加载"}

try:
    from answer_extractors import try_direct_answer
except Exception:
    def try_direct_answer(question: str, results: list) -> dict | None:
        return None

try:
    from model_backends import (
        generate_openai_compatible,
        is_openai_compatible_backend,
        normalize_api_base,
        ping_openai_compatible,
    )
except Exception:
    def is_openai_compatible_backend(name: str | None) -> bool:
        return False

    def normalize_api_base(api_base: str) -> str:
        return api_base or ""

    def ping_openai_compatible(api_base: str, timeout: float = 3.0) -> dict:
        return {"ok": False, "error": "model_backends.py 未加载"}

    def generate_openai_compatible(**kwargs) -> str:
        raise RuntimeError("model_backends.py 未加载，无法使用 OpenAI-compatible 后端。")

try:
    from web_evidence import (
        attach_web_evidence_to_results,
        extract_web_blocks_from_html,
        format_item_evidence_for_model,
        format_web_evidence_chain_for_user,
        validate_evidence_citations,
    )
except Exception:
    def attach_web_evidence_to_results(results, question, max_total=8, per_source=2):
        return results, []

    def extract_web_blocks_from_html(html_text: str) -> list:
        return []

    def format_item_evidence_for_model(item: dict) -> str:
        return ""

    def format_web_evidence_chain_for_user(cards: list) -> str:
        return ""

    def validate_evidence_citations(answer: str, cards: list) -> dict:
        return {"ok": True, "has_evidence": bool(cards), "cited_ids": [], "invalid_ids": []}

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# 基础配置
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CFG_PATH = SCRIPT_DIR / "cfg.txt"


def load_cfg() -> dict:
    """
    读取和 run.py 同目录下的 cfg.txt。
    格式：
    MODEL_PATH=C:\\work\\search\\model\\Qwen3.5-2B
    SEARXNG_URL=http://localhost:18080/search
    SERVER_PORT=8080

    优先级：
    环境变量 > cfg.txt > 默认值
    """
    cfg = {}

    if not CFG_PATH.exists():
        return cfg

    for line in CFG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            cfg[key] = value

    return cfg


CFG = load_cfg()


def get_cfg_value(key: str, default: str) -> str:
    return os.getenv(key, CFG.get(key, default))


def parse_bool(value: str, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if text in {"1", "true", "yes", "y", "on", "启用", "是"}:
        return True
    if text in {"0", "false", "no", "n", "off", "禁用", "否"}:
        return False
    return default


def resolve_cfg_path(value: str, base_dir: Path) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return (base_dir / path).resolve()


base_dir_text = get_cfg_value("BASE_DIR", str(SCRIPT_DIR))
BASE_DIR = resolve_cfg_path(base_dir_text, SCRIPT_DIR)

model_path_text = get_cfg_value("MODEL_PATH", str(BASE_DIR / "model" / "Qwen3.5-2B"))
MODEL_PATH = resolve_cfg_path(model_path_text, BASE_DIR)

SKILL_PATH = BASE_DIR / "skill" / "search.txt"
QUESTION_PATH = BASE_DIR / "question.txt"
RESULT_PATH = BASE_DIR / "result.txt"
DEBUG_PATH = BASE_DIR / "debug.txt"
WEBSITE_PATH = BASE_DIR / "website.txt"

SEARXNG_URL = get_cfg_value("SEARXNG_URL", "http://localhost:18080/search")

# 默认 all，避免强制 zh-CN 导致结果过度偏向中文站点。
# 如需强制中文，可在 cfg.txt 里写：SEARXNG_LANGUAGE=zh-CN
SEARCH_LANGUAGE = get_cfg_value("SEARXNG_LANGUAGE", "all").strip()

# 可选：用于 GitHub API 直连解析。公开仓库不填也能查，但填了速率限制更宽。
GITHUB_TOKEN = get_cfg_value("GITHUB_TOKEN", "").strip()

# 应用档位：local/web/fast/full。当前版本不强制覆盖配置，仅用于状态展示和启动脚本判断。
APP_PROFILE = get_cfg_value("APP_PROFILE", "local").strip().lower()

# 模型后端：
# transformers：沿用当前 HF/Transformers 本地模型；
# openai_compatible：调用本机或局域网 OpenAI-compatible 服务，例如 llama-server / Ollama / LM Studio。
MODEL_BACKEND = get_cfg_value("MODEL_BACKEND", "transformers").strip().lower()
LLM_API_BASE = get_cfg_value("LLM_API_BASE", "http://127.0.0.1:8001/v1/chat/completions").strip()
LLM_MODEL_NAME = get_cfg_value("LLM_MODEL_NAME", "local-model").strip()
LLM_API_KEY = get_cfg_value("LLM_API_KEY", "").strip()
LLM_TIMEOUT_SECONDS = float(get_cfg_value("LLM_TIMEOUT_SECONDS", "120"))

# 本地题库/结构化答案直抽。命中“正确答案:”这类结构时可跳过模型生成，速度会明显提升。
DIRECT_ANSWER_ENABLED = parse_bool(get_cfg_value("DIRECT_ANSWER_ENABLED", "1"), default=True)

# 本地知识库：第一版保持轻量，默认使用 kb/documents + kb/kb.sqlite。
LOCAL_KB_ENABLED = parse_bool(get_cfg_value("LOCAL_KB_ENABLED", "1"), default=True)
LOCAL_KB_DOCS_DIR = resolve_cfg_path(get_cfg_value("LOCAL_KB_DOCS_DIR", str(BASE_DIR / "kb" / "documents")), BASE_DIR)
LOCAL_KB_DB_PATH = resolve_cfg_path(get_cfg_value("LOCAL_KB_DB_PATH", str(BASE_DIR / "kb" / "kb.sqlite")), BASE_DIR)
LOCAL_KB_SEARCH_LIMIT = int(get_cfg_value("LOCAL_KB_SEARCH_LIMIT", "8"))
LOCAL_KB_AUTO_ENOUGH_RESULTS = int(get_cfg_value("LOCAL_KB_AUTO_ENOUGH_RESULTS", "4"))
LOCAL_KB_AUTO_MIN_SCORE = float(get_cfg_value("LOCAL_KB_AUTO_MIN_SCORE", "8"))
LOCAL_KB_AUTO_SYNC = parse_bool(get_cfg_value("LOCAL_KB_AUTO_SYNC", "1"), default=True)
LOCAL_KB_SYNC_BEFORE_SEARCH = parse_bool(get_cfg_value("LOCAL_KB_SYNC_BEFORE_SEARCH", "1"), default=True)
LOCAL_KB_SYNC_ON_START = parse_bool(get_cfg_value("LOCAL_KB_SYNC_ON_START", "0"), default=False)
LOCAL_KB_AUTO_SYNC_INTERVAL_SECONDS = float(get_cfg_value("LOCAL_KB_AUTO_SYNC_INTERVAL_SECONDS", "5"))
DEFAULT_SEARCH_MODE = get_cfg_value("SEARCH_MODE", "AUTO").strip().lower()

SERVER_HOST = get_cfg_value("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(get_cfg_value("SERVER_PORT", "8080"))
INDEX_URL = f"http://localhost:{SERVER_PORT}/index.html"

DEFAULT_SITE_WEIGHT = 1
MIN_RELEVANCE_SCORE = 1
DISABLED_SITE_WEIGHT = 0

# 速度相关配置
FAST_SEARCH_RESULTS_PER_QUERY = int(get_cfg_value("FAST_SEARCH_RESULTS_PER_QUERY", "6"))
PRECISE_SEARCH_RESULTS_PER_QUERY = int(get_cfg_value("PRECISE_SEARCH_RESULTS_PER_QUERY", "10"))

FAST_FINAL_RESULTS = int(get_cfg_value("FAST_FINAL_RESULTS", "6"))
PRECISE_FINAL_RESULTS = int(get_cfg_value("PRECISE_FINAL_RESULTS", "10"))

FAST_FINAL_TOKENS = int(get_cfg_value("FAST_FINAL_TOKENS", "256"))
PRECISE_FINAL_TOKENS = int(get_cfg_value("PRECISE_FINAL_TOKENS", "512"))

QUERY_MAX_TOKENS = int(get_cfg_value("QUERY_MAX_TOKENS", "96"))

# 可信搜索增强模块
# 1. 页面正文抓取：不能只依赖 SearXNG snippet。
# 2. 反证/验证查询：主动寻找限制、冲突、版本和官方证据。
# 3. 证据审计：把来源拆成可解释维度，避免“看起来很对”的强行总结。
# 4. 结论门控：证据不足时，宁可不给结论，也不编造。
# 5. 查询模板库：按官方、论文、代码、排错、版本等场景扩展搜索。
FAST_FETCH_TOP_N = int(get_cfg_value("FAST_FETCH_TOP_N", "3"))
PRECISE_FETCH_TOP_N = int(get_cfg_value("PRECISE_FETCH_TOP_N", "5"))
FETCH_TIMEOUT_SECONDS = float(get_cfg_value("FETCH_TIMEOUT_SECONDS", "8"))
FETCH_MAX_BYTES = int(get_cfg_value("FETCH_MAX_BYTES", "900000"))
FETCH_TEXT_MAX_CHARS = int(get_cfg_value("FETCH_TEXT_MAX_CHARS", "6000"))
QUERY_TEMPLATE_FAST_LIMIT = int(get_cfg_value("QUERY_TEMPLATE_FAST_LIMIT", "3"))
QUERY_TEMPLATE_PRECISE_LIMIT = int(get_cfg_value("QUERY_TEMPLATE_PRECISE_LIMIT", "5"))

# 并行搜索配置：只并行网络 I/O，不并行模型生成或知识库写入。
PARALLEL_SEARCH_ENABLED = parse_bool(get_cfg_value("PARALLEL_SEARCH_ENABLED", "1"), default=True)
SEARCH_WORKERS = max(1, int(get_cfg_value("SEARCH_WORKERS", "4")))
FETCH_WORKERS = max(1, int(get_cfg_value("FETCH_WORKERS", "4")))

WEB_EVIDENCE_ENABLED = parse_bool(get_cfg_value("WEB_EVIDENCE_ENABLED", "1"), default=True)
WEB_EVIDENCE_MAX_TOTAL = max(1, int(get_cfg_value("WEB_EVIDENCE_MAX_TOTAL", "8")))
WEB_EVIDENCE_PER_SOURCE = max(1, int(get_cfg_value("WEB_EVIDENCE_PER_SOURCE", "2")))

# 回答风格：plain 适合普通搜索；strict 适合科研/审计型回答。
ANSWER_STYLE = get_cfg_value("ANSWER_STYLE", "plain").strip().lower()
SHOW_EVIDENCE_AUDIT = parse_bool(get_cfg_value("SHOW_EVIDENCE_AUDIT", "0"), default=False)
SHOW_WEB_EVIDENCE_CHAIN = parse_bool(get_cfg_value("SHOW_WEB_EVIDENCE_CHAIN", "0"), default=False)
LOCAL_EVIDENCE_ENABLED = parse_bool(get_cfg_value("LOCAL_EVIDENCE_ENABLED", "1"), default=True)
SHOW_LOCAL_EVIDENCE_CHAIN = parse_bool(get_cfg_value("SHOW_LOCAL_EVIDENCE_CHAIN", "1"), default=True)
EVIDENCE_ONLY_MODE = parse_bool(get_cfg_value("EVIDENCE_ONLY_MODE", "0"), default=False)
REQUIRE_EVIDENCE_CITATIONS = parse_bool(get_cfg_value("REQUIRE_EVIDENCE_CITATIONS", "1"), default=True)

QUERY_MAX_TOKENS = 96

# 查询缓存，避免重复搜索同一问题时反复等 SearXNG
SEARCH_CACHE_TTL_SECONDS = 600

model = None
tokenizer = None
model_lock = threading.Lock()

search_cache_lock = threading.Lock()
search_cache = {}

progress_lock = threading.Lock()
kb_sync_lock = threading.Lock()
_last_kb_auto_sync_at = 0.0
progress_state = {
    "running": False,
    "percent": 0,
    "stage": "空闲",
    "detail": "",
}

cancel_lock = threading.Lock()
cancel_requested = False


# ============================================================
# 取消控制
# ============================================================

class UserCancelledError(Exception):
    pass


def reset_cancel():
    global cancel_requested
    with cancel_lock:
        cancel_requested = False


def request_cancel():
    global cancel_requested
    with cancel_lock:
        cancel_requested = True

    set_progress(
        100,
        "已请求终止",
        "正在尽快停止当前搜索任务",
        running=False,
    )


def is_cancel_requested() -> bool:
    with cancel_lock:
        return cancel_requested


def check_cancelled():
    if is_cancel_requested():
        raise UserCancelledError("搜索已被用户终止。")


class CancelStoppingCriteria(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return is_cancel_requested()


# ============================================================
# 进度状态
# ============================================================

def set_progress(percent: int, stage: str, detail: str = "", running: bool = True):
    with progress_lock:
        progress_state["running"] = running
        progress_state["percent"] = max(0, min(100, int(percent)))
        progress_state["stage"] = stage
        progress_state["detail"] = detail


def get_progress() -> dict:
    with progress_lock:
        state = dict(progress_state)

    state["cancel_requested"] = is_cancel_requested()
    return state


# ============================================================
# 文件读写
# ============================================================

def ensure_files():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "skill").mkdir(parents=True, exist_ok=True)

    QUESTION_PATH.touch(exist_ok=True)
    RESULT_PATH.touch(exist_ok=True)
    DEBUG_PATH.touch(exist_ok=True)
    WEBSITE_PATH.touch(exist_ok=True)
    SKILL_PATH.touch(exist_ok=True)

    if LOCAL_KB_ENABLED:
        ensure_kb_layout(
            BASE_DIR,
            docs_dir=LOCAL_KB_DOCS_DIR,
            db_path=LOCAL_KB_DB_PATH,
        )
        if LOCAL_KB_SYNC_ON_START:
            maybe_auto_sync_local_kb("start", force=True)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def write_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def clean_search_snippet(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\bMissing:\s*.*?(?=(?:Show results with:|$))", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bShow results with:\s*\S+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# 加载 skill/search.txt
# ============================================================

def load_skill() -> str:
    skill = read_text(SKILL_PATH)

    if skill:
        return skill

    return (
        "You are a rigorous AI search assistant. "
        "Your job is to understand the user's query, produce accurate search queries, "
        "filter search results, and answer clearly based only on verifiable evidence. "
        "Do not fabricate information. If the evidence is insufficient, say so explicitly."
    )


# ============================================================
# website.txt 权重
# 格式：
# github.com 5
# zhihu.com 2
# 未标注网站默认权重为 1
# ============================================================

def normalize_domain(domain: str) -> str:
    domain = (domain or "").strip().lower()

    if domain.startswith("http://") or domain.startswith("https://"):
        domain = urlparse(domain).netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def load_website_weights() -> dict:
    weights = {}

    if not WEBSITE_PATH.exists():
        return weights

    for line in WEBSITE_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            continue

        domain, weight_text = parts
        domain = normalize_domain(domain)

        try:
            weight = int(weight_text)
        except ValueError:
            continue

        weights[domain] = max(0, min(5, weight))

    return weights


def extract_domain(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def get_site_weight(domain: str, weights: dict) -> int:
    domain = normalize_domain(domain)

    if not domain:
        return DEFAULT_SITE_WEIGHT

    for configured_domain, weight in weights.items():
        configured_domain = normalize_domain(configured_domain)

        if domain == configured_domain:
            return weight

        if domain.endswith("." + configured_domain):
            return weight

    return DEFAULT_SITE_WEIGHT


def classify_source_type(domain: str) -> str:
    domain = normalize_domain(domain)

    official_suffixes = (
        ".gov",
        ".edu",
    )
    official_domains = {
        "openai.com",
        "docs.python.org",
        "python.org",
        "pytorch.org",
        "huggingface.co",
        "qwenlm.github.io",
        "alibabacloud.com",
        "learn.microsoft.com",
        "microsoft.com",
        "modelscope.cn",
    }
    doc_domains = {
        "readthedocs.io",
        "docs.python.org",
        "learn.microsoft.com",
    }
    paper_domains = {
        "arxiv.org",
        "aclanthology.org",
        "openreview.net",
    }
    repo_domains = {
        "github.com",
        "gitlab.com",
    }
    community_domains = {
        "zhihu.com",
        "zhuanlan.zhihu.com",
        "csdn.net",
        "blog.csdn.net",
        "cnblogs.com",
        "jianshu.com",
        "reddit.com",
        "cloud.tencent.com",
        "baidu.com",
        "wikipedia.org",
        "zh.wikipedia.org",
    }

    if any(domain.endswith(suffix) for suffix in official_suffixes):
        return "official"
    if domain in paper_domains:
        return "paper"
    if domain in repo_domains:
        return "repo"
    if domain in doc_domains:
        return "doc"
    if domain in official_domains:
        return "official"
    if domain in community_domains or ".wikipedia.org" in domain:
        return "community"
    return "general"


def get_source_type_bonus(source_type: str) -> int:
    bonus_map = {
        "local_kb": 42,
        "repo": 36,
        "paper": 32,
        "official": 28,
        "doc": 26,
        "general": 8,
        "community": -15,
    }
    return bonus_map.get(source_type, 0)


# ============================================================
# 模型加载与调用
# ============================================================

def load_model():
    global model, tokenizer

    if is_openai_compatible_backend(MODEL_BACKEND):
        model = None
        tokenizer = None
        endpoint = normalize_api_base(LLM_API_BASE)
        print("[模型] 使用外部 OpenAI-compatible 后端。")
        print(f"[模型] API：{endpoint}")
        print(f"[模型] 名称：{LLM_MODEL_NAME}")
        ping = ping_openai_compatible(endpoint, timeout=2.5)
        if ping.get("ok"):
            print("[模型] 外部模型服务可访问。")
        else:
            print(f"[模型] 警告：暂未确认外部模型服务可访问：{ping.get('error') or ping.get('warning') or ping}")
        return

    print("[模型] 正在加载本地 Transformers 模型...")
    print(f"[模型] 模型路径：{MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
    )

    model_kwargs = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = "auto"
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        **model_kwargs,
    )

    if not torch.cuda.is_available():
        model.to("cpu")

    model.eval()
    print("[模型] Transformers 模型加载完成。")


def build_chat_prompt(system_prompt: str, user_prompt: str) -> str:
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt,
        })

    messages.append({
        "role": "user",
        "content": user_prompt,
    })

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def gen_model_response(
    user_prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 768,
    do_sample: bool = False,
    temperature: float = 0.2,
) -> str:
    check_cancelled()

    if is_openai_compatible_backend(MODEL_BACKEND):
        # 外部模型服务不需要在 run.py 内加载 tokenizer/model。
        return generate_openai_compatible(
            api_base=LLM_API_BASE,
            model_name=LLM_MODEL_NAME,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            timeout=LLM_TIMEOUT_SECONDS,
            api_key=LLM_API_KEY,
        )

    if model is None or tokenizer is None:
        raise RuntimeError("模型尚未加载。")

    prompt = build_chat_prompt(system_prompt, user_prompt)

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": 1.06,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "stopping_criteria": StoppingCriteriaList([CancelStoppingCriteria()]),
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = 0.9
        generate_kwargs["top_k"] = 50

    with model_lock:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generate_kwargs,
            )

    check_cancelled()

    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True,
    )

    return response.strip()


# ============================================================
# 短词 / 歧义词搜索
# ============================================================

def is_ambiguous_short_query(question: str) -> bool:
    q = question.strip()
    q_clean = re.sub(r"[。！？!?，,\s]+", "", q)

    if not q_clean:
        return False

    if re.fullmatch(r"[\u4e00-\u9fff]{1,6}", q_clean):
        return True

    if re.fullmatch(r"[a-zA-Z0-9_\-\.\+#/]{1,12}", q_clean):
        return True

    return False


def summarize_short_query_results(question: str, search_query: str, ranked_results: list) -> str:
    if not ranked_results:
        return (
            "【核心结论】\n"
            f"没有找到与“{question}”足够相关的搜索结果。\n\n"
            "【简要依据】\n"
            "1. 这个输入较短，可能存在多个含义。\n"
            "2. 当前 SearXNG 返回结果不足以支持明确判断。\n\n"
            "【建议】\n"
            f"1. 可以补充限定词，例如“{question} 含义”。\n"
            f"2. 可以补充限定词，例如“{question} 歌曲”。\n"
            f"3. 可以补充限定词，例如“{question} 官网”。\n\n"
            "【不确定部分】\n"
            "无法仅凭短词判断用户真实搜索意图。"
        )

    lines = [
        "【核心结论】",
        f"你搜索的是短词：“{question}”。这个词可能有多种含义，因此这里只展示搜索结果，不强行推断你的真实意图。",
        "",
        "【搜索结果】",
    ]

    for index, item in enumerate(ranked_results, start=1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        domain = item.get("domain", "")
        content = clean_search_snippet(item.get("content", "") or "")
        weight = item.get("site_weight", DEFAULT_SITE_WEIGHT)
        relevance = item.get("relevance", 0)
        final_score = item.get("final_score", 0)

        lines.append(f"{index}. {title}")
        lines.append(f"来源：{domain}；权重：{weight}；相关性：{relevance}；总分：{final_score}")
        lines.append(f"链接：{url}")

        if content:
            lines.append(f"摘要：{content[:180]}")

        lines.append("")

    lines.extend([
        "【不确定部分】",
        "短词搜索不交给模型自由总结，避免模型根据歧义词脑补错误场景。",
    ])

    return "\n".join(lines)


# ============================================================
# 搜索词生成
# ============================================================

def clean_one_line(text: str, max_len: int = 240) -> str:
    text = text.strip()
    text = text.strip('"').strip("'").strip("`")

    prefixes = [
        "搜索词：", "搜索词:",
        "关键词：", "关键词:",
        "查询：", "查询:",
        "English search query:",
        "Search query:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text[:max_len].strip()


def extract_search_query(question: str, skill: str) -> str:
    prompt = f"""Rewrite the user's question into a concise search query suitable for SearXNG or a web search engine.

Requirements:
1. Preserve key entities, technical terms, version numbers, file paths, and error messages.
2. Do not make the query too short.
3. Do not rewrite the question into an unrelated topic.
4. Do not explain.
5. Do not output JSON.
6. Output exactly one line.

User question:
{question}

Search query:"""

    query = gen_model_response(
        user_prompt=prompt,
        system_prompt=skill,
        max_new_tokens=QUERY_MAX_TOKENS,
        do_sample=False,
    )

    query = clean_one_line(query)

    if not query:
        query = question.strip()

    if is_query_obviously_bad(question, query):
        print(f"[模型] 搜索词疑似跑偏，已退回原问题。模型输出：{query}")
        query = question.strip()

    print(f"[模型] 搜索词：{query}")
    return query


def extract_english_search_query(question: str, skill: str) -> str:
    """生成英文搜索词。优先用规则保护混合中英实体，避免模型脑补未知项目名。"""
    rule_query = rule_translate_mixed_query(question, for_search=True)
    if rule_query:
        print(f"[模型] 英文搜索词（规则）：{rule_query}")
        return rule_query[:220].strip()

    prompt = f"""请把下面的用户问题改写成适合英文搜索引擎使用的英文关键词查询。不要逐字翻译，要改写成英语用户真实会搜索的关键词。

要求：
1. 保留模型名、软件名、论文名、版本号、错误码、路径中的公开实体。
2. 技术词尽量使用英文表达。
3. 不要输出解释。
4. 不要输出 JSON。
5. 只输出一行英文搜索词。

用户问题：
{question}

English search query:"""

    query = gen_model_response(
        user_prompt=prompt,
        system_prompt=skill,
        max_new_tokens=QUERY_MAX_TOKENS,
        do_sample=False,
    )

    query = clean_one_line(query)

    if not query:
        query = question.strip()

    if translation_looks_like_answer(question, query):
        fallback = rule_translate_mixed_query(question, for_search=True) or build_preserving_english_query(question, for_search=True)
        if fallback:
            print(f"[模型] 英文搜索词疑似回答化或丢失关键信息，已回退规则搜索词。模型输出：{query}")
            query = fallback

    if is_query_obviously_bad(question, query):
        print(f"[模型] 英文搜索词疑似跑偏，回退原问题。模型输出：{query}")
        query = question.strip()

    print(f"[模型] 英文搜索词：{query}")
    return query[:220].strip()


def build_resource_query(question: str) -> str:
    """
    查资料模式默认不调用模型改写，直接做轻量规则清洗，速度更快。
    """
    q = question.strip()

    removable = [
        "帮我", "请", "查找", "查一下", "找一下", "找几篇", "推荐几篇",
        "找文章", "相关文章", "相关资料", "参考资料", "推荐文章",
        "推荐论文", "找论文", "论文", "资料", "链接", "网站",
        "有哪些", "给我", "关于",
    ]

    for word in removable:
        q = q.replace(word, " ")

    q = re.sub(r"\s+", " ", q).strip()
    q = re.sub(
        r"\b(find|search|lookup|look up|show me|give me|about|resources?|links?|articles?)\b",
        " ",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip(" -:：")

    return q or question.strip()


def detect_resource_intent(question: str) -> str:
    q = (question or "").lower()

    project_keywords = [
        "github", "gitlab", "repo", "repository", "repositories", "open source",
        "opensource", "library", "toolkit", "framework", "project", "source code",
        "开源", "项目", "仓库", "源码", "代码库", "库", "工具",
    ]
    paper_keywords = [
        "paper", "papers", "technical report", "whitepaper", "arxiv",
        "论文", "技术报告", "白皮书", "报告",
    ]
    docs_keywords = [
        "documentation", "docs", "api", "readme", "manual",
        "文档", "官方文档", "接口", "说明书", "手册",
    ]
    tutorial_keywords = [
        "tutorial", "guide", "example", "examples", "how to",
        "教程", "入门", "指南", "示例", "例子", "怎么用", "如何使用",
    ]
    troubleshooting_keywords = [
        "error", "errors", "exception", "traceback", "fix", "issue", "issues",
        "bug", "install", "setup", "config", "troubleshoot",
        "报错", "错误", "异常", "修复", "问题", "安装", "配置", "排查", "无法", "失败",
    ]

    if any(keyword in q for keyword in troubleshooting_keywords):
        return "troubleshooting"
    if any(keyword in q for keyword in project_keywords):
        return "project"
    if any(keyword in q for keyword in paper_keywords):
        return "paper"
    if any(keyword in q for keyword in docs_keywords):
        return "docs"
    if any(keyword in q for keyword in tutorial_keywords):
        return "tutorial"
    return "general"


def detect_answer_intent(question: str) -> str:
    q = (question or "").lower()

    troubleshooting_keywords = [
        "error", "exception", "traceback", "fix", "issue", "issues",
        "bug", "报错", "错误", "异常", "修复", "失败", "无法", "不工作",
    ]
    recommendation_keywords = [
        "recommend", "choose", "selection", "which is better", "best",
        "怎么选", "选哪个", "推荐", "适合", "值得", "应该用", "哪个好",
    ]
    comparison_keywords = [
        "vs", "versus", "compare", "comparison", "difference", "better than",
        "区别", "比较", "对比", "差异", "优劣", "是不是一样",
    ]

    if any(keyword in q for keyword in troubleshooting_keywords):
        return "troubleshooting"
    if any(keyword in q for keyword in recommendation_keywords):
        return "recommendation"
    if any(keyword in q for keyword in comparison_keywords):
        return "comparison"
    return "concept"


def build_resource_queries(question: str, intent: str) -> list:
    base = build_resource_query(question)
    queries = [base]

    intent_suffixes = {
        "paper": [
            f"{base} 论文",
            f"{base} technical report",
            f"{base} paper arxiv",
        ],
        "project": [
            f"{base} GitHub",
            f"{base} github repo",
            f"{base} source code",
            f"{base} README",
            f"{base} open source",
            f"{base} repository",
        ],
        "docs": [
            f"{base} 官方文档",
            f"{base} API documentation",
            f"{base} docs",
        ],
        "tutorial": [
            f"{base} 教程",
            f"{base} guide",
            f"{base} tutorial",
        ],
        "troubleshooting": [
            question.strip(),
            f"{base} error fix",
            f"{base} 报错",
        ],
        "general": [
            f"{base} 资料",
            f"{base} references",
        ],
    }

    for query in intent_suffixes.get(intent, []):
        query = query.strip()
        if query and query not in queries:
            queries.append(query)

    return queries


def is_technical_query(question: str) -> bool:
    text = (question or "").lower()
    technical_keywords = [
        "api", "sdk", "github", "gitlab", "huggingface", "modelscope",
        "arxiv", "paper", "technical report", "whitepaper", "docs",
        "documentation", "readme", "python", "pytorch", "cuda", "docker",
        "linux", "windows", "macos", "npm", "pip", "conda", "transformers",
        "llm", "rag", "agent", "prompt", "embedding", "quantization",
        "inference", "training", "benchmark", "error", "exception", "traceback",
        "报错", "错误", "异常", "技术报告", "论文", "源码", "仓库", "接口", "文档",
        "部署", "推理", "训练", "量化", "模型", "框架", "版本", "安装", "配置",
    ]
    return any(keyword in text for keyword in technical_keywords)


def should_generate_english_query(question: str, task_type: str) -> bool:
    if re.search(r"[a-zA-Z]{3,}", question or ""):
        return True
    if task_type == "resource":
        return is_technical_query(question)
    return is_technical_query(question)


def is_query_obviously_bad(question: str, query: str) -> bool:
    q1 = set(tokenize_for_relevance(question))
    q2 = set(tokenize_for_relevance(query))

    if not q2:
        return True

    if len(question.strip()) <= 4 and question.strip() not in query:
        return True

    english1 = {t for t in q1 if re.search(r"[a-zA-Z0-9]", t)}
    english2 = {t for t in q2 if re.search(r"[a-zA-Z0-9]", t)}

    if english1 and not (english1 & english2):
        return True

    return False


# ============================================================
# SearXNG 搜索
# ============================================================

def get_depth_config(depth: str) -> dict:
    if depth == "precise":
        return {
            "depth": "precise",
            "per_query": PRECISE_SEARCH_RESULTS_PER_QUERY,
            "final_results": PRECISE_FINAL_RESULTS,
            "final_tokens": PRECISE_FINAL_TOKENS,
            # 精准模式：原问题 + 中文改写搜索词 + 更多验证查询 + 更多正文抓取
            "use_chinese_rewrite": True,
            "fetch_top_n": PRECISE_FETCH_TOP_N,
            "query_template_limit": QUERY_TEMPLATE_PRECISE_LIMIT,
        }

    return {
        "depth": "fast",
        "per_query": FAST_SEARCH_RESULTS_PER_QUERY,
        "final_results": FAST_FINAL_RESULTS,
        "final_tokens": FAST_FINAL_TOKENS,
        # 快速模式：只搜原始输入或查资料清洗后的关键词，但仍做轻量验证。
        "use_chinese_rewrite": False,
        "fetch_top_n": FAST_FETCH_TOP_N,
        "query_template_limit": QUERY_TEMPLATE_FAST_LIMIT,
    }


def cache_get(key):
    now = time.time()

    with search_cache_lock:
        item = search_cache.get(key)

        if not item:
            return None

        timestamp, value = item

        if now - timestamp > SEARCH_CACHE_TTL_SECONDS:
            search_cache.pop(key, None)
            return None

        return list(value)


def cache_set(key, value):
    with search_cache_lock:
        search_cache[key] = (time.time(), list(value))


def choose_searxng_categories(resource_intent: str = "general") -> str:
    """按资料意图选择 SearXNG category，避免所有查询都挤到 general。"""
    if resource_intent in {"project", "docs", "tutorial", "troubleshooting"}:
        return "it"
    if resource_intent == "paper":
        return "science"
    return "general"


def search_searxng(
    query: str,
    num_results: int = FAST_SEARCH_RESULTS_PER_QUERY,
    categories: str = "general",
) -> list:
    check_cancelled()

    query = query.strip()
    if not query:
        return []

    cache_key = (query, SEARCH_LANGUAGE, num_results, categories)
    cached = cache_get(cache_key)

    if cached is not None:
        print(f"[搜索] 命中缓存：{query}")
        return cached[:num_results]

    print(f"[搜索] SearXNG 查询：{query} | categories={categories}")

    params = {
        "q": query,
        "format": "json",
        "categories": categories,
        "safesearch": 0,
    }

    if SEARCH_LANGUAGE:
        params["language"] = SEARCH_LANGUAGE

    try:
        response = requests.get(
            SEARXNG_URL,
            params=params,
            timeout=(3, 8),
        )

        response.raise_for_status()
        check_cancelled()

        try:
            data = response.json()
        except Exception:
            print("[搜索] SearXNG 没有返回 JSON。请确认 settings.yml 已启用 json 格式。")
            return []

        results = data.get("results", [])
        print(f"[搜索] 获取结果数：{len(results)}")

        check_cancelled()

        sliced = results[:num_results]
        cache_set(cache_key, sliced)

        return sliced

    except UserCancelledError:
        raise

    except Exception as e:
        print(f"[搜索] 请求失败：{e}")
        return []


def search_multiple_queries(
    question: str,
    search_query: str,
    english_search_query: str = "",
    extra_queries: list | None = None,
    per_query: int = FAST_SEARCH_RESULTS_PER_QUERY,
    categories: str = "general",
    include_direct: bool = True,
    include_web: bool = True,
) -> list:
    queries = []

    # 不再生成英文搜索词。
    # 快速模式：只搜原始输入 / 查资料清洗词。
    # 精准模式：先搜中文改写词，再搜原始输入。
    if search_query.strip():
        queries.append(search_query.strip())

    if question.strip() and question.strip() not in queries:
        queries.append(question.strip())

    english_search_query = (english_search_query or "").strip()
    if english_search_query and english_search_query not in queries:
        queries.append(english_search_query)

    for query in extra_queries or []:
        query = (query or "").strip()
        if query and query not in queries:
            queries.append(query)

    print("[搜索] 实际查询列表：")
    for index, query in enumerate(queries, start=1):
        print(f"       {index}. {query}")

    all_results = []
    seen_urls = set()

    def add_unique_results(items: list) -> None:
        for item in items:
            check_cancelled()
            url = item.get("url", "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            all_results.append(item)

    # URL 直连解析。离线/只本地模式下会跳过，避免任何外部请求。
    if include_direct:
        direct_config = {
            "github_token": GITHUB_TOKEN,
            "timeout": (3, FETCH_TIMEOUT_SECONDS),
            "max_bytes": FETCH_MAX_BYTES,
        }
        for query in queries:
            check_cancelled()
            direct_results = resolve_direct_sources(query, config=direct_config)
            add_unique_results(direct_results)

        if all_results:
            print(f"[直连] 已加入直接解析结果：{len(all_results)}")

    if not include_web:
        print("[搜索] 当前模式跳过 SearXNG 联网搜索。")
        return all_results

    def search_one_query(query_index: int, query: str) -> tuple[int, list]:
        check_cancelled()
        results = search_searxng(query, num_results=per_query, categories=categories)
        prepared = []
        for item in results:
            item = dict(item)
            item["_matched_query"] = query
            item["_query_index"] = query_index
            prepared.append(item)
        return query_index, prepared

    query_results_by_index: dict[int, list] = {}

    if PARALLEL_SEARCH_ENABLED and SEARCH_WORKERS > 1 and len(queries) > 1:
        worker_count = min(SEARCH_WORKERS, len(queries))
        print(f"[并行搜索] 启用 SearXNG 并行查询：workers={worker_count}")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(search_one_query, index, query): (index, query)
                for index, query in enumerate(queries)
            }

            for future in as_completed(futures):
                check_cancelled()
                index, query = futures[future]
                try:
                    result_index, items = future.result()
                    query_results_by_index[result_index] = items
                except UserCancelledError:
                    raise
                except Exception as e:
                    print(f"[并行搜索] 查询失败：{query} | {e}")
    else:
        for index, query in enumerate(queries):
            result_index, items = search_one_query(index, query)
            query_results_by_index[result_index] = items

    # 为了保持排序行为稳定，虽然搜索是并行的，合并仍按原查询顺序合并。
    for index in range(len(queries)):
        add_unique_results(query_results_by_index.get(index, []))

    print(f"[搜索] 合并去重后结果数：{len(all_results)}")
    return all_results


# ============================================================
# 可信搜索增强：查询模板、正文抓取、证据审计与结论门控
# ============================================================

class ReadableTextExtractor(HTMLParser):
    """轻量 HTML 正文抽取器，避免引入 BeautifulSoup 等额外依赖。"""

    BLOCK_TAGS = {
        "p", "div", "section", "article", "main", "li", "br", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6", "pre", "code", "blockquote",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "iframe", "nav", "footer"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        data = (data or "").strip()
        if data:
            self.parts.append(data)
            self.parts.append(" ")

    def get_text(self) -> str:
        text = "".join(self.parts)
        text = unescape(text)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()


def extract_html_title(html_text: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text or "", flags=re.I | re.S)
    if not match:
        return ""
    return clean_search_snippet(unescape(re.sub(r"<[^>]+>", " ", match.group(1))))[:180]


def extract_publication_date(html_text: str, url: str = "") -> str:
    html_text = html_text or ""
    meta_patterns = [
        r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|date|datePublished|dc.date|dc.date.issued|citation_publication_date|og:updated_time)["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\'](?:article:published_time|date|datePublished|dc.date|dc.date.issued|citation_publication_date|og:updated_time)["\']',
        r'"(?:datePublished|dateModified|uploadDate)"\s*:\s*"([^"\']+)"',
    ]
    for pattern in meta_patterns:
        match = re.search(pattern, html_text, flags=re.I | re.S)
        if match:
            return clean_search_snippet(match.group(1))[:40]

    for text in [url, html_text[:5000]]:
        match = re.search(r"(20\d{2}|19\d{2})[-/年](0?[1-9]|1[0-2])(?:[-/月](0?[1-9]|[12]\d|3[01]))?", text)
        if match:
            year = match.group(1)
            month = str(match.group(2)).zfill(2)
            day = str(match.group(3)).zfill(2) if match.group(3) else ""
            return f"{year}-{month}" + (f"-{day}" if day else "")
    return ""


def split_passages(text: str) -> list:
    text = re.sub(r"\s+", " ", text or " ").strip()
    if not text:
        return []
    rough = re.split(r"(?<=[。！？!?\.])\s+|\n+", text)
    passages = []
    for part in rough:
        part = clean_search_snippet(part)
        if 40 <= len(part) <= 900:
            passages.append(part)
        elif len(part) > 900:
            for i in range(0, min(len(part), 2600), 520):
                chunk = clean_search_snippet(part[i:i + 620])
                if len(chunk) >= 40:
                    passages.append(chunk)
    return passages


def extract_relevant_passages(question: str, text: str, max_passages: int = 4) -> list:
    passages = split_passages(text)
    if not passages:
        return []
    q_tokens = tokenize_for_relevance(question)[:16]
    scored = []
    for passage in passages[:220]:
        lower = passage.lower()
        score = 0
        for token in q_tokens:
            if token.lower() in lower:
                score += 4 if len(token) >= 4 else 2
        if re.search(r"\b(version|release|support|unsupported|deprecated|error|issue|install|compatible|limitation|official|docs?)\b", lower):
            score += 2
        if re.search(r"(版本|发布|支持|不支持|弃用|错误|报错|兼容|限制|官方|文档)", passage):
            score += 2
        if score > 0:
            scored.append((score, passage))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = []
    seen = set()
    for _, passage in scored:
        key = passage[:90]
        if key in seen:
            continue
        seen.add(key)
        selected.append(passage)
        if len(selected) >= max_passages:
            break
    return selected




def guess_response_encoding_from_bytes(response, raw: bytes) -> str:
    content_type = response.headers.get("Content-Type", "") or response.headers.get("content-type", "") or ""
    m = re.search(r"charset=([A-Za-z0-9_\-]+)", content_type, flags=re.I)
    if m:
        return m.group(1)
    head = raw[:8192]
    for pattern in (
        rb"<meta[^>]+charset=[\"']?\s*([A-Za-z0-9_\-]+)",
        rb"<meta[^>]+content=[\"'][^\"']*charset=([A-Za-z0-9_\-]+)",
    ):
        mm = re.search(pattern, head, flags=re.I)
        if mm:
            try:
                return mm.group(1).decode("ascii", errors="ignore")
            except Exception:
                pass
    return response.encoding or getattr(response, "apparent_encoding", None) or "utf-8"


def decode_response_bytes(response, raw: bytes) -> str:
    candidates = []
    preferred = guess_response_encoding_from_bytes(response, raw)
    for enc in [preferred, response.encoding, getattr(response, "apparent_encoding", None), "utf-8", "gb18030", "gbk", "gb2312"]:
        if not enc:
            continue
        try:
            decoded = raw.decode(enc, errors="replace")
            candidates.append((enc, decoded))
        except Exception:
            continue
    if not candidates:
        return raw.decode("utf-8", errors="ignore")

    def score(item):
        _, s = item
        zh = len(re.findall(r"[\u4e00-\u9fff]", s))
        replacement = s.count("�")
        mojibake = sum(s.count(ch) for ch in ("Ã", "Â", "â", "¤", "å", "æ", "ç"))
        return (zh, -replacement, -mojibake, len(s))

    return max(candidates, key=score)[1]

def fetch_page_text(url: str) -> dict:
    """抓取网页正文；失败时返回可解释错误，不中断主流程。"""
    check_cancelled()
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"ok": False, "error": "非 HTTP/HTTPS 链接，跳过正文抓取。"}

    lower_url = url.lower()
    if any(lower_url.endswith(ext) for ext in (".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dmg", ".whl")):
        return {"ok": False, "error": "下载类链接，跳过正文抓取。"}

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LocalResearchSearch/1.0; +https://localhost)",
        "Accept": "text/html,application/xhtml+xml,application/xml,text/plain;q=0.9,*/*;q=0.2",
    }

    try:
        with requests.get(url, headers=headers, timeout=(3, FETCH_TIMEOUT_SECONDS), stream=True, allow_redirects=True) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not any(x in content_type for x in ("text/", "html", "xml", "json")):
                return {"ok": False, "error": f"内容类型不是可读文本：{content_type[:80]}"}

            chunks = []
            total = 0
            for chunk in response.iter_content(chunk_size=16384):
                check_cancelled()
                if not chunk:
                    continue
                chunks.append(chunk)
                total += len(chunk)
                if total >= FETCH_MAX_BYTES:
                    break

            raw = b"".join(chunks)
            html_text = decode_response_bytes(response, raw)
            title = extract_html_title(html_text)
            date = extract_publication_date(html_text, url)

            web_blocks = extract_web_blocks_from_html(html_text)

            extractor = ReadableTextExtractor()
            try:
                extractor.feed(html_text)
                text = extractor.get_text()
            except Exception:
                text = clean_search_snippet(re.sub(r"<[^>]+>", " ", html_text))

            text = clean_search_snippet(text)
            if len(text) > FETCH_TEXT_MAX_CHARS:
                text = text[:FETCH_TEXT_MAX_CHARS]

            if len(text) < 80:
                return {"ok": False, "error": "页面正文过短或不可读。", "title": title, "published_at": date}

            return {
                "ok": True,
                "title": title,
                "published_at": date,
                "text": text,
                "blocks": web_blocks,
                "content_type": content_type[:80],
                "bytes_read": len(raw),
                "truncated": total >= FETCH_MAX_BYTES,
            }

    except UserCancelledError:
        raise
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:160]}


def enrich_results_with_page_text(results: list, question: str, top_n: int = FAST_FETCH_TOP_N) -> list:
    if not results:
        return results

    enriched = [None] * len(results)
    fetch_jobs = []
    top_n = max(0, top_n)

    def mark_local_item(index: int, item: dict) -> None:
        item["fetched_ok"] = True
        item["fetch_error"] = ""
        item["fetched_title"] = item.get("title", "")
        item["fetched_text"] = item.get("content", "")
        item["fetched_blocks"] = []
        item["fetched_passages"] = (
            extract_relevant_passages(question, item.get("content", ""), max_passages=4)
            or [item.get("content", "")[:760]]
        )
        enriched[index] = item

    def mark_unfetched_item(index: int, item: dict) -> None:
        item.setdefault("fetched_ok", False)
        item.setdefault("fetched_passages", [])
        enriched[index] = item

    def fetch_one_item(index: int, item: dict) -> tuple[int, dict]:
        check_cancelled()
        fetched = fetch_page_text(item.get("url", ""))
        item["fetched_ok"] = bool(fetched.get("ok"))
        item["fetch_error"] = fetched.get("error", "")
        item["fetched_title"] = fetched.get("title", "")
        item["published_at"] = fetched.get("published_at", "")
        item["fetched_text"] = fetched.get("text", "")
        item["fetched_blocks"] = fetched.get("blocks", []) or []
        item["fetched_passages"] = extract_relevant_passages(question, item.get("fetched_text", ""), max_passages=4)
        if item.get("fetched_passages"):
            item["content"] = clean_search_snippet(" ".join(item["fetched_passages"][:2])) or item.get("content", "")
        return index, item

    for index, original_item in enumerate(results):
        item = dict(original_item)
        if item.get("source_type") == "local_kb" or str(item.get("url", "")).startswith("file://"):
            mark_local_item(index, item)
        elif index < top_n:
            fetch_jobs.append((index, item))
        else:
            mark_unfetched_item(index, item)

    if fetch_jobs:
        if PARALLEL_SEARCH_ENABLED and FETCH_WORKERS > 1 and len(fetch_jobs) > 1:
            worker_count = min(FETCH_WORKERS, len(fetch_jobs))
            print(f"[并行抓取] 启用网页正文并行抓取：workers={worker_count}, pages={len(fetch_jobs)}")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(fetch_one_item, index, item): index
                    for index, item in fetch_jobs
                }

                for future in as_completed(futures):
                    check_cancelled()
                    try:
                        index, item = future.result()
                        enriched[index] = item
                    except UserCancelledError:
                        raise
                    except Exception as e:
                        index = futures[future]
                        failed_item = dict(results[index])
                        failed_item["fetched_ok"] = False
                        failed_item["fetch_error"] = str(e)[:160]
                        failed_item.setdefault("fetched_passages", [])
                        enriched[index] = failed_item
        else:
            for index, item in fetch_jobs:
                _, fetched_item = fetch_one_item(index, item)
                enriched[index] = fetched_item

    return [item for item in enriched if item is not None]


def extract_core_terms(question: str, max_terms: int = 8) -> list:
    tokens = tokenize_for_relevance(question)
    ranked = []
    bad_fragments = ["是否", "怎么", "如何", "什么", "为什么", "有没有", "可以", "请问"]
    for token in tokens:
        if any(fragment in token for fragment in bad_fragments):
            continue
        score = len(token)
        if re.search(r"[A-Z][A-Za-z0-9_\-.]+", token):
            score += 4
        if re.search(r"\d", token):
            score += 3
        if token.lower() in {"error", "exception", "traceback", "cuda", "pytorch", "python", "transformers", "github", "arxiv"}:
            score += 3
        ranked.append((score, token))
    ranked.sort(key=lambda x: x[0], reverse=True)
    terms = []
    for _, token in ranked:
        if token not in terms:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def build_query_templates(
    question: str,
    base_query: str,
    answer_intent: str = "concept",
    resource_intent: str = "general",
    limit: int = QUERY_TEMPLATE_FAST_LIMIT,
) -> list:
    """按场景生成验证查询。核心目的是找官方证据、反证、版本限制和失败案例。"""
    q = (question or "").strip()
    base = (base_query or q).strip()
    terms = extract_core_terms(q or base, max_terms=7)
    term_query = " ".join(terms) if terms else base

    candidates = []
    if base:
        candidates.append(base)

    lower = q.lower()
    asks_docs = any(x in lower for x in ["官方", "文档", "docs", "documentation", "api", "标准", "standard"])
    asks_version = any(x in lower for x in ["版本", "兼容", "支持", "不支持", "release", "changelog", "support", "supported", "compatib"])
    asks_issue = any(x in lower for x in ["报错", "错误", "异常", "失败", "error", "exception", "traceback", "install", "安装", "bug"])
    asks_paper = any(x in lower for x in ["论文", "paper", "arxiv", "openreview", "实验", "benchmark", "评测"])

    # 每个问答任务至少加一条官方/文档验证查询。
    candidates.append(f"{term_query} official documentation")
    if asks_docs or answer_intent in {"concept", "recommendation", "comparison", "troubleshooting"}:
        candidates.append(f"{term_query} docs")

    if asks_version:
        candidates.append(f"{term_query} release notes changelog")
        candidates.append(f"{term_query} compatibility support unsupported")

    if asks_issue or answer_intent == "troubleshooting":
        candidates.append(f"{term_query} error issue fix")
        candidates.append(f"{term_query} site:github.com issue")

    # 反证查询：即使用户没有明确要求，也主动寻找限制/相反说法。
    candidates.append(f"{term_query} limitation unsupported deprecated")
    candidates.append(f"{term_query} not supported issue")

    if asks_paper or resource_intent == "paper":
        candidates.append(f"{term_query} arxiv paper")
        candidates.append(f"{term_query} openreview")

    if resource_intent == "project" or "github" in lower:
        candidates.append(f"{term_query} site:github.com")

    unique = []
    for candidate in candidates:
        candidate = clean_one_line(candidate, max_len=220)
        if not candidate:
            continue
        if is_query_obviously_bad(q or base, candidate) and candidate != base:
            continue
        if candidate not in unique:
            unique.append(candidate)
        if len(unique) >= max(1, limit):
            break
    return unique


def is_time_sensitive_or_high_stakes(question: str) -> bool:
    text = (question or "").lower()
    keywords = [
        "最新", "当前", "现在", "today", "latest", "current", "recent",
        "价格", "费用", "政策", "法律", "法规", "许可证", "license", "pricing",
        "版本", "兼容", "支持", "不支持", "发布时间", "release", "changelog", "deprecated",
        "医疗", "药", "诊断", "法律", "投资", "股票", "汇率", "安全漏洞", "cve",
    ]
    return any(keyword in text for keyword in keywords)


def classify_evidence_role(item: dict) -> str:
    text = " ".join([
        item.get("title", ""),
        item.get("content", ""),
        item.get("fetched_text", "")[:1200],
        item.get("_matched_query", ""),
    ]).lower()
    negative_patterns = [
        "not supported", "unsupported", "deprecated", "limitation", "known issue", "bug", "failed", "failure",
        "不支持", "不兼容", "已弃用", "限制", "问题", "失败", "报错", "错误", "无法",
    ]
    if any(pattern in text for pattern in negative_patterns):
        return "限制/反证"
    if item.get("source_type") in {"local_kb", "official", "doc", "paper", "repo"}:
        return "支持证据"
    return "背景资料"


def score_evidence_item(question: str, item: dict, domain_count: int = 1) -> dict:
    source_type = item.get("source_type", "general")
    authority_map = {"local_kb": 5, "official": 5, "doc": 5, "paper": 5, "repo": 4, "general": 2, "community": 1}
    authority = authority_map.get(source_type, 2)

    relevance_raw = int(item.get("relevance", 0) or 0)
    relevance = 5 if relevance_raw >= 20 else 4 if relevance_raw >= 12 else 3 if relevance_raw >= 6 else 2 if relevance_raw >= 2 else 1

    q_tokens = extract_core_terms(question, max_terms=10)
    text = " ".join([
        item.get("title", ""),
        item.get("content", ""),
        " ".join(item.get("fetched_passages", []) or []),
        item.get("url", ""),
    ]).lower()
    token_hits = sum(1 for token in q_tokens if token.lower() in text)
    specificity = 5 if token_hits >= 5 else 4 if token_hits >= 3 else 3 if token_hits >= 2 else 2 if token_hits >= 1 else 1

    if item.get("published_at"):
        freshness = 4
    elif is_time_sensitive_or_high_stakes(question):
        freshness = 2
    else:
        freshness = 3

    content_depth = 5 if item.get("fetched_passages") else 3 if item.get("content") else 1
    independence = 5 if domain_count <= 1 else 3 if domain_count == 2 else 2

    total = authority * 2 + relevance * 2 + specificity + freshness + content_depth + independence
    if total >= 29:
        label = "强"
    elif total >= 22:
        label = "中"
    else:
        label = "弱"

    reasons = []
    reasons.append(f"来源权威性{authority}/5")
    reasons.append(f"相关性{relevance}/5")
    reasons.append(f"问题匹配度{specificity}/5")
    reasons.append(f"正文可读性{content_depth}/5")
    if item.get("published_at"):
        reasons.append(f"可见日期：{item.get('published_at')}")
    elif is_time_sensitive_or_high_stakes(question):
        reasons.append("未检测到明确日期")

    return {
        "authority": authority,
        "relevance": relevance,
        "specificity": specificity,
        "freshness": freshness,
        "content_depth": content_depth,
        "independence": independence,
        "total": total,
        "label": label,
        "role": classify_evidence_role(item),
        "reasons": reasons,
    }


def build_evidence_audit(question: str, ranked_results: list, search_queries: list | None = None) -> dict:
    search_queries = search_queries or []
    domain_counts = {}
    for item in ranked_results:
        domain = item.get("domain", "") or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    cards = []
    for index, item in enumerate(ranked_results, start=1):
        metrics = score_evidence_item(question, item, domain_counts.get(item.get("domain", ""), 1))
        item["evidence_score"] = metrics["total"]
        item["evidence_label"] = metrics["label"]
        item["evidence_role"] = metrics["role"]
        item["evidence_reasons"] = metrics["reasons"]
        passages = item.get("fetched_passages") or []
        excerpt = passages[0] if passages else clean_search_snippet(item.get("content", ""))[:360]
        cards.append({
            "index": index,
            "title": item.get("title", "无标题"),
            "url": item.get("url", ""),
            "domain": item.get("domain", ""),
            "source_type": item.get("source_type", "general"),
            "role": metrics["role"],
            "label": metrics["label"],
            "score": metrics["total"],
            "excerpt": excerpt,
            "published_at": item.get("published_at", ""),
            "reasons": metrics["reasons"],
            "fetched_ok": item.get("fetched_ok", False),
            "fetch_error": item.get("fetch_error", ""),
        })

    strong_cards = [c for c in cards if c["label"] == "强"]
    medium_cards = [c for c in cards if c["label"] == "中"]
    primary_cards = [c for c in cards if c["source_type"] in {"local_kb", "official", "doc", "paper", "repo"}]
    caution_cards = [c for c in cards if c["role"] == "限制/反证"]
    fetched_count = sum(1 for c in cards if c.get("fetched_ok"))
    unique_domains = len({c["domain"] for c in cards if c.get("domain")})

    warnings = []
    if not ranked_results:
        warnings.append("没有进入最终回答的来源。")
    if not primary_cards:
        warnings.append("缺少官方文档、论文、官方仓库或高权威技术来源。")
    if is_time_sensitive_or_high_stakes(question) and not any(c.get("published_at") for c in cards):
        warnings.append("问题具有时效性/版本性，但未检测到明确发布日期或版本页。")
    if fetched_count == 0 and ranked_results:
        warnings.append("没有成功抓取正文，只能依赖搜索摘要，结论必须保守。")
    if caution_cards:
        warnings.append("检索到可能的限制、反例、报错或不支持信息，需要在结论中保留条件。")
    if unique_domains <= 1 and len(cards) >= 3:
        warnings.append("来源域名过于集中，独立性不足。")

    # 综合强度：不只看数量，也看来源类型、正文、时效和独立性。
    if len(strong_cards) >= 2 and len(primary_cards) >= 2 and unique_domains >= 2:
        strength = "高"
    elif (strong_cards or len(medium_cards) >= 2) and primary_cards:
        strength = "中"
    elif cards:
        strength = "低"
    else:
        strength = "不足"

    must_not_assert = False
    if strength in {"低", "不足"} and is_time_sensitive_or_high_stakes(question):
        must_not_assert = True
    if not primary_cards and len(cards) <= 2:
        must_not_assert = True

    return {
        "strength": strength,
        "cards": cards,
        "warnings": warnings,
        "must_not_assert": must_not_assert,
        "fetched_count": fetched_count,
        "unique_domains": unique_domains,
        "primary_count": len(primary_cards),
        "caution_count": len(caution_cards),
        "search_queries": [q for q in search_queries if q],
    }


def format_evidence_context_for_model(audit: dict) -> str:
    lines = [
        f"综合证据强度：{audit.get('strength', '不足')}",
        f"是否禁止强结论：{'是' if audit.get('must_not_assert') else '否'}",
        f"成功抓取正文来源数：{audit.get('fetched_count', 0)}；独立域名数：{audit.get('unique_domains', 0)}；主要来源数：{audit.get('primary_count', 0)}",
    ]
    if audit.get("warnings"):
        lines.append("审计警告：")
        for warning in audit["warnings"][:6]:
            lines.append(f"- {warning}")
    lines.append("证据卡片：")
    for card in audit.get("cards", [])[:8]:
        lines.append(
            f"[{card['index']}] {card['title']} | {card['domain']} | {card['source_type']} | "
            f"角色：{card['role']} | 强度：{card['label']} | 分数：{card['score']}"
        )
        if card.get("published_at"):
            lines.append(f"日期：{card['published_at']}")
        if card.get("excerpt"):
            lines.append(f"摘录：{card['excerpt'][:520]}")
    return "\n".join(lines)


def format_evidence_audit_for_user(audit: dict) -> str:
    lines = [
        "【证据链】",
    ]
    cards = audit.get("cards", [])
    if not cards:
        lines.append("无可用证据链。")
    else:
        for card in cards[:6]:
            lines.append(
                f"{card['index']}. {card['title']}\n"
                f"角色：{card['role']}；强度：{card['label']}；分数：{card['score']}；来源：{card['domain']}（{card['source_type']}）\n"
                f"链接：{card['url']}"
            )
            if card.get("published_at"):
                lines.append(f"可见日期：{card['published_at']}")
            if card.get("excerpt"):
                lines.append(f"证据摘录：{card['excerpt'][:260]}")
            if card.get("reasons"):
                lines.append("评分依据：" + "；".join(card["reasons"][:4]))
            lines.append("")

    lines.extend([
        "【反证/限制】",
    ])
    warnings = audit.get("warnings", [])
    if warnings:
        for index, warning in enumerate(warnings, start=1):
            lines.append(f"{index}. {warning}")
    else:
        lines.append("未检测到明显反证或限制，但这不等于不存在反例。")

    lines.extend([
        "",
        "【证据强度】",
        f"综合：{audit.get('strength', '不足')}",
        f"正文抓取成功：{audit.get('fetched_count', 0)} 个来源；独立域名：{audit.get('unique_domains', 0)}；主要来源：{audit.get('primary_count', 0)}；限制/反证：{audit.get('caution_count', 0)}。",
    ])
    if audit.get("must_not_assert"):
        lines.append("门控结论：当前证据不足以支持强结论，最终回答必须保持保守或只给验证路径。")
    return "\n".join(lines).strip()


def build_insufficient_evidence_answer(question: str, audit: dict, search_queries: list) -> str:
    return "\n".join([
        "【核心结论】",
        "当前搜索结果不足以支持一个可靠结论；强行回答会有误导风险。",
        "",
        "【简要依据】",
        f"1. 综合证据强度：{audit.get('strength', '不足')}。",
        f"2. 主要来源数量：{audit.get('primary_count', 0)}；正文抓取成功：{audit.get('fetched_count', 0)}。",
        "3. 对版本、兼容性、政策、价格、法律或最新状态类问题，需要官方/原始来源直接支持。",
        "",
        "【详细分析】",
        "1. 当前结果可以作为继续检索的线索，但不足以证明具体事实。",
        "2. 建议优先补充官方文档、release notes、原始论文、仓库 issue/PR 或明确版本号。",
        "3. 如果这是排错问题，请同时提供完整报错、系统环境、软件版本和安装命令。",
        "",
        "【不确定部分】",
        "关键结论缺少足够强的直接证据。",
        "",
        format_evidence_audit_for_user(audit),
        "",
        format_search_list(search_queries),
    ]).strip()


# ============================================================
# 结果过滤与排序
# ============================================================

def looks_like_noise(result: dict) -> bool:
    title = result.get("title", "") or ""
    content = result.get("content", "") or result.get("snippet", "") or ""
    url = result.get("url", "") or ""

    text = f"{title} {content} {url}".lower()

    noise_keywords = [
        "广告",
        "推广",
        "赞助",
        "sponsored",
        "advertisement",
        "doubleclick",
        "adservice",
        "tracking",
        "utm_source",
        "utm_medium",
        "utm_campaign",
    ]

    return any(keyword in text for keyword in noise_keywords)


STOPWORDS = {
    "你好", "您好", "请问", "请", "帮我", "一下", "一个", "这个", "那个",
    "怎么", "如何", "什么", "为什么", "是否", "可以", "有没有", "关于",
    "介绍", "推荐", "说明", "方法", "步骤", "问题", "搜索", "查询",
}


def chinese_ngrams(text: str) -> list:
    grams = []
    chunks = re.findall(r"[\u4e00-\u9fff]+", text)

    for chunk in chunks:
        if len(chunk) <= 1:
            continue

        if 2 <= len(chunk) <= 6:
            grams.append(chunk)

        if len(chunk) > 2:
            for n in (2, 3, 4):
                for i in range(0, len(chunk) - n + 1):
                    grams.append(chunk[i:i+n])

    return grams


def tokenize_for_relevance(text: str) -> list:
    text = (text or "").lower()

    tokens = []
    tokens.extend(re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_\-\.\+#/]{1,}", text))
    tokens.extend(chinese_ngrams(text))

    cleaned = []

    for token in tokens:
        token = token.strip().lower()

        if len(token) < 2:
            continue

        if token in STOPWORDS:
            continue

        cleaned.append(token)

    seen = set()
    unique = []

    for token in cleaned:
        if token not in seen:
            seen.add(token)
            unique.append(token)

    return unique


def calc_relevance_score(question: str, result: dict) -> int:
    title = result.get("title", "") or ""
    content = result.get("content", "") or result.get("snippet", "") or ""
    url = result.get("url", "") or ""

    title_l = title.lower()
    content_l = content.lower()
    url_l = url.lower()

    question_tokens = tokenize_for_relevance(question)

    if not question_tokens:
        q = question.strip().lower()

        if q and q in title_l:
            return 10

        if q and q in content_l:
            return 5

        if q and q in url_l:
            return 2

        return 0

    score = 0

    q_full = question.strip().lower()

    if len(q_full) >= 2:
        if q_full in title_l:
            score += 20
        elif q_full in content_l:
            score += 12

    for token in question_tokens:
        if token in title_l:
            score += 5
        elif token in content_l:
            score += 2
        elif token in url_l:
            score += 1

    return score


def calc_query_match_bonus(result: dict) -> int:
    matched_query = (result.get("_matched_query", "") or "").strip().lower()
    if not matched_query:
        return 0

    title = (result.get("title", "") or "").lower()
    content = (result.get("content", "") or result.get("snippet", "") or "").lower()
    url = (result.get("url", "") or "").lower()

    bonus = 0
    if matched_query in title:
        bonus += 12
    elif matched_query in content:
        bonus += 6
    elif matched_query in url:
        bonus += 3

    for token in tokenize_for_relevance(matched_query)[:8]:
        if token in title:
            bonus += 2
        elif token in content:
            bonus += 1

    return bonus


def calc_resource_intent_bonus(result: dict, resource_intent: str, source_type: str) -> int:
    if resource_intent == "general":
        return 0

    title = (result.get("title", "") or "").lower()
    url = (result.get("url", "") or "").lower()
    domain = (result.get("domain", "") or "").lower()
    text = f"{title} {url} {domain}"

    if resource_intent == "paper":
        patterns = ["arxiv", "paper", "technical report", ".pdf", "abs/", "pdf/"]
    elif resource_intent == "docs":
        patterns = ["docs", "documentation", "api", "readme", "manual"]
    elif resource_intent == "tutorial":
        patterns = ["tutorial", "guide", "example", "examples", "how to", "quickstart"]
    elif resource_intent == "troubleshooting":
        patterns = ["error", "exception", "traceback", "issue", "fix", "faq", "troubleshoot"]
    elif resource_intent == "project":
        patterns = ["github", "gitlab", "repository", "repo", "open source", "readme", "stars", "开源", "仓库"]
    else:
        patterns = []

    bonus = 0
    for pattern in patterns:
        if pattern in text:
            bonus += 10

    if resource_intent == "docs":
        if source_type in {"official", "doc"}:
            bonus += 35
        elif source_type == "repo":
            bonus -= 10
    elif resource_intent == "paper":
        if source_type in {"paper", "official", "repo"}:
            bonus += 20
    elif resource_intent == "troubleshooting":
        if source_type == "repo":
            bonus += 20
    elif resource_intent == "project":
        if source_type == "repo":
            bonus += 90
        elif source_type in {"official", "doc"}:
            bonus += 10
        elif source_type == "paper":
            bonus -= 10
        elif source_type == "general":
            bonus -= 15

    return bonus


def calc_technical_source_bonus(source_type: str, resource_intent: str) -> int:
    if resource_intent == "project":
        bonus_map = {
            "repo": 70,
            "official": 16,
            "doc": 14,
            "paper": 8,
            "general": -5,
            "community": -35,
        }
    else:
        bonus_map = {
            "repo": 48,
            "official": 42,
            "paper": 40,
            "doc": 38,
            "general": 8,
            "community": -35,
        }
    return bonus_map.get(source_type, 0)


def rank_results(
    results: list,
    weights: dict,
    question: str,
    resource_intent: str = "general",
    min_relevance: int = MIN_RELEVANCE_SCORE,
    limit: int = FAST_FINAL_RESULTS,
) -> list:
    scored = []
    technical_query = is_technical_query(question)
    direct_url_domains = get_direct_url_domains(question)

    for result in results:
        if looks_like_noise(result):
            continue

        url = result.get("url", "")
        domain = extract_domain(url)

        # 用户给了明确 URL 时，避免搜索兜底结果跨域跑偏。
        # 例如输入 github.com/owner/repo，不应把 youtube.com 的泛化结果排进最终答案。
        if direct_url_domains and domain and domain not in direct_url_domains:
            continue

        site_weight = get_site_weight(domain, weights)
        if site_weight == DISABLED_SITE_WEIGHT:
            continue

        source_type = result.get("_source_type") or classify_source_type(domain)
        relevance = calc_relevance_score(question, result)
        query_bonus = calc_query_match_bonus(result)
        source_bonus = get_source_type_bonus(source_type)
        intent_bonus = calc_resource_intent_bonus(result, resource_intent, source_type)
        if technical_query:
            source_bonus += calc_technical_source_bonus(source_type, resource_intent)

        if relevance < min_relevance:
            continue

        local_kb_score = float(result.get("_kb_score", 0) or 0)
        final_score = relevance * 6 + query_bonus + source_bonus + intent_bonus + site_weight * 6 + min(local_kb_score * 2, 40)

        normalized = {
            "title": result.get("title", "无标题"),
            "url": url,
            "content": clean_search_snippet(result.get("content", "") or result.get("snippet", "")),
            "domain": domain,
            "site_weight": site_weight,
            "source_type": source_type,
            "relevance": relevance,
            "query_bonus": query_bonus,
            "source_bonus": source_bonus,
            "intent_bonus": intent_bonus,
            "final_score": final_score,
            "_source_type": result.get("_source_type", ""),
            "_direct_source": result.get("_direct_source", ""),
            "_kb_score": result.get("_kb_score", 0),
            "_kb_path": result.get("_kb_path", ""),
            "_kb_ext": result.get("_kb_ext", ""),
            "_kb_chunk_id": result.get("_kb_chunk_id", ""),
            "_kb_chunk_index": result.get("_kb_chunk_index", ""),
            "_kb_doc_id": result.get("_kb_doc_id", ""),
            "_kb_paragraph_start": result.get("_kb_paragraph_start", ""),
            "_kb_paragraph_end": result.get("_kb_paragraph_end", ""),
            "_kb_char_start": result.get("_kb_char_start", ""),
            "_kb_char_end": result.get("_kb_char_end", ""),
            "_kb_location": result.get("_kb_location", ""),
            "_kb_quote": result.get("_kb_quote", ""),
        }

        scored.append(normalized)

    scored.sort(
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return scored[:limit]


def format_results_for_model(results: list) -> str:
    if not results:
        return "没有可用搜索结果。"

    parts = []

    for index, item in enumerate(results, start=1):
        content = clean_search_snippet(item.get("content", ""))
        fetched_passages = item.get("fetched_passages", []) or []
        fetched_text = " ".join(fetched_passages[:3])
        if fetched_text:
            fetched_text = clean_search_snippet(fetched_text)[:760]
        located_evidence = format_item_evidence_for_model(item)
        local_located_evidence = format_local_evidence_for_model(item)

        parts.append(
            f"[{index}]\n"
            f"标题：{item.get('title', '')}\n"
            f"域名：{item.get('domain', '')}\n"
            f"来源类型：{item.get('source_type', 'general')}\n"
            f"网站权重：{item.get('site_weight', DEFAULT_SITE_WEIGHT)}\n"
            f"相关性分：{item.get('relevance', 0)}\n"
            f"总分：{item.get('final_score', 0)}\n"
            f"证据强度：{item.get('evidence_label', '未审计')}；证据角色：{item.get('evidence_role', '未审计')}；证据分：{item.get('evidence_score', 0)}\n"
            f"可见日期：{item.get('published_at', '') or '未检测到'}\n"
            f"链接：{item.get('url', '')}\n"
            f"搜索摘要：{content[:420]}\n"
            f"正文摘录：{fetched_text or '未抓取到可用正文'}\n"
            f"本地定位证据：\n{local_located_evidence or '未生成段落级本地证据'}\n"
            f"网页定位证据：\n{located_evidence or '未生成段落级网页证据'}\n"
        )

    return "\n".join(parts)


def format_scored_sources_for_user(results: list) -> str:
    if not results:
        return "【来源评分】\n无进入最终回答的相关结果。"

    lines = ["【来源评分】"]

    for index, item in enumerate(results, start=1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        domain = item.get("domain", "")
        weight = item.get("site_weight", DEFAULT_SITE_WEIGHT)
        relevance = item.get("relevance", 0)
        final_score = item.get("final_score", 0)

        source_type = item.get("source_type", "general")
        evidence_label = item.get("evidence_label", "未审计")
        evidence_score = item.get("evidence_score", 0)

        lines.append(
            f"{index}. {title}\n"
            f"域名：{domain}；类型：{source_type}；权重：{weight}；相关性：{relevance}；总分：{final_score}；证据：{evidence_label}/{evidence_score}\n"
            f"链接：{url}"
        )

    return "\n\n".join(lines)


# ============================================================
# 输出生成
# ============================================================

def detect_task_type(question: str, mode: str = "auto") -> str:
    mode = (mode or "auto").strip().lower()

    if mode in {"answer", "resource"}:
        return mode

    q = question.strip().lower()
    normalized_resource_keywords = [
        "相关文章", "相关资料", "参考资料", "推荐文章",
        "推荐论文", "找论文", "论文", "paper", "papers",
        "文档", "documentation", "资料", "链接", "网站",
        "博客", "教程", "guide", "references", "sources",
        "找几篇", "推荐几篇", "有哪些资料", "技术报告", "白皮书",
    ]

    if any(keyword in q for keyword in normalized_resource_keywords):
        return "resource"

    resource_keywords = [
        "找文章", "相关文章", "相关资料", "参考资料", "推荐文章",
        "推荐论文", "找论文", "论文", "paper", "papers",
        "文档", "documentation", "资料", "链接", "网站",
        "博客", "教程", "guide", "references", "sources",
        "找几篇", "推荐几篇", "有哪些资料",
    ]

    if any(keyword in q for keyword in resource_keywords):
        return "resource"

    return "answer"


def summarize_resource_results(question: str, search_query: str, ranked_results: list) -> str:
    if not ranked_results:
        return (
            "【核心结论】\n"
            "没有找到足够相关的资料。\n\n"
            "【使用建议】\n"
            "1. 换一个更具体的关键词。\n"
            "2. 尝试加入英文关键词。\n"
            "3. 检查 SearXNG 是否启用了 GitHub、arXiv、Wikipedia、Bing、DuckDuckGo 等来源。\n\n"
            "【来源评分】\n"
            "无可用来源。"
        )

    lines = [
        "【核心结论】",
        f"已按“查资料”模式检索“{question}”。下面只列出资料、摘要和评分，不强行生成综合分析。",
        "",
        "【找到的资料】",
    ]

    for index, item in enumerate(ranked_results, start=1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        domain = item.get("domain", "")
        content = clean_search_snippet(item.get("content", "") or "")
        weight = item.get("site_weight", DEFAULT_SITE_WEIGHT)
        relevance = item.get("relevance", 0)
        final_score = item.get("final_score", 0)

        lines.append(f"{index}. {title}")
        lines.append(f"链接：{url}")
        lines.append(f"来源：{domain}")
        lines.append(f"评分：权重 {weight}；相关性 {relevance}；总分 {final_score}")

        if content:
            lines.append(f"摘要：{content[:260]}")

        lines.append("")

    lines.extend([
        "【使用建议】",
        "1. 优先看权重和相关性都高的来源。",
        "2. 官方文档、论文、GitHub、模型平台优先于普通博客和论坛。",
        "3. 如果你需要结论分析，请切换到“问问题”模式。",
        "",
        format_scored_sources_for_user(ranked_results),
    ])

    return "\n".join(lines)


def clean_model_answer(text: str) -> str:
    text = (text or "").strip()
    cleaned_lines = []

    for line in text.splitlines():
        current = line.rstrip()

        current = re.sub(r"^\s{0,3}#{1,6}\s*", "", current)
        current = current.replace("**", "")

        if re.fullmatch(r"\s*[-=_]{3,}\s*", current):
            continue

        cleaned_lines.append(current)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def summarize_results(
    question: str,
    skill: str,
    search_query: str,
    ranked_results: list,
    answer_intent: str = "concept",
    max_tokens: int = FAST_FINAL_TOKENS,
    evidence_audit: dict | None = None,
) -> str:
    if not ranked_results:
        return (
            "【核心结论】\n"
            "没有找到与问题足够相关的搜索结果，不能强行给出结论。\n\n"
            "【简要依据】\n"
            f"1. 原始问题：{question}\n"
            f"2. 实际搜索词：{search_query}\n"
            "3. 当前 SearXNG 返回结果与问题的相关性不足。\n\n"
            "【详细分析】\n"
            "1. 可能是问题太短、实体不明确，或搜索词没有覆盖关键概念。\n"
            "2. 也可能是搜索结果摘要中没有包含足够可验证信息。\n\n"
            "【不确定部分】\n"
            "资料不足。建议补充系统版本、软件名称、错误提示、模型名或目标场景后重新搜索。\n\n"
            "【主要来源】\n"
            "未找到足够相关的来源。"
        )

    context = format_results_for_model(ranked_results)
    audit_context = format_evidence_context_for_model(evidence_audit or {})
    answer_intent_label = format_answer_intent_label(answer_intent)
    intent_instructions = {
        "concept": (
            "For concept questions, define the object first, then explain its purpose, boundaries, and common misunderstandings."
        ),
        "comparison": (
            "For comparison questions, do not force a winner. Compare by scenario, tradeoff, cost, capability, and failure mode."
        ),
        "troubleshooting": (
            "For troubleshooting questions, identify the most likely root causes, required environment details, and the order of checks."
        ),
        "recommendation": (
            "For recommendation questions, recommend by scenario and constraints such as budget, complexity, maintenance cost, and team skill."
        ),
    }.get(answer_intent, "")

    if ANSWER_STYLE in {"plain", "normal", "simple", "concise"}:
        prompt = f"""请用自然、简洁的中文回答用户问题。

要求：
1. 面向普通用户，先直接回答，不要写成研究报告。
2. 不要使用“核心结论 / 证据链 / 不确定部分 / 主要来源”这类固定大标题。
3. 如果问题是“是什么、为什么、怎么做”，默认给出通俗解释。
4. 除非结果明显不相关，或问题涉及医疗、法律、金融、安全、重大决策，否则不要反复强调“证据不足”。
5. 如果有证据编号如 [L1] 或 [E1]，关键事实后轻量标注 1 到 2 个编号；不要每句话都标。
6. 不要编造具体事实、日期、机构、项目名称。搜索结果不支持的地方，用一句话说明即可。
7. 控制在 3 到 6 段，语言直接一点。

建议格式：
先用一小段直接解释。
再用 2 到 4 个要点补充关键点。
最后可选写“参考：……”，列 1 到 3 个最相关来源或证据编号。

用户问题：
{question}

实际搜索词：
{search_query}

证据审计摘要：
{audit_context}

搜索结果和正文摘录：
{context}

请开始回答："""
    else:
        prompt = f"""Answer the user's question using the search results below.

You are not a simple search-result rewriter. Your task is:
1. First check whether the premise of the user's question is valid.
2. Then inspect the evidence audit. If the audit says strong conclusions are forbidden, do not give a strong factual conclusion.
3. Then provide a conclusion with clear boundaries and conditions.
4. Then analyze the key points using the search results, page excerpts, and reliable technical reasoning.

Pay special attention:
1. If the user's question contains an absolute or comparative premise, such as "why is A more important than B", "is A always better than B", "are they the same", or "is it necessarily true", do not blindly accept the premise.
2. State the necessary context or scope before giving the conclusion.
3. For technical questions, distinguish scenarios such as training, inference, deployment, quantization, long context, memory usage, computation, and model capability.
4. Do not sacrifice accuracy for brevity.
5. Do not treat "a search result says so" as a definitive fact.
6. If the search results are weak, you may use basic technical reasoning, but do not fabricate concrete facts, dates, paper conclusions, or official statements.
7. Non-official sources are only supporting evidence. Do not treat Zhihu, CSDN, personal blogs, or forums as strong evidence.
8. If there are counterexamples or exceptions, mention them.
9. Current answer intent: {answer_intent_label}.
10. {intent_instructions}
11. If the search results contain local evidence IDs like [L1] or web evidence IDs like [E1], cite them after every key factual claim.
12. Do not invent evidence IDs. Only cite evidence IDs that appear in “本地定位证据” or “网页定位证据”. If no evidence ID supports a claim, state uncertainty instead of citing.

Output format must be followed exactly. Section names must each occupy a separate line and must appear in this order:

【核心结论】
Answer directly in 2 to 4 sentences. Give a scoped conclusion, not an absolute one.
If the user's premise is problematic, say so in the first sentence.

【简要依据】
Use 2 to 5 short points to explain the basis for the conclusion. Each point should include its scope or condition when needed.

【证据链】
List 2 to 5 evidence cards. For each card, state whether it is support, limitation/counter-evidence, or background. If paragraph-level evidence IDs are available, include the ID such as [E1].

【详细分析】
Explain the key reasons in points. For technical questions, explain applicable scenarios and exception scenarios.
If the question involves two concepts, explain where each concept matters.

【不确定部分】
State any insufficient evidence, source conflict, ambiguous concept, or unsupported part.
If there is no obvious uncertainty, write: 暂无明显不确定部分。

【主要来源】
List the 3 to 6 most important sources.
Format:
1. Source title - URL（域名：xxx；权重：x；相关性：x；总分：x）

Hard formatting requirements:
1. Do not use Markdown heading markers such as #, ##, or ###.
2. Do not use Markdown bold markers such as **.
3. Do not use long divider lines made of hyphens, equals signs, or asterisks.
4. Do not output debug logs.
5. Do not fabricate concrete facts that are not supported by the search results or page excerpts.
6. If the search results do not match the question, say they do not match; do not force an answer.
7. If evidence audit says evidence strength is low or insufficient, answer conservatively and give verification steps instead of pretending certainty.
8. Summarize first, then analyze, but do not oversimplify the summary.

Additional intent requirements:
- If intent is 排错解释, give troubleshooting steps in the likely order of verification.
- If intent is 对比判断, separate the comparison by scenario instead of giving a blanket conclusion.
- If intent is 选型建议, state the recommendation conditions before the conclusion.
- If intent is 概念解释, start with a concise definition before expanding.


When answering technical questions, do not use a single paper or a single mechanism to prove a broad general claim.

Avoid unsupported strong claims such as:
- “directly determines”
- “proves”
- “destroys generation ability”
- “is always more important”
unless the search results explicitly support them.

User question:
{question}

Actual search query:
{search_query}

Evidence audit:
{audit_context}

Search results and page excerpts:
{context}

Final answer:"""

    answer = gen_model_response(
        user_prompt=prompt,
        system_prompt=skill,
        max_new_tokens=max_tokens,
        do_sample=False,
    )

    answer = clean_model_answer(answer)

    if not answer:
        answer = "【核心结论】\n模型没有生成有效回答。\n\n【不确定部分】\n资料不足。\n\n【主要来源】\n未找到可用来源。"

    return answer




def append_web_evidence_output(answer: str, web_evidence_cards: list) -> str:
    if not web_evidence_cards or not SHOW_WEB_EVIDENCE_CHAIN:
        return answer
    validation = validate_evidence_citations(answer, web_evidence_cards)
    extra = []
    if validation.get("invalid_ids"):
        extra.append("【引用校验】\n模型回答中出现了不存在的网页证据编号：" + ", ".join(validation.get("invalid_ids", [])) + "。请以网页证据定位列表为准。")
    elif not validation.get("has_any_citation"):
        extra.append("【引用校验】\n模型回答未显式引用网页证据编号；下面列出本次检索得到的段落级网页证据，供人工核验。")

    chain = format_web_evidence_chain_for_user(web_evidence_cards)
    if chain:
        extra.append(chain)
    if extra:
        return answer.rstrip() + "\n\n" + "\n\n".join(extra)
    return answer



def _evidence_quote(text: str, max_len: int = 520) -> str:
    text = clean_search_snippet(text or "")
    if len(text) > max_len:
        return text[:max_len].rstrip() + "…"
    return text


def build_local_evidence_cards(results: list, max_total: int = 8) -> tuple[list, list]:
    """从本地知识库结果生成段落级证据卡，并回写到结果项。"""
    if not LOCAL_EVIDENCE_ENABLED:
        return results, []

    cards = []
    updated = []
    seen = set()

    for item in results or []:
        item = dict(item)
        is_local = item.get("source_type") == "local_kb" or item.get("_source_type") == "local_kb" or str(item.get("url", "")).startswith("file://")
        if not is_local:
            updated.append(item)
            continue

        key = item.get("_kb_chunk_id") or (item.get("url"), item.get("content", "")[:100])
        if key in seen:
            item["_local_evidence"] = []
            updated.append(item)
            continue
        seen.add(key)

        if len(cards) < max_total:
            evidence_id = f"L{len(cards) + 1}"
            location = item.get("_kb_location") or item.get("_kb_path") or item.get("title", "本地文档")
            quote = _evidence_quote(item.get("_kb_quote") or item.get("content") or item.get("snippet") or "")
            card = {
                "evidence_id": evidence_id,
                "source_type": "local_kb",
                "title": item.get("title", "本地知识库"),
                "url": item.get("url", ""),
                "location": location,
                "quote": quote,
                "score": float(item.get("_kb_score", 0) or 0),
                "chunk_id": item.get("_kb_chunk_id"),
                "paragraph_start": item.get("_kb_paragraph_start"),
                "paragraph_end": item.get("_kb_paragraph_end"),
            }
            cards.append(card)
            item["_local_evidence"] = [card]
        else:
            item["_local_evidence"] = []

        updated.append(item)

    return updated, cards


def format_local_evidence_for_model(item: dict) -> str:
    cards = item.get("_local_evidence") or []
    if not cards:
        return ""
    lines = []
    for card in cards:
        lines.append(f"- [{card.get('evidence_id')}] {card.get('location')}：{card.get('quote', '')[:420]}")
    return "\n".join(lines)


def format_local_evidence_chain_for_user(cards: list) -> str:
    if not cards:
        return ""
    lines = ["【本地证据定位】", "说明：第 N 段按工具从本地文档抽取的文本段落计数；DOCX 页码不稳定，因此优先显示段落号。"]
    for card in cards:
        lines.append(
            f"- [{card.get('evidence_id')}] {card.get('title', '')}\n"
            f"  位置：{card.get('location', '')}\n"
            f"  原文摘录：{card.get('quote', '')}\n"
            f"  文件链接：{card.get('url', '')}"
        )
    return "\n\n".join(lines)


def validate_local_evidence_citations(answer: str, cards: list) -> dict:
    valid = {card.get("evidence_id") for card in cards or []}
    cited = set(re.findall(r"\[(L\d+)\]", answer or ""))
    return {
        "has_evidence": bool(cards),
        "valid_ids": sorted(valid),
        "cited_ids": sorted(cited),
        "invalid_ids": sorted(cited - valid),
        "has_any_citation": bool(cited),
        "ok": (not cards) or (bool(cited) and not (cited - valid)),
    }


def append_local_evidence_output(answer: str, local_evidence_cards: list) -> str:
    if not local_evidence_cards or not SHOW_LOCAL_EVIDENCE_CHAIN:
        return answer
    validation = validate_local_evidence_citations(answer, local_evidence_cards)
    extra = []
    if validation.get("invalid_ids"):
        extra.append("【引用校验】\n模型回答中出现了不存在的本地证据编号：" + ", ".join(validation.get("invalid_ids", [])) + "。请以本地证据定位列表为准。")
    elif REQUIRE_EVIDENCE_CITATIONS and not validation.get("has_any_citation"):
        extra.append("【引用校验】\n模型回答未显式引用本地证据编号；下面列出本次检索得到的段落级本地证据，供人工核验。")
    chain = format_local_evidence_chain_for_user(local_evidence_cards)
    if chain:
        extra.append(chain)
    if extra:
        return (answer or "").rstrip() + "\n\n" + "\n\n".join(extra)
    return answer


def format_evidence_only_answer(question: str, local_cards: list, web_cards: list, ranked_results: list) -> str:
    lines = ["【证据检索结果】", f"问题：{question}", ""]
    if not local_cards and not web_cards:
        lines.append("没有检索到可定位的证据。")
        if ranked_results:
            lines.append("\n下面是相关来源，但未能形成段落级证据：")
            for idx, item in enumerate(ranked_results[:5], start=1):
                lines.append(f"{idx}. {item.get('title', '')} - {item.get('url', '')}")
        return "\n".join(lines)

    if local_cards:
        lines.append(format_local_evidence_chain_for_user(local_cards))
        lines.append("")
    if web_cards:
        lines.append(format_web_evidence_chain_for_user(web_cards))
    return "\n".join(line for line in lines if line is not None).strip()

def format_actual_searches(question: str, search_query: str) -> str:
    queries = []

    for q in [search_query, question]:
        q = (q or "").strip()
        if q and q not in queries:
            queries.append(q)

    lines = ["【实际搜索】"]
    for index, q in enumerate(queries, start=1):
        lines.append(f"{index}. {q}")

    return "\n".join(lines)


def format_search_list(queries: list) -> str:
    unique_queries = []
    for query in queries:
        query = (query or "").strip()
        if query and query not in unique_queries:
            unique_queries.append(query)

    lines = ["【实际搜索】"]
    for index, query in enumerate(unique_queries, start=1):
        lines.append(f"{index}. {query}")

    return "\n".join(lines)


def assess_evidence_strength(ranked_results: list) -> str:
    if not ranked_results:
        return "不足"

    labels = [item.get("evidence_label", "") for item in ranked_results[:6]]
    strong = labels.count("强")
    medium = labels.count("中")
    primary = sum(
        1
        for item in ranked_results[:6]
        if item.get("source_type") in {"official", "paper", "repo", "doc"}
    )

    if strong >= 2 and primary >= 2:
        return "高"
    if strong >= 1 or (medium >= 2 and primary >= 1):
        return "中"
    return "低"


def format_resource_intent_label(resource_intent: str) -> str:
    labels = {
        "paper": "论文/技术报告",
        "project": "开源项目/仓库",
        "docs": "官方文档/API",
        "tutorial": "教程/指南",
        "troubleshooting": "排错/问题解决",
        "general": "通用资料",
    }
    return labels.get(resource_intent, "通用资料")


def format_answer_intent_label(answer_intent: str) -> str:
    labels = {
        "concept": "概念解释",
        "comparison": "对比判断",
        "troubleshooting": "排错解释",
        "recommendation": "选型建议",
    }
    return labels.get(answer_intent, "概念解释")


def build_resource_suggestions(resource_intent: str) -> list:
    common = [
        "优先看官方文档、论文、GitHub 和模型平台。",
        "如果结果仍然偏杂，继续补充版本号、产品名或错误信息。",
    ]
    specific = {
        "paper": [
            "优先打开 arXiv、官方博客、GitHub PDF 或模型卡。",
            "如果要快速判断可信度，先看摘要、发布时间和作者机构。",
        ],
        "project": [
            "优先打开 GitHub/GitLab 仓库、README、Release 和 Issue。",
            "如果要判断项目质量，先看维护频率、stars、license 和最近提交。",
        ],
        "docs": [
            "优先打开 docs、API reference、README 和 Quickstart。",
            "如果需要落地接入，继续加上 SDK 名称、版本号或语言名搜索。",
        ],
        "tutorial": [
            "优先打开 Quickstart、Guide、Example，再看社区教程。",
            "如果你要动手实践，继续补充框架名或运行环境。",
        ],
        "troubleshooting": [
            "优先打开 issue、FAQ、官方文档和带完整报错的页面。",
            "如果问题还不够聚焦，继续补充报错原文、版本号和操作系统。",
        ],
        "general": [
            "优先看评分高且来源类型更强的结果。",
        ],
    }
    return specific.get(resource_intent, []) + common


def summarize_resource_results_v2(
    question: str,
    search_queries: list,
    ranked_results: list,
    resource_intent: str,
) -> str:
    intent_label = format_resource_intent_label(resource_intent)

    if not ranked_results:
        lines = [
            "【核心结论】",
            f"已按“{intent_label}”模式检索“{question}”，但没有找到足够可信的资料。",
            "",
            "【使用建议】",
            "1. 补充更具体的实体名、版本号、报错原文或产品名。",
            "2. 对技术主题优先尝试英文关键词。",
            "3. 检查 SearXNG 是否启用了 GitHub、arXiv、官方站和文档源。",
            "",
            "【来源评分】",
            "无可用来源。",
        ]
        return "\n".join(lines)

    lines = [
        "【核心结论】",
        f"已按“{intent_label}”模式检索“{question}”。下面只列出资料、摘要和评分，不强行生成综合分析。",
        "",
        "【找到的资料】",
    ]

    for index, item in enumerate(ranked_results, start=1):
        title = item.get("title", "无标题")
        url = item.get("url", "")
        domain = item.get("domain", "")
        source_type = item.get("source_type", "general")
        content = clean_search_snippet(item.get("content", "") or "")
        weight = item.get("site_weight", DEFAULT_SITE_WEIGHT)
        relevance = item.get("relevance", 0)
        final_score = item.get("final_score", 0)

        lines.append(f"{index}. {title}")
        lines.append(f"链接：{url}")
        lines.append(f"来源：{domain}")
        lines.append(f"类型：{source_type}")
        lines.append(f"评分：权重 {weight}；相关性 {relevance}；总分 {final_score}")
        if content:
            lines.append(f"摘要：{content[:260]}")
        lines.append("")

    lines.append("【使用建议】")
    for index, suggestion in enumerate(build_resource_suggestions(resource_intent), start=1):
        lines.append(f"{index}. {suggestion}")

    lines.extend([
        "",
        format_scored_sources_for_user(ranked_results),
        "",
        format_search_list(search_queries),
    ])
    return "\n".join(lines)




# ============================================================
# 搜索框转译
# 中文 -> 英文；英文 -> 中文
# ============================================================

def contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def extract_ascii_entities(text: str) -> list[str]:
    """提取用户输入中的英文实体/项目名，避免转译时被模型改写或补全。"""
    text = text or ""
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_.:@/\-+]{1,}", text)
    entities = []
    stopwords = {
        "http", "https", "www", "com", "org", "net", "cn", "io",
        "what", "how", "why", "when", "where", "which", "tool",
        "github", "gitlab", "huggingface", "arxiv",
    }
    for item in candidates:
        cleaned = item.strip(".,;:!?，。；：！？()[]{}<>《》\"'")
        if not cleaned:
            continue
        if cleaned.lower() in stopwords:
            continue
        if cleaned not in entities:
            entities.append(cleaned)
    return entities


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = (item or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_preserving_english_query(text: str, for_search: bool = False) -> str:
    """
    轻量规则转译：保留用户原始约束，不把复杂查询压缩成“某实体的 GitHub 仓库”。
    主要用于技术/开源项目检索场景，避免小模型丢掉 ChatGPT、Codex、用量、提交时间等关键信息。
    """
    text = (text or "").strip()
    if not contains_chinese(text):
        return ""

    compact = re.sub(r"\s+", "", text)
    entities = extract_ascii_entities(text)

    terms = []

    wants_open_source = "开源" in compact
    wants_project = any(k in compact for k in ["项目", "仓库", "源码", "代码"])
    has_service_degradation_level = "服务降级风险等级" in compact or "降级风险等级" in compact
    has_service_degradation = has_service_degradation_level or "服务降级" in compact or "降级风险" in compact
    has_deep_research = "深度研究" in compact
    has_usage = any(k in compact for k in ["用量", "使用量", "额度"])
    has_fetch = any(k in compact for k in ["获取", "查询", "查看"])
    has_info = any(k in compact for k in ["信息", "数据"])
    has_latest_commit = any(k in compact for k in ["最新一次提交", "最后一次提交", "最近一次提交"])
    has_commit = has_latest_commit or "提交" in compact
    has_three_weeks = any(k in compact for k in ["三周之前", "3周之前", "三周前", "3周前"])
    has_two_weeks = any(k in compact for k in ["两周之前", "2周之前", "两周前", "2周前"])
    has_one_month = any(k in compact for k in ["一个月之前", "1个月之前", "一个月前", "1个月前"])

    # 目标类型
    if wants_open_source:
        terms.append("open source")
    if wants_project:
        terms.append("project")
        terms.append("GitHub repository")

    # 保留英文实体
    terms.extend(entities)

    if has_service_degradation_level:
        terms.append("service degradation risk level")
    elif has_service_degradation:
        terms.append("service degradation risk")
    if has_deep_research:
        terms.append("Deep Research")
    if has_usage:
        terms.append("usage")
    if has_fetch:
        terms.append("fetch")
    if has_info:
        terms.append("information")
    if has_latest_commit:
        terms.append("latest commit")
    elif has_commit:
        terms.append("commit")
    if has_three_weeks:
        terms.append("three weeks ago")
    elif has_two_weeks:
        terms.append("two weeks ago")
    elif has_one_month:
        terms.append("one month ago")

    terms = unique_keep_order(terms)

    # 没有足够可保护的信息时，不强行规则转译，交给后面的模型翻译/原文回退。
    if len(terms) < 3:
        return ""

    if for_search:
        return " ".join(terms)

    # 作为转译按钮输出时，用接近自然语言的句子，但仍然保留所有关键约束。
    if "找" in compact or "搜索" in compact or "查找" in compact or "需要" in compact:
        if wants_open_source and wants_project:
            details = []
            if "ChatGPT" in entities and has_service_degradation:
                details.append("ChatGPT service degradation risk level")
            if has_deep_research:
                details.append("Deep Research")
            if "Codex" in entities and has_usage:
                details.append("Codex usage")
            elif has_usage:
                details.append("usage")

            detail_text = ", ".join(details) if details else " ".join(terms)
            tail = ""
            if has_three_weeks:
                tail = ", with its latest commit three weeks ago"
            elif has_two_weeks:
                tail = ", with its latest commit two weeks ago"
            elif has_one_month:
                tail = ", with its latest commit one month ago"
            elif has_latest_commit:
                tail = ", with its latest commit matching the query"

            return f"Find an open-source project that fetches {detail_text} information{tail}"

        return "Find " + " ".join(terms)
    return " ".join(terms)


def rule_translate_mixed_query(text: str, for_search: bool = False) -> str:
    """
    对“英文实体 + 中文问题”的常见输入做规则转译。
    短问题可以压缩成搜索词；长问题必须保留约束，不能只剩“GitHub repository for X”。
    """
    text = (text or "").strip()
    if not contains_chinese(text):
        return ""

    entities = extract_ascii_entities(text)
    if not entities:
        return build_preserving_english_query(text, for_search=for_search)

    compact = re.sub(r"\s+", "", text)

    # 长查询/多实体查询通常带有多个限定条件，不能用“entity + GitHub repository”粗暴概括。
    preserving = build_preserving_english_query(text, for_search=for_search)
    if preserving and (len(compact) >= 28 or len(entities) >= 2 or "开源" in compact or "最新" in compact or "用量" in compact):
        return preserving

    entity = entities[0]

    if "工具" in compact and ("什么样" in compact or "是什么" in compact or "什么" in compact):
        return f"{entity} tool" if for_search else f"what kind of tool is {entity}"
    if "是什么" in compact or "是啥" in compact or compact.endswith("是什么"):
        return f"{entity} what is" if for_search else f"what is {entity}"
    if "怎么用" in compact or "如何使用" in compact or "怎么使用" in compact:
        return f"{entity} how to use" if for_search else f"how to use {entity}"
    if "安装" in compact or "部署" in compact:
        return f"{entity} install setup" if for_search else f"how to install and set up {entity}"
    if "报错" in compact or "错误" in compact or "异常" in compact or "失败" in compact:
        return f"{entity} error fix" if for_search else f"how to fix errors in {entity}"
    if "论文" in compact or "paper" in compact.lower():
        return f"{entity} paper" if for_search else f"papers about {entity}"
    if "项目" in compact or "仓库" in compact or "源码" in compact or "代码" in compact:
        # 只有非常短的“X 项目/仓库”才允许压缩；复杂项目检索必须走 preserving。
        if len(compact) <= 24:
            return f"{entity} GitHub repository" if for_search else f"GitHub repository for {entity}"
        return preserving or ""

    return preserving or ""


def translation_looks_like_answer(original: str, translated: str) -> bool:
    """检测模型是否把“翻译/搜索词”变成了回答。"""
    original = original or ""
    translated = (translated or "").strip()
    if not translated:
        return True

    if not contains_chinese(original):
        return False

    entities = extract_ascii_entities(original)
    lower = translated.lower()

    # 典型跑偏：ST-research 是什么工具 -> ST-research is a tool for scientific research.
    if entities and re.search(r"\bis\s+(a|an|the)\b", lower) and "?" not in translated:
        question_words = ("what", "how", "why", "when", "where", "which", "whether")
        if not lower.strip().startswith(question_words):
            return True

    # 模型添加了明显的解释性后缀，通常不是纯转译。
    suspicious_phrases = [
        "is a tool for", "is used for", "designed to", "refers to",
        "是一种", "用于", "旨在", "指的是",
    ]
    if any(phrase in lower for phrase in suspicious_phrases):
        return True

    # 复杂检索句不能丢掉关键实体和约束。
    if translation_loses_key_information(original, translated):
        return True

    return False


def translation_loses_key_information(original: str, translated: str) -> bool:
    """粗检模型转译是否把复杂查询压缩到丢失关键信息。"""
    original = original or ""
    translated = translated or ""
    if not contains_chinese(original):
        return False

    lower = translated.lower()

    # 英文实体必须尽量保留。
    entities = extract_ascii_entities(original)
    if entities:
        missing = [e for e in entities if e.lower() not in lower]
        if missing:
            return True

    compact = re.sub(r"\s+", "", original)
    required_groups = [
        (["开源"], ["open source", "opensource"]),
        (["项目", "仓库", "源码", "代码"], ["project", "repository", "repo", "github", "source code"]),
        (["服务降级", "降级风险"], ["degradation", "downgrade", "service status", "risk"]),
        (["深度研究"], ["deep research"]),
        (["用量", "额度", "使用量"], ["usage", "quota", "limit"]),
        (["最新一次提交", "最后一次提交", "最近一次提交"], ["latest commit", "last commit", "recent commit"]),
        (["三周", "3周"], ["three weeks", "3 weeks"]),
    ]

    for source_keys, target_keys in required_groups:
        if any(k in compact for k in source_keys):
            if not any(k in lower for k in target_keys):
                return True

    # 如果原问题很长但译文极短，通常是搜索词被过度压缩。
    if len(compact) >= 35 and len(translated.strip()) < 35:
        return True

    return False


def translate_input_text(text: str, skill: str = "") -> str:
    text = (text or "").strip()

    if not text:
        return ""

    # 对“ST-research 是一个什么样的工具”这类混合输入，优先用规则转译。
    # 这比让小模型自由翻译更稳，可以避免把未知实体脑补成“scientific research”。
    rule_translation = rule_translate_mixed_query(text, for_search=False)
    if rule_translation:
        return rule_translation

    if contains_chinese(text):
        prompt = f"""Translate the following Chinese text into English.

Hard requirements:
1. Translate only. Do not answer the question.
2. Preserve all product names, project names, model names, software names, paper titles, version numbers, file paths, URLs, and error codes exactly.
3. If the input asks what something is, keep it as a question. Do not turn it into a statement.
4. Do not infer what an unknown name means.
5. Do not explain.
6. Do not output JSON.
7. Output only one English line.

Original text:
{text}

English translation:"""
    else:
        prompt = f"""Translate the following English text into natural, accurate Chinese.

Hard requirements:
1. Translate only. Do not answer the question.
2. Preserve all product names, project names, model names, software names, paper titles, version numbers, file paths, URLs, and error codes exactly.
3. If the input asks what something is, keep it as a question. Do not turn it into a statement.
4. Do not infer what an unknown name means.
5. Do not explain.
6. Do not output JSON.
7. Output only one Chinese line.

Original text:
{text}

Chinese translation:"""

    translated = gen_model_response(
        user_prompt=prompt,
        system_prompt=skill or load_skill(),
        max_new_tokens=128,
        do_sample=False,
    )

    translated = clean_one_line(translated, max_len=500)

    if translation_looks_like_answer(text, translated):
        fallback = rule_translate_mixed_query(text, for_search=False) or build_preserving_english_query(text, for_search=False)
        if fallback:
            print(f"[转译] 模型疑似把翻译变成回答或丢失关键信息，已回退规则转译。模型输出：{translated}")
            return fallback
        print(f"[转译] 模型转译疑似跑偏，已回退原文。模型输出：{translated}")
        return text

    return translated or text

# ============================================================
# 本地知识库 / 离线模式
# ============================================================

def normalize_source_mode(source_mode: str | None = None) -> str:
    mode = (source_mode or DEFAULT_SEARCH_MODE or "auto").strip().lower()
    aliases = {
        "offline": "local",
        "local_only": "local",
        "kb": "local",
        "web": "online",
        "internet": "online",
        "online_only": "online",
        "both": "hybrid",
        "all": "hybrid",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"auto", "local", "online", "hybrid"}:
        mode = "auto"
    return mode


def source_mode_flags(source_mode: str, local_results: list | None = None) -> tuple[bool, bool, bool]:
    """Return (use_local, use_direct, use_web)."""
    mode = normalize_source_mode(source_mode)

    if mode == "local":
        return True, False, False
    if mode == "online":
        return False, True, True
    if mode == "hybrid":
        return True, True, True

    # AUTO：先查本地；如果本地结果足够，就不联网。
    enough = local_results_are_enough(local_results or [])
    return True, not enough, not enough


def local_results_are_enough(local_results: list) -> bool:
    if not local_results:
        return False
    best = max(float(item.get("_kb_score", 0) or 0) for item in local_results)
    return len(local_results) >= LOCAL_KB_AUTO_ENOUGH_RESULTS and best >= LOCAL_KB_AUTO_MIN_SCORE



def maybe_auto_sync_local_kb(reason: str = "before_search", force: bool = False) -> dict:
    """Incrementally refresh the local KB without a file watcher or extra dependency."""
    global _last_kb_auto_sync_at

    if not LOCAL_KB_ENABLED or not LOCAL_KB_AUTO_SYNC:
        return {"ok": False, "skipped": True, "reason": "local_kb_auto_sync_disabled"}

    if reason == "before_search" and not LOCAL_KB_SYNC_BEFORE_SEARCH:
        return {"ok": False, "skipped": True, "reason": "sync_before_search_disabled"}

    now = time.time()
    if not force and now - _last_kb_auto_sync_at < LOCAL_KB_AUTO_SYNC_INTERVAL_SECONDS:
        return {"ok": True, "skipped": True, "reason": "sync_interval"}

    if not kb_sync_lock.acquire(blocking=False):
        return {"ok": True, "skipped": True, "reason": "sync_already_running"}

    try:
        stats = sync_kb_index(
            BASE_DIR,
            docs_dir=LOCAL_KB_DOCS_DIR,
            db_path=LOCAL_KB_DB_PATH,
        )
        _last_kb_auto_sync_at = time.time()
        if stats.get("changed"):
            print(
                "[本地知识库] 自动同步："
                f"新增 {stats.get('added_count', 0)}，"
                f"更新 {stats.get('updated_count', 0)}，"
                f"删除 {stats.get('removed_count', 0)}，"
                f"跳过 {stats.get('skipped_count', 0)}，"
                f"耗时 {stats.get('seconds', 0)}s"
            )
        return stats
    except Exception as exc:
        print(f"[本地知识库] 自动同步失败：{exc}")
        return {"ok": False, "error": str(exc)}
    finally:
        kb_sync_lock.release()

def expand_local_kb_queries(question: str) -> list[str]:
    """Small bilingual fallback for the lightweight keyword KB.

    This is intentionally not a full translation system. It only maps common
    English technical/academic terms to Chinese terms so a translated query can
    still hit a Chinese local knowledge base.
    """
    text = (question or "").strip()
    if not text:
        return []

    lower = text.lower()
    replacements = {
        "marx": "马克思",
        "marxism": "马克思主义",
        "labor process": "劳动过程",
        "labour process": "劳动过程",
        "labor": "劳动",
        "labour": "劳动",
        "basic elements": "基本要素",
        "three basic elements": "三个基本要素",
        "means of labor": "劳动资料",
        "means of labour": "劳动资料",
        "object of labor": "劳动对象",
        "object of labour": "劳动对象",
        "tool": "工具",
        "tools": "工具",
        "correct answer": "正确答案",
        "labor law": "劳动法",
        "labour law": "劳动法",
        "garbage classification": "垃圾分类",
    }

    mapped_terms = []
    for key, value in replacements.items():
        if key in lower and value not in mapped_terms:
            mapped_terms.append(value)

    if not mapped_terms:
        return []

    return [" ".join(mapped_terms)]


def collect_local_kb_results(
    question: str,
    search_query: str = "",
    extra_queries: list | None = None,
    limit: int = LOCAL_KB_SEARCH_LIMIT,
) -> list:
    if not LOCAL_KB_ENABLED:
        return []

    maybe_auto_sync_local_kb("before_search")

    queries = []
    for query in [question, search_query, *(extra_queries or [])]:
        query = (query or "").strip()
        if query and query not in queries:
            queries.append(query)

    for query in expand_local_kb_queries(question):
        query = (query or "").strip()
        if query and query not in queries:
            queries.append(query)

    results = []
    seen = set()
    for query in queries[:4]:
        check_cancelled()
        try:
            items = search_local_kb(
                query=query,
                base_dir=BASE_DIR,
                docs_dir=LOCAL_KB_DOCS_DIR,
                db_path=LOCAL_KB_DB_PATH,
                limit=limit,
            )
        except Exception as exc:
            print(f"[本地知识库] 检索失败：{exc}")
            items = []

        for item in items:
            key = item.get("_kb_chunk_id") or item.get("url")
            if key in seen:
                continue
            seen.add(key)
            results.append(item)

    if results:
        print(f"[本地知识库] 命中片段：{len(results)}")
    return results[:limit]


def prepend_search_mode_note(answer: str, source_mode: str, used_local: bool, used_web: bool, local_count: int) -> str:
    if ANSWER_STYLE in {"plain", "normal", "simple", "concise"}:
        # 普通搜索模式下不要把“检索模式”做成醒目的结论卡片。
        return answer or ""

    mode = normalize_source_mode(source_mode)
    if mode == "local":
        note = f"【检索模式】\n仅使用本地知识库；命中 {local_count} 个本地片段，未访问互联网。\n\n"
    elif mode == "online":
        note = "【检索模式】\n仅使用联网搜索/直连解析，未混入本地知识库。\n\n"
    elif mode == "auto" and used_local and not used_web:
        note = f"【检索模式】\n自动模式：本地知识库结果已达到阈值，命中 {local_count} 个本地片段，本次未联网。\n\n"
    elif used_local and used_web:
        note = f"【检索模式】\n本地知识库 + 联网搜索混合检索；本地命中 {local_count} 个片段。\n\n"
    elif used_local:
        note = f"【检索模式】\n使用本地知识库；命中 {local_count} 个片段。\n\n"
    else:
        note = ""
    return note + (answer or "")


# ============================================================
# 主流程
# ============================================================

def process_question(question: str, mode: str = "answer", depth: str = "fast", source_mode: str | None = None) -> str:
    english_search_query = ""
    extra_queries = []
    effective_source_mode = normalize_source_mode(source_mode)
    print("\n" + "=" * 60)
    print(f"[流程] 收到问题：{question}")
    print(f"[流程] 检索来源模式：{effective_source_mode}")

    ensure_files()
    write_text(QUESTION_PATH, question)

    config = get_depth_config(depth)

    set_progress(8, "读取本地配置", "读取 search.txt 和 website.txt")

    skill = load_skill()
    weights = load_website_weights()
    task_type = detect_task_type(question, mode)
    resource_intent = detect_resource_intent(question) if task_type == "resource" else "general"
    answer_intent = detect_answer_intent(question) if task_type == "answer" else "concept"
    searxng_categories = choose_searxng_categories(resource_intent)

    print(f"[流程] 任务模式：{task_type}")
    print(f"[流程] 搜索深度：{config['depth']}")
    if task_type == "resource":
        print(f"[流程] 资料意图：{resource_intent}")
    else:
        print(f"[流程] 问答意图：{answer_intent}")

    check_cancelled()

    # 短词：允许搜索，但不让模型乱扩写、不强行总结
    if is_ambiguous_short_query(question):
        set_progress(30, "短词搜索", "检测到短词，直接使用原词搜索")
        search_query = question.strip()
        local_results = collect_local_kb_results(question, search_query, limit=LOCAL_KB_SEARCH_LIMIT)
        use_local, use_direct, use_web = source_mode_flags(effective_source_mode, local_results)
        if not use_local:
            local_results = []

        set_progress(55, "检索资料", f"本地知识库 + 联网搜索：{search_query}")
        if effective_source_mode != "local" and should_generate_english_query(question, task_type):
            english_search_query = extract_english_search_query(question, skill)

        web_results = search_multiple_queries(
            question=question,
            search_query=search_query,
            english_search_query=english_search_query,
            per_query=config["per_query"],
            categories=searxng_categories,
            include_direct=use_direct,
            include_web=use_web,
        )
        raw_results = [*local_results, *web_results]
        check_cancelled()

        set_progress(76, "排序搜索结果", "根据相关性和 website.txt 权重排序")
        ranked_results = rank_results(
            results=raw_results,
            weights=weights,
            question=question,
            resource_intent=resource_intent,
            min_relevance=0,
            limit=config["final_results"],
        )
        check_cancelled()

        set_progress(92, "整理短词结果", "短词搜索不做自由总结")

        if task_type == "resource":
            final_answer = summarize_resource_results_v2(
                question=question,
                search_queries=[search_query, english_search_query, question],
                ranked_results=ranked_results,
                resource_intent=resource_intent,
            )
        else:
            final_answer = summarize_short_query_results(
                question=question,
                search_query=search_query,
                ranked_results=ranked_results,
            )

        final_answer = prepend_search_mode_note(
            final_answer,
            effective_source_mode,
            used_local=bool(local_results),
            used_web=use_web,
            local_count=len(local_results),
        )

        write_text(RESULT_PATH, final_answer)
        set_progress(100, "完成", "结果已写入 result.txt", running=False)
        return final_answer

    # 已移除“中文改写成英文”步骤。
    # 快速模式：直接搜原始输入；查资料模式先做轻量关键词清洗。
    # 精准模式：额外生成中文搜索词，不生成英文搜索词。
    if task_type == "resource":
        set_progress(22, "准备资料检索", "清洗关键词，并按资料意图扩展搜索词")
        resource_queries = build_resource_queries(question, resource_intent)
        search_query = resource_queries[0]
        extra_queries = resource_queries[1:]
    elif config.get("use_chinese_rewrite"):
        set_progress(25, "生成中文搜索词", "精准模式会额外生成中文改写搜索词")
        search_query = extract_search_query(question, skill)
        check_cancelled()
    else:
        set_progress(22, "准备快速搜索", "直接使用原始输入搜索")
        search_query = question.strip()

    set_progress(43, "扩展验证查询", "生成官方、反证、版本和排错查询模板")
    if task_type == "answer":
        verification_queries = build_query_templates(
            question=question,
            base_query=search_query,
            answer_intent=answer_intent,
            resource_intent=resource_intent,
            limit=config.get("query_template_limit", QUERY_TEMPLATE_FAST_LIMIT),
        )
        for query in verification_queries[1:]:
            if query not in extra_queries:
                extra_queries.append(query)

    set_progress(45, "检索资料", "正在检索本地知识库和允许的外部来源")
    if effective_source_mode != "local" and should_generate_english_query(question, task_type):
        english_search_query = extract_english_search_query(question, skill)

    local_results = collect_local_kb_results(
        question=question,
        search_query=search_query,
        extra_queries=extra_queries,
        limit=LOCAL_KB_SEARCH_LIMIT,
    )
    use_local, use_direct, use_web = source_mode_flags(effective_source_mode, local_results)
    if not use_local:
        local_results = []

    web_results = search_multiple_queries(
        question=question,
        search_query=search_query,
        english_search_query=english_search_query,
        extra_queries=extra_queries,
        per_query=config["per_query"],
        categories=searxng_categories,
        include_direct=use_direct,
        include_web=use_web,
    )
    raw_results = [*local_results, *web_results]
    check_cancelled()

    set_progress(65, "排序搜索结果", "根据相关性和 website.txt 权重排序")
    ranked_results = rank_results(
        results=raw_results,
        weights=weights,
        question=question,
        resource_intent=resource_intent,
        min_relevance=MIN_RELEVANCE_SCORE,
        limit=config["final_results"],
    )
    check_cancelled()

    set_progress(72, "抓取正文", "读取高分网页正文，提取可验证摘录")
    ranked_results = enrich_results_with_page_text(
        ranked_results,
        question=question,
        top_n=config.get("fetch_top_n", FAST_FETCH_TOP_N),
    )
    check_cancelled()

    local_evidence_cards = []
    if LOCAL_EVIDENCE_ENABLED:
        ranked_results, local_evidence_cards = build_local_evidence_cards(
            ranked_results,
            max_total=WEB_EVIDENCE_MAX_TOTAL,
        )

    web_evidence_cards = []
    if WEB_EVIDENCE_ENABLED:
        set_progress(76, "生成网页证据", "为网页正文生成段落级证据编号")
        ranked_results, web_evidence_cards = attach_web_evidence_to_results(
            ranked_results,
            question=question,
            max_total=WEB_EVIDENCE_MAX_TOTAL,
            per_source=WEB_EVIDENCE_PER_SOURCE,
        )
        check_cancelled()

    all_search_queries = [search_query, english_search_query, *extra_queries, question]

    if EVIDENCE_ONLY_MODE:
        set_progress(92, "整理证据", "仅输出证据定位，不调用模型生成答案")
        final_answer = format_evidence_only_answer(question, local_evidence_cards, web_evidence_cards, ranked_results)
        write_text(RESULT_PATH, final_answer)
        set_progress(100, "完成", "已输出证据定位结果", running=False)
        return final_answer

    # 本地题库/结构化文档极速路径：如果命中的本地片段里已有“正确答案:”，
    # 直接抽取并返回，避免最后一步模型长时间生成。
    if DIRECT_ANSWER_ENABLED and task_type == "answer" and ranked_results:
        set_progress(78, "尝试直接抽答案", "检查本地知识库片段中的结构化正确答案")
        direct_answer = try_direct_answer(question, ranked_results)
        if direct_answer and direct_answer.get("ok"):
            final_answer = direct_answer.get("answer", "")
            final_answer = append_local_evidence_output(final_answer, local_evidence_cards)
            final_answer = final_answer + "\n\n" + format_scored_sources_for_user(ranked_results[:3])
            final_answer = prepend_search_mode_note(
                final_answer,
                effective_source_mode,
                used_local=bool(local_results),
                used_web=use_web,
                local_count=len(local_results),
            )

            debug_sources = "【调试信息：排序后的来源】\n"
            for index, item in enumerate(ranked_results, start=1):
                debug_sources += (
                    f"{index}. {item['title']}\n"
                    f"   域名：{item['domain']} | 权重：{item['site_weight']} | "
                    f"相关性：{item['relevance']} | 总分：{item['final_score']}\n"
                    f"   {item['url']}\n"
                )
            write_text(DEBUG_PATH, debug_sources)
            write_text(RESULT_PATH, final_answer)
            set_progress(100, "完成", "已直接抽取本地知识库答案", running=False)
            return final_answer

    evidence_audit = build_evidence_audit(question, ranked_results, all_search_queries)

    if task_type == "resource":
        set_progress(82, "整理资料列表", "按资料模式输出链接、摘要和评分")
        final_answer = summarize_resource_results_v2(
            question=question,
            search_queries=all_search_queries,
            ranked_results=ranked_results,
            resource_intent=resource_intent,
        )
        if SHOW_EVIDENCE_AUDIT:
            final_answer = final_answer + "\n\n" + format_evidence_audit_for_user(evidence_audit)
        final_answer = append_local_evidence_output(final_answer, local_evidence_cards)
        final_answer = append_web_evidence_output(final_answer, web_evidence_cards)
    else:
        if evidence_audit.get("must_not_assert"):
            set_progress(82, "证据门控", "证据不足，改为输出保守结论和验证路径")
            final_answer = build_insufficient_evidence_answer(question, evidence_audit, all_search_queries)
        else:
            set_progress(82, "生成最终答案", "本地模型正在基于正文摘录和证据审计整理回答")
            final_answer = summarize_results(
                question=question,
                skill=skill,
                search_query=search_query,
                ranked_results=ranked_results,
                answer_intent=answer_intent,
                max_tokens=config["final_tokens"],
                evidence_audit=evidence_audit,
            )
            if SHOW_EVIDENCE_AUDIT:
                final_answer = final_answer + "\n\n" + format_evidence_audit_for_user(evidence_audit)

        if ANSWER_STYLE not in {"plain", "normal", "simple", "concise"}:
            final_answer = final_answer + "\n\n" + format_scored_sources_for_user(ranked_results)
        final_answer = append_local_evidence_output(final_answer, local_evidence_cards)
        final_answer = append_web_evidence_output(final_answer, web_evidence_cards)

    final_answer = prepend_search_mode_note(
        final_answer,
        effective_source_mode,
        used_local=bool(local_results),
        used_web=use_web,
        local_count=len(local_results),
    )

    debug_sources = "【调试信息：排序后的来源】\n"
    if ranked_results:
        for index, item in enumerate(ranked_results, start=1):
            debug_sources += (
                f"{index}. {item['title']}\n"
                f"   域名：{item['domain']} | 权重：{item['site_weight']} | "
                f"相关性：{item['relevance']} | 总分：{item['final_score']} | "
                f"证据：{item.get('evidence_label', '未审计')}/{item.get('evidence_score', 0)}\n"
                f"   {item['url']}\n"
            )
    else:
        debug_sources += "无进入最终回答的相关结果。\n"

    if task_type != "resource":
        actual_searches = format_search_list(all_search_queries)
        if "【实际搜索】" not in final_answer:
            final_answer = final_answer + "\n\n" + actual_searches

    write_text(DEBUG_PATH, debug_sources)
    write_text(RESULT_PATH, final_answer)

    set_progress(100, "完成", "结果已写入 result.txt", running=False)

    print("[流程] 已写入 result.txt")
    print("=" * 60 + "\n")

    return final_answer


# ============================================================
# HTTP 服务
# ============================================================

class SearchHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/translate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                data = json.loads(post_data.decode("utf-8"))
                text = data.get("text", "").strip()
            except Exception:
                self._send_json({
                    "error": "请求体不是合法 JSON。"
                }, status=400)
                return

            if not text:
                self._send_json({
                    "error": "文本不能为空。"
                }, status=400)
                return

            try:
                translated = translate_input_text(text, load_skill())
                self._send_json({
                    "translated": translated
                })
            except UserCancelledError as e:
                self._send_json({
                    "cancelled": True,
                    "error": str(e)
                }, status=499)
            except Exception as e:
                self._send_json({
                    "error": f"转译失败：{str(e)}"
                }, status=500)
            return

        if self.path == "/kb/rebuild":
            try:
                ensure_files()
                set_progress(10, "重建知识库", "正在扫描 kb/documents")
                stats = rebuild_kb_index(
                    BASE_DIR,
                    docs_dir=LOCAL_KB_DOCS_DIR,
                    db_path=LOCAL_KB_DB_PATH,
                )
                set_progress(100, "知识库已重建", f"文档 {stats.get('doc_count', 0)}，片段 {stats.get('chunk_count', 0)}", running=False)
                self._send_json(stats)
            except Exception as e:
                self._send_json({"ok": False, "error": f"知识库重建失败：{e}"}, status=500)
            return

        if self.path == "/kb/sync":
            try:
                ensure_files()
                set_progress(10, "同步知识库", "正在增量扫描 kb/documents")
                stats = maybe_auto_sync_local_kb("manual", force=True)
                if stats.get("ok"):
                    set_progress(100, "知识库已同步", f"新增 {stats.get('added_count', 0)}，更新 {stats.get('updated_count', 0)}，删除 {stats.get('removed_count', 0)}", running=False)
                self._send_json(stats)
            except Exception as e:
                self._send_json({"ok": False, "error": f"知识库同步失败：{e}"}, status=500)
            return

        if self.path == "/cancel":
            request_cancel()
            self._send_json({
                "ok": True,
                "message": "已请求终止当前搜索任务。",
            })
            return

        if self.path != "/ask":
            self.send_error(404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode("utf-8"))
            question = data.get("question", "").strip()
            mode = data.get("mode", "answer")
            depth = data.get("depth", "fast")
            source_mode = data.get("source_mode", data.get("sourceMode", DEFAULT_SEARCH_MODE))
        except Exception:
            self._send_json({
                "error": "请求体不是合法 JSON。",
            }, status=400)
            return

        if not question:
            self._send_json({
                "error": "问题不能为空。",
            }, status=400)
            return

        try:
            reset_cancel()
            set_progress(3, "开始处理", "收到用户问题")
            result = process_question(question, mode=mode, depth=depth, source_mode=source_mode)

            self._send_json({
                "result": result,
            })

        except UserCancelledError as e:
            error_message = str(e)
            set_progress(100, "已终止", error_message, running=False)
            write_text(RESULT_PATH, error_message)

            self._send_json({
                "cancelled": True,
                "error": error_message,
            }, status=499)

        except Exception as e:
            error_message = f"处理出错：{str(e)}"
            set_progress(100, "失败", error_message, running=False)
            write_text(RESULT_PATH, error_message)

            self._send_json({
                "error": error_message,
            }, status=500)

    def do_GET(self):
        if self.path == "/result":
            result = read_text(RESULT_PATH)
            self._send_json({
                "result": result,
            })
            return

        if self.path == "/progress":
            self._send_json(get_progress())
            return

        if self.path == "/kb/status":
            try:
                self._send_json(get_kb_status(
                    BASE_DIR,
                    docs_dir=LOCAL_KB_DOCS_DIR,
                    db_path=LOCAL_KB_DB_PATH,
                ))
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, status=500)
            return

        if self.path == "/ping":
            self._send_json({
                "status": "ok",
                "searxng_url": SEARXNG_URL,
                "search_language": SEARCH_LANGUAGE,
                "base_dir": str(BASE_DIR),
                "model_path": str(MODEL_PATH),
                "default_site_weight": DEFAULT_SITE_WEIGHT,
                "cache_ttl_seconds": SEARCH_CACHE_TTL_SECONDS,
                "fast_fetch_top_n": FAST_FETCH_TOP_N,
                "precise_fetch_top_n": PRECISE_FETCH_TOP_N,
                "query_template_fast_limit": QUERY_TEMPLATE_FAST_LIMIT,
                "query_template_precise_limit": QUERY_TEMPLATE_PRECISE_LIMIT,
                "local_kb_enabled": LOCAL_KB_ENABLED,
                "local_kb_docs_dir": str(LOCAL_KB_DOCS_DIR),
                "local_kb_db_path": str(LOCAL_KB_DB_PATH),
                "default_search_mode": DEFAULT_SEARCH_MODE,
                "local_kb_auto_sync": LOCAL_KB_AUTO_SYNC,
                "local_kb_sync_before_search": LOCAL_KB_SYNC_BEFORE_SEARCH,
                "local_kb_sync_on_start": LOCAL_KB_SYNC_ON_START,
                "local_kb_auto_sync_interval_seconds": LOCAL_KB_AUTO_SYNC_INTERVAL_SECONDS,
                "app_profile": APP_PROFILE,
                "model_backend": MODEL_BACKEND,
                "llm_api_base": normalize_api_base(LLM_API_BASE),
                "llm_model_name": LLM_MODEL_NAME,
                "direct_answer_enabled": DIRECT_ANSWER_ENABLED,
                "parallel_search_enabled": PARALLEL_SEARCH_ENABLED,
                "search_workers": SEARCH_WORKERS,
                "fetch_workers": FETCH_WORKERS,
                "web_evidence_enabled": WEB_EVIDENCE_ENABLED,
                "web_evidence_max_total": WEB_EVIDENCE_MAX_TOTAL,
                "answer_style": ANSWER_STYLE,
                "show_evidence_audit": SHOW_EVIDENCE_AUDIT,
                "show_web_evidence_chain": SHOW_WEB_EVIDENCE_CHAIN,
                "web_evidence_per_source": WEB_EVIDENCE_PER_SOURCE,
                "local_evidence_enabled": LOCAL_EVIDENCE_ENABLED,
                "show_local_evidence_chain": SHOW_LOCAL_EVIDENCE_CHAIN,
                "evidence_only_mode": EVIDENCE_ONLY_MODE,
                "local_kb_status": get_kb_status(
                    BASE_DIR,
                    docs_dir=LOCAL_KB_DOCS_DIR,
                    db_path=LOCAL_KB_DB_PATH,
                ) if LOCAL_KB_ENABLED else {"ok": False, "error": "LOCAL_KB_ENABLED=0"},
            })
            return

        super().do_GET()

    def guess_type(self, path):
        if path.endswith(".html"):
            return "text/html; charset=utf-8"
        return super().guess_type(path)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


def open_browser():
    time.sleep(1.5)
    print(f"[服务器] 打开浏览器：{INDEX_URL}")
    webbrowser.open(INDEX_URL)


def main():
    ensure_files()

    print("=" * 60)
    print("  AI 本地搜索工具")
    print("=" * 60)
    print(f"[目录] BASE_DIR: {BASE_DIR}")
    print(f"[搜索] SearXNG: {SEARXNG_URL}")
    print(f"[搜索] Language: {SEARCH_LANGUAGE}")
    print(f"[前端] {INDEX_URL}")
    print(f"[知识库] enabled={LOCAL_KB_ENABLED} docs={LOCAL_KB_DOCS_DIR}")
    print(f"[知识库] auto_sync={LOCAL_KB_AUTO_SYNC} before_search={LOCAL_KB_SYNC_BEFORE_SEARCH} on_start={LOCAL_KB_SYNC_ON_START}")
    print(f"[检索模式] 默认 SEARCH_MODE={DEFAULT_SEARCH_MODE}")
    print(f"[应用档位] APP_PROFILE={APP_PROFILE}")
    print(f"[模型后端] MODEL_BACKEND={MODEL_BACKEND}")
    if is_openai_compatible_backend(MODEL_BACKEND):
        print(f"[模型后端] LLM_API_BASE={normalize_api_base(LLM_API_BASE)}")
    print(f"[权重] 未标注网站默认权重：{DEFAULT_SITE_WEIGHT}")
    print(f"[增强] 快速/精准模式均启用验证查询、正文抓取、证据审计和证据门控")
    print("=" * 60)

    load_model()

    os.chdir(BASE_DIR)

    server = ThreadingHTTPServer(
        (SERVER_HOST, SERVER_PORT),
        SearchHandler,
    )

    print(f"[服务器] 已启动：http://localhost:{SERVER_PORT}")

    threading.Thread(
        target=open_browser,
        daemon=True,
    ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[服务器] 正在关闭...")
        server.shutdown()


if __name__ == "__main__":
    main()
