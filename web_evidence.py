import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import quote


# 只把真正的文本节点作为正文块；不要把 article/main/section 这类容器也算进去，
# 否则会出现“浏览器看是第 5 段，工具显示第 8 段”的偏差。
WEB_PARAGRAPH_TAGS = {"p", "li", "blockquote", "pre", "code", "td", "th"}
WEB_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
WEB_BLOCK_TAGS = WEB_PARAGRAPH_TAGS | WEB_HEADING_TAGS

SKIP_TAGS = {
    "script", "style", "noscript", "svg", "canvas", "iframe",
    "nav", "footer", "header", "form", "button"
}


def clean_text(text: str, max_len: int | None = None) -> str:
    text = unescape(text or "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = text.strip()
    if max_len and len(text) > max_len:
        return text[:max_len].rstrip() + "…"
    return text


class HTMLBlockExtractor(HTMLParser):
    """把网页正文抽成可编号的正文块。只用标准库，避免引入额外依赖。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.current_tag = ""
        self.current_parts: list[str] = []
        self.raw_blocks: list[dict] = []

    def _flush(self):
        if not self.current_parts:
            return
        text = clean_text(" ".join(self.current_parts))
        self.current_parts = []
        tag = self.current_tag or "text"
        self.current_tag = ""

        # 过滤导航、按钮、短碎片和重复空白。
        if len(text) < 20:
            return
        if len(text) > 1800:
            # 长块拆成较小证据块，便于定位和引用。
            parts = split_long_text(text, max_len=700)
            for part in parts:
                if len(part) >= 20:
                    self.raw_blocks.append({"tag": tag, "text": part})
        else:
            self.raw_blocks.append({"tag": tag, "text": text})

    def handle_starttag(self, tag, attrs):
        tag = (tag or "").lower()
        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return
        if tag in WEB_BLOCK_TAGS:
            self._flush()
            self.current_tag = tag

    def handle_endtag(self, tag):
        tag = (tag or "").lower()
        if tag in SKIP_TAGS and self.skip_depth > 0:
            self.skip_depth -= 1
            return
        if self.skip_depth > 0:
            return
        if tag in WEB_BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        data = clean_text(data or "")
        if data:
            self.current_parts.append(data)

    def close(self):
        super().close()
        self._flush()


def split_long_text(text: str, max_len: int = 700) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_len:
        return [text] if text else []

    sentences = re.split(r"(?<=[。！？!?\.])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_len:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            if len(sentence) > max_len:
                for i in range(0, len(sentence), max_len):
                    part = sentence[i:i + max_len].strip()
                    if part:
                        chunks.append(part)
                current = ""
            else:
                current = sentence
    if current:
        chunks.append(current)
    return chunks


def normalize_blocks(raw_blocks: list[dict]) -> list[dict]:
    blocks = []
    seen = set()
    char_pos = 0
    paragraph_pos = 0

    for raw in raw_blocks:
        text = clean_text(raw.get("text", ""))
        if len(text) < 20:
            continue

        tag = (raw.get("tag", "text") or "text").lower()

        # 去重，避免同一段同时被容器和 p 标签重复收录。
        key = re.sub(r"\s+", " ", text[:220]).lower()
        if key in seen:
            continue
        seen.add(key)

        block_index = len(blocks) + 1
        start = char_pos
        end = start + len(text)

        paragraph_index = None
        if tag in WEB_PARAGRAPH_TAGS:
            paragraph_pos += 1
            paragraph_index = paragraph_pos

        if paragraph_index is not None:
            location = f"网页正文第 {paragraph_index} 段"
        elif tag in WEB_HEADING_TAGS:
            location = f"网页正文标题块 {block_index}"
        else:
            location = f"网页正文块 {block_index}"

        blocks.append({
            "block_id": f"P{block_index}",
            "block_index": block_index,
            "paragraph_index": paragraph_index,
            "location": location,
            "tag": tag,
            "text": text,
            "char_start": start,
            "char_end": end,
        })
        char_pos = end + 1

    return blocks


def extract_web_blocks_from_html(html_text: str) -> list[dict]:
    if not html_text:
        return []
    parser = HTMLBlockExtractor()
    try:
        parser.feed(html_text)
        parser.close()
    except Exception:
        return []
    return normalize_blocks(parser.raw_blocks)


def split_text_to_blocks(text: str) -> list[dict]:
    text = clean_text(text)
    if not text:
        return []
    rough = re.split(r"\n+", text)
    if len(rough) <= 1:
        rough = re.split(r"(?<=[。！？!?\.])\s+", text)
    raw = []
    for part in rough:
        part = clean_text(part)
        if len(part) > 900:
            for sub in split_long_text(part, max_len=650):
                raw.append({"tag": "text", "text": sub})
        elif len(part) >= 20:
            raw.append({"tag": "text", "text": part})
    return normalize_blocks(raw)


def tokenize_query(text: str, max_terms: int = 16) -> list[str]:
    text = text or ""
    terms = []

    # 英文/数字/符号实体，例如 ChatGPT、Codex、Deep Research、Qwen3.5。
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_+\-.]{1,}|\d+(?:\.\d+)?", text):
        token = token.strip()
        if len(token) >= 2 and token.lower() not in {"the", "and", "for", "with", "what", "how", "why", "this", "that"}:
            terms.append(token)

    # 连续中文词组；轻量做法，不依赖分词库。
    for token in re.findall(r"[\u4e00-\u9fff]{2,12}", text):
        if token not in {"什么", "怎么", "如何", "为什么", "一个", "这个", "那个", "用户", "工具", "项目", "文章"}:
            terms.append(token)

    # 常见中英映射，提升跨语言网页证据召回。
    mapping = {
        "chatgpt": ["ChatGPT"],
        "codex": ["Codex"],
        "deep research": ["Deep Research", "深度研究"],
        "服务降级": ["degradation", "degraded", "服务降级"],
        "风险等级": ["risk", "risk level", "风险等级"],
        "用量": ["usage", "quota", "limit", "用量"],
        "开源": ["open source", "GitHub", "开源"],
    }
    lower = text.lower()
    for key, values in mapping.items():
        if key in lower or key in text:
            terms.extend(values)

    unique = []
    seen = set()
    for term in terms:
        norm = term.lower()
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(term)
        if len(unique) >= max_terms:
            break
    return unique


def score_block(question: str, block: dict, title: str = "", url: str = "") -> float:
    terms = tokenize_query(question, max_terms=20)
    if not terms:
        return 0.0

    text = block.get("text", "") or ""
    hay = " ".join([title, url, text])
    lower = hay.lower()
    block_lower = text.lower()
    score = 0.0

    for term in terms:
        t = term.lower()
        if not t:
            continue
        if t in block_lower:
            score += 5.0 if len(term) >= 4 else 3.0
            # 专名/英文实体更重要。
            if re.search(r"[A-Za-z]", term):
                score += 2.5
        elif t in lower:
            score += 1.0

    # 标题/正文标签略加权。
    tag = block.get("tag", "")
    if tag in {"h1", "h2", "h3"}:
        score += 1.5

    # 过短/过长都不利于作为证据。
    length = len(text)
    if 40 <= length <= 600:
        score += 2.0
    elif length > 1200:
        score -= 2.0

    # 含有明显证据词。
    if re.search(r"(according to|supports|provides|usage|quota|risk|degradation|service|功能|支持|提供|获取|用量|风险|等级|服务降级|开源)", text, flags=re.I):
        score += 2.0

    return score


def _best_text_fragment_snippet(text: str) -> str:
    """为浏览器 Text Fragment 生成更短、更容易命中的定位文本。"""
    text = clean_text(text)
    if not text:
        return ""

    # 优先取第一句；太短再取前 90 字符。避免整段过长导致浏览器定位失败。
    parts = re.split(r"(?<=[。！？!?\.])\s+", text)
    for part in parts:
        part = clean_text(part)
        if 24 <= len(part) <= 120:
            return part

    snippet = text[:100].strip()
    # 尽量不要在单词中间截断。
    if re.search(r"[A-Za-z]", snippet) and " " in snippet:
        snippet = snippet.rsplit(" ", 1)[0]
    return snippet


def make_text_fragment_url(url: str, quote_text: str) -> str:
    url = url or ""
    quote_text = clean_text(quote_text)
    if not url.startswith(("http://", "https://")) or len(quote_text) < 12:
        return url

    # Text Fragment 对动态页面不保证有效，但对静态文章一般可以直接跳转并高亮。
    # 去掉原 fragment，避免和 #:~:text= 冲突。
    base_url = url.split("#", 1)[0]
    snippet = _best_text_fragment_snippet(quote_text)
    if len(snippet) < 12:
        return base_url
    return base_url + "#:~:text=" + quote(snippet, safe="")


def build_web_evidence_for_item(item: dict, question: str, per_source: int = 2) -> list[dict]:
    url = item.get("url", "") or ""
    if not url.startswith(("http://", "https://")):
        return []
    if item.get("source_type") == "local_kb":
        return []

    blocks = item.get("fetched_blocks") or []
    if not blocks:
        blocks = split_text_to_blocks(item.get("fetched_text", "") or " ".join(item.get("fetched_passages", []) or []))
    if not blocks:
        return []

    title = item.get("fetched_title") or item.get("title", "")
    scored = []
    for block in blocks[:260]:
        score = score_block(question, block, title=title, url=url)
        if score > 0:
            scored.append((score, block))

    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for score, block in scored:
        quote_text = clean_text(block.get("text", ""), max_len=520)
        key = quote_text[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "source_type": "web",
            "title": title or item.get("title", ""),
            "url": url,
            "domain": item.get("domain", ""),
            "location": block.get("location") or f"网页正文块 {block.get('block_index')}",
            "block_start": block.get("block_index"),
            "block_end": block.get("block_index"),
            "paragraph_index": block.get("paragraph_index"),
            "block_id": block.get("block_id"),
            "tag": block.get("tag", ""),
            "quote": quote_text,
            "score": round(float(score), 2),
            "text_fragment_url": make_text_fragment_url(url, quote_text),
        })
        if len(selected) >= max(1, per_source):
            break
    return selected


def attach_web_evidence_to_results(
    results: list,
    question: str,
    max_total: int = 8,
    per_source: int = 2,
) -> tuple[list, list]:
    cards = []
    updated = []
    for item in results:
        item = dict(item)
        item_cards = build_web_evidence_for_item(item, question, per_source=per_source)
        for card in item_cards:
            if len(cards) >= max_total:
                break
            card = dict(card)
            card["evidence_id"] = f"E{len(cards) + 1}"
            cards.append(card)
        item["_web_evidence"] = [card for card in cards if card.get("url") == item.get("url")]
        updated.append(item)
        if len(cards) >= max_total:
            # 后续结果仍保留，但不再添加 evidence。
            continue
    return updated, cards


def format_evidence_cards_for_model(cards: list) -> str:
    if not cards:
        return ""
    parts = []
    for card in cards:
        parts.append(
            f"[证据 {card.get('evidence_id')} ]\n"
            f"来源：{card.get('title', '')}\n"
            f"URL：{card.get('url', '')}\n"
            f"位置：{card.get('location', '')}\n"
            f"原文：{card.get('quote', '')}\n"
        )
    return "\n".join(parts)


def format_item_evidence_for_model(item: dict) -> str:
    cards = item.get("_web_evidence") or []
    if not cards:
        return ""
    lines = []
    for card in cards:
        lines.append(
            f"- [{card.get('evidence_id')}] {card.get('location')}：{card.get('quote', '')[:420]}"
        )
    return "\n".join(lines)


def format_web_evidence_chain_for_user(cards: list) -> str:
    if not cards:
        return ""
    lines = [
        "【网页证据定位】",
        "说明：“第 N 段”按工具抽取出的正文文本段落计数；标题、导航、页脚、表单等不计入。定位链接使用浏览器 Text Fragment，支持时会直接跳转并高亮原文。"
    ]
    for card in cards:
        url = card.get("text_fragment_url") or card.get("url", "")
        original_url = card.get("url", "")
        lines.append(
            f"- [{card.get('evidence_id')}] {card.get('title', '')}\n"
            f"  位置：{card.get('location', '')}\n"
            f"  原文摘录：{card.get('quote', '')}\n"
            f"  定位链接：{url}\n"
            f"  原文链接：{original_url}"
        )
    return "\n".join(lines)


def validate_evidence_citations(answer: str, cards: list) -> dict:
    valid = {card.get("evidence_id") for card in cards}
    cited = set(re.findall(r"\[(E\d+)\]", answer or ""))
    invalid = sorted(cited - valid)
    return {
        "has_evidence": bool(cards),
        "valid_ids": sorted(valid),
        "cited_ids": sorted(cited),
        "invalid_ids": invalid,
        "has_any_citation": bool(cited),
        "ok": (not cards) or (bool(cited) and not invalid),
    }
