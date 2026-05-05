"""Lightweight direct-answer extractors for local quiz/document chunks.

These rules are deliberately conservative. If a structured answer cannot be
found with enough confidence, return None and let the model answer normally.
"""

from __future__ import annotations

import re
from typing import Any


_CN_STOPWORDS = set("的了和是一个一种以及对于关于认为下面哪个哪些什么如何是否其中不包括包括根据下列以下属于不属于正确错误".split())


def _clean(text: str) -> str:
    text = re.sub(r"[\t\r\f\v]+", " ", text or "")
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    return text.strip()


def _terms(text: str) -> set[str]:
    text = text or ""
    terms = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,}|\d+|[\u4e00-\u9fff]{2,}", text):
        token = token.lower()
        if token and token not in _CN_STOPWORDS:
            terms.add(token)
    # Add short Chinese bigrams so queries like “劳动过程基本要素” match better.
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(max(0, len(cjk) - 1)):
        pair = cjk[i:i + 2]
        if pair not in _CN_STOPWORDS:
            terms.add(pair)
    return terms


def _overlap_score(question: str, block: str) -> int:
    q_terms = _terms(question)
    if not q_terms:
        return 0
    b_terms = _terms(block)
    return len(q_terms & b_terms)


def _split_question_blocks(text: str) -> list[str]:
    text = _clean(text)
    if not text:
        return []

    # Try to split before numbered quiz questions.
    parts = re.split(r"(?=\n?\s*\d+\.\s*[\(（]?(?:单选题|判断题|多选题)[\)）]?)", text)
    blocks = [_clean(p) for p in parts if len(_clean(p)) >= 20]
    if blocks:
        return blocks

    return [text]


def _extract_options(block: str) -> dict[str, str]:
    options: dict[str, str] = {}
    # Handles “A. xxx”, “· A. xxx”, “A:xxx”, and Chinese fullwidth punctuation.
    pattern = re.compile(
        r"(?:^|\n|\s|·)\s*([A-H])\s*[\.．、:：]\s*([^\n;；]+)",
        flags=re.I,
    )
    for letter, value in pattern.findall(block):
        value = _clean(value)
        value = re.sub(r"\s*我的答案.*$", "", value).strip()
        if value:
            options[letter.upper()] = value
    return options


def _extract_correct(block: str) -> tuple[str, str]:
    # Examples:
    # 正确答案:D:劳动休息时间;
    # 正确答案:对
    # 正确答案：A
    match = re.search(r"正确答案\s*[:：]\s*([A-H]|对|错|正确|错误)\s*(?:[:：]\s*([^;；\n]+))?", block, flags=re.I)
    if not match:
        return "", ""
    key = match.group(1).strip().upper()
    text = _clean(match.group(2) or "")
    return key, text


def _extract_stem(block: str) -> str:
    # Remove answer/options tail where possible.
    head = re.split(r"(?:\n|\s|·)\s*A\s*[\.．、:：]|我的答案|正确答案", block, maxsplit=1, flags=re.I)[0]
    head = re.sub(r"^\s*\d+\.\s*", "", head)
    return _clean(head)[:260]


def _select_best_block(question: str, results: list[dict[str, Any]]) -> tuple[dict[str, Any], str, int] | None:
    best: tuple[dict[str, Any], str, int] | None = None
    for item in results or []:
        text = item.get("fetched_text") or item.get("content") or item.get("snippet") or ""
        for block in _split_question_blocks(text):
            if "正确答案" not in block:
                continue
            score = _overlap_score(question, block)
            # Local KB ranking already narrowed the chunk; allow low overlap for short questions.
            if best is None or score > best[2]:
                best = (item, block, score)
    return best


def try_direct_answer(question: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
    selected = _select_best_block(question, results)
    if not selected:
        return None

    item, block, score = selected
    if score <= 0 and len(question.strip()) > 10:
        # Avoid answering from an unrelated chunk just because it has 正确答案.
        return None

    correct_key, correct_text = _extract_correct(block)
    if not correct_key:
        return None

    options = _extract_options(block)
    if not correct_text and correct_key in options:
        correct_text = options[correct_key]

    stem = _extract_stem(block)
    source_title = item.get("title", "本地知识库")
    source_url = item.get("url", "")

    lines = ["【直接答案】"]

    if correct_key in {"对", "错", "正确", "错误"}:
        answer_text = "对" if correct_key in {"对", "正确"} else "错"
        lines.append(f"根据本地知识库，正确答案是：{answer_text}。")
    else:
        if correct_text:
            lines.append(f"根据本地知识库，正确答案是 {correct_key}：{correct_text}。")
        else:
            lines.append(f"根据本地知识库，正确答案是 {correct_key}。")

    # Special case: user asks for “three elements”, while the matched quiz asks “不包括”.
    if re.search(r"三个|三.*要素|哪些|哪三", question) and "不包括" in block and correct_key in options:
        included = [(k, v) for k, v in options.items() if k != correct_key]
        if included:
            lines.append("")
            lines.append("该题题干问的是“不包括”，因此除正确选项外，其余选项可作为题干所指的基本要素：")
            for _, value in included[:6]:
                lines.append(f"- {value}")

    lines.append("")
    lines.append("【依据】")
    if stem:
        lines.append(f"命中的题目：{stem}")
    if options:
        option_text = "；".join(f"{k}. {v}" for k, v in options.items())
        lines.append(f"选项：{option_text}")
    lines.append(f"来源：{source_title}" + (f"（{source_url}）" if source_url else ""))
    lines.append("")
    lines.append("【说明】")
    lines.append("这是从本地知识库中的结构化“正确答案”字段直接抽取的结果，未调用模型生成长回答。")

    return {
        "ok": True,
        "answer": "\n".join(lines).strip(),
        "source_title": source_title,
        "source_url": source_url,
        "score": score,
    }
