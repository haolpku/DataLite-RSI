"""从 LLM/Agent 的自由文本输出里稳健地抽取 JSON 与代码块。"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any | None:
    """从文本中抽取第一个 JSON 对象/数组。

    依次尝试：① ```json 围栏内容（先整体解析，失败再对围栏内容做平衡扫描——
    旧实现用非贪婪 `\\{.*?\\}` 对嵌套对象会截断，故改为平衡扫描）；② 整段直接
    json.loads；③ 全文第一个平衡的 {...} / []。失败返回 None。
    """
    if not text:
        return None

    m = _FENCE.search(text)
    if m:
        inner = m.group(1).strip()
        for cand in (inner, _first_balanced(inner)):
            if cand:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    pass

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    snippet = _first_balanced(text)
    if snippet is not None:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def extract_code(text: str, lang: str = "python") -> str | None:
    """抽取第一个代码围栏的内容；无围栏时若整段像代码则原样返回。"""
    if not text:
        return None
    m = _CODE_FENCE.search(text)
    if m:
        return m.group(1).strip()
    stripped = text.strip()
    if stripped and not stripped.startswith("```"):
        return stripped
    return None


def _first_balanced(text: str) -> str | None:
    """找到第一个括号平衡的 {...} 或 [...] 子串（忽略字符串字面量内的括号）。"""
    start = None
    open_ch = close_ch = ""
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            open_ch = ch
            close_ch = "}" if ch == "{" else "]"
            break
    if start is None:
        return None

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
