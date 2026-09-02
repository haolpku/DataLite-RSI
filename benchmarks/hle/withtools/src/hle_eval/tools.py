from __future__ import annotations

import ipaddress
import os
import random
import resource
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx

DEFAULT_FETCH_CHARS = 12_000
MAX_REDIRECTS = 5
NO_SEARCH_RESULTS = "I couldn't find any relevant information on the web."

HLE_URL_BLOCKLIST = (
    "huggingface.co",
    "hf.co",
    "hf-mirror.com",
    "promptfoo.dev",
    "://scale.com",
    ".scale.com",
    "lastexam.ai",
    "agi.safe.ai",
    "last-exam",
    "hle-exam",
    "askfilo.com",
    "studocu.com",
    "coursehero.com",
    "qiita.com",
    "2501.14249",
    "2507.05241",
    "2508.10173",
    "2510.08959",
    "2605.02442",
    "nature.com/articles/s41586-025-09962-4",
    "openreview.net/pdf?id=46UGfq8kMI",
    "researchgate.net/publication/394488269_benchmark-driven_selection_of_ai_evidence_from_deepseek-r1",
    "openreview.net/pdf/a94b1a66a55ab89d0e45eb8ed891b115db8bf760.pdf",
    "scribd.com/document/866099862",
    "x.com/tbenst/status/1951089655191122204",
    "x.com/andrewwhite01/status/1948056183115493725",
    "news.ycombinator.com/item?id=44694191",
    "github.com/supaihq/hle",
    "github.com/centerforaisafety/hle",
    "mveteanu/hle_pdf",
    "researchgate.net/scientific-contributions/petr-spelda-2170307851",
    "medium.com/@82deutschmark/o3-quiet-breakthrough-1bf9f0bafc84",
    "rahulpowar.medium.com/deepseek-triggers-1-trillion-slump-but-paves-a-bigger-future-for-ai",
    "bincial.com/news/tztechnology/421026",
    "36kr.com/p/3481854274280581",
    "jb243.github.io",
    "github.com/deepwriter-ai/hle-gemini-3-0",
    "github.com/ruc-nlpir/webthinker/blob/main/data/hle",
    "github.com/hanjanghoon/deer",
    "github.com/repos/hanjanghoon/deer",
    "xiaowenz.com/episodes/humanity-last-exam-and-agi",
    "research-collection.ethz.ch/server/api/core/bitstreams/1902b5a9-4209-4529-b278-c258aad557ba/content",
    "news.qq.com/rain/a/20260228a00wdr00",
    # Direct HLE and HLE-Verified data mirrors.
    "raw.githubusercontent.com/skylenage-ai/hle-verified",
    "media.githubusercontent.com/media/skylenage-ai/hle-verified",
    "raw.githubusercontent.com/mveteanu/hle_pdf",
    "cdn.jsdelivr.net/gh/mveteanu/hle_pdf",
    "gcore.jsdelivr.net/gh/mveteanu/hle_pdf",
    "data.jsdelivr.com/v1/package/gh/mveteanu/hle_pdf",
    "raw.githack.com/mveteanu/hle_pdf",
    "cdn.statically.io/gh/mveteanu/hle_pdf",
    "ghproxy.net/mveteanu/hle_pdf",
    "raw.gitmirror.com/mveteanu/hle_pdf",
    # Dataset pages, APIs, mirrors, and query services.
    "huggingface.co/datasets/cais/hle",
    "huggingface.co/api/datasets/cais/hle",
    "huggingface.co/datasets/skylenage/hle-verified",
    "huggingface.co/datasets/skylenage-ai/hle-verified",
    "datasets-server.huggingface.co",
    "hf-mirror.com/datasets/cais/hle",
    "hf-mirror.com/api/datasets/cais/hle",
    "hf.co/api/datasets",
    "modelscope.cn/datasets/hle-verified",
    "modelscope.cn/api/v1/datasets/hle-verified",
    "www.modelscope.cn/datasets/hle-verified",
    "www.modelscope.cn/api/v1/datasets/hle-verified",
    "openreward.ai/generalreasoning/hle-verified",
)

_NORMALIZED_BLOCKLIST = tuple(
    value.replace("/", "").casefold() for value in HLE_URL_BLOCKLIST
)


class ToolError(RuntimeError):
    pass


class ToolRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._rejected_search_keys: set[str] = set()

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "web_search":
            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ToolError("query must be a non-empty string")
            return self._web_search(query)
        if name == "web_fetch":
            return web_fetch(
                url=_string_argument(arguments, "url"),
                max_length=arguments.get("max_length"),
                start_index=arguments.get("start_index", 0),
                raw=arguments.get("raw", False),
                config=self.config.get("web_fetch", {}),
            )
        if name == "code_execution":
            return execute_python(
                _string_argument(arguments, "code"),
                self.config.get("code_execution", {}),
            )
        raise ToolError(f"Unknown tool: {name}")

    def _web_search(self, query: str) -> str:
        config = self.config.get("web_search") or {}
        provider = config.get("provider")
        if provider == "tavily":
            return self._tavily(query, config)
        if provider == "exa":
            return self._exa(query, config)
        if provider == "google":
            return self._google(query, config)
        if provider == "http_json":
            return self._http_json(query, config)
        raise ToolError("web_search.provider must be tavily, exa, google, or http_json")

    def _tavily(self, query: str, config: dict[str, Any]) -> str:
        keys = list(_api_keys(config, "TAVILY_API_KEY"))
        random.shuffle(keys)
        last_error: Exception | None = None
        for key in keys:
            if key in self._rejected_search_keys:
                continue
            options = _provider_options(config)
            options.setdefault("include_answer", True)
            response = httpx.post(
                config.get("url", "https://api.tavily.com/search"),
                headers={"Authorization": f"Bearer {key}"},
                json={"query": query, **options},
                timeout=float(config.get("timeout", 60)),
            )
            if response.status_code == 401:
                self._rejected_search_keys.add(key)
                last_error = ToolError("Tavily rejected an API key with HTTP 401")
                continue
            try:
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise ToolError(f"Tavily search failed: {exc}") from exc
            data = response.json()
            return _format_search(data.get("answer"), data.get("results", []))
        raise ToolError(str(last_error or "Tavily has no usable API keys"))

    def _exa(self, query: str, config: dict[str, Any]) -> str:
        key = _api_keys(config, "EXA_API_KEY")[0]
        options = _provider_options(config)
        options.setdefault("text", True)
        response = httpx.post(
            config.get("url", "https://api.exa.ai/answer"),
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json={"query": query, **options},
            timeout=float(config.get("timeout", 60)),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"Exa search failed: {exc}") from exc
        data = response.json()
        results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("text"),
            }
            for item in data.get("citations", [])
        ]
        return _format_search(data.get("answer"), results)

    def _google(self, query: str, config: dict[str, Any]) -> str:
        key = _api_keys(config, "GOOGLE_CSE_API_KEY")[0]
        cse_env = str(config.get("cse_id_env", "GOOGLE_CSE_ID"))
        cse_id = os.getenv(cse_env)
        if not cse_id:
            raise ToolError(f"environment variable {cse_env!r} is not set")
        response = httpx.get(
            config.get("url", "https://www.googleapis.com/customsearch/v1"),
            params={
                "q": query,
                "key": key,
                "cx": cse_id,
                "num": int(config.get("num_results", 5)),
            },
            timeout=float(config.get("timeout", 60)),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"Google search failed: {exc}") from exc
        results = [
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "content": item.get("snippet"),
            }
            for item in response.json().get("items", [])
        ]
        return _format_search(None, results)

    def _http_json(self, query: str, config: dict[str, Any]) -> str:
        url = config.get("url")
        if not isinstance(url, str) or not url:
            raise ToolError("http_json search requires url")
        headers = {
            str(key): str(value) for key, value in config.get("headers", {}).items()
        }
        key_env = config.get("api_key_env")
        if key_env:
            key = os.getenv(str(key_env))
            if not key:
                raise ToolError(f"environment variable {key_env!r} is not set")
            headers.setdefault("Authorization", f"Bearer {key}")
        response = httpx.post(
            url,
            headers=headers,
            json={"query": query},
            timeout=float(config.get("timeout", 60)),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ToolError(f"HTTP search provider failed: {exc}") from exc
        data = response.json()
        if isinstance(data, str):
            return data
        if not isinstance(data, dict):
            raise ToolError("HTTP search response must be a string or JSON object")
        if isinstance(data.get("text"), str):
            return data["text"]
        return _format_search(data.get("answer"), data.get("results", []))


def execute_python(code: str, config: dict[str, Any]) -> str:
    if config.get("provider", "builtin") != "builtin":
        raise ToolError("code_execution.provider must be builtin")
    timeout = float(config.get("timeout", 300))
    max_output = int(config.get("max_output_chars", 100_000))
    with tempfile.TemporaryDirectory(prefix="hle-code-") as directory:
        workspace = Path(directory)
        sandbox_uid = int(config.get("uid", 65534))
        sandbox_gid = int(config.get("gid", 65534))
        if os.geteuid() == 0:
            os.chown(workspace, sandbox_uid, sandbox_gid)
        env = {
            "HOME": str(workspace),
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
        }
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-"],
                input=code,
                text=True,
                capture_output=True,
                cwd=workspace,
                env=env,
                timeout=timeout,
                check=False,
                preexec_fn=lambda: _limit_process(sandbox_uid, sandbox_gid, timeout),
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"Python execution timed out after {timeout:g} seconds"
            ) from exc
        output = result.stdout + result.stderr
        if result.returncode and not output:
            output = f"Process exited with status {result.returncode}."
        if len(output) > max_output:
            output = output[:max_output] + "\n<output truncated>"
        return output


def _limit_process(uid: int, gid: int, timeout: float) -> None:
    memory = 2 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
    cpu = max(1, int(timeout) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)


def web_fetch(
    *,
    url: str,
    max_length: int | None,
    start_index: int,
    raw: bool,
    config: dict[str, Any],
) -> str:
    if config.get("provider", "builtin") != "builtin":
        raise ToolError("web_fetch.provider must be builtin")
    max_length = DEFAULT_FETCH_CHARS if max_length is None else max_length
    if not isinstance(max_length, int) or max_length < 1:
        raise ToolError("max_length must be positive")
    if not isinstance(start_index, int) or start_index < 0:
        raise ToolError("start_index must be non-negative")
    if not isinstance(raw, bool):
        raise ToolError("raw must be boolean")
    validate_public_url(url)
    timeout = float(config.get("timeout", 20))
    current_url = url
    headers = {
        "User-Agent": "ModelContextProtocol/1.0 (Autonomous; +https://github.com/modelcontextprotocol/servers)"
    }
    try:
        with httpx.Client(
            follow_redirects=False, timeout=timeout, headers=headers
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                response = client.get(current_url)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ToolError(
                            "web_fetch received a redirect without a location"
                        )
                    current_url = urljoin(current_url, location)
                    validate_public_url(current_url)
                    continue
                response.raise_for_status()
                break
            else:
                raise ToolError("web_fetch followed too many redirects")
    except httpx.HTTPStatusError as exc:
        raise ToolError(f"web_fetch received HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ToolError(f"web_fetch request failed: {exc}") from exc

    content = response.text
    if not raw and _is_html_response(content, response.headers):
        content = extract_content_from_html(content)
    if start_index >= len(content):
        return "<error>No more content available.</error>"
    truncated = content[start_index : start_index + max_length]
    remaining = len(content) - start_index - len(truncated)
    if len(truncated) == max_length and remaining > 0:
        truncated += (
            "\n\n<error>Content truncated. Call the web_fetch tool with a "
            f"start_index of {start_index + len(truncated)} to get more content.</error>"
        )
    return truncated or "<error>No more content available.</error>"


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ToolError("web_fetch accepts only absolute http or https URLs")
    if parsed.username or parsed.password:
        raise ToolError("web_fetch does not accept URLs with embedded credentials")
    hostname = (parsed.hostname or "").rstrip(".").lower()
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(
        ".local"
    ):
        raise ToolError("web_fetch accepts only public web hosts")
    if address is not None and not address.is_global:
        raise ToolError("web_fetch accepts only public web hosts")
    if is_blocked_url(url):
        raise ToolError("web_fetch rejected a URL on the HLE blocklist")


def is_blocked_url(url: str) -> bool:
    decoded = url
    for _ in range(2):
        decoded = unquote(decoded)
    decoded = (
        decoded.replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
    )
    parsed = urlparse(decoded)
    hostname = (parsed.hostname or "").rstrip(".").casefold()
    path = parsed.path.casefold()
    if hostname in {"modelscope.cn", "www.modelscope.cn"} and (
        path.startswith("/datasets/") or path.startswith("/api/v1/datasets/")
    ) and "/hle-verified/" in f"{path.rstrip('/')}/":
        return True
    normalized = decoded.replace("/", "").casefold()
    return any(value in normalized for value in _NORMALIZED_BLOCKLIST)


def extract_content_from_html(html: str) -> str:
    from markdownify import ATX, markdownify
    from readabilipy import simple_json

    javascript_dir = Path(simple_json.__file__).with_name("javascript")
    simplified = simple_json.simple_json_from_html_string(
        html,
        use_readability=(javascript_dir / "node_modules").is_dir(),
    )
    if not simplified["content"]:
        return "<error>Page failed to be simplified from HTML</error>"
    return markdownify(simplified["content"], heading_style=ATX)


def _is_html_response(content: str, headers: Mapping[str, Any]) -> bool:
    content_type = str(headers.get("content-type", ""))
    return (
        "<html" in content[:100].lower()
        or "text/html" in content_type.lower()
        or not content_type
    )


def _format_search(answer: Any, results: Any) -> str:
    lines: list[str] = []
    if isinstance(answer, str) and answer.strip():
        lines.append(answer.strip())
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or item.get("link")
            if not isinstance(url, str) or is_blocked_url(url):
                continue
            title = item.get("title") or url
            content = (
                item.get("content") or item.get("text") or item.get("snippet") or ""
            )
            lines.append(f"{title}\n{url}\n{content}".strip())
    return "\n\n".join(lines) or NO_SEARCH_RESULTS


def _api_keys(config: dict[str, Any], default_env: str) -> tuple[str, ...]:
    env_name = str(config.get("api_key_env", default_env))
    value = os.getenv(env_name, "")
    keys = tuple(
        item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()
    )
    if not keys:
        raise ToolError(f"environment variable {env_name!r} is not set")
    return keys


def _provider_options(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"provider", "url", "api_key_env", "timeout", "headers", "cse_id_env"}
    return {key: value for key, value in config.items() if key not in ignored}


def _string_argument(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolError(f"{key} must be a string")
    return value
