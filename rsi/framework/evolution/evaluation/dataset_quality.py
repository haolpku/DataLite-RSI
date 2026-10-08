"""Dataset-level embedding evaluation with optional proxy MMD/DAS and diversity.

The implementation follows Data-Preparation-Bench's DAS definition:
``DAS = -sqrt(max(0, biased_MMD_squared))`` with an RBF kernel.  Candidate
records and proxy records are encoded as the same two-turn user/assistant
conversation so the comparison is not confounded by field names.

The candidate embeddings are also reused for two threshold-free, codebook-free
diversity views: cosine-kernel Vendi Score and nearest-neighbour cosine statistics.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from rsi.framework.evolution.models import DatasetQualityResult
from rsi.framework.evolution.telemetry.pipeline_usage import normalize_usage
from rsi.framework.evolution.utils.logging import get_logger


Conversation = list[dict[str, str]]
logger = get_logger("core.dataset_quality")


@dataclass(frozen=True)
class ProxyDatasetConfig:
    source: str
    path: str
    user_field: str
    assistant_field: str
    name: str = ""
    split: str = "train"


@dataclass
class EmbeddingBatch:
    vectors: np.ndarray
    fail_rate: float = 0.0
    usage: dict[str, Any] | None = None


class EmbeddingBackend(Protocol):
    model_name: str

    def embed(self, conversations: list[Conversation]) -> EmbeddingBatch: ...


class OpenAIChatEmbeddingBackend:
    """Qwen/vLLM-compatible chat embedding client with bounded concurrency."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str = "EMPTY",
        max_concurrent_requests: int = 128,
        max_retries: int = 3,
        truncate_prompt_tokens: int = 40960,
        truncation_side: str = "right",
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrent_requests = max(1, max_concurrent_requests)
        self.max_retries = max(1, max_retries)
        self.truncate_prompt_tokens = truncate_prompt_tokens
        self.truncation_side = truncation_side

    def embed(self, conversations: list[Conversation]) -> EmbeddingBatch:
        logger.info(
            "请求 embedding：model=%s samples=%d concurrency=%d",
            self.model_name,
            len(conversations),
            self.max_concurrent_requests,
        )
        return asyncio.run(self._embed_async(conversations))

    async def _embed_async(self, conversations: list[Conversation]) -> EmbeddingBatch:
        from openai import AsyncOpenAI
        from openai.types.create_embedding_response import CreateEmbeddingResponse

        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def one(
            messages: Conversation,
        ) -> tuple[list[float] | None, dict[str, int] | None, int]:
            body: dict[str, Any] = {
                "messages": messages,
                "model": self.model_name,
                "truncate_prompt_tokens": self.truncate_prompt_tokens,
                "truncation_side": self.truncation_side,
            }
            for attempt in range(self.max_retries):
                async with semaphore:
                    try:
                        response: CreateEmbeddingResponse = await client.post(
                            "/embeddings",
                            cast_to=CreateEmbeddingResponse,
                            body=body,
                        )
                        usage = normalize_usage(_model_dump(getattr(response, "usage", None)))
                        return list(response.data[0].embedding), usage, attempt + 1
                    except Exception:
                        if attempt + 1 >= self.max_retries:
                            return None, None, attempt + 1
                await asyncio.sleep(0.1 * (attempt + 1))
            return None, None, self.max_retries

        try:
            results = await asyncio.gather(*(one(messages) for messages in conversations))
        finally:
            await client.close()

        successful = [value[0] for value in results if value[0] is not None]
        fail_rate = 1.0 - (len(successful) / len(conversations)) if conversations else 0.0
        usage = _embedding_batch_usage(results, self.model_name, len(conversations))
        if not successful:
            return EmbeddingBatch(
                np.empty((0, 0), dtype=np.float32), fail_rate, usage
            )
        return EmbeddingBatch(
            np.asarray(successful, dtype=np.float32), fail_rate, usage
        )


def _model_dump(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json", by_alias=True, exclude_none=True)
        return payload if isinstance(payload, dict) else None
    return None


def _empty_embedding_tokens() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens": 0,
    }


def _embedding_batch_usage(
    results: list[tuple[list[float] | None, dict[str, int] | None, int]],
    model: str,
    item_count: int,
) -> dict[str, Any]:
    tokens = _empty_embedding_tokens()
    request_count = sum(attempts for _, _, attempts in results)
    success_count = sum(vector is not None for vector, _, _ in results)
    failure_count = max(request_count - success_count, 0)
    missing_usage_count = failure_count
    exact_usage_count = 0
    for vector, usage, _ in results:
        if vector is None:
            continue
        if usage is None:
            missing_usage_count += 1
            continue
        exact_usage_count += 1
        for key in tokens:
            tokens[key] += int(usage.get(key) or 0)
    return {
        "model": model,
        "item_count": item_count,
        "request_count": request_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "missing_usage_count": missing_usage_count,
        "exact_usage_count": exact_usage_count,
        "usage_status": "exact" if missing_usage_count == 0 else "partial",
        "tokens": tokens,
    }


def _combine_embedding_usage(
    events: list[dict[str, Any]],
    *,
    model: str,
) -> dict[str, Any]:
    totals = _empty_embedding_tokens()
    combined: dict[str, Any] = {
        "schema_version": 1,
        "kind": "embedding",
        "model": model,
        "batch_count": len(events),
        "cache_hit_count": sum(bool(event.get("cache_hit")) for event in events),
        "item_count": 0,
        "request_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "missing_usage_count": 0,
        "exact_usage_count": 0,
        "wall_time_seconds": 0.0,
        "unavailable_batch_count": sum(
            event.get("usage_status") == "unavailable" for event in events
        ),
        "tokens": totals,
        "batches": events,
    }
    for event in events:
        for key in (
            "item_count",
            "request_count",
            "success_count",
            "failure_count",
            "missing_usage_count",
            "exact_usage_count",
        ):
            combined[key] += int(event.get(key) or 0)
        event_tokens = event.get("tokens")
        if isinstance(event_tokens, dict):
            for key in totals:
                totals[key] += int(event_tokens.get(key) or 0)
        combined["wall_time_seconds"] += float(event.get("wall_time_seconds") or 0.0)
    combined["wall_time_seconds"] = round(combined["wall_time_seconds"], 6)
    if not events:
        combined["usage_status"] = "not_requested"
    elif all(event.get("cache_hit") for event in events):
        combined["usage_status"] = "cache_only"
    elif combined["unavailable_batch_count"]:
        combined["usage_status"] = "unavailable"
    elif combined["missing_usage_count"]:
        combined["usage_status"] = "partial"
    else:
        combined["usage_status"] = "exact"
    return combined


class DatasetQualityEvaluator:
    """Evaluate candidate diversity and optionally compare with a quality proxy."""

    def __init__(
        self,
        *,
        proxy: ProxyDatasetConfig | None,
        embedder: EmbeddingBackend,
        sample_size: int = 5000,
        seed: int = 42,
        sigma: float = 1.0,
        biased: bool = True,
        normalize_embeddings: bool = True,
        max_embedding_fail_rate: float = 0.02,
        cache_dir: str | Path | None = None,
        require_full_sample: bool = True,
        proxy_comparison_enabled: bool = True,
    ) -> None:
        if sample_size < 1:
            raise ValueError("dataset_quality.sample_size 必须至少为 1")
        if sigma <= 0:
            raise ValueError("dataset_quality.sigma 必须大于 0")
        if proxy_comparison_enabled and proxy is None:
            raise ValueError("DAS/MMD enabled but no proxy dataset was configured")
        self.proxy = proxy
        self.proxy_comparison_enabled = proxy_comparison_enabled
        self.embedder = embedder
        self.sample_size = sample_size
        self.seed = seed
        self.sigma = sigma
        self.biased = biased
        self.normalize_embeddings = normalize_embeddings
        self.max_embedding_fail_rate = max_embedding_fail_rate
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.require_full_sample = require_full_sample
        self._proxy_messages: list[Conversation] | None = None
        self._embedding_usage_events: list[dict[str, Any]] = []

    @property
    def proxy_name(self) -> str:
        if self.proxy is None:
            return "disabled (candidate-only diversity)"
        if self.proxy.name and self.proxy.name != "default":
            return f"{self.proxy.path}:{self.proxy.name}"
        return self.proxy.path

    def evaluate(
        self,
        candidate_path: str,
        *,
        candidate_user_field: str,
        candidate_assistant_field: str,
    ) -> DatasetQualityResult:
        self._embedding_usage_events = []
        base = DatasetQualityResult(
            enabled=True,
            status="error",
            metric=("DAS" if self.proxy_comparison_enabled else "candidate_embedding_diversity"),
            proxy_name=self.proxy_name,
            proxy_sample_size=(self.sample_size if self.proxy_comparison_enabled else 0),
            embedding_model=self.embedder.model_name,
            normalized=self.normalize_embeddings,
            sigma=self.sigma,
            estimator="biased" if self.biased else "unbiased",
        )
        try:
            candidate_messages, total, valid = _sample_jsonl_conversations(
                candidate_path,
                sample_size=self.sample_size,
                seed=self.seed,
                user_field=candidate_user_field,
                assistant_field=candidate_assistant_field,
            )
            base.candidate_total = total
            base.candidate_sample_size = len(candidate_messages)
            logger.info(
                "DAS 候选采样：path=%s total=%d valid=%d sampled=%d target=%d",
                candidate_path,
                total,
                valid,
                len(candidate_messages),
                self.sample_size,
            )
            if self.require_full_sample and valid < self.sample_size:
                base.status = "insufficient_samples"
                base.error = (
                    f"候选数据只有 {valid} 条可编码记录，DAS 配置要求固定采样 "
                    f"{self.sample_size} 条"
                )
                return base
            if not candidate_messages:
                base.status = "insufficient_samples"
                base.error = "候选数据没有可编码记录"
                return base

            candidate_batch = self._embed_cached(candidate_messages, scope="candidate")
            base.embedding_fail_rate = candidate_batch.fail_rate
            proxy_batch = None
            if self.proxy_comparison_enabled:
                proxy_messages = self._load_proxy_messages()
                base.proxy_sample_size = len(proxy_messages)
                if self.require_full_sample and len(proxy_messages) < self.sample_size:
                    base.status = "insufficient_proxy_samples"
                    base.error = (
                        f"代理数据只有 {len(proxy_messages)} 条可编码记录，配置要求 "
                        f"{self.sample_size} 条"
                    )
                    return base
                proxy_batch = self._embed_cached(proxy_messages, scope="proxy")
                base.embedding_fail_rate = max(
                    candidate_batch.fail_rate, proxy_batch.fail_rate
                )
            if base.embedding_fail_rate > self.max_embedding_fail_rate:
                base.error = (
                    f"embedding 失败率 {base.embedding_fail_rate:.2%} 超过上限 "
                    f"{self.max_embedding_fail_rate:.2%}"
                )
                return base

            base.status = "ok"
            base.embedding_dim = int(candidate_batch.vectors.shape[1])
            base.embedding_diversity = compute_embedding_diversity(
                candidate_batch.vectors
            )
            if proxy_batch is not None:
                mmd, terms = compute_rbf_mmd(
                    candidate_batch.vectors,
                    proxy_batch.vectors,
                    sigma=self.sigma,
                    biased=self.biased,
                    normalize=self.normalize_embeddings,
                )
                base.mmd = mmd
                base.das = -mmd
                base.kernel_terms = terms
            logger.info(
                "Dataset embedding 完成：proxy=%s candidate=%d "
                "MMD=%s DAS=%s Vendi=%.3f NN-cos-mean=%.4f",
                self.proxy_name,
                len(candidate_batch.vectors),
                f"{base.mmd:.6f}" if base.mmd is not None else "disabled",
                f"{base.das:.6f}" if base.das is not None else "disabled",
                base.embedding_diversity.get("cosine_vendi_score", 0.0),
                base.embedding_diversity.get("nearest_neighbor_cosine_mean", 0.0),
            )
            return base
        except Exception as exc:  # Evidence must survive service/data failures.
            base.error = f"{type(exc).__name__}: {exc}"
            logger.exception("DAS 计算失败：%s", base.error)
            return base
        finally:
            base.embedding_usage = _combine_embedding_usage(
                self._embedding_usage_events,
                model=self.embedder.model_name,
            )
            base.embedding_wall_time_seconds = float(
                base.embedding_usage.get("wall_time_seconds") or 0.0
            )

    def _load_proxy_messages(self) -> list[Conversation]:
        if self._proxy_messages is not None:
            return self._proxy_messages
        if self.proxy is None:
            raise ValueError("proxy comparison is disabled")
        source = self.proxy.source.strip().lower()
        if source in {"local", "jsonl", "json"}:
            messages, _, _ = _sample_local_proxy(
                self.proxy.path,
                sample_size=self.sample_size,
                seed=self.seed,
                user_field=self.proxy.user_field,
                assistant_field=self.proxy.assistant_field,
            )
        elif source in {"huggingface", "hf", "datasets"}:
            messages = _sample_huggingface_proxy(self.proxy, self.sample_size, self.seed)
        else:
            raise ValueError(f"不支持的代理数据来源：{self.proxy.source}")
        self._proxy_messages = messages
        return messages

    def _embed_cached(
        self,
        conversations: list[Conversation],
        *,
        scope: str,
    ) -> EmbeddingBatch:
        started = time.perf_counter()
        cache_path = self._cache_path(conversations)
        if cache_path is not None and cache_path.exists():
            logger.info("命中 dataset embedding 缓存：%s", cache_path)
            with np.load(cache_path, allow_pickle=False) as payload:
                batch = EmbeddingBatch(
                    np.asarray(payload["embeddings"], dtype=np.float32),
                    float(payload["fail_rate"]),
                )
            self._embedding_usage_events.append(
                {
                    "scope": scope,
                    "model": self.embedder.model_name,
                    "cache_hit": True,
                    "item_count": len(conversations),
                    "request_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "missing_usage_count": 0,
                    "wall_time_seconds": round(time.perf_counter() - started, 6),
                    "tokens": _empty_embedding_tokens(),
                }
            )
            return batch

        logger.info("未命中 dataset embedding 缓存，开始编码 %d 条样本", len(conversations))
        batch = self.embedder.embed(conversations)
        usage = dict(batch.usage or {})
        usage.update(
            {
                "scope": scope,
                "model": str(usage.get("model") or self.embedder.model_name),
                "cache_hit": False,
                "item_count": int(usage.get("item_count") or len(conversations)),
                "wall_time_seconds": round(time.perf_counter() - started, 6),
            }
        )
        if not batch.usage:
            usage.update(
                {
                    "usage_status": "unavailable",
                    "request_count": 0,
                    "success_count": 0,
                    "failure_count": 0,
                    "missing_usage_count": 0,
                    "tokens": _empty_embedding_tokens(),
                }
            )
        self._embedding_usage_events.append(usage)
        if batch.vectors.ndim != 2 or batch.vectors.shape[0] == 0:
            raise RuntimeError("embedding 服务没有返回有效的二维向量")
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temp_path,
                embeddings=np.asarray(batch.vectors, dtype=np.float32),
                fail_rate=np.asarray(batch.fail_rate, dtype=np.float64),
            )
            os.replace(temp_path, cache_path)
        return batch

    def _cache_path(self, conversations: list[Conversation]) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "version": 1,
                    "model": self.embedder.model_name,
                    "backend": type(self.embedder).__name__,
                    "base_url": getattr(self.embedder, "base_url", ""),
                    "truncate_prompt_tokens": getattr(
                        self.embedder, "truncate_prompt_tokens", None
                    ),
                    "truncation_side": getattr(self.embedder, "truncation_side", None),
                    "normalized": self.normalize_embeddings,
                    "conversation_format": "user_assistant_messages",
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for messages in conversations:
            digest.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
        return self.cache_dir / f"{digest.hexdigest()}.npz"


def compute_rbf_mmd(
    candidate: np.ndarray,
    proxy: np.ndarray,
    *,
    sigma: float = 1.0,
    biased: bool = True,
    normalize: bool = True,
) -> tuple[float, dict[str, float]]:
    """Compute exact RBF MMD using the same estimator as Data-Preparation-Bench."""
    x = np.asarray(candidate, dtype=np.float64)
    y = np.asarray(proxy, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or not len(x) or not len(y):
        raise ValueError("MMD 输入必须是非空二维 embedding 数组")
    if x.shape[1] != y.shape[1]:
        raise ValueError("候选和代理 embedding 维度不一致")
    if sigma <= 0:
        raise ValueError("RBF sigma 必须大于 0")
    if not biased and (len(x) < 2 or len(y) < 2):
        raise ValueError("unbiased MMD 每组至少需要两条样本")
    if normalize:
        x = _l2_normalize(x)
        y = _l2_normalize(y)

    k_xx = _rbf_kernel(x, x, sigma)
    k_yy = _rbf_kernel(y, y, sigma)
    k_xy = _rbf_kernel(x, y, sigma)
    if biased:
        xx = float(k_xx.mean())
        yy = float(k_yy.mean())
    else:
        xx = float((k_xx.sum() - np.trace(k_xx)) / (len(x) * (len(x) - 1)))
        yy = float((k_yy.sum() - np.trace(k_yy)) / (len(y) * (len(y) - 1)))
    xy = float(k_xy.mean())
    mmd_squared = xx + yy - 2.0 * xy
    mmd = float(np.sqrt(max(0.0, mmd_squared)))
    return mmd, {
        "candidate_internal": xx,
        "proxy_internal": yy,
        "cross": xy,
        "mmd_squared": mmd_squared,
    }


def compute_embedding_diversity(embeddings: np.ndarray) -> dict[str, float]:
    """Compute threshold-free semantic-diversity evidence from one embedding batch.

    Vendi Score is the exponential entropy of the eigenvalues of the normalized
    cosine Gram matrix.  It is an effective number of distinct samples: identical
    vectors score 1, while orthogonal vectors score ``n``.  The nearest-neighbour
    profile exposes concentration without choosing a SemDeDup cutoff in advance.
    """
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("多样性指标输入必须是非空二维 embedding 数组")
    normalized = _l2_normalize(values)
    cosine = normalized @ normalized.T
    cosine = np.clip((cosine + cosine.T) * 0.5, -1.0, 1.0)

    eigenvalues = np.linalg.eigvalsh(cosine)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        raise ValueError("cosine Gram matrix 没有正特征值")
    probabilities = eigenvalues / total
    positive = probabilities[probabilities > np.finfo(np.float64).eps]
    vendi = float(np.exp(-np.sum(positive * np.log(positive))))
    sample_count = len(normalized)
    vendi = min(max(vendi, 1.0), float(sample_count))

    metrics = {
        "cosine_vendi_score": vendi,
        "cosine_vendi_ratio": vendi / sample_count,
    }
    if sample_count == 1:
        return metrics

    off_diagonal = cosine[~np.eye(sample_count, dtype=bool)]
    nearest = cosine.copy()
    np.fill_diagonal(nearest, -np.inf)
    nearest = nearest.max(axis=1)
    metrics.update(
        {
            "mean_pairwise_cosine": float(off_diagonal.mean()),
            "nearest_neighbor_cosine_mean": float(nearest.mean()),
            "nearest_neighbor_cosine_p50": float(np.quantile(nearest, 0.50)),
            "nearest_neighbor_cosine_p90": float(np.quantile(nearest, 0.90)),
            "nearest_neighbor_cosine_p95": float(np.quantile(nearest, 0.95)),
            "nearest_neighbor_cosine_p99": float(np.quantile(nearest, 0.99)),
        }
    )
    return metrics


def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    distances = (
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * (x @ y.T)
    )
    np.maximum(distances, 0.0, out=distances)
    distances *= -1.0 / (2.0 * sigma * sigma)
    np.exp(distances, out=distances)
    return distances


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding 中包含零向量，无法 L2 归一化")
    return values / norms


def _conversation(record: dict[str, Any], user_field: str, assistant_field: str) -> Conversation | None:
    user = record.get(user_field)
    assistant = record.get(assistant_field)
    if user_field == assistant_field and isinstance(user, list):
        messages: Conversation = []
        if user and all(isinstance(message, str) for message in user):
            if len(user) % 2 != 0 or any(not message.strip() for message in user):
                return None
            # Alternating string-list proxies such as EduChat carry a repeated
            # branding system prompt. Exclude that metadata so MMD compares the
            # educational exchanges instead of a shared boilerplate prefix.
            messages.extend(
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": content.strip(),
                }
                for index, content in enumerate(user)
            )
        else:
            for message in user:
                if not isinstance(message, dict):
                    return None
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    return None
                if not isinstance(content, str) or not content.strip():
                    return None
                messages.append({"role": role, "content": content.strip()})
        dialog = messages[1:] if messages and messages[0]["role"] == "system" else messages
        expected = ["user" if index % 2 == 0 else "assistant" for index in range(len(dialog))]
        if (
            len(dialog) < 2
            or dialog[-1]["role"] != "assistant"
            or [message["role"] for message in dialog] != expected
        ):
            return None
        return messages
    if not isinstance(user, str) or not user.strip():
        return None
    if not isinstance(assistant, str) or not assistant.strip():
        return None
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def _sample_jsonl_conversations(
    path: str,
    *,
    sample_size: int,
    seed: int,
    user_field: str,
    assistant_field: str,
) -> tuple[list[Conversation], int, int]:
    return _sample_jsonl_files(
        [Path(path)],
        sample_size=sample_size,
        seed=seed,
        user_field=user_field,
        assistant_field=assistant_field,
    )


def _sample_jsonl_files(
    paths: list[Path],
    *,
    sample_size: int,
    seed: int,
    user_field: str,
    assistant_field: str,
) -> tuple[list[Conversation], int, int]:
    rng = random.Random(seed)
    reservoir: list[Conversation] = []
    total = 0
    valid = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(record, dict):
                    continue
                messages = _conversation(record, user_field, assistant_field)
                if messages is None:
                    continue
                valid += 1
                if len(reservoir) < sample_size:
                    reservoir.append(messages)
                else:
                    index = rng.randint(0, valid - 1)
                    if index < sample_size:
                        reservoir[index] = messages
    return reservoir, total, valid


def _sample_local_proxy(
    path: str,
    *,
    sample_size: int,
    seed: int,
    user_field: str,
    assistant_field: str,
) -> tuple[list[Conversation], int, int]:
    proxy_path = Path(path)
    if proxy_path.is_dir():
        jsonl_files = sorted(
            file
            for pattern in ("*.jsonl", "*.ndjson")
            for file in proxy_path.rglob(pattern)
        )
        if jsonl_files:
            return _sample_jsonl_files(
                jsonl_files,
                sample_size=sample_size,
                seed=seed,
                user_field=user_field,
                assistant_field=assistant_field,
            )
        parquet_files = sorted(proxy_path.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"本地代理目录中没有 JSONL/NDJSON 或 Parquet 分片：{proxy_path}")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "读取本地 Parquet 代理集需要安装 dataset-quality 可选依赖"
            ) from exc
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(file) for file in parquet_files]},
            split="train",
        )
        messages = _sample_dataset_conversations(
            dataset,
            sample_size=sample_size,
            seed=seed,
            user_field=user_field,
            assistant_field=assistant_field,
        )
        return messages, len(dataset), len(messages)
    if proxy_path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _sample_jsonl_conversations(
            path,
            sample_size=sample_size,
            seed=seed,
            user_field=user_field,
            assistant_field=assistant_field,
        )
    raw = json.loads(proxy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("本地代理 JSON 必须是对象数组；逐行数据请使用 JSONL")
    messages = [
        conv
        for item in raw
        if isinstance(item, dict)
        for conv in [_conversation(item, user_field, assistant_field)]
        if conv is not None
    ]
    rng = random.Random(seed)
    sampled = rng.sample(messages, sample_size) if len(messages) > sample_size else messages
    return sampled, len(raw), len(messages)


def _sample_huggingface_proxy(
    proxy: ProxyDatasetConfig, sample_size: int, seed: int
) -> list[Conversation]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace 代理数据需要安装 datasets；请安装项目 dataset-quality 可选依赖"
        ) from exc
    kwargs: dict[str, Any] = {"path": proxy.path, "split": proxy.split}
    if proxy.name:
        kwargs["name"] = proxy.name
    dataset = load_dataset(**kwargs)
    return _sample_dataset_conversations(
        dataset,
        sample_size=sample_size,
        seed=seed,
        user_field=proxy.user_field,
        assistant_field=proxy.assistant_field,
    )


def _sample_dataset_conversations(
    dataset: Any,
    *,
    sample_size: int,
    seed: int,
    user_field: str,
    assistant_field: str,
) -> list[Conversation]:
    """Deterministically sample a map-style datasets.Dataset without network access."""
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    if len(indices) > sample_size:
        indices = rng.sample(indices, sample_size)
    messages: list[Conversation] = []
    for index in indices:
        record = dataset[index]
        if isinstance(record, dict):
            conversation = _conversation(record, user_field, assistant_field)
            if conversation is not None:
                messages.append(conversation)
    return messages
