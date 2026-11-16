"""去污染：输出数据与用户提供的参考集之间的 n-gram 重合检测。

去污染属于确定性硬指标，不能只依赖 prompt 约束。ReviewAgent 会测量参考集重合率；超过配置
阈值的候选直接判为不合格，并把命中证据反馈给下一轮 PipelineAgent。

判定口径沿用主流去污染做法：**归一化后的 13-gram 精确重合**。
- 归一化（小写、去标点、压空白）：避免"改个标点就绕过去"。
- n-gram 而不是整句：改写过的题目仍会保留长片段。
- 短文本（token 数 < n）无法构成 n-gram，退回整串精确匹配，否则短题目永远检不出来。

宁可漏报不要误报：n-gram 命中就判污染会有假阳（常见套话、同一道经典题的公开变体），但阈值
默认给到"允许极低比例"，而不是零容忍——否则正常数据会被噪声判死。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from rsi.framework.evolution.utils.logging import get_logger

logger = get_logger("core.decontamination")

DEFAULT_NGRAM = 13
DEFAULT_MAX_RATE = 0.02      # 允许的污染率上限；超过即判不通过
MAX_EXAMPLES = 3             # 报告里带几条命中样例（给 Agent 定位用）

_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")
# 中日韩表意文字与假名/谚文：这些语言不用空格分词，需按字切。
_CJK_CHAR = re.compile(r"([\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af])")


def normalize(text: str) -> str:
    """小写 + 去标点 + 压空白。让"改标点/改空格"这类表面变形绕不过去。"""
    return _SPACES.sub(" ", _NON_WORD.sub(" ", text.lower())).strip()


def _tokens(text: str) -> list[str]:
    """拉丁文按词切、CJK 按字切，中英混排也稳定。

    CJK 没有词间空格，纯按空白切会把整句变成一个 token，n-gram 恒为空——中文数据集将完全
    检不出泄露。这里先给每个 CJK 字符两侧补空格再统一按空白切，因此同一段文本无论出现在
    基准侧还是生成侧、无论周围有没有其它字段拼接，切出来的 token 序列都一致。
    英文按词、中文按字使 n=13 的含义不同（13 词 vs 13 字），但都落在"长到不会偶然撞上"
    的量级，作为泄露信号够用。
    """
    norm = normalize(text)
    if not norm:
        return []
    return _CJK_CHAR.sub(r" \1 ", norm).split()


def _ngrams(tokens: list[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class ContaminationReport:
    """一次扫描的结果。`rate` 是对外口径，其余供日志与 Agent 定位。"""

    rate: float = 0.0
    hits: int = 0
    total: int = 0
    examples: list[str] = field(default_factory=list)


class ContaminationIndex:
    """基准测试集的 n-gram 索引；`enabled=False` 时所有扫描直接返回干净。"""

    def __init__(self, n: int = DEFAULT_NGRAM, max_rate: float = DEFAULT_MAX_RATE) -> None:
        self.n = max(2, int(n))
        self.max_rate = float(max_rate)
        self._ngrams: set[str] = set()
        self._exact: set[str] = set()          # 短文本（不足 n 个 token）的整串匹配兜底
        self.sources: list[str] = []

    @property
    def enabled(self) -> bool:
        return bool(self._ngrams or self._exact)

    def __len__(self) -> int:
        return len(self._ngrams) + len(self._exact)

    # ---- 建索引 ----
    def add_text(self, text: str) -> None:
        toks = _tokens(text)
        if not toks:
            return
        grams = _ngrams(toks, self.n)
        if grams:
            self._ngrams |= grams
        else:
            self._exact.add(" ".join(toks))

    def add_file(self, path: str | Path, fields: list[str] | None = None) -> int:
        """把一个基准文件加进索引，返回吸收的文本条数。

        读不到就告警跳过而不是抛：去污染索引缺一个文件应当以"警告 + 少覆盖一部分"收场，
        不该让整轮搜索起不来；但完全没建起索引会在 `enabled` 上体现，调用方能察觉。
        """
        p = Path(path).expanduser()
        if not p.is_file():
            logger.warning("去污染参考文件不存在，已跳过：%s", p)
            return 0
        count = 0
        try:
            for text in _iter_texts(p, fields):
                self.add_text(text)
                count += 1
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("读取去污染参考文件 %s 失败（已跳过）：%s", p, exc)
            return count
        self.sources.append(str(p))
        logger.info("去污染索引：从 %s 吸收 %d 条文本（累计 %d 个 %d-gram）",
                    p.name, count, len(self), self.n)
        return count

    # ---- 扫描 ----
    def scan(self, dataset_path: str | Path, fields: list[str] | None = None) -> ContaminationReport:
        """逐行判断生成数据是否与基准重合，返回污染率。"""
        report = ContaminationReport()
        if not self.enabled:
            return report
        p = Path(dataset_path)
        if not p.is_file():
            return report
        examples: list[str] = []
        try:
            with p.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    report.total += 1
                    row = _loads(line)
                    if row is None:
                        continue
                    hit = self._hit(_row_text(row, fields))
                    if hit:
                        report.hits += 1
                        if len(examples) < MAX_EXAMPLES:
                            examples.append(hit)
        except OSError as exc:
            logger.warning("扫描去污染失败（视为未检出）：%s", exc)
            return ContaminationReport()
        report.examples = examples
        report.rate = report.hits / report.total if report.total else 0.0
        if report.hits:
            logger.warning(
                "检出与基准测试集重合：%d/%d（%.2f%%），样例片段：%s",
                report.hits, report.total, report.rate * 100, examples[:1],
            )
        return report

    def _hit(self, text: str) -> str | None:
        """返回命中的片段（供定位），未命中返回 None。"""
        toks = _tokens(text)
        if not toks:
            return None
        grams = _ngrams(toks, self.n)
        if grams:
            for g in grams:
                if g in self._ngrams:
                    return g
            return None
        joined = " ".join(toks)
        return joined if joined in self._exact else None


def build(
    reference_files: list[str] | None,
    n: int = DEFAULT_NGRAM,
    max_rate: float = DEFAULT_MAX_RATE,
    fields: list[str] | None = None,
) -> ContaminationIndex:
    """按配置建索引；没给参考文件就是一个 disabled 的空索引（扫描恒返回干净）。"""
    index = ContaminationIndex(n=n, max_rate=max_rate)
    for path in reference_files or []:
        index.add_file(path, fields)
    if reference_files and not index.enabled:
        logger.warning("配置了去污染参考文件但索引为空，泄露检测本轮不会生效。")
    return index


# ---- 文本抽取 --------------------------------------------------------------
def _iter_texts(path: Path, fields: list[str] | None):
    # utf-8-sig：基准文件常带 BOM（如 GSM8K test），按 utf-8 读会让首行变成 "\ufeff{...}"
    # 而解析失败——首题被静默漏掉，恰恰是最该建进索引的那一条。无 BOM 时行为与 utf-8 相同。
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                row = _loads(line)
                if isinstance(row, dict):
                    yield _row_text(row, fields)
                elif isinstance(row, str):
                    yield row
    elif suffix == ".json":
        data = _loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if isinstance(row, dict):
                yield _row_text(row, fields)
            elif isinstance(row, str):
                yield row
    else:                                   # 纯文本：整行作为一条
        with path.open("r", encoding="utf-8-sig", errors="replace") as f:
            for line in f:
                if line.strip():
                    yield line


def _row_text(row: dict, fields: list[str] | None) -> str:
    """把一行拼成待比对文本。

    默认取全部字符串字段：污染既可能在题干也可能在答案里，而字段名在生成数据与基准之间
    往往对不上（question/instruction/problem…），按名字挑反而容易漏。
    """
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from strings(nested)

    values = [row.get(k) for k in fields] if fields else row.values()
    picked = [text for value in values for text in strings(value)]
    return " ".join(p for p in picked if p)


def _loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
