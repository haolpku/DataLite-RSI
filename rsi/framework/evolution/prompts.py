"""Prompts for the fixed-corpus PipelineAgent and candidate feedback loop."""
from __future__ import annotations

import json

from rsi.framework.evolution.evaluation.downstream_eval import DownstreamEvaluationResult
from rsi.framework.evolution.corpus import InputCorpus
from rsi.framework.evolution.models import PipelineConfig, ReviewResult, TaskSpec

_BAD_CASE_RESPONSE_HEAD_CHARS = 1200
_BAD_CASE_RESPONSE_TAIL_CHARS = 1800
_RUNAWAY_RESPONSE_HEAD_CHARS = 500
_RUNAWAY_RESPONSE_TAIL_CHARS = 500
_SELFCORRECT_PROMPT_MAX_CHARS = 900_000
_SELFCORRECT_TRACEBACK_MAX_CHARS = 32 * 1024
_SELFCORRECT_DIAGNOSTICS_MAX_CHARS = 64 * 1024


def _bounded_repair_section(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    marker = f"\n...[section truncated to {limit} chars; use the log paths for full context]...\n"
    budget = max(0, limit - len(marker))
    head = budget // 4
    return text[:head] + marker + text[-(budget - head):]


_PIPELINE_RUNTIME_BOUNDARY = """
【最高优先级运行边界】
先加载并遵循 `$dataflow-evolver-pipeline`。生成代码从 `rsi.framework` 导入
`OperatorABC`、`PipelineABC`、`FileStorage`、`PipelineLLMServing`；算子由本任务
本地生成，compile 预检、执行、产物发现、缓存与验收由框架提供。不得导入
`dataflow` 包（未安装，且框架会拒绝该 artifact）。生成代码与 Serving 契约均以该
skill 为准。
"""

PIPELINE_PROMPT = _PIPELINE_RUNTIME_BOUNDARY + "\n\n" + """为下述目标编写一条完整、可直接运行的 DataFlow 流水线（含自定义算子）。

Target: {target}
Expected outputs: {expected_outputs}
Entry file: {entry_path}
{input_block}
{modality_block}

核心要求：
1. 这是固定输入记录加工/增强任务。将上面的 Entry file 作为 `FileStorage` 的
   `first_entry_file_name` 固定输入（经 `FrameworkStepCache.from_env` 注入）。
   框架验证父轮 step manifest 并选择可复用前缀；未命中时从固定 raw data 开始。
   流水线不能绕开输入另起一套纯生成数据。
   在设计流水线之前，必须先对 Entry file 做只读的输入审查。检查来源分布、schema、模态引用、
   角色顺序、轮次数、语言、长度、占位符和重复等任务相关字段；不得把全量数据加载进内存或 Agent 上下文，也不得只依据 Prompt 中的前三行
   预览、当前 iteration 目录为空或主观猜测直接设计。将实际检查到的关键证据和对应设计取舍写入
   `decision.json.reason`。
2. 不允许联网拉取外部数据，包括 `load_dataset`、`hf_hub_download`、`requests`/curl，以及
   远程存储入口。调用配置好的 LLM serving API 不受此限。
3. 目标 schema：{target_schema}。最终 JSONL 至少包含这些字段且非空；可以保留额外的中间字段
   和溯源信息。注意 `storage.read("dict")` 的取值类型由 pandas 推断决定：含 null 的整数列
   会读成 float，纯数字字符串会读成数字。请按字段实际取值校验，不要假定其仍是源文件中的 JSON 类型。
4. 每个处理步骤放在 `{iteration_dir}/operators/` 下的独立自定义算子文件中，主文件写到
   `{iteration_dir}/pipeline.py`；算子继承 `rsi.framework.OperatorABC` 并实现
   `run(self, storage)`，通过 `storage.read("dict")` / `storage.write(rows)` 读写每个 step。
5. 同步写 `{iteration_dir}/decision.json`，schema 必须为：
   {{"ops": [{{"name": "算子类名", "purpose": "作用", "params": {{}}}}, ...],
     "field_flow": "raw fields -> ... -> target fields", "reason": "设计理由"}}
   `ops[].name` 必须与 `OPERATOR_NAMES` 及算子 Python 类名一致、顺序一致。
6. 以下配置声明本轮可用的 LLM Serving 能力，你需要根据任务与反馈设计是否需要调用 LLM、在哪个步骤调用 LLM，例如你可以使用 LLM 来做数据清洗、质量控制、数据转换与增强等等；若使用 LLM，
   必须在 Pipeline 层按专用 skill 构造并注入 `PipelineLLMServing`，且不得根据模型名称自行推断响应模式：
{llm_block}
   如果流水线包含 LLM 算子，也请把该算子的 prompt 作为可调整的设计部分之一：根据 Review 和下游归因适度改进任务说明、上下文组织和输出约束。prompt 应保持任务级、可复用、可审计，不得针对单个 benchmark 题目或答案硬编码。
7. `FileStorage` 把每个算子产物按 `pipeline_step_step{{N}}.jsonl` 写入
   `{iteration_dir}/cache`；正式运行产生的最后一个 step JSONL 是本轮最终数据集产物。
8. 主入口必须使用专用 skill 中唯一的
   `pipeline.compile()` → `DF_COMPILE_ONLY` 守卫 → `pipeline.forward()` 契约。
   只写入所需产物，不要自行执行 `pipeline.py`（包括 compile-only 或 smoke test）；框架会在
   Agent 返回后依次执行确定性的 artifact 校验、compile 预检和正式运行。
9. 以解决当前任务评估与下游归因报告指出的问题为首要目标，不以最少代码、文件或算子为目标，也不为节省调用成本而牺牲数据质量；LLM 调用范围应根据任务与反馈合理设计。
   根据职责边界和反馈自由增加、删除、拆分或合并算子；不要为了缓存复用而保留不合理的父流水线结构。
{init_block}{parent_block}{review_block}{execution_block}{diagnostic_block}{downstream_block}{history_block}
约束：无头自动化运行，禁止 AskUserQuestion 与 git/gh 子流程，不能获取新数据；所有实现
依据均来自专用 skill、本轮 Prompt、明确提供的父轮上下文、对固定 Entry file 的只读审查。
当前候选是进化循环中的一轮新输出，目录 `{iteration_dir}` 是本轮 pipeline 的工作目录；父轮来源
以显式的“父流水线”block 为准，不要根据历史编号自行选择父轮。"""


DECISION_SYNC_NOTE = """如果修复改变了算子、字段流或设计取舍，请同步覆盖
`{iteration_dir}/decision.json`（键名仍为 `ops` / `field_flow` / `reason`）。"""


PIPELINE_SELFCORRECT_PROMPT = _PIPELINE_RUNTIME_BOUNDARY + "\n\n" + """流水线 `{iteration_dir}/pipeline.py` 未通过框架检查。
失败阶段：`{failure_stage}`。

只依据专用 skill、错误栈、逐步诊断、下面列出的完整子进程日志（仅按需读取），以及当前 iteration
的 `pipeline.py` 和 `operators/` 定位问题。

只修改 `pipeline.py`、`operators/` 和必要的 `decision.json`，不要自行执行 `pipeline.py`、
compile-only 或 smoke test。Agent 返回后，框架会重新执行 artifact 校验、公共 Pipeline compile 预检与正式运行。
{decision_sync_note}

错误栈：
```
{traceback}
```

逐步产物诊断：
```
{diagnostics}
```

按失败阶段定位：`artifact_validation` 优先修正语法及
`compile()` → `DF_COMPILE_ONLY` 守卫 → `forward()` 的主入口契约；
`compile` 优先检查 import、算子初始化、参数和字段流；`execution` / `output_validation` 则先找
第一个行数变为 0、字段丢失或 JSONL 异常的步骤，修复产生问题的上游算子，不要只在下游吞掉异常。

只修复执行问题，不改变任务目标，不联网取数，不进行交互或 git/gh 操作。"""


FAILURE_SIGNALS: dict[str, str] = {
    "F1": "抽样描述过窄：可见样本围绕极少数结构展开",
    "F2": "抽样华而不实：表面变化很多但可见逻辑结构高度一致",
    "F3": "抽样模板锁定：可见样本存在近重复或模板化雷同",
    "F4": "抽样覆盖不足：可见样本缺少目标任务的关键能力维度",
    "F5": "约束漂移：偏离目标任务、领域或答案格式",
    "F6": "有形无实：形式满足但训练价值低",
    "F7": "描述空泛：内容语义松散或逐渐跑题",
}

_DIM_LABELS: tuple[tuple[str, str], ...] = (
    ("correctness_score", "正确性"),
    ("relevance_score", "相关性"),
    ("difficulty_score", "难度"),
    ("schema_score", "schema 符合度"),
)


REVIEW_PROMPT = """你是数据质量评审员。下面是候选数据集的有界随机抽样。请评估该抽样中
可直接观察到的微调质量，每个维度取 0.0-1.0。F1-F4 只能在抽样本身提供重复、模板化或覆盖不足
的直接证据时报告，不能外推整个数据集；全量多样性由框架使用 dataset embedding 单独评估。

目标任务：{task_description}
期望 schema：{target_schema}
质量要点：{quality_criteria}

数据抽样（每行一条 JSON）：
{samples}

评分维度：
- correctness_score：输出内容、标签或解答满足任务要求；如果包含解释或推理，内容应保持自洽。
- relevance_score：与目标任务和领域相关。
- difficulty_score：对目标任务具有适当的信息量、覆盖度和可学习挑战。
- schema_score：必需字段齐全、非空且类型正确；额外元数据不扣分。

失败信号：
{failure_signal_block}

只输出以下 JSON object：
{{
  "correctness_score": 0.0,
  "relevance_score": 0.0,
  "difficulty_score": 0.0,
  "schema_score": 0.0,
  "failure_signals": ["F3"],
  "issues": ["具体问题及样本证据"]
}}
没有失败信号时使用空数组。不要输出 JSON 以外的内容。"""


def build_pipeline_diagnostic_prompt(
    *,
    iteration_dir: str,
    task: TaskSpec,
    observation_path: str | None,
    source_files: list[str],
    config: PipelineConfig,
    review: ReviewResult,
    max_prompt_chars: int = 80_000,
) -> str:
    """Build the non-scoring, read-only Call B prompt."""
    review_context = {
        "correctness_score": review.correctness_score,
        "relevance_score": review.relevance_score,
        "difficulty_score": review.difficulty_score,
        "schema_score": review.schema_score,
        "failure_signals": list(review.failure_signals),
        "issues": list(review.issues),
        "hard_metrics": {
            "dup_rate": review.dup_rate,
            "null_rate": review.null_rate,
            "parse_valid_rate": review.parse_valid_rate,
            "contamination_rate": review.contamination_rate,
        },
    }
    declared_operators = [
        item for item in config.operators[:20] if isinstance(item, dict)
    ]
    prompt = (
        "你是只读的流水线执行诊断 Agent。\n\n"
        "请判断已执行的流水线是否符合其声明的设计，包括算子职责、筛选规则和配额。"
        "以框架生成的 execution_observation.json 作为主要执行证据，"
        "必要时再只读检查列出的 pipeline.py、operators/ 和 decision.json。\n"
        "请逐步比较实际行数、保留率、字段、低基数字段分布和有界样本，定位声明与实际结果"
        "首次出现偏差的算子。\n"
        "不要修改、创建、删除或执行流水线文件，也不要读取完整输入或输出 JSONL；"
        "只使用 "
        "execution_observation.json 中的有界样本。\n"
        "这只是机制性诊断：不要给数据质量打分、预测 benchmark 提升，"
        "也不要为了影响 Review 而提出建议。\n\n"
        "严格只返回一个 JSON object，格式如下（键名保持不变）：\n"
        '{"status":"ok","summary":"...","findings":[{"step":"...",'
        '"severity":"low|medium|high","evidence":"...","diagnosis":"...",'
        '"recommended_action":"..."}],"contract_mismatches":[]}\n\n'
        f"当前 iteration 目录：{iteration_dir}\n"
        f"执行观测文件：{observation_path or '缺失'}\n"
        "可只读检查的产物：\n"
        + "\n".join(f"- {path}" for path in source_files)
        + "\n\n任务：\n"
        + str(task.task_description)
        + "\n声明的算子：\n"
        + json.dumps(declared_operators, ensure_ascii=False, default=str)
        + "\nReview 证据（仅作上下文，不是评分目标）：\n"
        + json.dumps(review_context, ensure_ascii=False, default=str)
        + "\n"
    )
    limit = max(1000, int(max_prompt_chars))
    if len(prompt) > limit:
        marker = "\n[诊断提示已由框架截断]\n"
        prompt = prompt[: max(0, limit - len(marker))] + marker
    return prompt

def build_pipeline_prompt(
    task: TaskSpec,
    corpus: InputCorpus,
    iteration_dir: str,
    llm_block: str = "",
    parent_code: str | None = None,
    parent_rationale: str | None = None,
    parent_iteration_dir: str | None = None,
    parent_review: ReviewResult | None = None,
    execution_observation_path: str | None = None,
    pipeline_diagnostic: object | None = None,
    downstream_feedback: DownstreamEvaluationResult | None = None,
    evolution_history: list[dict] | None = None,
    is_init: bool = False,
) -> str:
    if not corpus.grounded:
        raise ValueError("轻量 Pipeline prompt 要求固定 raw corpus")
    return PIPELINE_PROMPT.format(
        target=task.task_description,
        expected_outputs=list(task.target_schema),
        entry_path=corpus.entry_path,
        input_block=corpus.prompt_block(),
        modality_block=(
            "【输入契约】" + json.dumps(task.input_contract.to_dict(), ensure_ascii=False)
            + "。图像/视频字段是 artifact reference，不能把二进制内容或本地路径直接写进生成代码；"
            "同一记录可同时含文本、图像和视频。评估由当前任务注入的 evaluator 完成。\n"
            if task.input_contract else ""
        ),
        target_schema=task.target_schema,
        iteration_dir=iteration_dir,
        llm_block=llm_block or "   使用项目已有的默认 serving 配置。",
        init_block=(
            "【初始流水线】本轮没有父流水线。当前 iteration 目录是新流水线的输出位置，"
            "先完成上面要求的有界输入审查，再从固定 raw data 设计稳定、可解释的处理基线。\n"
            if is_init else ""
        ),
        parent_block=_parent_block(
            parent_code,
            parent_rationale,
            iteration_dir,
            parent_iteration_dir,
        ),
        review_block=(
            _parent_quality_block(parent_review, task) if parent_review is not None else ""
        ),
        execution_block=_execution_observation_block(execution_observation_path),
        diagnostic_block=_pipeline_diagnostic_block(pipeline_diagnostic),
        downstream_block=_downstream_feedback_block(downstream_feedback),
        history_block=_history_block(evolution_history),
    )

def build_selfcorrect_prompt(
    iteration_dir: str,
    traceback: str,
    diagnostics: str = "not available",
    failure_stage: str = "execution",
    log_paths: dict[str, str] | None = None,
) -> str:
    logs = ""
    if log_paths:
        logs = "\n完整子进程日志（仅按需读取，不要默认全文加载）：\n" + "\n".join(
            f"- {name}: `{path}`" for name, path in sorted(log_paths.items())
        )
    prompt = PIPELINE_SELFCORRECT_PROMPT.format(
        iteration_dir=iteration_dir,
        failure_stage=failure_stage,
        traceback=_bounded_repair_section(traceback, _SELFCORRECT_TRACEBACK_MAX_CHARS),
        diagnostics=_bounded_repair_section(diagnostics, _SELFCORRECT_DIAGNOSTICS_MAX_CHARS),
        decision_sync_note=DECISION_SYNC_NOTE.format(iteration_dir=iteration_dir),
    ) + logs
    if len(prompt) <= _SELFCORRECT_PROMPT_MAX_CHARS:
        return prompt
    marker = "\n...[repair context truncated; inspect the complete logs above]...\n"
    return prompt[:_SELFCORRECT_PROMPT_MAX_CHARS - len(marker)] + marker


def build_review_prompt(task: TaskSpec, samples_text: str) -> str:
    signals = "\n".join(f"- {key}: {value}" for key, value in FAILURE_SIGNALS.items())
    return REVIEW_PROMPT.format(
        task_description=task.task_description,
        target_schema=task.target_schema,
        quality_criteria=task.quality_criteria,
        samples=samples_text,
        failure_signal_block=signals,
    )


def _parent_block(
    code: str | None,
    rationale: str | None,
    iteration_dir: str,
    parent_iteration_dir: str | None = None,
) -> str:
    if not code and not rationale:
        return ""
    lines = ["【父流水线（当前最佳 incumbent）】"]
    if parent_iteration_dir:
        lines.append(
            f"父轮 incumbent 目录：`{parent_iteration_dir}`。这是本轮唯一允许继承的父轮目录；"
            "不要按 iteration 编号猜测父轮。"
        )
    if code:
        lines.append(
            "下面是本轮实际继承的父流水线，仅作为可复用的参考，不限制本轮设计空间。根据任务与反馈自由设计，并可按需保留、替换、拆分、合并、增加或删除算子；不要为了缓存复用而维持原结构：\n"
            f"```python\n{code}\n```"
        )
    if rationale:
        lines.append(f"父流水线设计理由：{rationale}")
    if code:
        lines.append(
            "【Framework-owned 父轮复用】只判断哪些连续前缀算子在语义上保持不变，并复制这些算子"
            f"的本地源码到当前 `{iteration_dir}/operators/`。不要读取、选择、验证或硬编码父轮 cache "
            "JSONL；不要通过 `pipeline_step_stepN` 推断 producer，也不要自行实现 cache hit/miss。\n"
            "每条 Pipeline 都必须使用 skill 中的 `compile()` → `DF_COMPILE_ONLY` 守卫 → "
            "`forward()` 标准入口。框架会在"
            "执行前比较 raw input、逻辑算子顺序、算子名、`decision.json.params` 与初始化调用的语义"
            "身份，并从"
            "父轮 `step_manifest.json` 注入经过验证的 entry path 和 prefix count；无法证明时自动从 raw "
            "input 完整重算。`forward()` 只对框架确认后的后缀调用 `should_run(i)` 为真的算子。"
            "含 run 内部 blob 引用的 step 不能跨 iteration 复用，因为其相对路径属于原 run。\n"
            "在 `decision.json.reason` 只记录哪些前缀算子未变化以及首个变化算子；缓存 artifact、step "
            "编号、lineage 与命中结论全部由框架记录，Agent 不得自行决定。框架会比较本地算子类、"
            "共享 helper 源码、算子构造调用、params 和父产物哈希。外部服务、模型或包的行为若变化，"
            "仍需更新算子版本或配置以主动使缓存失效。"
        )
    return "\n".join(lines) + "\n"


def _parent_quality_block(review: ReviewResult, task: TaskSpec) -> str:
    if task.evaluator_kind != "review_agent":
        lines = [
            "【当前最佳 incumbent 的任务评估反馈——本轮必须直接回应】",
            f"候选得分：{review.review_score:.3f}；评估{'通过' if review.passed else '未通过'}",
        ]
        if review.issues:
            lines.append("具体问题：")
            lines.extend(f"- {issue}" for issue in review.issues)
        if review.domain_feedback:
            lines.append(
                "领域反馈："
                + json.dumps(review.domain_feedback, ensure_ascii=False, default=str)[:12000]
            )
        lines.append("只依据当前任务的评估证据修改相应算子，并在 decision.json.reason 中说明。")
        return "\n".join(lines) + "\n"
    dims = [
        (label, float(getattr(review, attr, 0.0) or 0.0))
        for attr, label in _DIM_LABELS
    ]
    lines = [
        "【当前最佳 incumbent 的候选评估反馈——本轮必须直接回应】",
        f"Review：{review.review_score:.3f}；评审{'通过' if review.passed else '未通过'}",
        "抽样四维：" + " | ".join(f"{label} {score:.2f}" for label, score in dims),
        f"全量硬指标：重复率 {review.dup_rate:.2f} | 空值率 {review.null_rate:.2f} | "
        f"可解析率 {review.parse_valid_rate:.2f} | 污染率 {review.contamination_rate:.2f}",
    ]
    dataset_quality = review.dataset_quality
    if dataset_quality.enabled:
        if dataset_quality.status == "ok" and dataset_quality.mmd is not None:
            delta_mmd = (
                f"{dataset_quality.delta_mmd_vs_parent:+.6f}"
                if dataset_quality.delta_mmd_vs_parent is not None
                else "baseline"
            )
            diversity = dataset_quality.embedding_diversity
            if diversity:
                delta_diversity = dataset_quality.delta_embedding_diversity_vs_parent
                delta_vendi = (
                    f"{delta_diversity['cosine_vendi_ratio']:+.6f}"
                    if "cosine_vendi_ratio" in delta_diversity
                    else "baseline"
                )
                delta_p95 = (
                    f"{delta_diversity['nearest_neighbor_cosine_p95']:+.6f}"
                    if "nearest_neighbor_cosine_p95" in delta_diversity
                    else "baseline"
                )
                lines.append(
                    f"Dataset embedding：ΔMMD {delta_mmd} | "
                    f"normalized Vendi {diversity.get('cosine_vendi_ratio', 0.0):.6f} "
                    f"(Δ {delta_vendi}) | "
                    f"NN cosine p95 {diversity.get('nearest_neighbor_cosine_p95', 0.0):.6f} "
                    f"(Δ {delta_p95})"
                )
                lines.append(
                    "ΔMMD 越负越好；normalized Vendi 越高越好；NN cosine p95 越低越好。"
                )
        else:
            lines.append(
                f"Dataset-level DAS：状态 {dataset_quality.status}；"
                f"原因：{dataset_quality.error or '未知'}。"
                "请优先保证输出至少含足够数量且字段可编码的有效记录。"
            )
    if review.issues:
        lines.append("具体问题：")
        lines.extend(f"- {issue}" for issue in review.issues)
    if review.domain_feedback:
        lines.append(
            "领域反馈（由当前任务的 evaluator 提供）："
            + json.dumps(review.domain_feedback, ensure_ascii=False, default=str)[:12000]
        )
    lines.append(
        "优先修改能够解释上述反馈的步骤，同时保留无关且有效的行为；若证据要求，可增加、拆分、"
        "合并或替换算子。请在 decision.json 的 reason 中说明反馈、改动与预期作用的对应关系。"
    )
    return "\n".join(lines) + "\n"


def _history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    visible = history[-12:]
    lines = [
        "【简要进化历史——不要重复已拒绝尝试的同类策略】",
        "已接受历史只含摘要；被拒历史若附带 observation/diagnostic，仅用于避免重复失败，不是可继承的 pipeline。",
        "当前父流水线的执行观测和诊断只看上方独立的父流水线 block。",
    ]
    if len(history) > len(visible):
        lines.append(f"- 更早的 {len(history) - len(visible)} 次尝试仅保存在运行记录中。")
    for item in visible:
        score = item.get("review_score")
        score_text = f"{float(score):.3f}" if score is not None else "无"
        operators = ">".join(str(name) for name in item.get("operators", [])) or "未知"
        issues = "；".join(str(issue) for issue in item.get("issues", [])) or "无"
        outcome = str(item.get("outcome") or "未知")
        outcome = {"accepted": "已接受", "rejected": "已拒绝"}.get(outcome, outcome)
        lines.append(
            f"- 第 {item.get('iteration')} 轮 [{outcome}]，评审分数={score_text}，"
            f"算子={operators}，decision={item.get('decision') or '无'}，问题={issues}"
        )
        observation = item.get("execution_observation")
        diagnostic = item.get("pipeline_diagnostic")
        if item.get("outcome") == "rejected" and observation:
            lines.append(f"  被拒尝试执行观测（仅用于避免重复失败）：{observation}")
        if item.get("outcome") == "rejected" and diagnostic:
            lines.append(
                "  被拒尝试执行诊断（仅用于避免重复失败，不参与评分）："
                + json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"))
            )
    return "\n".join(lines) + "\n"


def _execution_observation_block(path: str | None) -> str:
    if not path:
        return ""
    return (
        "【父流水线执行观测（当前最佳 incumbent）】框架已经完成本轮实际继承的父流水线。"
        "以下是确定性执行证据，不是 Review 分数。修改算子前请检查各步骤的实际行数、"
        "保留率、字段、低基数字段分布和有界样本；必要时读取完整文件：\n"
        + "- execution_observation.json：" + path + "\n"
    )


def _pipeline_diagnostic_block(diagnostic: object | None) -> str:
    if not diagnostic:
        return ""
    if isinstance(diagnostic, dict):
        payload = json.dumps(diagnostic, ensure_ascii=False, indent=2)
    else:
        payload = str(diagnostic)
    if len(payload) > 12000:
        payload = payload[:11950] + "\n[诊断内容已由框架截断]"
    return (
        "【父流水线执行诊断（当前最佳 incumbent）】这是独立诊断 Agent 给出的机制性意见。"
        "它不影响 review_score、passed 或 incumbent 选择。"
        "采取行动前请结合 execution_observation 和产物自行核实：\n" + payload + "\n"
    )


def _downstream_feedback_block(
    feedback: DownstreamEvaluationResult | None,
) -> str:
    if feedback is None:
        return ""
    if not feedback.succeeded:
        return (
            "【下游 SFT/评测检查点】外部评测未成功，因此本轮没有可用的下游质量信号。\n"
            f"失败原因：{feedback.error or 'unknown error'}。不要根据基础设施失败臆测数据问题。\n"
        )

    if feedback.stage == "baseline":
        title = "【第 0 轮下游模型诊断——初始流水线必须分析并回应】"
        checkpoint = (
            "检查点：迭代开始前；未进行 SFT；"
            f"下游模型：{feedback.model}"
        )
    else:
        title = "【当前最佳 incumbent 的下游 SFT/评测反馈——本轮必须分析并回应】"
        checkpoint = (
            f"检查点：完成第 {feedback.checkpoint_iteration} 轮后；"
            f"被评测 incumbent：第 {feedback.incumbent_iteration} 轮；"
            f"下游模型：{feedback.model}"
        )

    lines = [
        title,
        checkpoint,
        "指标含义：delta_previous 是当前 checkpoint 相对上一个成功下游 checkpoint 的分数变化；"
        "delta_baseline 是相对 checkpoint 000（未训练 backbone）的分数变化；正数表示提升，"
        "负数表示下降，baseline 阶段因没有对照而显示 unknown。",
        "下面只提供聚合指标和 Downstream Attribution Agent 生成的结构化归因报告。",
        "归因报告中的 Evidence ID 只是不可解析的审计引用，不代表可访问的数据源。"
        "只能归纳可复用的能力缺口并改进 raw-data pipeline。",
    ]
    attribution = feedback.attribution
    for benchmark_name, benchmark in feedback.benchmarks.items():
        delta = feedback.score_deltas.get(benchmark_name)
        baseline_delta = feedback.score_deltas_vs_baseline.get(benchmark_name)
        lines.append(
            f"\n### {benchmark_name}: {benchmark.metric}={benchmark.score:.6f}; "
            f"delta_previous={delta if delta is not None else 'unknown'}; "
            f"delta_baseline={baseline_delta if baseline_delta is not None else 'unknown'}; "
            f"correct_count={benchmark.correct_count}; total_count={benchmark.total_count}; "
            f"question_count={benchmark.question_count}; n_sampling={benchmark.n_sampling}; "
            f"incorrect={benchmark.incorrect_count}; wrong={benchmark.wrong_answer_count}; "
            f"parse_failure={benchmark.parse_failure_count}; runaway={benchmark.runaway_count}"
        )
    if feedback.summary:
        lines.extend(["", "外部评测摘要：", feedback.summary])
    if attribution is None or attribution.status != "ok":
        lines.extend(
            [
                "",
                "下游 bad case 归因报告不可用；本轮只保留 benchmark 统计，不向 PipelineAgent 暴露原始 bad case。",
                f"归因失败原因：{attribution.error if attribution is not None else 'not generated'}。",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(["", "下游 bad case 归因报告：", attribution.summary or "无总体结论。"])
    for index, finding in enumerate(attribution.findings, start=1):
        category = str(finding.get("category", "other"))
        confidence = finding.get("confidence", 0.0)
        is_data_side = bool(finding.get("data_side", True))
        data_side = "data-side" if is_data_side else "non-data"
        evidence_ids = ", ".join(str(item) for item in finding.get("evidence_ids", [])) or "none"
        recommended_actions = (
            "；".join(str(item) for item in finding.get("recommended_actions", []))
            if is_data_side
            else "[suppressed: non-data cause; do not modify pipeline]"
        )
        lines.extend(
            [
                f"Finding {index} [{category}; {data_side}; confidence={confidence}]",
                f"Evidence: {evidence_ids}",
                f"Diagnosis: {finding.get('diagnosis', '')}",
                "Recommended pipeline actions: " + recommended_actions,
            ]
        )
    if attribution.non_data_causes:
        lines.extend(["非数据原因（不要通过修改 pipeline 解决）：", *attribution.non_data_causes])
    if attribution.recommended_pipeline_changes:
        lines.extend(["归因器建议的 pipeline 改动：", *attribution.recommended_pipeline_changes])
    lines.append("只能依据归因报告中的跨案例模式改进 raw-data pipeline，不得针对单个 benchmark 题目硬编码。")
    return "\n".join(lines) + "\n"



def _project_bad_case_response(
    response: str,
    *,
    length_truncated: bool = False,
) -> tuple[str, int]:
    """Render bounded evidence while keeping the full response in result.json."""
    head_chars = (
        _RUNAWAY_RESPONSE_HEAD_CHARS
        if length_truncated
        else _BAD_CASE_RESPONSE_HEAD_CHARS
    )
    tail_chars = (
        _RUNAWAY_RESPONSE_TAIL_CHARS
        if length_truncated
        else _BAD_CASE_RESPONSE_TAIL_CHARS
    )
    limit = head_chars + tail_chars
    if len(response) <= limit:
        return response, 0
    omitted = len(response) - limit
    marker = (
        f"\n\n...[{omitted} chars omitted from prompt; full response retained for human audit only]...\n\n"
    )
    return (
        response[:head_chars]
        + marker
        + response[-tail_chars:],
        omitted,
    )

def _compact_prompt_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

def build_downstream_attribution_prompt(
    *,
    task: TaskSpec,
    feedback: DownstreamEvaluationResult,
    pipeline_operators: list[dict[str, object]],
    parent_review: ReviewResult | None = None,
    max_cases_per_benchmark: int = 3,
    max_findings: int = 6,
    max_actions: int = 8,
) -> str:
    """Build a structured, evidence-bound diagnosis prompt.

    This is deliberately separate from the PipelineAgent prompt: the
    attribution model sees bounded bad cases, while the generator sees only
    the resulting diagnosis.
    """
    operator_names = [
        str(item.get("name", "?"))
        for item in pipeline_operators[:12]
        if isinstance(item, dict)
    ]
    review_text = "无静态 Review 记录。"
    if parent_review is not None:
        review_text = (
            f"review_score={parent_review.review_score:.4f}; "
            f"correctness={parent_review.correctness_score:.3f}; "
            f"relevance={parent_review.relevance_score:.3f}; "
            f"difficulty={parent_review.difficulty_score:.3f}; "
            f"schema={parent_review.schema_score:.3f}; "
            f"issues={'；'.join(parent_review.issues[:4]) or 'none'}"
        )
    lines = [
        "你是下游评测归因器，负责把模型 benchmark 失败归因到“可由数据流水线改进的原因”。",
        "不要把单个 benchmark 题目当作训练数据，不要复述或硬编码题目/答案。",
        "必须区分 data_side=true（数据生成可修复）与 data_side=false（模型、解析器或基础设施问题）。",
        "优先从多个案例中归纳可复用模式，单个孤立案例可能需要降低 confidence。",
        "每条 finding 必须绑定 evidence_ids，并给出可执行的过滤、上下文、重写、覆盖或难度策略。",
        "目标任务：",
        task.task_description,
        f"目标 schema：{task.target_schema}",
        f"质量标准：{task.quality_criteria}",
        f"当前 pipeline operators：{'>'.join(operator_names) or 'unknown'}",
        f"当前静态 Review：{review_text}",
        f"下游检查点：stage={feedback.stage}, checkpoint={feedback.checkpoint_iteration}, "
        f"incumbent={feedback.incumbent_iteration}, model={feedback.model}",
    ]
    for benchmark_name, benchmark in feedback.benchmarks.items():
        delta = feedback.score_deltas.get(benchmark_name)
        baseline_delta = feedback.score_deltas_vs_baseline.get(benchmark_name)
        lines.append(
            f"\n### benchmark={benchmark_name}; metric={benchmark.metric}; "
            f"score={benchmark.score:.6f}; incorrect={benchmark.incorrect_count}; "
            f"wrong={benchmark.wrong_answer_count}; parse_failure={benchmark.parse_failure_count}; "
            f"runaway={benchmark.runaway_count}; "
            f"delta_previous={delta if delta is not None else 'unknown'}; "
            f"delta_baseline={baseline_delta if baseline_delta is not None else 'unknown'}"
        )
        for index, case in enumerate(
            benchmark.bad_cases[: max(0, max_cases_per_benchmark)], start=1
        ):
            is_runaway = case.length_truncated or case.finish_reason not in {"", "stop"}
            response, omitted = _project_bad_case_response(
                case.model_response,
                length_truncated=is_runaway,
            )
            case_id = f"{benchmark_name}#{index}"
            lines.extend(
                [
                    f"Evidence {case_id}:",
                    f"Dimension: {case.dimension or 'unspecified'}",
                    f"Question: {_compact_prompt_text(case.question, 1200)}",
                    (
                        "Response metadata: "
                        f"tokens={case.response_token_count if case.response_token_count is not None else 'unknown'}, "
                        f"finish_reason={case.finish_reason or 'unknown'}, "
                        f"length_truncated={str(case.length_truncated).lower()}, "
                        f"omitted_chars={omitted}"
                    ),
                    "Model response (bounded):",
                    response,
                    f"Parsed answer: {_compact_prompt_text(case.parsed_answer, 800) or '[empty]'}",
                    f"Correct answer: {_compact_prompt_text(case.correct_answer, 800)}",
                ]
            )
    example_benchmark = next(
        (name for name, benchmark in feedback.benchmarks.items() if benchmark.bad_cases),
        None,
    )
    example_evidence_ids = [f"{example_benchmark}#1"] if example_benchmark else []
    max_findings = max(1, int(max_findings))
    max_actions = max(1, int(max_actions))
    lines.extend(
        [
            "",
            "只输出 JSON object，不要输出 markdown：",
            json.dumps(
                {
                    "summary": "跨案例、面向数据流水线的总体结论",
                    "findings": [
                        {
                            "category": "missing_context",
                            "data_side": True,
                            "confidence": 0.0,
                            "evidence_ids": example_evidence_ids,
                            "diagnosis": "可复用的失败模式及其证据解释",
                            "recommended_actions": ["具体 pipeline 改动"],
                        }
                    ],
                    "non_data_causes": ["应避免误改数据的模型或基础设施原因"],
                    "recommended_pipeline_changes": ["按优先级排列的 1-3 个改动"],
                },
                ensure_ascii=False,
            ),
            f"JSON 字段约束：findings 最多 {max_findings} 条；每条 finding 的 recommended_actions 最多 {max_actions} 条；"
            "confidence 为 0-1；evidence_ids 必须引用上面已有的 Evidence；"
            "recommended_actions 只能描述数据流水线动作。若证据不足，返回空 findings 并说明原因。",
        ]
    )
    return "\n".join(lines) + "\n"
