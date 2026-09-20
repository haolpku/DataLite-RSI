from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.context import RunContext
from ..core.schema import (
    DeltaStep,
    Inputs,
    Sample,
    Sampling,
    TaskType,
)
from ..core.storage import DirStorage
from ..operators.filter import ThresholdFilter
from ..operators.generate import EditApply, GenBefore
from ..operators.recipe import SeedEditCompatibilityGate, VLMInstructionPlanner
from ..operators.verify import IFVCVQVerifier
from ..serving.base import ImageGenServingABC, LLMServingABC, VLMServingABC
from .campaign_spec import CampaignSpec
from .coverage import CoverageTracker
from .diagnose import FailureDiagnoser
from .ledger import RunLedger
from .overhead import OverheadLedger
from .planners import (
    DEFAULT_ENVIRONMENT_SNAPSHOT,
    BatchJob,
    BatchPlanner,
    PlannerServices,
    make_planner,
)

log = logging.getLogger("deltasynth.harness")


@dataclass(slots=True)
class BatchSummary:
    batch_id: str
    settings_id: str
    samples: int = 0
    goodcase: int = 0
    badcase: int = 0
    api_calls: int = 0
    failure_tags_top: list[tuple[str, int]] = field(default_factory=list)


CALLS_PER_EPISODE = 4


def _make_sample_id(
    *,
    run_tag: str,
    setting_id: str,
    batch_idx: int,
    job_idx: int,
    edit_code: str,
    scene_id: str,
) -> str:
    clean_scene = scene_id.replace("/", "_").replace(" ", "_")
    return (
        f"{run_tag}_{setting_id}_b{batch_idx:03d}_{edit_code}_"
        f"{clean_scene}_{job_idx:03d}"
    )


class AgenticLoop:

    def __init__(
        self,
        spec: CampaignSpec,
        storage: DirStorage,
        ctx: RunContext,
        *,
        llm_serving: LLMServingABC,
        image_serving: ImageGenServingABC,
        vlm_serving: VLMServingABC,
        rewriter_serving: Optional[LLMServingABC] = None,
        run_tag: Optional[str] = None,
        size: str = "1024x1024",
        edit_input_max_edge: int = 1024,
        environment_snapshot: Optional[Path] = None,
        extend_mode: bool = False,
        before_bank: Optional[Path] = None,
    ) -> None:
        self.spec = spec
        self.storage = storage
        self.ctx = ctx

        self.llm = llm_serving
        self.image = image_serving
        self.vlm = vlm_serving
        self.rewriter_llm = rewriter_serving or llm_serving

        self.run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.size = size
        self.edit_input_max_edge = edit_input_max_edge
        self.extend_mode = extend_mode

        snapshot = environment_snapshot or (
            Path(spec.environment_snapshot)
            if spec.environment_snapshot
            else DEFAULT_ENVIRONMENT_SNAPSHOT
        )
        self.environment_snapshot = snapshot
        self.planner: BatchPlanner = make_planner(
            spec,
            snapshot_path=snapshot,
            services=PlannerServices(
                storage_root=storage.root, llm=self.rewriter_llm
            ),
        )

        self._instruction_planner = VLMInstructionPlanner(
            self.vlm, environment_snapshot=snapshot
        )
        self._compatibility_gate = SeedEditCompatibilityGate()
        self._gen_before = GenBefore(
            self.image,
            before_bank=before_bank,
            generate_on_bank_miss=self.planner.needs_generated_before,
        )
        if before_bank:
            print(f"[harness] before-image bank: {before_bank}")
        self._edit_apply = EditApply(
            self.image, edit_input_max_edge=self.edit_input_max_edge
        )
        rubric_id = self.spec.verifier_stack[0].rubric_id
        if rubric_id != "if_vc_vq":
            raise ValueError(
                f"verifier_stack[0].rubric_id={rubric_id!r} is no longer "
                "supported; if_vc_vq is the only rubric."
            )
        self._verifier = IFVCVQVerifier(self.vlm)
        self._filter = ThresholdFilter(
            min_score=self.spec.goodcase_threshold.aggregate_score
        )
        self._diagnoser = FailureDiagnoser()

        self.ledger = RunLedger(storage)
        self.overhead = OverheadLedger(storage.root)
        self.coverage = CoverageTracker(
            spec.coverage_axes,
            target_per_slice=max(1, spec.budget.batch_size // 4),
        )

        self.batch_summaries: list[BatchSummary] = []


    def run(self) -> dict[str, Any]:
        log.info(
            "[harness] start campaign=%s setting=%s out=%s",
            self.spec.campaign_id,
            self.spec.setting_id,
            self.storage.root,
        )


        for batch_idx in range(self.spec.budget.max_batches):
            ledger_rows = self.ledger.scan(setting_id=self.spec.setting_id)
            goodcase_now = sum(1 for r in ledger_rows if r.is_goodcase)
            overhead = self.overhead.totals(setting_id=self.spec.setting_id)
            calls_used = RunLedger.total_api_calls(ledger_rows) + overhead.api_calls
            if (
                self.spec.budget.target_goodcase is not None
                and goodcase_now >= self.spec.budget.target_goodcase
            ):
                log.info("[harness] target_goodcase reached: %d", goodcase_now)
                break
            if calls_used >= self.spec.budget.max_api_calls:
                log.info("[harness] max_api_calls hit: %d", calls_used)
                break
            if self.planner.is_exhausted():
                log.info("[harness] planner exhausted at batch %d", batch_idx)
                break

            budget_remaining = max(
                0,
                (self.spec.budget.max_api_calls - calls_used) // CALLS_PER_EPISODE,
            )

            self.planner.prepare(self._instruction_planner)
            cov_report = self.coverage.update(ledger_rows)
            jobs = self.planner.plan(
                batch_id=batch_idx,
                ledger_rows=ledger_rows,
                coverage=cov_report,
                budget_remaining=budget_remaining,
            )
            if not jobs:
                log.info("[harness] planner returned 0 jobs at batch %d", batch_idx)
                break

            batch_id_str = f"b{batch_idx:03d}"
            new_ids = self._materialise_jobs(
                jobs, batch_idx=batch_idx, batch_id_str=batch_id_str
            )

            active_ids = self._run_pipeline_on(new_ids)

            if active_ids:
                self._diagnoser.run(self.storage, self.ctx, sample_ids=active_ids)

            batch_rows = [
                r for r in self.ledger.scan(setting_id=self.spec.setting_id)
                if r.batch_id == batch_id_str
            ]
            summary = BatchSummary(
                batch_id=batch_id_str,
                settings_id=self.spec.setting_id,
                samples=len(batch_rows),
                goodcase=sum(1 for r in batch_rows if r.is_goodcase),
                badcase=sum(1 for r in batch_rows if r.judgment == "BadCase"),
                api_calls=sum(r.api_calls for r in batch_rows),
                failure_tags_top=RunLedger.top_failure_tags(batch_rows, top_k=5),
            )
            self.batch_summaries.append(summary)

            try:
                self.planner.observe(
                    batch_id=batch_id_str,
                    run_tag=self.run_tag,
                    sample_ids=list(new_ids),
                    ledger_rows=self.ledger.scan(setting_id=self.spec.setting_id),
                    storage=self.storage,
                    ctx=self.ctx,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("[harness] reflection failed on %s: %r", batch_id_str, exc)

            self._append_curve(batch_id_str)

            log.info(
                "[harness] batch=%s samples=%d good=%d bad=%d calls=%d tags=%s",
                batch_id_str,
                summary.samples,
                summary.goodcase,
                summary.badcase,
                summary.api_calls,
                summary.failure_tags_top[:3],
            )

        all_rows = self.ledger.scan(setting_id=self.spec.setting_id)
        overhead = self.overhead.totals(setting_id=self.spec.setting_id)
        summary = RunLedger.summary(
            all_rows,
            setting_id=self.spec.setting_id,
            coverage_axis=self.spec.coverage_axes[0],
            target_goodcase=self.spec.budget.target_goodcase or 100,
        )
        total_calls = summary["api_calls"] + overhead.api_calls
        accepted = summary["goodcase"]
        summary["overhead_api_calls"] = overhead.api_calls
        summary["total_api_calls"] = total_calls
        summary["calls_per_goodcase"] = (
            total_calls / accepted if accepted else None
        )
        out: dict[str, Any] = {
            "campaign_id": self.spec.campaign_id,
            "setting_id": self.spec.setting_id,
            "run_tag": self.run_tag,
            "batches": [asdict(s) for s in self.batch_summaries],
            "summary": summary,
        }
        return out


    def _append_curve(self, batch_id_str: str) -> None:
        rows = self.ledger.scan(setting_id=self.spec.setting_id)
        overhead = self.overhead.totals(setting_id=self.spec.setting_id)
        sample_calls = RunLedger.total_api_calls(rows)
        point = {
            "setting_id": self.spec.setting_id,
            "run_tag": self.run_tag,
            "batch_id": batch_id_str,
            "cum_raw": len(rows),
            "cum_api_calls": sample_calls + overhead.api_calls,
            "cum_sample_api_calls": sample_calls,
            "cum_overhead_api_calls": overhead.api_calls,
            "cum_goodcase": RunLedger.goodcase_count(rows),
            "coverage_entropy": RunLedger.coverage_entropy(
                rows, axis=self.spec.coverage_axes[0]
            ),
            "redundancy": RunLedger.redundancy(rows),
            "repair_rate": RunLedger.repair_rate(rows),
        }
        path = self.storage.root / "_harness_curve.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(point, ensure_ascii=False) + "\n")


    def _materialise_jobs(
        self,
        jobs: list[BatchJob],
        *,
        batch_idx: int,
        batch_id_str: str,
    ) -> list[str]:
        ids: list[str] = []
        for j_idx, job in enumerate(jobs):
            sid = _make_sample_id(
                run_tag=self.run_tag,
                setting_id=self.spec.setting_id,
                batch_idx=batch_idx,
                job_idx=j_idx,
                edit_code=job.edit_code,
                scene_id=job.scene_id,
            )
            sample = Sample(
                sample_id=sid,
                task_type=TaskType.T2I,
                category=job.category,
                edit_codes=[job.edit_code],
                rewrite_parent_id=job.rewrite_parent_id,
                inputs=Inputs(
                    base_prompt=job.base_prompt,
                    sampling=Sampling(size=self.size),
                ),
                steps=[DeltaStep(index=1)],
            )
            sample.meta.setting_id = self.spec.setting_id
            sample.meta.batch_id = batch_id_str
            sample.meta.created_at = datetime.now()
            sample.meta.slice_id = job.slice_id
            sample.meta.pair_meta = {
                "pair_id": job.pair_id,
                "seed_id": job.pair_id,
                "scene_id": job.scene_id,
                "edit_code": job.edit_code,
                "category": job.category,
                "category_hint": job.category,
                "style": job.style,
                "composition": job.composition,
                "lighting": job.lighting,
                "rewrite_failure_tags": job.rewrite_failure_tags,
            }
            sample.meta.tags = ["harness", self.spec.setting_id, batch_id_str]
            self.storage.write_sample(sample)
            ids.append(sid)
        return ids


    def _run_pipeline_on(self, sample_ids: list[str]) -> list[str]:
        rewrite_ids: list[str] = []
        new_ids: list[str] = []
        for sid in sample_ids:
            s = self.storage.read_sample(sid)
            if s.rewrite_parent_id:
                rewrite_ids.append(sid)
            else:
                new_ids.append(sid)

        ready_ids: list[str] = []
        if new_ids:
            ready_ids.extend(
                self._gen_before.run(self.storage, self.ctx, sample_ids=new_ids)
            )
        for sid in rewrite_ids:
            if self._reuse_parent_before(sid):
                ready_ids.append(sid)

        if not ready_ids:
            return []
        planned_ids = self._instruction_planner.run(
            self.storage, self.ctx, sample_ids=ready_ids
        )
        if not planned_ids:
            return []

        active_ids = self._compatibility_gate.run(
            self.storage, self.ctx, sample_ids=planned_ids
        )
        if not active_ids:
            return []
        self._edit_apply.run(self.storage, self.ctx, sample_ids=active_ids)
        self._verifier.run(self.storage, self.ctx, sample_ids=active_ids)
        self._filter.run(self.storage, self.ctx, sample_ids=active_ids)
        return active_ids

    def _reuse_parent_before(self, sid: str) -> bool:
        sample = self.storage.read_sample(sid)
        if not sample.rewrite_parent_id:
            return False
        try:
            parent = self.storage.read_sample(sample.rewrite_parent_id)
        except FileNotFoundError:
            return False
        if parent.inputs.init_image is None:
            return False
        parent_path = self.storage.artifact_path(
            parent.sample_id, parent.inputs.init_image.path
        )
        if not parent_path.exists():
            return False
        self.storage.write_artifact(sid, "before.jpg", parent_path.read_bytes())
        payload = sample.model_dump()
        payload["task_type"] = TaskType.EDIT.value
        input_payload = payload.setdefault("inputs", {})
        before_ref = parent.inputs.init_image.model_dump()
        before_ref["path"] = "before.jpg"
        before_ref["source"] = "reference"
        before_ref["method"] = "rewrite_parent_before"
        input_payload["init_image"] = before_ref
        sample = Sample.model_validate(payload)
        self.storage.write_sample(sample, update_index=True)
        return True
