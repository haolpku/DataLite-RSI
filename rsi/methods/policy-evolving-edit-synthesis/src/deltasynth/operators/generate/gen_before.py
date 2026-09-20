from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from PIL import Image as PILImage

from ...core.imageops import fit_size, parse_size, read_size, require_size
from ...core.operator import OperatorABC
from ...core.parallel import ParallelMixin
from ...core.schema import ImageRef, ImageSource, Sample, TaskType
from ...serving.base import ImageGenServingABC


def _target_size(sample: Sample) -> Optional[tuple[int, int]]:
    return parse_size(
        sample.inputs.sampling.size if sample.inputs.sampling else None
    )


class GenBefore(ParallelMixin, OperatorABC):
    name = "gen_before"

    def __init__(
        self,
        image_serving: ImageGenServingABC,
        *,
        before_bank: Optional[Path] = None,
        generate_on_bank_miss: bool = False,
    ) -> None:
        super().__init__()
        self.image_serving = image_serving
        self.before_bank = before_bank
        self.generate_on_bank_miss = generate_on_bank_miss

    def run(
        self,
        storage,
        ctx,
        *,
        sample_ids: Optional[list[str]] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> list[str]:
        sample_ids = sample_ids or list(storage.list_samples())
        pending = self._filter_pending(storage, sample_ids)

        if self.before_bank is not None:
            ready = self._run_from_bank(storage, sample_ids, pending)
            if not self.generate_on_bank_miss:
                return ready
            missed = [sid for sid in sample_ids if sid not in set(ready)]
            if not missed:
                return ready
            ok = set(ready) | set(self._run_generate(storage, ctx, missed))
            return [sid for sid in sample_ids if sid in ok]

        if not pending:
            return sample_ids
        ok = set(self._run_generate(storage, ctx, pending)) | (
            set(sample_ids) - set(pending)
        )
        return [sid for sid in sample_ids if sid in ok]

    def _run_generate(self, storage, ctx, pending: list[str]) -> list[str]:
        if not pending:
            return []
        max_workers = max(1, getattr(ctx, "max_workers", 1))
        _, failed = self.run_parallel_samples(
            pending,
            self._process_one,
            storage=storage,
            ctx=ctx,
            max_workers=max_workers,
            progress_desc=self.name,
        )
        failed_ids = {result.sample_id for result in failed}
        return [sid for sid in pending if sid not in failed_ids]


    def _run_from_bank(
        self,
        storage,
        sample_ids: list[str],
        pending: list[str],
    ) -> list[str]:
        assert self.before_bank is not None
        ready: list[str] = []
        already_done = [s for s in sample_ids if s not in set(pending)]

        for sid in pending:
            try:
                if self._adopt_from_bank(storage, sid):
                    ready.append(sid)
            except Exception as exc:  # noqa: BLE001 — one sample must not sink the batch
                print(f"[gen_before] {sid}: bank adoption failed: {exc!r}")

        for sid in already_done:
            sample = storage.read_sample(sid)
            if sample.inputs.init_image is not None:
                ready.append(sid)
            else:
                print(
                    f"[gen_before] {sid}: marked complete but has no init_image; "
                    f"re-adopting from the bank"
                )
                try:
                    if self._adopt_from_bank(storage, sid):
                        ready.append(sid)
                except Exception as exc:  # noqa: BLE001
                    print(f"[gen_before] {sid}: re-adoption failed: {exc!r}")

        return [sid for sid in sample_ids if sid in set(ready)]

    def _adopt_from_bank(self, storage, sid: str) -> bool:
        assert self.before_bank is not None
        sample = storage.read_sample(sid)
        scene_id = (sample.meta.pair_meta or {}).get("scene_id") or ""

        bank_path = self.before_bank / f"{scene_id}.jpg"
        if not bank_path.exists():
            bank_path = self.before_bank / f"{scene_id}.png"
        if not bank_path.exists():
            print(
                f"[gen_before] bank miss: no image for scene {scene_id!r} "
                f"in {self.before_bank}"
            )
            return False

        data = bank_path.read_bytes()
        with PILImage.open(bank_path) as img:
            width, height = img.size
        target = _target_size(sample)
        if target is not None and (width, height) != target:
            print(
                f"[gen_before] {sid}: bank image for {scene_id!r} is "
                f"{width}x{height}, not the declared {target[0]}x{target[1]}; "
                f"using it as-is — re-render or resize the banked image to "
                f"the declared size to fix this"
            )
        storage.write_artifact(sid, "before.jpg", data)

        sample.inputs.init_image = ImageRef(
            path="before.jpg",
            width=width,
            height=height,
            source=ImageSource.REFERENCE,
            method="before_bank",
            prompt=sample.inputs.base_prompt,
        )
        sample = _retype_to_edit(sample)
        storage.write_sample(sample, update_index=False)

        if storage.read_sample(sid).inputs.init_image is None:
            print(f"[gen_before] {sid}: metadata write did not take effect")
            return False

        self._mark_done(storage, sid)
        return True


    def _process_one(self, storage, ctx, sid: str) -> None:
        sample = storage.read_sample(sid)
        assert sample.inputs.base_prompt, f"{sid} missing base_prompt"
        size = (
            sample.inputs.sampling.size if sample.inputs.sampling else None
        ) or "1024x1024"

        result = ctx.call_with_retry(
            self.image_serving.generate,
            sample.inputs.base_prompt,
            size=size,
            op_name=f"gen_before:{sid}",
        )

        target = parse_size(size)
        data = result.image_bytes
        current = read_size(data)
        if current is not None and target is not None and current != target:
            if current[0] * target[1] == current[1] * target[0]:
                data, _ = fit_size(data, target)
        width, height = require_size(data, target)
        storage.write_artifact(sid, "before.jpg", data)

        sample.inputs.init_image = ImageRef(
            path="before.jpg",
            width=width,
            height=height,
            source=ImageSource.GENERATED,
            method="generations_api",
            prompt=sample.inputs.base_prompt,
        )

        sample.cost.add(result.cost)
        ctx.cost_tracker.add(sid, result.cost)

        sample = _retype_to_edit(sample)

        storage.write_sample(sample, update_index=True)
        self._mark_done(storage, sid)


def _retype_to_edit(sample: Sample) -> Sample:
    payload = sample.model_dump()
    payload["task_type"] = TaskType.EDIT.value
    return Sample.model_validate(payload)
