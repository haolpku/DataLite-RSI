from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

from ...core.imageops import fit_size, parse_size, read_size, require_size
from ...core.operator import OperatorABC
from ...core.parallel import ParallelMixin
from ...core.schema import ImageRef, ImageSource, Sample
from ...serving.base import ImageGenServingABC


def _to_png_bytes(jpeg_bytes: bytes, *, max_edge: int = 1024) -> bytes:
    try:
        from PIL import Image  # type: ignore
    except ImportError:  # pragma: no cover
        return jpeg_bytes

    with Image.open(BytesIO(jpeg_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max_edge / float(max(w, h))
        if scale < 1.0:
            new_size = (int(round(w * scale)), int(round(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()


class EditApply(ParallelMixin, OperatorABC):
    name = "edit_apply"

    def __init__(
        self,
        image_serving: ImageGenServingABC,
        *,
        edit_input_max_edge: int = 1024,
    ) -> None:
        super().__init__()
        self.image_serving = image_serving
        self.edit_input_max_edge = edit_input_max_edge

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
        if not pending:
            return sample_ids

        max_workers = max(1, getattr(ctx, "max_workers", 1))
        succeeded, failed = self.run_parallel_samples(
            pending,
            self._process_one,
            storage=storage,
            ctx=ctx,
            max_workers=max_workers,
            progress_desc=self.name,
        )
        return [s for s in sample_ids if s not in {r.sample_id for r in failed}]


    def _process_one(self, storage, ctx, sid: str) -> None:
        sample = storage.read_sample(sid)
        assert (
            sample.inputs.init_image is not None
        ), f"{sid} missing init_image — run gen_before first"
        assert (
            sample.steps[0].instruction.en is not None
        ), f"{sid} missing instruction — run edit_derive first"

        before_path = storage.artifact_path(sid, sample.inputs.init_image.path)
        jpeg_bytes = before_path.read_bytes()
        png_bytes = _to_png_bytes(jpeg_bytes, max_edge=self.edit_input_max_edge)

        instruction = sample.steps[0].instruction.en
        size = (
            sample.inputs.sampling.size if sample.inputs.sampling else None
        ) or "1024x1024"

        result = ctx.call_with_retry(
            self.image_serving.edit,
            png_bytes,
            instruction,
            size=size,
            op_name=f"edit_apply:{sid}",
        )

        if len(result.image_bytes) < 1024:
            raise RuntimeError(
                f"edits API returned suspiciously small image ({len(result.image_bytes)} B)"
            )

        target = parse_size(size) or read_size(jpeg_bytes)
        data = result.image_bytes
        current = read_size(data)
        if current is not None and target is not None and current != target:
            if current[0] * target[1] == current[1] * target[0]:
                data, _ = fit_size(data, target)
        width, height = require_size(data, target)
        storage.write_artifact(sid, "after.jpg", data)

        after_ref = ImageRef(
            path="after.jpg",
            width=width,
            height=height,
            source=ImageSource.GENERATED,
            method="edits_api",
            prompt=instruction,
        )
        sample.steps[0].image = after_ref
        sample.steps[0].cost = result.cost
        sample.final_image = after_ref

        sample.cost.add(result.cost)
        ctx.cost_tracker.add(sid, result.cost)

        sample = Sample.model_validate(sample.model_dump())
        storage.write_sample(sample, update_index=True)
        self._mark_done(storage, sid)
