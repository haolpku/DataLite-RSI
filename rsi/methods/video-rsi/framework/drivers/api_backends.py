"""OpenAI-compatible backends used by the real RSI experiment driver."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from video_rsi.api_client import OpenAICompatibleJSONClient


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class APIQuestionGenerator:
    name = "api:gpt-5.5"

    def __init__(self, client: OpenAICompatibleJSONClient) -> None:
        self.client = client

    def generate(self, *, candidate, evidence, rewatch, prompt):
        del rewatch
        try:
            result = self.client.complete_json(
                prompt=prompt,
                system="Return only the requested JSON object. Do not add markdown.",
                max_tokens=1800,
                temperature=0.2,
            )
        except Exception as exc:
            return {"question": "", "answer": "", "evidence_ids": [],
                    "generation_error": f"{self.name}: {exc}"}
        if result.get("reject"):
            return {"question": "", "answer": "", "evidence_ids": []}
        options = result.get("options") or result.get("choices") or {}
        if isinstance(options, dict):
            choices = [str(options[key]).strip() for key in "ABCD" if options.get(key)]
        else:
            choices = [str(item).strip() for item in options]
        answer = str(result.get("answer") or "").strip()
        if not answer and result.get("correct_option") in options:
            answer = str(options[result["correct_option"]]).strip()
        evidence_ids = list(result.get("evidence_ids") or [])
        if not evidence_ids:
            raw = result.get("evidence") or result.get("citations") or []
            if isinstance(raw, list):
                evidence_ids = [
                    item.get("evidence_id") if isinstance(item, dict) else str(item)
                    for item in raw
                ]
                evidence_ids = [item for item in evidence_ids if item]
        # The operator performs the final allow-list check.  Preserve the
        # candidate's grounded IDs as a safe recovery when a local model omits
        # an otherwise non-semantic citation field.
        if not evidence_ids:
            evidence_ids = list(candidate.get("evidence_ids") or [])
        return {
            "question": str(result.get("question") or "").strip(),
            "answer": answer,
            "choices": choices,
            "rationale": str(result.get("rationale") or "").strip(),
            "evidence_ids": evidence_ids,
        }


class APIDistractorEnhancer:
    """One-shot hard-negative rewriting using the question-generation model."""

    name = "api:glm-5.3-flash:distractor-enhancement"

    def __init__(self, client: OpenAICompatibleJSONClient) -> None:
        self.client = client

    def enhance(self, *, sample, evidence, prompt):
        del sample, evidence
        return self.client.complete_json(
            prompt=prompt,
            system="Return only JSON with a choices array of exactly four strings.",
            max_tokens=700,
            temperature=0.2,
        )


class APIGeneralVerifier:
    name = "api:gemini-3.1-flash-lite"

    def __init__(self, client: OpenAICompatibleJSONClient) -> None:
        self.client = client

    def verify(self, *, sample, evidence, rewatch):
        del rewatch
        prompt = """You are a general verifier for a grounded video question.
Check the question, options, answer and cited timestamped evidence together.
Do one holistic verification: validity, answer support, uniqueness, absence of
visual hallucination, and training value. Do not repair the sample. Return only
JSON with boolean keys valid, answer_supported, unambiguous,
video_training_value and a concise reason.

SAMPLE:
{sample}
EVIDENCE:
{evidence}
""".format(sample=_json(sample), evidence=_json(evidence))
        try:
            result = self.client.complete_json(
                prompt=prompt,
                system="Return only valid JSON.",
                max_tokens=700,
                temperature=0.0,
            )
        except Exception as exc:
            return {"valid": False, "answer_supported": False, "unambiguous": False,
                    "video_training_value": False,
                    "reason": f"{self.name}: malformed verifier output: {exc}"}
        return {
            "valid": bool(result.get("valid")),
            "answer_supported": bool(result.get("answer_supported")),
            "unambiguous": bool(result.get("unambiguous")),
            "video_training_value": bool(result.get("video_training_value")),
            "reason": str(result.get("reason") or ""),
        }


class APITextOnly:
    name = "api:glm-5.3-flash"

    def __init__(self, client: OpenAICompatibleJSONClient) -> None:
        self.client = client

    def answer_text(self, *, sample) -> str:
        prompt = """Answer this multiple-choice question using only the text.
Return JSON: {{"answer":"the selected option text"}}. If the text is
insufficient, answer {{"answer":"CANNOT_DETERMINE"}}.
QUESTION: {question}
OPTIONS: {choices}
""".format(question=sample["question"], choices=_json(sample.get("choices", [])))
        try:
            result = self.client.complete_json(
                prompt=prompt,
                system="Return only valid JSON.",
                max_tokens=180,
                temperature=0.0,
            )
        except Exception as exc:
            return "CANNOT_DETERMINE"
        return str(result.get("answer") or result.get("prediction") or "").strip()


def _sample_frames(video_path: Path, start: float, end: float, count: int = 16) -> list[str]:
    """Uniformly sample the evidence interval for the frozen target.

    Frame count is a caller-controlled inference policy.  Sampling the cited
    interval rather than the full source video keeps the target's observation
    aligned with the evidence envelope used to construct the question.
    """
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    with tempfile.TemporaryDirectory(prefix="rsi_frames_") as tmp:
        pattern = str(Path(tmp) / "frame_%02d.jpg")
        duration = max(0.1, end - start)
        fps = count / duration
        # The cluster does not expose ffmpeg on PATH.  Keep the binary
        # configurable so a personal/conda installation can be used without
        # modifying the process-wide PATH or relying on a system package.
        ffmpeg = os.environ.get("RSI_FFMPEG_BIN", "ffmpeg")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", str(start),
            "-t", str(duration), "-i", str(video_path), "-vf", f"fps={fps},scale=640:-1",
            "-frames:v", str(count), "-q:v", "5", pattern,
        ]
        # Long batch runs occasionally see transient ffmpeg failures (for
        # example exit status 228 under concurrent decoding) even though the
        # same file and command succeed immediately afterwards. Retry only the
        # local extraction step; this does not repeat any paid model request.
        # Remove partial JPEGs before each retry so ffmpeg never encounters
        # stale output from the preceding attempt.
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(3):
            for frame in Path(tmp).glob("frame_*.jpg"):
                frame.unlink()
            try:
                subprocess.run(command, check=True, capture_output=True)
                last_error = None
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
        if last_error is not None:
            stderr = (last_error.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg frame extraction failed after 3 attempts for {video_path}: "
                f"exit={last_error.returncode}; stderr={stderr or '<empty>'}"
            ) from last_error
        urls = []
        for frame in sorted(Path(tmp).glob("frame_*.jpg")):
            encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
            urls.append("data:image/jpeg;base64," + encoded)
        return urls


class APIFrozenTarget:
    """Vision target adapter; sends sampled frames through the same API."""

    name = "api:qwen3-vl-8b-instruct"

    def __init__(self, client: OpenAICompatibleJSONClient, model_name: str = "qwen3-vl-8b-instruct", allow_text_fallback: bool = False) -> None:
        self.client = client
        self.allow_text_fallback = allow_text_fallback
        suffix = ":text-fallback" if allow_text_fallback else ""
        self.name = "api:" + model_name + suffix

    def answer_video(self, *, sample, video, trial_index) -> str:
        del trial_index
        video_value = video.get("path") or video.get("video_path")
        if not video_value:
            raise ValueError("frozen target requires video.path or video.video_path")
        video_path = Path(str(video_value))
        start = min(float(item["start_sec"]) for item in sample["selected_evidence"])
        end = max(float(item["end_sec"]) for item in sample["selected_evidence"])
        try:
            frames = _sample_frames(video_path, start, end, count=16)
        except FileNotFoundError:
            if not self.allow_text_fallback:
                raise
            frames = []
        content: list[dict[str, Any]] = [{"type": "text", "text": (
            "Answer the video question. Return JSON {\"answer\":\"selected option text\"}.\n"
            f"Question: {sample['question']}\\nOptions: {_json(sample.get('choices', []))}"
        )}]
        if frames:
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in frames)
        else:
            content[0]["text"] += "\nEvidence descriptions (degraded text-only smoke fallback): " + _json(sample["selected_evidence"])
        try:
            result = self.client.complete_multimodal_json(content=content, max_tokens=180)
        except Exception as exc:
            return "CANNOT_DETERMINE"
        answer = str(result.get("answer") or result.get("prediction") or "").strip()
        # Gateways/models commonly answer with A/B/C/D even when prompted for
        # option text.  Convert a letter to the corresponding choice so the
        # frontier comparator evaluates the model's actual decision.
        letter = re.fullmatch(r"[\[(]?\s*([ABCD])\s*[\])]?(?:[.。:]|$)", answer, re.I)
        if letter:
            index = ord(letter.group(1).upper()) - ord("A")
            choices = sample.get("choices") or []
            if index < len(choices):
                return str(choices[index]).strip()
        # Also handle concise outputs such as "The answer is B".
        match = re.search(r"\b(?:answer|option|choice)\s*(?:is|:)?\s*([ABCD])\b", answer, re.I)
        if match:
            index = ord(match.group(1).upper()) - ord("A")
            choices = sample.get("choices") or []
            if index < len(choices):
                return str(choices[index]).strip()
        return answer
