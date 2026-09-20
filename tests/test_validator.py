import copy
import shutil
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPOSITORY_ROOT / "scripts" / "validate_contributions.py"
SPEC = importlib.util.spec_from_file_location("lite_rsi_validator", VALIDATOR_PATH)
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


class ValidatorTests(unittest.TestCase):
    def test_empty_registry_scaffold_is_valid(self):
        self.assertEqual(validator.validate_repository(REPOSITORY_ROOT), [])

    def test_manifest_id_must_match_directory(self):
        benchmark_spec = next(
            spec for spec in validator.SPECS if spec.kind == "benchmark"
        )
        manifest = json.loads(
            (REPOSITORY_ROOT / "templates" / "benchmark.json").read_text()
        )
        manifest["id"] = "different-id"
        errors = validator.validate_manifest(manifest, benchmark_spec, "expected-id")
        self.assertTrue(any("must match directory" in error for error in errors))

    def test_invalid_result_types_are_rejected_without_crashing(self):
        spec = next(s for s in validator.SPECS if s.kind == "result")
        record = json.loads((REPOSITORY_ROOT / "results/submissions/opsd-data-lite-qwen3-8b/result.json").read_text())
        for field, value in (("track", []), ("method_id", {}), ("status", [])):
            bad = copy.deepcopy(record)
            bad[field] = value
            self.assertTrue(validator.validate_manifest(bad, spec, record["submission_id"]))
        for value in ("0.6", True, float("nan"), float("inf")):
            bad = copy.deepcopy(record)
            bad["metrics"]["final"]["primary_score"] = value
            self.assertTrue(validator.validate_manifest(bad, spec, record["submission_id"]))
        for field, value in (("seeds", "42"), ("seeds", [True]), ("iterations", 0), ("iterations", True)):
            bad = copy.deepcopy(record)
            bad["settings"][field] = value
            self.assertTrue(validator.validate_manifest(bad, spec, record["submission_id"]))

    def test_result_benchmark_version_must_match_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for spec in validator.SPECS:
                for path in REPOSITORY_ROOT.glob(spec.pattern):
                    dest = root / path.relative_to(REPOSITORY_ROOT)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, dest)
            path = root / "results/submissions/opsd-data-lite-qwen3-8b/result.json"
            record = json.loads(path.read_text())
            record["benchmark"]["version"] = "9.9.9"
            path.write_text(json.dumps(record))
            self.assertTrue(any("version does not match" in e for e in validator.validate_repository(root)))

    def test_invalid_json_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "datasets" / "broken" / "dataset.json"
            path.parent.mkdir(parents=True)
            path.write_text("{not-json", encoding="utf-8")
            errors = validator.validate_repository(root)
        self.assertTrue(any("cannot read valid UTF-8 JSON" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
