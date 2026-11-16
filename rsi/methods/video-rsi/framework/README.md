# VideoRSI-Dev

Main development tree for the Video Data RSI framework.  It contains the
composable operator runtime, task-centric RSI pipeline, prompts, model/backend
contracts, tests, documentation and round-0 configuration.  Generated data,
API credentials and generated runtime outputs are intentionally excluded.

## Layout

```text
src/video_rsi/       framework and operators
drivers/             reproducible run, evaluation and Codex-evolution controller
tests/               unit and integration tests
docs/                architecture and evolution design
configs/             versioned task priors/configuration
```

Run the tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The reproducible experiment drivers live in `drivers/`. They snapshot this
framework into `versions/v0`, `versions/v1`, and so on, while keeping generated
results in a separate experiment root.

Start with [`drivers/README.md`](drivers/README.md) for the three execution
surfaces: data synthesis, fixed-policy evaluation and bounded Codex evolution.
The public framework deliberately does not prescribe cluster scheduling,
networking, credentials, or machine-specific operational procedures.

## Composable task pipelines

The framework now provides an operator library, serializable pipeline graphs,
and task skill packs.  A graph references immutable operator versions and can
be routed by task type without changing v0:

```python
from video_rsi import OperatorLibrary, PipelineGraph, PipelineNode, SkillRegistry

graph = PipelineGraph(
    "state_change", "v1", "state_change",
    (PipelineNode("EvidenceSelectionOperator@1"),
     PipelineNode("QuestionGenerationOperator@2")),
)
```

Load the example skill packs from `configs/skills/`.  `PipelineMigrationPlanner`
turns per-task/operator failure reports into explicit actions such as forking a
task graph or replacing an operator; the controller still owns tests and
promotion decisions.

## Scope boundary

This package intentionally stops at versioned data-pipeline search. Training
recipes, model-family adapters, benchmark wrappers, and vendor patches belong
to separate downstream experiment packages. Generated data, predictions,
checkpoints and server logs remain outside Git.
