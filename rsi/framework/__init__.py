"""DataFlow-centered execution framework for text and multimodal tasks."""

from .core.contracts import ArtifactRef, InputContract, OutputContract, TaskEnvelope
from .core.operator import Operator, OperatorSpec
from .runtime.framework import run
from .core.pipeline import (
    Pipeline,
    PipelineSpec,
    RunContext,
    bounded_repair_context,
)
from .core.dataflow_operator import LLMServingABC, OperatorABC
from .core.dataflow_pipeline import (
    BatchedPipelineABC,
    PipelineABC,
    StreamBatchedPipelineABC,
)
from .runtime.registry import MethodPlugin
from .io.storage import ArtifactStore, BlobStore, RecordStore, RunStore, StorageBundle
from .io.dataflow_storage import (
    BatchedFileStorage,
    DataFlowStorage,
    FileStorage,
    StreamBatchedFileStorage,
)
from .core.checkpoint import CheckpointStore
from .core.provenance import ProvenanceStore
from .runtime.agents import AgentRequest, AgentResult, AgentRuntime, MissingProviderDependency, build_agent_runtime
from .runtime.skills import SkillRef, SkillRegistry
from .io.serving import PipelineLLMServing
from .evolution.feedback import CandidateFeedback, CandidateEvaluator
from .evolution.media import artifact_store_for, resolve_local_input_artifact

__all__ = [
    "ArtifactRef", "InputContract", "OutputContract", "TaskEnvelope", "Operator",
    "OperatorSpec", "Pipeline", "PipelineSpec", "RunContext",
    "RecordStore", "BlobStore", "ArtifactStore", "CheckpointStore",
    "RunStore", "ProvenanceStore", "StorageBundle", "MethodPlugin",
    "bounded_repair_context", "run",
    "AgentRequest", "AgentResult", "AgentRuntime", "MissingProviderDependency",
    "build_agent_runtime", "SkillRef", "SkillRegistry", "PipelineLLMServing",
    "CandidateFeedback", "CandidateEvaluator", "resolve_local_input_artifact",
    # Operator, pipeline and storage contracts for generated code, aligned with
    # open-dataflow 1.0.10 observable behavior.
    "OperatorABC", "LLMServingABC", "PipelineABC", "BatchedPipelineABC",
    "StreamBatchedPipelineABC", "DataFlowStorage", "FileStorage",
    "BatchedFileStorage", "StreamBatchedFileStorage",
    # DataLite extension, not part of the reference contract.
    "artifact_store_for",
]
