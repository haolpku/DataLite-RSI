# VideoRSI

## Motivation

Improving video understanding through supervised fine-tuning depends strongly on
the quality, grounding, and diversity of the training data. For video tasks,
constructing such data is difficult because a useful training example must
connect a question and answer to visual evidence distributed across time.

VideoRSI studies recursive self-improvement at the **data-pipeline level**. The
pipeline, rather than model weights, is iteratively improved to construct better
video-grounded SFT data. This design avoids repeatedly training a target model
while exploring pipeline changes, reducing the cost of recursive improvement for
multimodal video tasks.

## Method

VideoRSI transforms a video corpus into a high-quality SFT pool through five
stages.

### 1. Temporal visual evidence

Each source video is divided into chronological 60-second chunks.
`CaptionOperator` produces detailed captions with relative timestamps.
`EntityExtractionOperator` extracts entities, entity updates, and events from
the current chunk and its caption, while maintaining stable entity identities
across chunks. The resulting captions, entity tracks, and events form a
timestamped evidence store for every video.

### 2. Evidence-based task construction

The pipeline indexes caption spans, entity observations, and events as evidence
units. Task candidate mining retrieves evidence units appropriate for video
understanding tasks, including action recognition, object reasoning, counting,
temporal reasoning, OCR, and spatial reasoning.

For each candidate, the grounded task builder creates a task record with the
selected evidence, task type, and answer target. The question generator then
produces a multiple-choice video question with evidence references and source
video provenance.

### 3. Question and option refinement

Generated questions are checked for schema validity, answer consistency, and
evidence references. A distractor-enhancement stage constructs more challenging
incorrect options, followed by a distractor-quality check. This stage removes
duplicate, malformed, and weak question-option sets.

### 4. Selection and data aggregation

VideoRSI evaluates whether a question can be solved from text alone and measures
its difficulty with a frozen target model. The frontier filter combines these
signals with task eligibility and duplicate detection. Local and global
deduplication produce a diverse SFT data pool while preserving evidence lineage
for each accepted example.

### 5. Recursive pipeline improvement

Feedback is collected from candidate yield, evidence and schema validation,
question filtering, text-only checks, frozen-target difficulty, duplicate
rejection, and accepted frontier-and-novel samples under a fixed budget. These
signals are used to update task definitions, prompts, operators, and pipeline
routing in the next iteration. The target model is not repeatedly trained during
this pipeline-search process.

## Evaluation

Starting from an initial data pipeline, VideoRSI performed five pipeline
iterations and used the final best pipeline to synthesize a 4.5k video SFT
dataset. The dataset was used to fine-tune models from five video-language model
families and evaluated on Video-MME.

| Model | Base | VideoRSI SFT | Gain |
| --- | ---: | ---: | ---: |
| Qwen3-VL-8B-Instruct | 56.56 | 58.89 | +2.33 |
| LLaVA-OneVision-7B | 58.52 | 59.48 | +0.96 |
| Qwen2.5-VL-3B-Instruct | 42.44 | 59.04 | +16.60 |
| Gemma-4-E4B | 47.74 | 48.37 | +0.63 |
| InternVL3-2B-hf | 34.85 | 55.22 | +20.37 |

VideoRSI improves the Video-MME overall score for every evaluated model family.
