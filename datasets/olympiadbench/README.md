# OlympiadBench

675 English-language open-ended mathematical olympiad problems.

**Source:** Qwen2.5-Math evaluation harness, `https://github.com/QwenLM/Qwen2.5-Math/tree/a45202bd16f1ec06f433442dc1152d0074773465/evaluation/data`.
Corresponds to the text-only English math competition subset
(`OE_TO_maths_en_COMP` configuration) of
[Hothan/OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench)
(sha `91184b52131e7fc9455fef848035173aea8cc01a`)
after filtering and field mapping. Not a verbatim HuggingFace split.

**Metric:** accuracy, greedy decoding.
