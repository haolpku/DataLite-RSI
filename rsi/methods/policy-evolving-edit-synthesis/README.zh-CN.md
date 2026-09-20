# 策略演化的编辑合成（Policy-Evolving Edit Synthesis）

一种生成式轨道的 RSI 方法。它合成图像编辑训练对 — 源图、编辑指令与编辑结果 — 并根据评判器的决策递归演化**指令撰写策略**，而非演化数据集或模型。

## 循环流程

一次运行只覆盖一种编辑类型，并按固定顺序遍历其场景列表。样本以每批十个处理。

```text
      策略库 --> 指令规划器 --> 编辑模型 --> 三轴评判
             ^                 （每个样本一个场景）                |
             |                                                       v
      若在第 2 批中得到支持则晋升 <-- 分析师阅读已决批次
             |
             +-- 未获支持：保持为开放论断；连续 2 批未决：退役
```

每批按以下顺序执行：

1. **规划。** 从调度中取出接下来十个尚未尝试的 scene/edit 对。活跃策略追加到指令规划器的提示词中。
2. **撰写。** 视觉语言规划器看到源图与编辑类型定义，撰写指令、保留列表以及预期终态。
3. **编辑。** 图像模型应用该指令。
4. **评判。** 冻结的三轴评判标准接受或拒绝该样本对。仅当源图通过环境有效性预检，且各轴既达到各自下限又满足聚合条件时，样本对才被接受：`instruction_following >= 4.0`、`visual_consistency >= 3.0`、`visual_quality >= 3.0`，等权重均值 `>= 3.5`。任一轴低于其下限即判失败，与均值无关。
5. **反思。** 分析师阅读已决批次 — 评判结果、各轴得分、失败原因、指令文本 — 并提出、修订或撤回关于「何种指令在该编辑类型上能成功」的论断。
6. **门控。** 论断仅在两个不同批次中得到支持后才成为活跃策略；在矛盾次数至少与支持次数相同时被拒绝；连续两批中没有任何样本能对其做出裁决时退役。

分析师上有一条常驻约束：禁止任何通过降低编辑质量来提高通过率的策略 — 典型情况是要求微小、几乎不可见的变更。该约束不做裁决，且任何策略不得覆盖它。

分析师从不看图像，评判器从不看策略，二者无法互相争辩。除策略外的一切 — 基础提示词、评判标准、场景调度 — 在各实验臂之间保持固定。

## 仓库结构

```text
method.json                    方法清单
README.md                      英文说明
README.zh-CN.md                中文说明
configs/
  rubrics/if_vc_vq.json        冻结的接受标准
  environment/edits.jsonl      23 种编辑类型定义
src/deltasynth/
  harness/evolve/              RSI 核心：分析师、门控、策略状态、
                               批次统计、提交器
  harness/                     批次规划器、AgenticLoop 编排器、
                               失败诊断、运行账本、覆盖率
  operators/recipe/            指令规划器与兼容性门控
  operators/generate/          源图生成与编辑应用
  operators/verify/            三轴评判器
  operators/filter/            按阈值标记接受/拒绝
  core/, environment/          样本 schema、存储、并行、任务宇宙
  serving/                     抽象模型接口
tests/                         门控、退役与评判标准的单元测试
```

实验臂通过 `setting_id` 选择：`random` 与 `stratified` 为基线，`evolving` 增加策略演化，`policy-evolving` 在此基础上增加上述常驻约束 — 后者即本方法。

有两项未包含在内。具体 provider 适配器被排除，因为它们是端点与凭证 plumbing；请针对自有模型实现 `src/deltasynth/serving/base.py` 中的 `LLMServingABC`、`VLMServingABC` 与 `ImageGenServingABC`。场景池是数据集而非方法的一部分，因此此处不随附环境快照；`src/deltasynth/environment/schema.py` 规定了快照须包含的内容。

评估与训练环境固定于 [`docker/generative-image-edit/`](../../../docker/generative-image-edit/)。

## 外部服务

需要图像生成或图像编辑端点、视觉语言规划器、视觉语言评判器，以及纯文本分析师
LLM。具体 provider 适配器不在本目录中：请针对自有模型实现
`LLMServingABC`、`VLMServingABC` 与 `ImageGenServingABC`。凭证只能通过运行时
环境注入，不得写入配置。

## 安全、回滚与预算

- **冻结评判器。** 三轴标准从不看到已演化的策略，分析师无法把接受规则争辩成更高通过率。
- **常驻约束。** 禁止通过降低编辑质量来提高产量的策略（典型是要求几乎不可见的改动），且不得被覆盖。
- **晋升降级门控。** 论断须在两个不同批次中得到支持才成为活跃策略；矛盾次数不少于支持次数时拒绝晋升；连续两批无法裁决则退役。
- **回滚是固有的：** 被拒绝的论断保持为开放假设或被退役，不会写入规划器提示词。
- **预算** 为固定批次日程（`max_batches × batch_size`）外加 API 调用硬上限。没有收敛判据。最后一批不做反思，因为没有后续批次能使用该结果。

## 已知限制

- **Provider 适配器与场景池不在本贡献中。** 复现需要自备 serving 实现，以及符合 `src/deltasynth/environment/schema.py` 的环境快照。
- **参考提交未记录每次运行的随机种子。**
- **三种编辑类型的接受率相对 stratified 臂下降**（H1、H2、G2），均值产量仍上升。
- **目标多样性均值排除了 H3 与 O9**，见表格注；不要把它和 23 类平均直接比较。
- **下游提升相对循环内产量差距较小。** 迁移由托管 VLM 评判器测量，不是仓库内的确定性评分器。

## 结果

机器可读提交（`unverified`）：

- [`policy-evolving-edit-synthesis-qwen-image-edit-2511`](../../../results/submissions/policy-evolving-edit-synthesis-qwen-image-edit-2511/)
- [`policy-evolving-edit-synthesis-flux2-klein-9b`](../../../results/submissions/policy-evolving-edit-synthesis-flux2-klein-9b/)

迁移评测套件是 [`image-edit-transfer-suite`](../../../benchmarks/image-edit-transfer-suite/)。
`primary_score` 为 `0.5 * (gedit_bench / 10 + imgedit_bench / 5)`。

两条臂在相同环境、评判标准与场景调度下，对 23 种编辑类型各产出 1,000 对 accepted 样本。

### 接受率（%）

| 编辑类型 | Stratified | Policy-evolving | Δ |
| --- | ---: | ---: | ---: |
| O1 background_change | 81.2 | 82.0 | +0.8 |
| O2 material_change | 73.4 | 90.0 | +16.6 |
| O3 subject_replace | 90.8 | 92.0 | +1.2 |
| O4 object_remove | 75.3 | 86.0 | +10.7 |
| O5 object_add | 92.0 | 94.0 | +2.0 |
| O6 style_transfer_object | 77.1 | 84.0 | +6.9 |
| O7 color_change | 80.9 | 92.0 | +11.1 |
| O8 texture_change | 68.0 | 80.0 | +12.0 |
| O9 object_extract | 79.4 | 80.0 | +0.6 |
| T1 text_modify | 85.4 | 94.0 | +8.6 |
| T2 text_remove | 69.6 | 90.0 | +20.4 |
| T3 text_add | 75.0 | 84.0 | +9.0 |
| T4 text_translate | 45.8 | 80.0 | +34.2 |
| T5 text_replace | 87.0 | 96.0 | +9.0 |
| T6 text_layout | 53.5 | 76.0 | +22.5 |
| T7 text_style_change | 79.0 | 96.0 | +17.0 |
| H1 action_change | 95.1 | 90.0 | -5.1 |
| H2 appearance_change | 95.2 | 94.0 | -1.2 |
| H3 portrait_beautify | 65.2 | 80.0 | +14.8 |
| G1 style_transfer | 96.7 | 98.0 | +1.3 |
| G2 layout_change | 80.2 | 76.0 | -4.2 |
| G3 scene_change | 83.4 | 92.0 | +8.6 |
| G4 lighting_change | 92.2 | 94.0 | +1.8 |
| **Mean** | **79.2** | **87.8** | **+8.6** |

### 编辑目标多样性

编辑目标的归一化 Shannon 熵，取值 0 到 1，越高表示分布越均匀。将两臂的目标合并，在固定种子下洗牌，由分类器盲分 — 分类器在未提供类别列表的情况下自建类别 — 再按臂拆分。

| 编辑类型 | Stratified | Policy-evolving | Δ |
| --- | ---: | ---: | ---: |
| O1 background_change | 0.974 | 0.958 | -0.016 |
| O2 material_change | 0.912 | 0.950 | +0.038 |
| O3 subject_replace | 0.947 | 0.958 | +0.011 |
| O4 object_remove | 0.898 | 0.821 | -0.077 |
| O5 object_add | 0.982 | 0.962 | -0.020 |
| O6 style_transfer_object | 0.629 | 0.868 | +0.239 |
| O7 color_change | 0.897 | 0.941 | +0.044 |
| O8 texture_change | 0.678 | 0.860 | +0.182 |
| O9 object_extract | 0.998 | 0.996 | -0.001 |
| T1 text_modify | 0.969 | 0.967 | -0.003 |
| T2 text_remove | 0.810 | 0.810 | +0.000 |
| T3 text_add | 0.984 | 0.980 | -0.004 |
| T4 text_translate | 0.624 | 0.713 | +0.089 |
| T5 text_replace | 0.881 | 0.843 | -0.038 |
| T6 text_layout | 0.895 | 0.876 | -0.019 |
| T7 text_style_change | 0.634 | 0.812 | +0.177 |
| H1 action_change | 0.922 | 0.911 | -0.011 |
| H2 appearance_change | 0.818 | 0.921 | +0.104 |
| H3 portrait_beautify | 0.509 | 0.840 | +0.331 |
| G1 style_transfer | 0.734 | 0.876 | +0.142 |
| G2 layout_change | 0.962 | 0.993 | +0.031 |
| G3 scene_change | 0.908 | 0.886 | -0.022 |
| G4 lighting_change | 0.849 | 0.858 | +0.009 |
| **Mean (excluding H3 and O9)** | **0.853** | **0.894** | **+0.041** |

### 下游训练

相同 LoRA 配方、两种基座模型，分别在两臂的 1,000 对上训练。

| 基座模型 | 训练数据 | GEdit-Bench | ImgEdit-Bench |
| --- | --- | ---: | ---: |
| qwen-image-edit-2511 | none (baseline) | 8.104 | 4.449 |
| qwen-image-edit-2511 | stratified pipeline | 8.138 | 4.578 |
| qwen-image-edit-2511 | **policy-evolving** | **8.171** | **4.598** |
| flux2-klein-9b-base | none (baseline) | 7.653 | 4.111 |
| flux2-klein-9b-base | stratified pipeline | 7.844 | 4.188 |
| flux2-klein-9b-base | **policy-evolving** | **7.920** | **4.280** |
