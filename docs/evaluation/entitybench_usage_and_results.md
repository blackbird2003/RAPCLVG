# EntityBench 使用说明与当前初步结果

本文记录 StoryMem 项目中使用 EntityBench 进行长视频一致性评测的基本信息、使用方法和截至 2026-07-12 的初步结果。当前结果主要用于方法迭代和趋势判断；若要写入论文，需要在同一 split 上重新运行各 baseline，避免与官方全量 140 episodes 数字直接作严格横向比较。

## 1. EntityBench 基本信息

EntityBench 是一个面向多镜头长视频生成的实体一致性评测基准，目标是评估模型在较长故事序列中维持角色、物体、地点等实体一致性的能力。

本地路径：

- 代码与数据：`/home/wxh/world_model_projects/EntityBench`
- 论文 PDF：`/home/wxh/world_model_projects/EntityBench/EntityBench.pdf`
- 官方 README：`/home/wxh/world_model_projects/EntityBench/README.md`

官方数据规模：

- 140 个 episodes
- 2491 个 shots
- 难度分为 easy / medium / hard
- 每个剧本包含 entity registry 与 per-shot entity schedule，用于标注每个镜头中应该出现的角色、物体、地点与动作信息

官方评测分为三个 pillar：

| Pillar | 关注点 | 典型指标 |
|---|---|---|
| Pillar 1 | 单镜头基础视频质量 | subject consistency, temporal flickering, motion smoothness, dynamic degree, aesthetic quality, imaging quality |
| Pillar 2 | shot 内 prompt/entity/action 对齐 | entity presence, entity fidelity, action alignment |
| Pillar 3 | 跨 shot 实体一致性 | CS-Face, CS-Object, transition boundary, LLM face/object/scene accuracy 和 mean score |

我们的研究重点主要是 Pillar 2 / 3，尤其是跨镜头实体一致性和边界连续性。Pillar 1 更偏向基础视频模型能力与生成风格，不一定能直接反映参考帧检索算法质量。

## 2. StoryMem 接入方式

EntityBench 要求生成视频按如下格式组织：

```text
<results_dir>/<episode_id>/<scene>_<shot>.mp4
```

例如：

```text
generated_videos/<run_name>/<episode_id>/001_001.mp4
generated_videos/<run_name>/<episode_id>/001_002.mp4
```

我们额外支持将完整拼接视频放在 episode 目录下：

```text
generated_videos/<run_name>/<episode_id>/full_concatenated.mp4
```

这用于 `cs_transition_boundary` 的 full-video 边界帧评测：

```bash
--transition_boundary_source full
--full_video_name full_concatenated.mp4
```

这样可以评估 StoryMem 实际最终输出中的拼接边界，而不是只比较原始相邻 shot 文件的首尾帧。对于包含 Smooth / RIFE 插帧等后处理的实验，这是更合理的设置。

## 3. 推荐运行环境

目前建议在 A6000 服务器上运行完整 EntityBench 评测，避免占用 4090 上的生成任务。

A6000 路径：

```text
/home/lzg/wxh/world_model_projects/EntityBench
/data3/lzg/wxh/world_model_projects/EntityBench
```

推荐把大文件放在 `/data3`：

- 生成视频：`/data3/lzg/wxh/world_model_projects/EntityBench/generated_videos`
- 测评结果：`/data3/lzg/wxh/world_model_projects/EntityBench/eval_results`

常用环境设置：

```bash
cd /home/lzg/wxh/world_model_projects/EntityBench
source ~/miniconda3/etc/profile.d/conda.sh
source ./env.a6000.sh

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

当前 VLM judge 使用：

```bash
--vlm_model doubao-seed-2-1-turbo-260628
```

VBench 模型缓存：

```bash
--vbench_cache_dir /home/lzg/.cache/vbench
```

如果只做方法迭代，优先跑 Pillar 2 / 3；需要完整表格时再补 Pillar 1。

## 4. 单次评测命令模板

单 GPU 完整评测模板：

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n py311 \
python eval/evaluate_benchmark.py \
  --results_dir /data3/lzg/wxh/world_model_projects/EntityBench/generated_videos/<run_name> \
  --scripts_dir /home/lzg/wxh/world_model_projects/EntityBench/data/scripts \
  --split_json /home/lzg/wxh/world_model_projects/EntityBench/data/splits/<split_name>.json \
  --out_dir /data3/lzg/wxh/world_model_projects/EntityBench/eval_results/<eval_name> \
  --method_name <method_name> \
  --pillars 1,2,3 \
  --device cuda \
  --transition_boundary_source full \
  --full_video_name full_concatenated.mp4 \
  --vlm_model doubao-seed-2-1-turbo-260628 \
  --vbench_cache_dir /home/lzg/.cache/vbench \
  --allow_config_override \
  --allow_partial \
  --llm_concurrency 5 \
  --resume
```

只跑 Pillar 2 / 3：

```bash
--pillars 2,3
```

日志建议写入：

```text
/home/lzg/wxh/world_model_projects/EntityBench/logs/<eval_name>.log
```

评测过程中的常见耗时来源：

- Pillar 1 会加载 VBench 相关模型，显存压力较大。
- Pillar 2 会进行 GroundingDINO / CLIP / DINOv2 等视觉匹配。
- Pillar 2 / 3 中的 VLM judge 受 API 限流影响，可能出现等待和重试。
- hard episode 镜头数多，VLM 调用数量会明显增加。

## 5. 当前 1/10 子集

当前 StoryMem 选取了一个 1/10 子集：

- Easy: 8 episodes
- Medium: 4 episodes
- Hard: 2 episodes
- 总计：14 episodes / 243 shots

筛选原则：

- 避免明显版权 IP 人物。
- 避免可能触发敏感审核的血腥、裸露等内容。
- 避免强写实风格描述，降低生成真实人脸导致后续参考审核失败的概率。

本地/远端 split：

```text
/home/lzg/wxh/world_model_projects/EntityBench/data/splits/storymem_1of10_easy8_mid4_ids.json
/home/lzg/wxh/world_model_projects/EntityBench/data/splits/storymem_1of10_hard2_ids.json
```

已完成结果：

```text
/data3/lzg/wxh/world_model_projects/EntityBench/eval_results/storymem_1of10_videogen_easy8_mid4_full_20260712_000500/report_storymem_1of10_videogen_easy8_mid4.json
/data3/lzg/wxh/world_model_projects/EntityBench/eval_results/storymem_1of10_videogen_hard2_full_20260712_111944/report_storymem_1of10_videogen_hard2.json
```

VLM 调用状态：

| Split | VLM total | Success | Rate limit hits | Exhausted |
|---|---:|---:|---:|---:|
| Easy 8 + Medium 4 | 1175 | 1171 | 49 | 4 |
| Hard 2 | 771 | 771 | 0 | 0 |

## 6. 当前初步结果

下面的 `Ours current 14` 为 StoryMem 当前方法在 Easy 8 + Medium 4 + Hard 2 上的 episode-weighted 汇总结果。baseline 数字来自 EntityBench 论文的全量 140 episodes 表格，因此只适合做方向性参考。

### Pillar 1

| Method | Subject | Temporal | Motion | Dynamic | Aesthetic | Imaging |
|---|---:|---:|---:|---:|---:|---:|
| Ours current 14 | 0.9346 | 0.9856 | 0.9908 | 0.3891 | 0.6448 | 74.42 |
| EntityMem | 0.881 | 0.976 | 0.988 | 0.657 | 0.593 | 66.00 |
| StoryMem | 0.759 | 0.838 | 0.849 | 0.562 | 0.475 | 56.41 |
| HoloCine | 0.860 | 0.957 | 0.964 | 0.721 | 0.518 | 49.97 |
| CineTrans | 0.968 | 0.979 | 0.990 | 0.688 | 0.596 | 68.57 |

### Pillar 2

| Method | Char Presence | Obj Presence | Loc Presence | Face Fidelity | Object Fidelity | Location Fidelity | Action Overall | Action Subject | Action Interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours current 14 | 0.9220 | 0.8690 | 0.6749 | 0.6585 | 0.6865 | 0.6482 | 0.7270 | 0.8059 | 0.7950 |
| EntityMem | 0.967 | 0.888 | 0.687 | 0.740 | 0.601 | 0.555 | 0.618 | 0.706 | 0.781 |
| StoryMem | 0.849 | 0.893 | 0.681 | 0.452 | 0.618 | 0.547 | 0.568 | 0.631 | 0.694 |
| HoloCine | 0.882 | 0.723 | 0.624 | 0.349 | 0.569 | 0.616 | 0.535 | 0.630 | 0.724 |
| CineTrans | 0.796 | 0.776 | 0.651 | 0.327 | 0.273 | 0.528 | 0.374 | 0.518 | 0.618 |

### Pillar 3

| Method | CS Face | CS Object | CS Boundary | LLM Face Acc | LLM Face Score | LLM Obj Acc | LLM Obj Score | LLM Scene Acc | LLM Scene Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours current 14 | 0.8151 | 0.8830 | 0.8004 | 0.9001 | 0.8443 | 0.9506 | 0.8617 | 0.9833 | 0.8900 |
| EntityMem | 0.737 | 0.798 | 0.738 | 0.406 | 0.426 | 0.164 | 0.202 | 0.309 | 0.659 |
| StoryMem | 0.792 | 0.839 | 0.663 | 0.226 | 0.234 | 0.203 | 0.222 | 0.398 | 0.671 |
| HoloCine | 0.751 | 0.803 | 0.498 | 0.228 | 0.242 | 0.088 | 0.094 | 0.304 | 0.616 |
| CineTrans | 0.772 | 0.794 | 0.508 | 0.091 | 0.145 | 0.092 | 0.145 | 0.119 | 0.432 |

## 7. 初步观察

当前方法在 Pillar 3 上表现最突出，尤其是 VLM judge 的 face / object / scene accuracy 和 mean score。这与我们的方法设计一致：通过视觉元素集合、候选关键帧库、VLM 标注和基于覆盖的参考帧选择，提高跨镜头实体引用的准确性和可解释性。

Pillar 2 中，action 和 object/location fidelity 已经较强；face fidelity 低于 EntityMem，但明显高于 StoryMem / HoloCine / CineTrans。presence 指标仍有提升空间，说明部分应该出现的实体可能没有稳定生成或没有被评测管线识别出来。

Pillar 1 中，temporal / motion / aesthetic / imaging 较高，dynamic degree 偏低。可能原因包括：动画风格提示较强、Seedance 生成较保守、稳定镜头和连续性约束会抑制大幅运动。这一指标更偏基础视频动态性，后续需要结合肉眼观察和任务目标判断是否真的构成问题。

## 8. 后续建议

短期建议：

- 在同一 1/10 split 上补跑 EntityMem / StoryMem / HoloCine / CineTrans，得到严格可比 baseline。
- 对 `presence` 失败样本做人工抽查，区分生成失败、检测失败、VLM judge 失败和剧本标注本身的问题。
- 继续保留 `transition_boundary_source=full`，因为它更贴近最终用户看到的拼接结果。
- 对 hard episode 单独分析，当前 hard 的 `cs_transition_boundary` 和 face 相关指标明显更低，可能暴露长距离引用和长序列边界处理的瓶颈。

中期建议：

- 将 EntityBench 的 entity schedule 与我们的视觉元素集合进行映射，分析我们的元素管理是否遗漏或过度合并实体。
- 输出每个 episode 的失败案例可视化报告，包括选中参考帧、指代 prompt、VLM 标注框和最终失败指标。
- 在论文实验中区分 `top-k static score`、`greedy coverage`、`smooth mode`、`without VLM annotation` 等消融设置。

## 9. Easy 8 + Medium 4 消融实验结果

截至 2026-07-13，如果不强制三种设置使用完全相同的 split，而是直接对当时已有 report 取平均，则阶段性总表如下。`greedy` 与 `topk` 当时已有 Easy 8 + Medium 4 + Hard 2；`noref` 当时已有 Easy 8 + Medium 4。Pillar 1 中 `imaging_quality` 先除以 100 后参与平均。

| Method | 当前已有范围 | P1 avg | P2 avg | P3 avg | Overall avg |
|---|---:|---:|---:|---:|---:|
| Greedy | Easy8+Mid4+Hard2 | 0.7815 | 0.7541 | 0.8809 | 0.8055 |
| TopK | Easy8+Mid4+Hard2 | 0.7826 | 0.7498 | 0.8625 | 0.7983 |
| NoRef | Easy8+Mid4 | 0.7819 | 0.7544 | 0.7520 | 0.7628 |

截至 2026-07-13，`greedy / topk / noref` 三种设置在 Easy 8 + Medium 4 上均已完成，范围一致，均为 `12 episodes / 143 shots`。

下面先给出三个 Pillar 的平均分。Pillar 1 中 `imaging_quality` 原始量纲约为 0-100，计算平均分时先除以 100；其余指标均直接使用 0-1 分数。`Overall avg` 是三个 Pillar 平均分的简单平均。

| Method | Scope | P1 avg | P2 avg | P3 avg | Overall avg |
|---|---:|---:|---:|---:|---:|
| Greedy | Easy8+Mid4 | 0.7848 | 0.7523 | 0.8906 | 0.8092 |
| TopK | Easy8+Mid4 | 0.7830 | 0.7466 | 0.8675 | 0.7990 |
| NoRef | Easy8+Mid4 | 0.7819 | 0.7544 | 0.7509 | 0.7624 |

### Pillar 1

| Method | Episodes/Shots | Subject | Temporal | Motion | Dynamic | Aesthetic | Imaging |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 12/143 | 0.9326 | 0.9855 | 0.9903 | 0.4073 | 0.6427 | 75.04 |
| topk | 12/143 | 0.9300 | 0.9842 | 0.9900 | 0.4183 | 0.6273 | 74.80 |
| noref | 12/143 | 0.9237 | 0.9817 | 0.9898 | 0.4439 | 0.6249 | 72.72 |

### Pillar 2

| Method | Episodes/Shots | Char Pres. | Obj Pres. | Loc Pres. | Face Fid. | Obj Fid. | Loc Fid. | Action | Act Subj. | Act Obj. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 12/143 | 0.9204 | 0.8704 | 0.6677 | 0.6578 | 0.6836 | 0.6642 | 0.7189 | 0.8003 | 0.7872 |
| topk | 12/143 | 0.9154 | 0.8545 | 0.6228 | 0.6407 | 0.6515 | 0.6659 | 0.7447 | 0.8285 | 0.7951 |
| noref | 12/143 | 0.9124 | 0.8736 | 0.7110 | 0.6220 | 0.7059 | 0.6040 | 0.7466 | 0.8034 | 0.8107 |

### Pillar 3

| Method | Episodes/Shots | CS Face | CS Obj. | CS Boundary | LLM Face Acc. | LLM Face Score | LLM Obj. Acc. | LLM Obj. Score | LLM Scene Acc. | LLM Scene Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 12/143 | 0.8298 | 0.8864 | 0.8187 | 0.9180 | 0.8565 | 0.9500 | 0.8627 | 1.0000 | 0.8950 |
| topk | 12/143 | 0.8192 | 0.8838 | 0.7840 | 0.8870 | 0.8280 | 0.9375 | 0.8774 | 0.9500 | 0.8450 |
| noref | 12/143 | 0.7835 | 0.8394 | 0.8871 | 0.6014 | 0.6589 | 0.7220 | 0.7211 | 0.8000 | 0.7550 |

观察：`greedy` 的总体平均分最高，优势主要来自 Pillar 3 跨镜头一致性；`topk` 接近 `greedy`，但 LLM face / scene 和 transition boundary 稍低；`noref` 在部分单镜头 prompt 对齐指标上不差，且 `cs_transition_boundary` 最高，但跨镜头实体一致性明显下降，尤其是 LLM face / object 相关指标。这支持“参考帧检索与指代 prompt 对跨镜头一致性有贡献”的消融判断。

## 10. 当前采用的正式消融框架

为了让论文实验部分更简洁、清晰，并且降低补实验成本，当前主文采用三组逐步增强的消融设置：

| 顺序 | 消融设置 | 去掉的核心模块 | 保留内容 | 主要验证问题 |
|---:|---|---|---|---|
| 1 | w/o Complementary Coverage Retrieval | 贪心互补覆盖选帧 | 视觉元素表、VLM 标注、结构化 multimodal prompt | 互补覆盖选帧是否优于一次性静态 Top-K |
| 2 | w/o Structured Multimodal Prompt Grounding | 参考帧元素级指代、排除说明、整体参考指引等结构化 prompt | 视觉元素表、VLM 标注、互补覆盖选帧 | 选中参考帧后，是否需要明确告诉生成模型如何使用它们 |
| 3 | w/o Agentic Visual Element Planning | 视觉元素表与 shot-level 元素状态规划 | 历史关键帧库、朴素 prompt-keyframe 相似度检索 | 整套 agentic visual element planning 是否优于朴素 CLIP/RAG 式检索 |

说明：第三项是“大削”消融。由于视觉元素表是覆盖选帧与结构化指代的基础，去掉它后系统自然退化为基于当前 shot video prompt 与历史关键帧库做 CLIP/语义相似度 Top-K 的朴素检索 baseline。

下一步 TODO：

1. 在相同 split、相同 Seedance 生成设置、相同 VLM judge 下补跑 `w/o Structured Multimodal Prompt Grounding`。
2. 实现并补跑 `w/o Agentic Visual Element Planning`：不维护元素表，不使用元素状态，仅用当前 video prompt 与历史关键帧做语义相似度 Top-K。
3. 将已完成的 `TopK` 明确对应到 `w/o Complementary Coverage Retrieval`，确认其余配置与 full method 完全一致。
4. 论文主表只保留 `Full / w/o Coverage / w/o Structured Grounding / w/o Element Planning` 四行；`NoRef / NoElementsGuide / NoHolisticGuide / Smooth` 等旧设置放入补充材料或内部分析。
5. 对最终四组结果补充 2-3 个定性案例：覆盖选帧优于 Top-K、结构化指代引导正确使用参考帧、朴素检索引入错误历史元素。

## 11. 阶段性消融方法总表

截至 2026-07-14，`greedy / topk / noref / no_elements_guide / no_holistic_guide` 五种设置均已完成同一 1/10 子集，即 `Easy8 + Mid4 + Hard2`，共 `14 episodes / 243 shots`。下表使用与上一节一致的简化平均口径：Pillar 1 中 `imaging_quality` 先除以 100；Pillar 2 取 9 个 headline 指标平均；Pillar 3 取 9 个跨镜头一致性指标平均；`Overall avg` 为三个 Pillar 的简单平均。

| Method | 含义 | Scope | P1 avg | P2 avg | P3 avg | Overall avg |
|---|---|---:|---:|---:|---:|---:|
| Greedy | 完整方法，启用视觉元素集合、holistic guide 与贪心覆盖选帧 | 14/243 | 0.7815 | 0.7541 | **0.8813** | **0.8056** |
| TopK | 不启用贪心覆盖，仅按静态分 TopK 选帧 | 14/243 | 0.7826 | 0.7498 | 0.8625 | 0.7983 |
| NoRef | 不使用历史参考帧 | 14/243 | 0.7798 | 0.7573 | 0.7471 | 0.7614 |
| NoElementsGuide | 去掉视觉元素计划/元素指引相关 prompt | 14/243 | 0.7761 | **0.7575** | 0.8615 | 0.7984 |
| NoHolisticGuide | 去掉整体剧情/全局上下文指引相关 prompt | 14/243 | **0.7829** | 0.7516 | 0.8759 | 0.8035 |

Hard-only 结果如下，便于观察长序列场景下不同 prompt 组件的影响：

| Method | Scope | P1 avg | P2 avg | P3 avg | Overall avg |
|---|---:|---:|---:|---:|---:|
| NoElementsGuide Hard | 2/100 | 0.7716 | 0.7491 | 0.7490 | 0.7566 |
| NoHolisticGuide Hard | 2/100 | 0.7675 | 0.7497 | 0.8292 | 0.7821 |

当前最清晰的趋势是：完整 `Greedy` 的总体分最高，主要优势仍来自 Pillar 3；`NoRef` 的 Pillar 3 明显下降，说明参考帧检索和指代 prompt 对跨镜头一致性非常关键。`NoHolisticGuide` 接近完整方法，甚至高于 `TopK / NoElementsGuide`，说明当前 holistic guide 的收益可能不如元素级指引稳定，后续值得做更细的 prompt ablation 和失败样本抽查。

## 12. 按论文主表 24 指标展开的消融结果

下表按 EntityBench 论文 Table 4 中展示的 24 个代表性指标展开。五种方法均为同一 `Easy8 + Mid4 + Hard2` 子集，即 `14 episodes / 243 shots`。所有指标均按“越高越好”标注，**加粗**为最优，<u>下划线</u>为次优；`imaging_quality` 保持 EntityBench / VBench 原始量纲。

| Group | Metric | Greedy | TopK | NoRef | NoElementsGuide | NoHolisticGuide |
|---|---|---:|---:|---:|---:|---:|
| P1: Quality | Imaging Quality | 74.42 | 74.21 | 72.21 | <u>74.85</u> | **75.00** |
| P1: Quality | Aesthetic Quality | **0.6449** | <u>0.6314</u> | 0.6235 | 0.6283 | 0.6177 |
| P1: Quality | Motion Smoothness | <u>0.9908</u> | 0.9904 | 0.9901 | **0.9915** | 0.9898 |
| P2: Presence | Char Presence | <u>0.9220</u> | 0.9185 | 0.9117 | **0.9288** | 0.9206 |
| P2: Presence | Object Presence | 0.8690 | 0.8571 | <u>0.8802</u> | **0.8890** | 0.8702 |
| P2: Presence | Location Presence | 0.6749 | 0.6377 | <u>0.7206</u> | **0.7213** | 0.6987 |
| P2: Fidelity | Face Fidelity | <u>0.6584</u> | 0.6445 | 0.6160 | **0.6623** | 0.6484 |
| P2: Fidelity | Object Fidelity | 0.6865 | 0.6496 | **0.7018** | <u>0.6877</u> | 0.6863 |
| P2: Fidelity | Location Fidelity | 0.6482 | <u>0.6542</u> | 0.5937 | 0.6113 | **0.6583** |
| P2: Action | Action Overall | 0.7270 | <u>0.7519</u> | **0.7564** | 0.7314 | 0.7199 |
| P2: Action | Action Subject | 0.8060 | **0.8341** | <u>0.8153</u> | 0.8090 | 0.7967 |
| P2: Action | Action Interaction | 0.7950 | <u>0.8008</u> | **0.8200** | 0.7770 | 0.7655 |
| P3: DINOv2 | CS Face | **0.8151** | 0.8047 | 0.7661 | 0.8085 | <u>0.8088</u> |
| P3: DINOv2 | CS Object | **0.8830** | <u>0.8774</u> | 0.8329 | 0.8754 | 0.8726 |
| P3: DINOv2 | CS Boundary | 0.8003 | 0.7826 | **0.8879** | 0.8002 | <u>0.8120</u> |
| P3: LLM Characters | LLM Face Accuracy | **0.9000** | 0.8868 | 0.5909 | <u>0.8965</u> | 0.8874 |
| P3: LLM Characters | LLM Face Mean Score | <u>0.8442</u> | 0.8264 | 0.6510 | 0.8417 | **0.8450** |
| P3: LLM Characters | LLM Face Face | **0.8275** | 0.7838 | 0.6071 | <u>0.8259</u> | 0.8184 |
| P3: LLM Objects | LLM Object Accuracy | **0.9505** | 0.9300 | 0.7296 | <u>0.9433</u> | 0.9404 |
| P3: LLM Objects | LLM Object Mean Score | 0.8617 | **0.8703** | 0.7241 | <u>0.8631</u> | 0.8501 |
| P3: LLM Objects | LLM Object Shape | 0.8729 | **0.8859** | 0.7560 | <u>0.8753</u> | 0.8700 |
| P3: LLM Scenes | LLM Scene Accuracy | **0.9857** | 0.9444 | 0.7917 | 0.8750 | <u>0.9714</u> |
| P3: LLM Scenes | LLM Scene Mean Score | <u>0.8907</u> | 0.8403 | 0.7500 | 0.8500 | **0.8950** |
| P3: LLM Scenes | LLM Scene Layout | <u>0.8950</u> | 0.8486 | 0.7750 | 0.8536 | **0.8965** |
