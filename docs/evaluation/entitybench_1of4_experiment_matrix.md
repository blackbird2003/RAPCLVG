# EntityBench 1/4 子集实验矩阵

本文记录当前 EntityBench 1/4 子集的实验约定、命名规则和测评完成状态。后续实验目标是补齐这里定义的 `35 × 6 = 210` 个 episode-method 测评结果。

更新时间：2026-07-23 05:20 America/New_York。

## 1. 子集定义

1/4 子集共 35 个 episode：

- Easy: `Easy_1` 到 `Easy_20`
- Mid: `Mid_1` 到 `Mid_10`
- Hard: `Hard_1` 到 `Hard_5`

本地剧本目录：

```text
/home/wxh/world_model_projects/EntityBench/data/entitybench_1of4_scripts
```

## 2. 关注的六种实验设置

| 设置 | 含义 |
|---|---|
| `full` | 当前完整方法：视觉元素集合、VLM 标注、grounded reference prompt、贪心覆盖选帧等均开启。 |
| `woCover` | 关闭贪心覆盖选帧，作为覆盖选择机制的消融。 |
| `woGrounding_noprefix` | 关闭/弱化参考图精确指代 prompt，并移除会触发 Seedance 内置 reference agent 的来源前缀。 |
| `woElements_noprefix` | 关闭/弱化视觉元素规划信息，并移除会触发 Seedance 内置 reference agent 的来源前缀。 |
| `_storymem` | 原始 StoryMem 风格策略对比实验，episode 名带 `_storymem` 后缀。 |
| `vimax` | ViMax 生成结果对比实验。 |

## 3. 测评约定

所有正式 P2/P3 对比应整理为 EntityBench 统一目录结构：

```text
generated_videos/<method>/<episode_id>/
  001_001.mp4
  001_002.mp4
  ...
  full_concatenated.mp4
```

测评统一使用：

```bash
--pillars 2,3
--transition_boundary_source full
--full_video_name full_concatenated.mp4
--vlm_model doubao-seed-2-1-turbo-260628
```

`cs_transition_boundary / CS-Bound.` 属于 Pillar 2/3 报告中的跨 shot 边界连续性指标；当前使用 `full_concatenated.mp4` 中的实际拼接边界帧计算。不同方法必须提供正确顺序、正确帧数的完整拼接视频，否则边界位置可能不可靠。

## 4. 命名归一化规则

统计 1/4 子集覆盖率时，以下名字视为同一 base episode：

- `Selected105_Easy_1` -> `Easy_1`
- `Easy_1_safe` -> `Easy_1`
- `Mid_1_storymem` -> `Mid_1`
- `Mid_1_woCover` -> `Mid_1`
- `Mid_1_woGrounding_noprefix` -> `Mid_1`
- `Mid_1_woElements_noprefix` -> `Mid_1`

“完成测评”的判定标准：对应 `eval_results/<run>/<method>/<episode_id>.json` 已存在。不把 `.work/eval_artifacts` 中正在生成的中间目录计为完成。

## 5. 当前完成状态

截至 2026-07-23 05:20，1/4 子集中各设置已完成测评如下：

| 设置 | 完成数 | Easy | Mid | Hard | 已完成范围 |
|---|---:|---:|---:|---:|---|
| `full` | 35/35 | 20 | 10 | 5 | 全部完成 |
| `woCover` | 10/35 | 0 | 10 | 0 | `Mid_1` 到 `Mid_10` |
| `woGrounding_noprefix` | 9/35 | 0 | 9 | 0 | `Mid_1, Mid_2, Mid_4, Mid_5, Mid_6, Mid_7, Mid_8, Mid_9, Mid_10` |
| `woElements_noprefix` | 13/35 | 0 | 10 | 3 | `Mid_1` 到 `Mid_10`; `Hard_1` 到 `Hard_3` |
| `_storymem` | 12/35 | 0 | 10 | 2 | `Mid_1` 到 `Mid_10`; `Hard_1` 到 `Hard_2` |
| `vimax` | 5/35 | 4 | 1 | 0 | `Easy_1` 到 `Easy_4`; `Mid_9` |

## 6. 当前交集

| 对比组合 | 交集数量 | Episodes |
|---|---:|---|
| `full + woCover + woGrounding_noprefix + woElements_noprefix` | 9 | `Mid_1, Mid_2, Mid_4, Mid_5, Mid_6, Mid_7, Mid_8, Mid_9, Mid_10` |
| `full + _storymem` | 12 | `Mid_1` 到 `Mid_10`; `Hard_1, Hard_2` |
| `full + vimax` | 5 | `Easy_1` 到 `Easy_4`; `Mid_9` |
| 六种设置全交集 | 1 | `Mid_9` |

## 7. 后续目标

后续任务以补齐如下矩阵为主：

```text
35 episodes × 6 settings = 210 episode-method results
```

优先级建议：

1. 继续补齐正在跑的 `_storymem` 与 `vimax` 在 1/4 子集中的已生成结果测评。
2. 补齐 `woCover`, `woGrounding_noprefix`, `woElements_noprefix` 在 Easy 与 Hard 上的生成和测评。
3. 每次新增测评后，更新本文档的完成状态与交集表。

用于结果表格时，应优先在同一 episode 交集上比较不同设置；不要直接把覆盖 episode 不同的均值作为严格横向结论。
