# StoryMem 文档索引

本文档用于说明 `docs/` 的当前分区。日常工程任务优先阅读
`AGENTS.md` 中的 Read Order；不要默认通读归档计划、周报或大型 PDF。

## 当前必读

- `current/videogen_notebook_user_guide.md`：当前 7870 Web UI 的使用说明。
- `current/videogen_notebook_design.md`：当前 Web UI、项目状态和数据结构设计。
- `current/seedance_pipeline_architecture.md`：Seedance pipeline 与 CLI/Web 共享架构。
- `current/visual_element_memory_methodology.md`：视觉元素记忆方法论与研究主线。
- `current/research_plan.md`：仍在推进的研究方向和实验问题。
- `current/history.md`：已完成的重要决策和里程碑摘要。

## 工程记录

- `engineering/task_report.md`：短工程任务完成记录。每次实现任务结束后追加简洁结果。
- `engineering/a6000_eval_queue_workflow.md`：A6000 测评队列与自动提交工作流。

## 归档计划

`plans/completed/` 保存已完成或已被当前实现吸收的设计/实施计划。它们可用于追溯
当时决策，但不应覆盖 `current/` 中的当前行为。

## 汇报与讨论

`reports/` 保存周报、组会提纲和汇报材料。

## 实验与评测

- `experiments/`：局部实验记录和策略讨论。
- `evaluation/`：EntityBench 等评测说明、指标和结果记录。
- `evaluation/entitybench_1of4_experiment_matrix.md`：当前 1/4 子集的
  `full / woCover / woGrounding_noprefix / woElements_noprefix / _storymem / vimax`
  六设置实验矩阵、命名归一化规则和完成状态。

## 外部参考

`references/` 保存 Seedance/火山方舟文档、论文 PDF、PPT 和算法参考材料。大型 PDF
只在需要精确查证接口或论文细节时阅读。

- `references/sponsored_video_api_survey.md`：赞助商提供的 Wan R2V / Veo
  代理视频生成 API 调研与 StoryMem 接入建议。
- `references/wan2_7_r2v_usage_for_storymem.md`：Wan2.7 R2V 官方
  reference-to-video 用法、first-frame non-cut 方案和 StoryMem 接入约定。
