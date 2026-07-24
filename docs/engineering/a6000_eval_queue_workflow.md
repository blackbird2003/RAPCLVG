# A6000 测评队列工作流

本文档记录 videogen_notebook 项目完成后自动提交 EntityBench 测评的轻量工作流。

## 1. 责任边界

4090 / StoryMem 只负责提交：

- 确认 videogen_notebook 项目已经 `completed`。
- 从项目目录导出 EntityBench 测评所需产物。
- 将产物同步到 A6000 数据盘。
- 在 A6000 队列目录写入一个 job JSON。
- 在本地项目目录写入 `eval_submitted.json`。

A6000 / EntityBench 负责管理：

- 测评任务排队。
- 单个 A6000 GPU 串行执行。
- LLM/VLM 临时失败后的等待重试。
- 状态文件、日志、报告路径。
- 成功或失败归档。

4090 侧不读取远端状态，也不在 Web UI 中展示远端测评进度。

## 2. 提交脚本

脚本位置：

```bash
/home/wxh/world_model_projects/StoryMem/tools/submit_videogen_eval_to_a6000.py
```

基本用法：

```bash
cd /home/wxh/world_model_projects/StoryMem
python tools/submit_videogen_eval_to_a6000.py \
  --project-dir /home/wxh/world_model_projects/StoryMem/.runtime/videogen_notebook/projects/<project_id>
```

常用参数：

```bash
--pillars 1,2,3
--pillars 2,3
--transition-boundary-source full
--tier easy|medium|hard
--episode-id Easy_1
--method-name greedy
--dry-run
```

默认远端：

```text
host: lzg@10.130.128.150
queue_root: /data3/lzg/wxh/world_model_projects/EntityBench/eval_queue
entitybench_root: /home/lzg/wxh/world_model_projects/EntityBench
```

可通过环境变量覆盖：

```bash
export STORYMEM_EVAL_REMOTE_HOST=lzg@10.130.128.150
export STORYMEM_EVAL_QUEUE_ROOT=/data3/lzg/wxh/world_model_projects/EntityBench/eval_queue
export STORYMEM_REMOTE_ENTITYBENCH_ROOT=/home/lzg/wxh/world_model_projects/EntityBench
```

## 3. EntityBench 元数据来源

提交脚本只从项目的 `story.json.source` 读取 EntityBench 所需原始字段。

需要存在：

```text
story.json
story.json.source.scenes
story.json.source.entity_schedule
story.json.source.entity_descriptions
```

这些字段来自通过 JSON 导入创建 videogen_notebook 项目时保留的原始 JSON 额外字段。当前算法流程和 Web UI 不使用这些字段，但测评需要它们。

如果 `story.json` 缺失，或 `story.json.source` 中缺少上述字段，提交脚本不会向 A6000 入队，而是在项目目录写入：

```text
eval_submitted.json
```

其中 `state` 为 `failed`，并记录失败原因。

## 4. 提交包格式

每个 job 在 A6000 上对应一个自包含 submission：

```text
eval_queue/submissions/<job_id>/
  scripts/<episode_id>.json
  results/<episode_id>/
    001_001.mp4
    001_002.mp4
    ...
    full_concatenated.mp4
  metadata/
    split.json
    export_manifest.json
    job.json
```

`scripts/<episode_id>.json` 来自 `story.json.source`，提交时会将 `story_name` 改成当前 `episode_id`，以匹配 EntityBench 的文件名和 split。

## 5. A6000 Worker

脚本位置：

```bash
/home/lzg/wxh/world_model_projects/EntityBench/scripts/eval_queue_worker.py
```

启动方式：

```bash
cd /home/lzg/wxh/world_model_projects/EntityBench
python scripts/eval_queue_worker.py \
  --queue-root /data3/lzg/wxh/world_model_projects/EntityBench/eval_queue \
  --gpu 0
```

单次消费一个任务后退出：

```bash
python scripts/eval_queue_worker.py \
  --queue-root /data3/lzg/wxh/world_model_projects/EntityBench/eval_queue \
  --gpu 0 \
  --once
```

默认行为：

- 使用 1 个 A6000 GPU，默认 `CUDA_VISIBLE_DEVICES=0`。
- 使用 A6000 当前已配置好的 `py311` conda 环境。
- 默认 VLM 为 `doubao-seed-2-1-turbo-260628`。
- 默认允许 `--allow_config_override`，以复用当前 Seed VLM 测评设置。
- 每个 job 最多尝试 3 次。
- 失败后等待 300 秒再重试。
- 使用 `--resume` 继续已有 episode 级中间结果。

## 6. 队列目录

```text
/data3/lzg/wxh/world_model_projects/EntityBench/eval_queue/
  jobs/          # submitter 原子写入的 job JSON
  submissions/   # 自包含测评输入包
  status/        # worker 写入状态
  logs/          # 每个 job 的完整执行日志
  reports/       # EntityBench 输出报告
  lock/          # worker.lock，避免多个 worker 抢同一队列
  archive/       # 预留
```

状态文件示例：

```json
{
  "job_id": "eval_...",
  "state": "running",
  "attempt": 1,
  "max_attempts": 3,
  "log_path": ".../logs/eval_....log",
  "report_json": ".../reports/eval_.../report_<method>.json",
  "error": null
}
```

## 7. 后续 Web UI 集成

第一版 Web UI 只需要在项目 `completed` 后调用提交脚本即可。`videogen_notebook`
项目设置中的 **Auto Submit Eval** 开关默认关闭；开启后，项目进入 `completed`
状态时 Web 后台会调用提交脚本。调用成功后，本地项目目录会出现
`eval_submitted.json`，表示任务已交给 A6000 管理。

Web UI 不需要查询 A6000 状态；远端结果由 A6000 的队列目录和 EntityBench 报告统一管理。
