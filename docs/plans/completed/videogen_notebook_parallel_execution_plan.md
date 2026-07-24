# videogen_notebook 多项目并行与 Step6 GPU 锁计划

本文档记录 2026-07-09 讨论形成的 `videogen_notebook` 并行执行升级方案。目标是在不重写项目数据结构、不引入复杂全局数据库的前提下，允许多个项目同时运行远端 LLM/VLM/Seedance 等待型流程，同时保证本地 GPU 密集的 Step 6 Keyframe Maintaining 串行排队。

## 1. 背景与判断

当前新版 `videogen_notebook` 的项目状态与资源已经基本自包含：

- 每个项目有独立目录：`project.json`、`settings.json`、`shots/`、`assets/`、`memory/`。
- 每个 Shot/Attempt 的输出写入本项目目录下，不依赖跨项目共享业务文件。
- Step 2/3/5 主要是远端 API 调用和轮询，等待时间远大于本地计算时间。
- Step 6 需要抽帧、HPS/质量筛选、VLM 标注，其中抽帧与质量模型可能占用本地 GPU，是主要的本地资源冲突点。

现有后台执行限制主要来自调度器实现，而不是项目结构本身：

- `videogen_notebook/jobs.py` 中 `ThreadPoolExecutor(max_workers=1)` 只允许一个后台任务。
- `BackgroundJobManager._start()` 会在任意项目已有未完成 Future 时拒绝新项目启动。
- 项目内部仍需要保持线性 Shot 前缀语义：后一个 Shot 依赖前一个 Shot 完整完成。

因此，本次改造应聚焦调度层：放宽项目级后台并行，新增 Step6 GPU 锁队列。

## 2. 目标行为

### 2.1 多项目并行

- 允许最多 `30` 个后台项目/Shot/Step 任务同时运行。
- 同一个项目仍只允许一个后台任务运行。
- 同一个 Shot 仍只允许一个 Step 处于 running。
- 一个项目内部仍按现有 Step 顺序和 Shot 顺序推进，不做项目内 Shot 并行。
- 项目之间互不阻塞，除非同时进入 Step 6。

### 2.2 Step 6 单 GPU 锁

Step 6 `keyframe_maintaining` 进入本地关键帧维护前必须获取全局 GPU 锁：

- 未获取锁时，Shot/Step 状态进入可见的等待状态。
- 获取锁后执行现有 Step 6 流程。
- 完成、失败、中断或异常退出时必须释放锁。
- 第一版只支持单 GPU worker，即同一时间最多一个 Step 6 运行。
- 其他项目在等待 Step 6 GPU 锁时，不影响别的项目继续执行 Step 2/3/4/5。

### 2.3 自动运行语义

项目级 `Run all` 或 Shot 级 `Run shot` 在自动推进到 Step 6 时：

1. 如果 GPU 锁空闲，立即进入 Step 6 running。
2. 如果 GPU 锁被占用，进入 `keyframe_maintaining_queued` 或同等等待状态。
3. 等锁可用后自动继续 Step 6。
4. Step 6 完成后继续当前项目的后续 Shot。

Step 级 `Run Step` 对 Step 6 也遵守同一锁规则。

## 3. 非目标

第一版暂不实现以下内容：

- 项目内多个 Shot 并行。
- 多 GPU 调度和 GPU 指定。
- 独立进程级 GPU worker 池。
- 分布式任务队列、Redis、数据库任务表。
- WebSocket 实时推送。
- 更细粒度地区分 HPS GPU 阶段和 VLM 远端阶段。
- 对旧 `storymem_web` 7860 UI 做并行改造。

## 4. 状态设计

现有 Step 状态已经是 `xxx_running / xxx_completed / xxx_failed` 风格。为 Step 6 增加等待状态：

```text
keyframe_maintaining_queued
```

语义：

- 表示 Step 6 前序条件满足，当前正在等待全局 GPU 锁。
- 属于 running-like 状态，项目仍视为 running。
- Stop Project 应能中断 queued 状态，取消等待并进入 interrupted。
- UI 可显示为 `Waiting for GPU`。

是否新增 `gpu_waiting_started_at` 可暂缓。第一版可以复用 Step 的 `started_at / updated_at`，并在 logs/details 中记录等待开始、获取锁、释放锁。

## 5. 调度设计

### 5.1 BackgroundJobManager

将 `videogen_notebook/jobs.py` 的后台线程池改为可配置：

```text
VIDEOGEN_NOTEBOOK_MAX_WORKERS=30
```

默认值：`30`。

启动任务时的限制从“任意项目只能有一个 Future”改为：

- 如果同一 `project_id` 已有未完成 Future，拒绝启动该项目的新任务。
- 不同项目可以同时提交到线程池。
- `_jobs` 继续以 `project_id -> Future` 管理即可。

这样保持轻量，不需要引入全局任务表。

### 5.2 Project/Shot 级保护

保留现有项目内保护：

- 项目状态为 running 时，不允许对同项目执行会破坏状态的编辑/删除。
- Shot 状态为 `*_running` 或 `keyframe_maintaining_queued` 时，不允许重复启动该 Shot。
- 后续 Shot 仍依赖前序 Shot `completed`。

需要将 `_is_running_status()` 扩展为识别 `keyframe_maintaining_queued`。

### 5.3 GPU 锁对象

新增轻量锁对象，例如：

```text
videogen_notebook/gpu_lock.py
```

建议第一版使用进程内 `threading.Condition`：

- 当前 Web UI 通常是单 uvicorn 进程，后台任务都在线程池内。
- Condition 可以支持排队等待和 stop/cancel 检查。
- 比文件锁更容易在等待期间定期检查 cancel flag。

接口草案：

```python
class KeyframeGpuLock:
    def acquire(self, project_id: str, shot_id: str, should_cancel: Callable[[], bool]) -> GpuLease:
        ...

@dataclass
class GpuLease:
    owner_project_id: str
    owner_shot_id: str
    acquired_at: str

    def release(self) -> None:
        ...
```

第一版只需要 FIFO-ish，不要求严格公平。为了避免长期饥饿，可以维护一个简单等待队列：

```text
queue: list[(project_id, shot_id, token)]
owner: token | None
```

等待线程只在自己是队首且 owner 为空时获得锁。

### 5.4 Step6 执行包装

在 `NotebookRunner._run_step_without_lock()` 或 `_run_keyframe_maintaining_step()` 周围增加锁包装。

推荐位置：`_run_step_without_lock()` 中进入 Step6 分支前后。

原因：

- 可以在进入实际 `_run_keyframe_maintaining_step()` 之前设置 queued 状态。
- 可以把锁等待视为 Step6 的一部分，保留 retry 逻辑不变。
- Step6 内部现有 retry 机制仍只覆盖关键帧维护失败，不覆盖排队等待。

伪流程：

```python
if step == "keyframe_maintaining":
    self._set_step_status(..., "keyframe_maintaining_queued")
    with self.gpu_lock.acquire(project_id, shot_id, should_cancel):
        self._set_step_status(..., "keyframe_maintaining_running")
        self._run_keyframe_maintaining_step(...)
```

取消处理：

- waiting 阶段检测 cancel event，抛 `RunnerCancelledError`。
- running 阶段沿用现有 cancel checkpoint。
- finally 中释放 GPU lease。

## 6. UI 更新

### 6.1 项目列表/项目页

最小 UI 改动：

- 状态标签支持 `keyframe_maintaining_queued`。
- Step 6 card 显示 `Waiting for GPU` 或 `Queued for GPU`。
- Details 日志显示：
  - queued time
  - acquired GPU lock
  - released GPU lock

不需要新增复杂队列面板。若后续实验规模继续扩大，可在首页增加“GPU queue”小面板。

### 6.2 用户文档

更新 `../../current/videogen_notebook_user_guide.md`：

- 删除“当前默认只允许一个项目 Running”的限制说明。
- 增加“可并行运行多个项目，但 Keyframe Maintaining 会排队等待本地 GPU”。
- 说明默认后台并行度为 30，可通过环境变量调整。

## 7. 数据与兼容

不需要迁移已有项目。

新增状态 `keyframe_maintaining_queued` 对旧项目无影响。旧项目打开时，如果不存在该状态则按原逻辑显示。

需要注意：

- `ProjectStore.recover_interrupted_runs()` 应把 queued 状态视为 running-like，服务重启后恢复为 interrupted。
- 项目复制/删除逻辑应将 queued 视为 running-like，避免复制出虚假的正在排队状态。
- Stop Project 应能中断 queued 项目。

## 8. 风险与处理

### 8.1 Seedance/LLM API 并发

放宽到 30 后，可能同时出现多个远端 API 请求和轮询。

第一版不做复杂限流，但保留两个简单缓冲：

- 并行度通过 `VIDEOGEN_NOTEBOOK_MAX_WORKERS` 可调。
- 已有 Step 级重试机制继续生效。

若后续遇到 API 限流，再增加 per-provider semaphore：

```text
LLM max concurrent
VLM max concurrent
Seedance submit max concurrent
Seedance poll max concurrent
```

### 8.2 Cloudflare Tunnel publisher

多个项目可能同时使用 Smooth 的 reference video publisher。当前 publisher 是进程内单例，复用一个 tunnel 和本地 serve dir，原则上可支持多个文件发布。

需要重点确认：

- publisher 初始化线程安全。
- 文件名使用 UUID，避免项目间冲突。
- 若 tunnel 启动失败，应只失败当前 Step5，不阻塞其他项目。

### 8.3 GPU 锁泄露

必须用 context manager/finally 释放锁。服务进程崩溃时进程内锁自然消失；项目状态恢复由 `recover_interrupted_runs()` 处理。

### 8.4 文件写冲突

不同项目写不同目录，风险低。同一项目禁止多个 job 即可避免项目内写冲突。

## 9. 推荐实现步骤

### Step A：调度器放宽

1. `BackgroundJobManager` 支持 `max_workers` 参数，默认读取 `VIDEOGEN_NOTEBOOK_MAX_WORKERS=30`。
2. `_start()` 改为只检查同一项目是否已有未完成 Future。
3. 添加 focused tests：
   - 同一项目重复启动仍报错；
   - 不同项目可以各自提交 Future。

### Step B：新增 GPU 锁

1. 新增 `videogen_notebook/gpu_lock.py`。
2. 实现进程内 FIFO-ish `KeyframeGpuLock`。
3. `create_app()` 或 runner 初始化时创建单例并注入 `NotebookRunner`。
4. 添加单元测试：
   - 同时两个线程 acquire 时第二个等待；
   - release 后第二个获得；
   - cancel 时等待线程退出。

### Step C：Step6 接入队列状态

1. 在 runner Step6 分支前设置 `keyframe_maintaining_queued`。
2. 获取锁后设置 `keyframe_maintaining_running`。
3. Step6 logs/details 记录排队、获取、释放。
4. `_is_running_status()` 识别 queued。
5. `recover_interrupted_runs()`、复制、删除、编辑保护同步识别 queued。

### Step D：UI 与文档

1. Step 6 状态组增加 queued 显示。
2. CSS 给 queued 一个 waiting/neutral 样式。
3. 用户文档更新并行运行说明。
4. `../../engineering/task_report.md` 记录实现结果。

### Step E：轻量验证

建议不跑真实付费 Seedance，只做：

- compileall
- focused tests for jobs/gpu lock/runner queued status
- fake backend 并行 smoke：
  - 启动两个项目 run all；
  - fake Step6 sleep 模拟 GPU；
  - 断言两个项目能同时走到等待/执行状态，Step6 不重叠。
- `git diff --check`

真实验证留给手动实验：

- 同时启动 2-3 个短项目；
- 观察 Step5 并行等待 Seedance；
- 观察 Step6 串行排队；
- 检查 GPU 显存不会出现两个关键帧维护进程同时占用。

## 10. 验收标准

- 可以在 Web UI 中启动多个不同项目运行。
- 同一项目重复 Run 仍被拒绝。
- 多项目同时到达 Step6 时，只有一个 Step6 进入 running，其余显示 queued。
- queued 项目可被 Stop 中断。
- Step6 完成/失败/中断后 GPU 锁释放，下一个 queued Step6 自动继续。
- 服务重启后 queued/running 状态不会永久卡住，恢复为 interrupted 或可重新运行状态。
- 不改变 Shot/Attempt/versioning、prompt、reference selection、keyframe selection 算法。

## 11. 后续扩展

当单 GPU 锁不足以支撑实验规模时，可再扩展：

- `VIDEOGEN_KEYFRAME_GPU_WORKERS=2`
- `VIDEOGEN_KEYFRAME_GPUS=0,1`
- 每个 GPU 一个独立 worker subprocess，通过环境变量绑定 `CUDA_VISIBLE_DEVICES`
- 首页显示 GPU queue 与 active owner
- Seedance/LLM/VLM provider 级并发限流

这些都不应进入第一版，以保持实现轻量和可维护。
