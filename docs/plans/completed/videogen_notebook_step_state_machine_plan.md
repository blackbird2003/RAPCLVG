# videogen_notebook Step 状态机与自动推进设计

本文档记录 2026-07-04 讨论形成的新版 `videogen_notebook` Step 状态机、自动推进倒计时、编辑/反思接口预留，以及 Seedance 提示词组织拆分计划。目标是让 Web UI 更接近 notebook：每个 Step 有清楚的局部状态，项目/Shot 级 Run 负责自动推进，用户可以在 Step 间倒计时窗口内检查并打断。

## 1. 设计目标

1. Step 状态表达 Step 自身真实状态，不混入“等待下一个 Step”这类 runner 状态。
2. Shot 顶层状态只作为摘要和首页显示，不作为唯一真相源。
3. 自动运行时，任何 Step 完成后进入统一的 Step 间倒计时，再进入下一个 Step。
4. 用户可在倒计时期间点击任意可编辑 Step 的 `Edit`，使自动运行暂停，并重置该 Step 之后的结果。
5. 为 `Run Step`、`Edit/Save`、`Reflect` 三类 Step 级按钮预留统一接口。
6. 初版只实现 Seedance 提示词组织 Step 的 Edit 功能；其他 Step 的 Edit/Reflect 先保留 UI/API 位置或后端接口形状。

## 2. Step 列表

建议 Step 序列为：

```text
shot_design
visual_plan
reference_selection
seedance_prompt
seedance_generation
keyframe_maintaining
```

含义：

- `shot_design`：人工/json 导入的 Shot 输入，包括 video prompt、cut、generation mode、duration。
- `visual_plan`：Visual Elements Plan，LLM 维护视觉元素集合与状态表。
- `reference_selection`：Historical Reference Selection，基于视觉元素状态选择历史参考帧。
- `seedance_prompt`：Seedance Prompt Composition，按规则组织完整 Seedance prompt，不调用外部模型。
- `seedance_generation`：Seedance Video Generation，真实或 fake 提交 Seedance 并下载视频。
- `keyframe_maintaining`：Keyframe Maintaining，提取关键帧并做 VLM 标注，维护 Produced Visual Memory。

## 3. Step 状态

普通算法 Step 使用统一状态集合：

```text
draft
running
completed
failed
interrupted
editing
reflecting
```

语义：

- `draft`：从未运行，或上游变更后被重置。
- `running`：正在执行初始算法流程。
- `completed`：已有可用产物。
- `failed`：运行失败，保留日志和可能的部分产物。
- `interrupted`：用户主动中断，可视为一种可重跑失败态。
- `editing`：用户正在人工编辑已有产物。
- `reflecting`：LLM 正在对已有产物进行自动复核/修正。

约束：

- `editing` 与 `reflecting` 的前提是该 Step 至少完成过一次 `running` 并存在初始产物。
- `editing -> completed` 表示用户保存人工修改。
- `reflecting -> completed` 表示 LLM 修正完成。
- `failed/interrupted -> running` 表示重跑。
- `completed -> running` 表示放弃当前产物并重新运行该 Step。

特殊 Step：

- `shot_design` 没有 `running`，通常是 `completed` 或 `editing`。
- `seedance_generation` 不提供 `editing` 和 `reflecting`，只支持 `draft/running/completed/failed/interrupted`。

## 4. Shot 与 Project 状态

Step 状态是精细真相源。Shot 顶层状态只作为摘要：

```text
draft
in_progress
waiting_next_step
paused_for_edit
completed
failed
interrupted
```

Project 顶层状态同样作为摘要：

```text
draft
running
waiting_next_step
paused_for_edit
partial
completed
failed
interrupted
```

建议后续逐步让 UI 和 runner 主要读取 `shot.steps[step].status`，顶层状态由 Step 状态与自动运行器状态推导或缓存。

## 5. 自动推进与 Step 间倒计时

倒计时不属于任何 Step。它属于项目/Shot 级自动运行器状态。

新增或扩展 Project 级 `run_state`：

```json
{
  "status": "waiting_next_step",
  "scope": "run_all",
  "shot_id": "0008",
  "from_step": "visual_plan",
  "to_step": "reference_selection",
  "deadline_at": "2026-07-04T12:00:30Z",
  "delay_seconds": 30
}
```

状态建议：

```text
idle
running
waiting_next_step
paused_for_edit
stopping
stopped
failed
completed
```

触发规则：

- 用户点击 `Run Step`：只运行当前 Step，完成后停止，不触发倒计时。
- 用户点击 `Run Shot`：从当前 Shot 第一个未完成 Step 开始；每个 Step 完成后进入倒计时，再进入下一个 Step。
- 用户点击 `Run all`：按 Shot 顺序运行；每个 Step 完成后进入倒计时，再进入下一个 Step 或下一 Shot。

倒计时设置：

```json
{
  "generation": {
    "auto_run_step_review_delay_seconds": 10,
    "require_human_confirmation_before_seedance": false
  }
}
```

UI 选项：

```text
0s / 10s / 30s / 60s / 120s，默认 10s
```

`auto_run_step_review_delay_seconds` 不再提供 `Never`。它只表示自动运行时每个 Step 完成后，进入下一个 Step 前的检查等待时间。

另设全局开关 `require_human_confirmation_before_seedance`，UI 名称建议为 `Seedance submit requires human confirmation`，默认关闭。

- 关闭时：自动运行按统一 Step 间倒计时推进，倒计时结束后可自动进入 `seedance_generation` 并提交 Seedance。
- 开启时：自动运行推进到 `seedance_generation` 前停止。用户必须手动点击 `seedance_generation` 的 `Run Step` 才会真实提交 Seedance。该 Step 完成后，如果用户再次点击高级别 `Run Shot/Run all`，流程可继续后续 Step。

倒计时期间用户操作：

- 用户点击任意可编辑 Step 的 `Edit`：
  - `run_state.status = paused_for_edit`
  - 被编辑 Step 进入 `editing`
  - 被编辑 Step 之后的 Step 全部 reset 为 `draft`
  - 当前自动运行停止，不自动恢复；用户需要手动再次点击 `Run Shot` 或 `Run all` 才能恢复自动执行状态
- 用户点击 Stop：
  - 当前运行/等待中断
  - `run_state.status = stopped` 或 `interrupted`

## 6. 下游重置规则

当 Step X 的产物发生变化时：

```text
Step X -> completed
Step X+1..end -> draft
```

例子：

编辑 `visual_plan` 保存后：

```text
visual_plan: completed
reference_selection: draft
seedance_prompt: draft
seedance_generation: draft
keyframe_maintaining: draft
```

编辑 `seedance_prompt` 保存后：

```text
seedance_prompt: completed
seedance_generation: draft
keyframe_maintaining: draft
```

这条规则应由统一 helper 实现，不应散落在各 route 中。

## 7. Step 级按钮与接口预留

四个主要可检查、可编辑、可反思 Step：

```text
visual_plan
reference_selection
seedance_prompt
keyframe_maintaining
```

Step 级按钮统一预留：

- `Run Step`
- `Edit` / `Save`
- `Reflect`

初版实现要求：

- `visual_plan`：已有 Edit/Save；Reflect 调用 Visual Plan Reflection，写回表格但不引入额外 Step 状态。
- `reference_selection`：初版只预留 Edit 与 Reflect 按钮/API，不实现表格编辑；
- `seedance_prompt`：初版必须支持 Edit/Save；Reflect 按钮/API 预留。
- `keyframe_maintaining`：初版只预留 Edit 与 Reflect 按钮/API，不实现 Produced Visual Memory 标签编辑；
`seedance_generation` 是付费视频提交 Step，不属于上述可编辑/反思 Step。初版仅提供 `Run Step`、Stop/Interrupt 和失败重跑；不提供 Edit/Save/Reflect。

`seedance_prompt` 初版行为：

- `Run Step` 瞬间按规则生成 prompt。
- `Edit` 显示 textarea。
- `Save` 保存用户版本并 reset `seedance_generation` 与 `keyframe_maintaining`。
- Reflect 仅预留按钮/API，不实际调用 LLM。

Visual Plan Reflect 补充语义：

- Reflect 的前提是 `visual_plan` 已经有一次初步结果。
- Reflect 本身不设置 `visual_plan_reflecting` 之类的 Step 状态；运行期间只通过项目 `run_state` 表示后台任务。
- 手动 Reflect 成功后，`visual_plan` 仍为 completed，并按 Visual Plan 输出变更规则重置下游 Step。
- 自动运行模式下，若 `auto_reflect_visual_plan=true`，`Run Shot` / `Run all` 在 Step2 初步结果完成后自动执行一次 Reflect，再继续 Step3。
- 自动 Reflect 失败时只记录日志，不阻断后续自动推进。

建议 API 形状：

```text
POST /projects/{project_id}/shots/{shot_id}/steps/{step}/run
POST /projects/{project_id}/shots/{shot_id}/steps/{step}/edit
POST /projects/{project_id}/shots/{shot_id}/steps/{step}/save
POST /projects/{project_id}/shots/{shot_id}/steps/{step}/reflect
```

对于预留的 Edit/Reflect，均实现为不可点击的按钮。

## 8. Seedance Prompt Composition Step

当前 Seedance prompt 是由前序结果按规则组装，不调用 LLM，也没有明显耗时。因此该 Step 可以非常轻：

```text
draft -> completed
completed -> editing -> completed
```

持久化建议：

```text
attempt/
  seedance_prompt/
    prompt.txt
    prompt.json
    logs.jsonl
```

`prompt.json`：

```json
{
  "system_prompt": "规则组装得到的初始 prompt",
  "final_prompt": "当前最终 prompt",
  "manual_edited": false,
  "updated_at": "...",
  "source": {
    "visual_plan_revision": 1,
    "reference_selection_revision": 1
  }
}
```

`seedance_generation` Step 必须只读取 `seedance_prompt/prompt.txt` 或 `prompt.json.final_prompt`，不再临时重新组装 prompt。

## 9. 实施步骤

建议分阶段实现。

### Phase 1：Prompt Step 拆分与编辑

1. 在 `STEP_SEQUENCE` 中插入 `seedance_prompt`。
2. 把现有 Seedance prompt 组装逻辑从 `seedance_generation` 中拆到新 runner 方法。
3. `seedance_prompt` Step 写入 `seedance_prompt/prompt.txt` 和 `prompt.json`。
4. `seedance_generation` Step 改为读取保存后的 final prompt。
5. 增加 prompt Edit/Save UI 与 route。
6. Save prompt 后 reset `seedance_generation` 与 `keyframe_maintaining`。
7. 增加 focused tests：Prompt Step 不触发 Seedance，编辑后的 prompt 被 Step5 使用。

### Phase 2：统一 Step 状态与下游 reset

1. 将 Step 状态统一为短状态：`draft/running/completed/failed/interrupted/editing/reflecting`。
2. 顶层 Shot 状态改为摘要，逐步减少对 `visual_plan_completed` 这类长状态的依赖。
3. 增加统一 helper：`set_step_status`、`reset_downstream_steps`、`derive_shot_status`。
4. 保持旧项目加载时的兼容映射。

### Phase 3：自动运行倒计时

1. 在 `settings.json` 增加 `auto_run_step_review_delay_seconds`。
2. 在 `settings.json` 增加 `require_human_confirmation_before_seedance`，默认 `false`。
3. 页面设置区增加 `Auto-run step review delay` select，选项为 `0s/10s/30s/60s/120s`。
4. 页面设置区增加 `Seedance submit requires human confirmation` 开关。
5. Project 增加 `run_state`。
6. `Run Shot/Run all` 每个 Step 完成后写入 `run_state.waiting_next_step` 并等待。
7. `Run Step` 不触发倒计时。
8. 若 `require_human_confirmation_before_seedance=true` 且下一步是 `seedance_generation`，自动运行停止并提示用户手动点击该 Step 的 `Run Step`。
9. 前端根据 `run_state.deadline_at` 展示倒计时提示。
10. Edit/Stop 能暂停 run_state。

### Phase 4：Reflect 预留与逐步实现

1. 添加 Reflect route 和按钮位。
2. 未实现 Step 返回 `501` 或 UI disabled。
3. 四个主要可检查 Step 的 Reflect 本轮全部只预留，不实现具体 LLM 逻辑；后续另行讨论设计。

## 10. 已确认决策

以下决策来自 2026-07-04 讨论：

1. `reference_selection` 与 `keyframe_maintaining` 的 Edit 初版仅预留按钮/API，不实现具体编辑。
2. `visual_plan` 的 Edit 已实现，继续保留。
3. `seedance_prompt` 的 Edit 本次应加入实现，因为它只是文本修改，复杂度低。
4. 四个主要可检查 Step 的 Reflect 暂时仅预留按钮/API，具体设计后续讨论。
5. 自动运行期间点击 Edit 后，Save 完成不自动恢复原来的 Run Shot/Run all；用户需要手动再次点击高级别 Run。
6. `auto_run_step_review_delay_seconds` 去掉 `Never` 选项。
7. 新增全局开关 `require_human_confirmation_before_seedance`。开启时，自动运行到 `seedance_generation` 前停止，必须用户手动点击该 Step 的 `Run Step` 才会提交 Seedance；完成后继续后续步骤需要用户再次点击高级别 Run。

## 11. 初版非目标

- 不需要一次性实现所有 Step 的人工编辑。
- 不需要让 Reflect 真正调用 LLM。
- 不需要复杂前端状态管理框架。
- 不需要迁移旧 Web UI 项目。
- 不需要改变 Visual Element Memory 核心算法。
