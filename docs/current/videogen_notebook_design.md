# videogen_notebook 新版 Web UI 设计文档

## 1. 目标定位

`videogen_notebook` 是面向下一阶段实验的新 Web UI。它延续当前 notebook 式交互，但不再承担旧 Web 项目的兼容负担，只服务最新的 Visual Element Memory pipeline。

核心目标是：

1. 保持当前 Web UI 的主要使用体验：导入剧本、逐 shot 编辑、Run Shot、Run All、查看参考帧、prompt、输出视频、关键帧记忆与最终拼接。
2. 采用更轻量清楚的数据结构，使每个项目都是自包含 bundle。
3. 原生支持项目复制、迁移、删除，以及从指定 shot 处分叉。
4. 避免全局数据库深度绑定 attempt、reference、asset 等项目内部状态。
5. 为后续多版本记忆策略、手动修正视觉元素、参考图上传、实验对比保留扩展空间。

新版不要求读取或迁移当前旧 Web 项目。旧 Web UI 可以继续作为历史实验查看器和兼容入口。`videogen_notebook` 的用户界面、命令和项目数据结构中不再出现旧项目名或旧品牌元素。

## 2. 设计原则

### 2.1 项目自包含

每个项目目录保存恢复该项目所需的全部状态和资源：

- 原始或编辑后的 story；
- shot 输入；
- attempt 快照；
- Seedance 请求与响应记录；
- 视觉元素集合状态；
- 历史参考帧选择结果；
- 生成视频；
- 生成后关键帧与标注；
- 最终拼接视频；
- 运行日志和费用统计。

项目复制应尽量接近目录级复制，而不是依赖复杂的全局 ID remap。

### 2.2 Workspace 只负责发现项目

第一版不设置全局业务数据库。Workspace 首页直接扫描：

```text
<workspace>/projects/*/project.json
```

`project.json` 提供首页所需的项目名称、状态、更新时间、缩略图和 active shot 信息。项目业务状态不进入全局数据库。

后续如果项目数量显著增加，可以增加一个可重建的索引缓存，但该索引不应成为真相源。

### 2.3 项目内部 ID 局部化

`shot_id`、`attempt_id`、`asset_id`、`reference_id` 都是项目内部 ID。不同项目可以重复使用相同 ID。

推荐形式：

```text
shot_id: 0001, 0002, 0003
attempt_id: a001, a002
asset_id: img_000001, vid_000001, thumb_000001
reference_id: ref_001
```

如果需要全局定位，用组合路径表达：

```text
project_id / shot_id / attempt_id
```

而不是要求 `attempt_id` 自身全局唯一。

### 2.4 Attempt 不可变

每次提交生成都创建新的 attempt。attempt 保存当时的输入、视觉元素状态、参考帧、提交 prompt、Seedance 任务、输出和后处理结果。

编辑 shot 输入不会覆盖已有 attempt；下一次运行会产生新 attempt，并更新该 shot 的 current attempt 指针。

### 2.5 媒体资源统一 asset 管理

所有图片、视频、缩略图、尾帧、smooth tail video、关键帧都进入项目内部 asset manifest。

业务 JSON 中优先引用 `asset_id`，避免散落绝对路径。路径通过 asset manifest 解析。

### 2.6 分叉是复制加 reset

从 shot N 分叉的语义是：

1. 复制整个项目 bundle；
2. 修改新项目的 `project_id`、名称与时间戳；
3. 保留 shot 1..N 的 current lineage；
4. reset shot N+1..end；
5. 更新项目状态并刷新首页。

这应是内建操作，而不是在全局数据库中重组一批互相耦合的记录。

## 3. 目录结构

默认 workspace：

```text
.runtime/videogen_notebook/
  projects/
    <project_id>/
      project.json
      story.json
      settings.json
      assets/
        manifest.json
        images/
        videos/
        thumbnails/
        temp/
      memory/
        visual_state.json
        lineage.json
      shots/
        0001/
          shot.json
          attempts/
            a001/
              attempt.json
              request.json
              response.json
              selected_references.json
              visual_element_status.json
              produced_visual_memory.json
              logs.jsonl
        0002/
          shot.json
          attempts/
            a001/
              ...
      final/
        current.mp4
        assembly.json
      logs/
        project.log.jsonl
  locks/
    global_run.lock
```

说明：

- `project.json` 是首页和项目入口文件。
- `story.json` 保存导入后规范化的剧本。
- `settings.json` 保存项目级生成设置。
- `assets/manifest.json` 是项目内媒体资源索引。
- `shot.json` 保存该 shot 的当前可编辑输入与 current attempt。
- `attempt.json` 保存一次运行的不可变快照。
- `memory/visual_state.json` 保存当前项目级视觉元素集合状态，便于恢复和调试。
- `final/prefix_<shot_id>_<attempt_id>.mp4` 是完成到某个 shot 的连续前缀拼接结果。每个 completed shot 的 `shot.state.current_final_video_asset_id` 和 attempt `outputs.current_final_video_asset_id` 都指向截至该 shot 的 Current output；`project.json.current_final_video_asset_id` 指向当前 completed prefix 的最后一个 shot 产物。后续 shot 生成时优先读取上一个 completed shot 保存的 Current output，并与本 shot 原始视频拼接出新的 Current output。
- 项目级运行锁用于禁止同一项目重复启动；不同项目可并行运行。

## 4. 核心数据结构

### 4.1 project.json

```json
{
  "schema_version": 1,
  "project_id": "p_20260701_abcdef",
  "name": "The Little Prince",
  "pipeline": "visual_element_memory_v1",
  "status": "draft",
  "active_shot_id": null,
  "created_at": "2026-07-01T12:00:00Z",
  "updated_at": "2026-07-01T12:00:00Z",
  "thumbnail_asset_id": null,
  "shot_count": 12,
  "completed_prefix": 0,
  "current_final_video_asset_id": null,
  "cost_summary": {
    "estimated_cny": 0.0,
    "seedance_tasks": 0
  }
}
```

首页只需要读取该文件，不需要扫描每个 attempt。

### 4.2 settings.json

```json
{
  "generation": {
    "default_duration_seconds": -1,
    "default_cut_mode": "default",
    "default_non_cut_mode": "smooth",
    "audio": true,
    "auto_run_step_review_delay_seconds": 10,
    "algorithm_step_max_attempts": 5,
    "smooth_reference_seconds": 2.0,
    "require_human_confirmation_before_seedance": false,
    "auto_reflect_visual_plan": true,
    "force_animation_style": true
  },
  "visual_element_memory": {
    "sink_frame_count": 0,
    "max_retrieved_frames": 4,
    "selection_mode": "greedy_coverage"
  },
  "seedance": {
    "model": "doubao-seedance-2-0-260128",
    "resolution": "720p",
    "ratio": "16:9"
  },
  "keyframes": {
    "profile": "loose"
  }
}
```

约束：

```text
predefined_reference_count + sink_frame_count + max_retrieved_frames <= 9
```

这是 Seedance 静态参考图片预算约束。Smooth 的 `reference_video` 不占用图片预算。
运行时会优先保留 Shot 级预定义参考图，再按剩余槽位收紧 sink 和历史检索数量。

### 4.3 story.json

导入剧本时将旧格式统一规范化成 ordered shots：

```json
{
  "title": "The Little Prince",
  "shots": [
    {
      "shot_id": "0001",
      "scene_num": 1,
      "shot_num": 1,
      "video_prompt": "...",
      "is_cut": true,
      "duration_seconds": 8,
      "generation_mode": "default",
      "predefined_references": [
        {
          "image_path": "assets/predefined_references/pref_0001_pref_0001.jpg",
          "label": "角色参考",
          "guidance": "保持该角色的动画造型与服装。"
        }
      ]
    },
    {
      "shot_id": "0002",
      "scene_num": 1,
      "shot_num": 2,
      "video_prompt": "...",
      "is_cut": false,
      "duration_seconds": 8,
      "generation_mode": "smooth"
    }
  ]
}
```

`scene` 只保留为展示和导入兼容概念，执行单位始终是 shot。

### 4.4 shot.json

```json
{
  "shot_id": "0002",
  "order_index": 2,
  "scene_num": 1,
  "shot_num": 2,
  "inputs": {
    "video_prompt": "...",
    "is_cut": false,
    "generation_mode": "smooth",
    "duration_seconds": 8,
    "predefined_references": []
  },
  "state": {
    "status": "draft",
    "revision": 1,
    "current_attempt_id": null,
    "updated_at": "2026-07-01T12:00:00Z"
  },
  "archived_attempt_ids": []
}
```

第一版只暴露最新 pipeline 所需输入：

- Video prompt；
- Cut；
- Generation mode；
- Generation duration。

旧版 Sink/Retrieve/Recent shot 级开关不出现在新版 UI。Sink 数量和最大检索历史帧数由项目级设置控制。

### 4.5 attempt.json

```json
{
  "attempt_id": "a001",
  "shot_id": "0002",
  "revision": 1,
  "status": "completed",
  "created_at": "2026-07-01T12:05:00Z",
  "updated_at": "2026-07-01T12:10:00Z",
  "input_snapshot": {
    "video_prompt": "...",
    "is_cut": false,
    "generation_mode": "smooth",
    "duration_seconds": 8,
    "project_settings": {}
  },
  "visual_element": {
    "state_before_path": "visual_element_status.json",
    "state_after_asset_id": null
  },
  "references": {
    "selected_references_path": "selected_references.json"
  },
  "prompt": {
    "submitted_prompt": "...",
    "prompt_asset_id": null
  },
  "seedance": {
    "task_id": "cgt-...",
    "request_path": "request.json",
    "response_path": "response.json",
    "usage": {}
  },
  "outputs": {
    "raw_video_asset_id": "vid_000004",
    "stitched_video_asset_id": null
  },
  "postprocess": {
    "produced_visual_memory_path": "produced_visual_memory.json",
    "keyframe_profile": "loose"
  },
  "error": null
}
```

`attempt.json` 中路径都是相对 attempt 目录或 asset manifest 的引用，不写绝对路径。

### 4.6 assets/manifest.json

```json
{
  "schema_version": 1,
  "assets": {
    "img_000001": {
      "kind": "image",
      "path": "assets/images/img_000001.jpg",
      "created_at": "2026-07-01T12:00:00Z",
      "source": {
        "type": "produced_keyframe",
        "shot_id": "0001",
        "attempt_id": "a001"
      },
      "metadata": {
        "width": 1920,
        "height": 1080,
        "thumbnail_asset_id": "thumb_000001"
      }
    },
    "vid_000004": {
      "kind": "video",
      "path": "assets/videos/vid_000004.mp4",
      "source": {
        "type": "seedance_output",
        "shot_id": "0002",
        "attempt_id": "a001"
      },
      "metadata": {
        "duration_seconds": 8,
        "has_audio": true
      }
    }
  }
}
```

Asset manifest 是复制、迁移、UI 渲染和后续导出的关键。业务状态只引用 `asset_id`，不直接依赖实际文件路径。

## 5. Notebook UI 布局

新版 UI 保持当前 notebook 页面心智模型，但只展示 Visual Element Memory 版本。

### 5.1 首页

项目列表通过扫描 `projects/*/project.json` 获取。

每个项目卡片展示：

- 项目名称；
- 状态；
- completed prefix；
- 更新时间；
- 缩略图；
- 操作：
  - Open；
  - Duplicate；
  - Delete。

第一版 Duplicate 可以默认完整复制项目现场。Fork 可以在项目页 shot 操作中提供。

### 5.2 项目顶部

项目页顶部展示：

- 项目名称，可重命名；
- Save all；
- Run all；
- Stop；
- 当前状态；
- 费用摘要；
- 项目设置：
  - Sink frame count；
  - Max retrieved frames；
  - 默认 duration；
  - Seedance 基础配置，只读或高级折叠。

### 5.3 Shot 块

每个 shot 的布局：

```text
Shot N
├─ Inputs
│  ├─ Video prompt
│  ├─ Cut
│  ├─ Generation mode
│  └─ Generation duration
├─ Shot Status
├─ Visual Element Status
│
├─ Selected Historical References
│  └─ Collection Maintenance Details
├─ Input Prompt
├─ Output Video
│  └─ Seedance Attempt Details
├─ Produced Visual Memory
│  └─ Keyframe And VLM Details
└─ Actions
   ├─ Save
   ├─ Run Shot
   ├─ Stop
   ├─ Reset from Here
   └─ Fork after This Shot
```

三个详情面板默认折叠，并分别服务于排障：

- `Collection Maintenance Details`：展示视觉元素集合维护与历史参考选择日志、视觉元素行、参考图行，以及当前 Attempt 记录的 LLM prompt/raw response/parse error 等 record。
- `Seedance Attempt Details`：展示 Attempt/backend/task、提交给 Seedance 的完整 prompt、请求摘要、脱敏响应、阶段日志和错误。
- `Keyframe And VLM Details`：展示关键帧提取/标注日志、postprocess 摘要、produced memory 行，以及当前 Attempt 记录的 VLM prompt/raw response/parse error。

每个 Attempt 目录可额外保存 `visual_element_details.json`。它不是项目运行状态真相源，而是为了历史 Attempt 可复制、可诊断而保存的本 shot 视觉记忆细节快照。

三个核心表格默认展开，但可折叠：

1. Visual Element Status
   展示元素名称、ID、类型、出现时期、备注、状态、状态原因。
   第一版暂不支持手动修改，但数据结构应预留人工 override、手动添加元素、手动删除元素的扩展字段。

2. Selected Historical References
   展示缩略图、来源 shot、引用角色、覆盖元素、冲突元素、分数、说明。该表不需要重复元素排除原因，排除原因已在 Visual Element Status 中展示。

3. Produced Visual Memory
   展示本 shot 生成后选出的关键帧缩略图、frame rank、可见元素、标注框信息、是否进入候选池。

Input Prompt 默认折叠，展开后展示完整 Seedance prompt。为避免当前旧 UI 的滚动问题，展开状态应是前端稳定状态，不被轮询刷新重置。

### 5.4 最底部项目结果

页面底部展示：

- 当前连续完成前缀拼接视频；
- 拼接日志；
- 费用信息；
- 已完成 shot 数；
- 失败或中断摘要。

## 6. 运行流程

### 6.1 Run Shot

执行单个 shot：

```text
读取 project.json/settings.json/shot.json
检查前置 shot 状态
冻结 input_snapshot
从上一 completed attempt 恢复 visual state
LLM 维护视觉元素集合与当前 shot 元素状态
VLM/规则评估历史候选帧
选择 Selected Historical References
按 generation_mode 添加连续性媒体
组成 Seedance prompt
提交 Seedance
轮询、下载视频、登记 video asset
提取关键帧
VLM 标注 Produced Visual Memory
更新 visual_state
更新 shot current_attempt_id
将上一个 completed shot 的 Current output 与本 shot 原始视频拼接
保存本 shot 的 Current output 并更新 completed prefix
更新 project.json
释放 global_run.lock
```

### 6.2 Run All

从第一个非 completed shot 开始顺序执行。Web 路由只启动后台任务并立即返回；每个 shot 完成后立即写入项目状态，浏览器刷新不影响已完成结果。

后台任务池允许多个不同项目同时 Running。由于等待主体是 Seedance/LLM/VLM
远端轮询，项目之间可以并行等待；本地 GPU 密集的 Step 6 Keyframe
Maintaining 通过单 GPU 锁串行排队，排队中的 Shot 显示
`keyframe_maintaining_queued`。

### 6.3 Stop

Stop 设置项目内 cancel flag，并让 worker 在安全检查点退出：

- Seedance 已提交的任务不强制取消；
- 当前 attempt 标记为 interrupted；
- 当前 shot 回到可重跑状态；
- 下游状态不自动修改，除非用户执行 reset from here。

运行中 shot 可能已经写入 `current_attempt_id`，但 `attempt.json` 尚未完成写入；项目读取逻辑必须容忍这种短暂状态并继续渲染页面。

如果 Web 服务重启后发现项目或 shot 仍处于 `running`，但当前进程没有对应后台任务，则启动时将这些孤儿运行态恢复为 `interrupted`，清空 `active_shot_id` 和 running shot 的 `current_attempt_id`，用户可手动重跑。

### 6.4 Reset from Shot

从 shot N reset：

- shot N..end 状态改为 draft；
- shot N..end 保留从剧本 JSON 导入及用户保存后的基础输入信息，包括 video prompt、cut、generation mode、duration；
- current attempt 清空；
- 旧 attempt 可移动到 `archived_attempt_ids` 或保留目录但不参与 current lineage；
- produced memory 从 current visual state 中移除；
- final video 回退到 N-1 的完成前缀；
- project completed prefix 更新。

第一版保留旧 attempts 文件，但 UI 默认隐藏 archived attempt，以便后续做对比查看。

## 7. 复制与分叉

### 7.1 Duplicate Project

完整复制项目现场：

```text
copytree old_project -> new_project
修改 new_project/project.json:
  project_id
  name
  created_at
  updated_at
  status 若原项目 running 则改为 interrupted 或 draft
删除 new_project 内运行锁/临时文件
```

因为项目内部 ID 是局部 ID，`attempt_id`、`asset_id`、`reference_id` 不需要改变。

### 7.2 Fork from Shot

从 shot N 分叉：

```text
Duplicate Project
Reset from shot N+1
修改项目名，例如 "原项目名 - fork from shot N"
```

第一版 fork 的默认且唯一语义是 `Fork after this shot`：

- 保留 shot 1..N 的 current attempt、输出视频、视觉元素状态、历史候选帧和最终前缀；
- shot N+1..end 保留从剧本 JSON 导入及用户保存后的基础输入信息，包括 video prompt、cut、generation mode、duration；
- shot N+1..end 的状态改为 draft；
- shot N+1..end 的旧 attempt 文件保留为 archived，但默认不在 UI 中显示；
- 新项目的 final video 回退到 shot N 的连续完成前缀。

第一版不提供“从当前 shot 开始重做”的 fork 按钮。若用户需要该效果，可在新项目中手动 Reset from shot N。

### 7.3 Delete Project

删除项目就是删除项目目录。为了避免误删，必须有确认弹窗。

如果项目正在 Running，则禁止删除，除非先 Stop 并释放 lock。

## 8. 与当前 pipeline 的关系

新版应复用已有稳定模块，而不是重写核心算法：

- Seedance API 客户端：`seedance_client.py`
- prompt 组织：现有 prompting 模块
- Visual Element Memory：现有 visual element memory 模块
- keyframe extraction：`extract_keyframes.py`
- keyframe settings：`keyframe_settings/*.json`
- Smooth / RIFE 逻辑可从当前实现中抽出可复用层

需要新增的是数据编排层：

```text
videogen_notebook/
  app.py
  project_store.py
  asset_store.py
  runner.py
  locks.py
  schemas.py
  views.py
  templates/
  static/
```

推荐职责：

- `project_store.py`：项目目录、JSON 原子读写、导入/复制/分叉/删除。
- `asset_store.py`：asset id 分配、文件登记、缩略图登记、相对路径解析。
- `runner.py`：shot 执行状态机，调用现有 pipeline 模块。
- `locks.py`：workspace 级 global run lock。
- `schemas.py`：JSON schema / dataclass / Pydantic 校验。
- `views.py`：把项目 bundle 转成模板视图模型。

## 9. JSON 写入与可靠性

所有状态文件写入必须使用原子写：

```text
write file.tmp
fsync
rename file.tmp -> file
```

运行中应周期更新：

- `project.json.status`
- `project.json.active_shot_id`
- 当前 `attempt.json.status`
- `logs.jsonl`

Worker 启动时检查 `global_run.lock`：

- pid 存活：禁止新任务；
- pid 不存在：清理 stale lock；
- lock 对应项目状态为 running 但 pid 不存在：标记为 interrupted 或 failed，需要用户确认后继续。

## 10. 第一版非目标

第一版不做：

- 旧 Web 项目迁移；
- classic pipeline；
- shot 级 Sink/Retrieve/Recent 开关；
- 多项目并发运行；
- 多用户权限；
- 复杂 DAG；
- 手动上传/删除/排序参考图；
- 手动编辑视觉元素状态；
- 生成结果对比视图；
- 项目 bundle 一键导出 zip；
- 云端数据库或账号系统。

这些能力应通过当前 bundle 结构自然扩展，而不是提前塞进第一版。

## 11. 分阶段实现计划

### Phase 0：只读原型

目标：验证 bundle 结构和 UI 渲染。

- 新建 `videogen_notebook/`；
- 实现 workspace 扫描；
- 新建空项目；
- 导入 JSON 剧本并规范化；
- 渲染项目页和 shot 块；
- 不执行真实生成。

验收：

- 项目目录结构符合本文；
- 首页可列出项目；
- 项目页可展示 Visual Element Memory 版 shot 布局。

### Phase 1：项目编辑与复制/分叉

目标：先把数据模型跑顺。

- Save all；
- 重命名；
- Duplicate Project；
- Delete Project；
- Reset from Shot；
- Fork after Shot；
- JSON 原子写；
- 基础单元测试覆盖复制和 fork 后路径独立。

验收：

- 复制项目后删除原项目，新项目仍可打开；
- fork 后前缀 shot 保留，后续 shot 保留基础输入并回到 draft。

### Phase 2：接入 fake runner

目标：验证 notebook 状态机。

- 使用 FakeSeedance / FakeLLM / FakeVLM；
- 生成占位视频或复用本地短视频；
- 写 attempt、asset、selected references、produced memory；
- 更新 final video；
- UI 轮询稳定，不重置展开/滚动状态。

验收：

- Run Shot / Run All / Stop / Reset 可端到端执行；
- 不需要网络和付费 API。

### Phase 3：接入真实 Visual Element Memory pipeline

目标：复用当前 CLI/Web 已验证的算法模块。

- LLM 维护视觉元素集合；
- 历史关键帧选择；
- precise reference prompt；
- Seedance 提交、轮询、下载；
- keyframe extraction；
- VLM 标注 produced memory；
- Smooth reference_video 与 RIFE 拼接。

验收：

- 不发起默认付费测试；
- 使用 fake client 覆盖请求内容顺序和 prompt；
- 使用本地小视频做 keyframe/RIFE smoke test。

### Phase 4：体验打磨

目标：让它成为日常实验入口。

- 缩略图缓存；
- 更好的日志折叠；
- 费用统计；
- 错误详情；
- 运行中状态恢复；
- 导入 project bundle。

## 12. 已确认产品决策

1. Fork 的默认语义是保留当前 shot 及其之前的现场，从下一 shot 开始重做。后续 shot 保留从剧本 JSON 导入及用户保存后的基础信息，状态为 draft。
2. Archived attempts 保留文件，但第一版 UI 默认隐藏。
3. Visual Element Status 第一版暂不支持手动修正，但数据结构预留人工 override、手动添加元素、手动删除元素的可能。
4. Project bundle 第一版不需要一键导出 zip。
5. 新 UI 与旧 Web UI 使用分开的启动命令和端口。Python 包名使用 `videogen_notebook`，启动命令为 `python -m videogen_notebook start`，也可使用包装脚本 `./run_videogen_notebook.sh`。默认端口使用 `7870`，避免与旧 Web UI 的 `7860` 混淆。
6. `real` runner 是默认启动后端，并默认执行真实付费提交：Visual Element Memory LLM/VLM、Seedance 提交、Smooth reference_video 发布和关键帧提取都会实际运行。若需要 dry-run 调试，显式设置 `VIDEOGEN_NOTEBOOK_REAL_SUBMIT=0`。

## 13. Step Pipeline 更新

新版 shot 不再把算法流程作为一个黑盒 Attempt 一次性执行，而是拆成六个从上到下推进的 notebook step：

1. `Shot Design`：人工或 JSON 导入的基础设计，包括 video prompt、cut、generation mode、duration 和 Shot 级预定义参考图。该 step 没有运行按钮。
2. `Visual Elements Plan`：调用 LLM 维护跨 shot 视觉元素集合，并为当前 shot 判断已有元素状态和新增元素。
3. `Historical Reference Selection`：以上一步的元素状态为输入，选择历史关键帧和 sink 参考帧，并用 LLM 为带有整体描述的已选参考帧补充 `reference_guidance`。如果当前 Shot 已有预定义参考图，本 Step 只使用剩余静态图片槽位。
4. `Seedance Prompt Composition`：基于当前 shot 输入、预定义参考图说明、历史参考帧和 generation mode 组装完整 Seedance prompt。该 step 不调用外部模型，可人工编辑保存。Prompt 中 `[预定义参考图说明]` 位于 `[历史参考图说明]` 之前，Seedance 静态图片提交顺序与 Prompt Image 编号一致。
5. `Seedance Video Generation`：读取已保存的 Seedance prompt，提交/下载视频。
6. `Keyframe Maintaining`：对生成视频提取候选关键帧，调用 VLM 标注可见视觉元素和 `holistic_description`，并把结果纳入当前 completed prefix 的记忆快照。

第 2、3、4、5、6 步均有独立 `Run Step` 按钮和默认折叠日志。三张主体表格默认展开但可折叠：

- `Visual Elements Plan`
- `Historical Reference Selection`
- `Keyframe Maintaining`

第 2、3、4、6 步预留 `Edit` 与 `Reflect` 操作位置。当前已实现第 2 步表格编辑、第 2 步 Visual Plan Reflect、第 3 步 `reference_guidance` 编辑/删除参考帧、第 4 步 prompt 文本编辑，以及第 6 步 `holistic_description` 编辑/删除关键帧；其余 Reflect 流程暂为后续扩展。

`Run Shot` 从当前 shot 第一个未完成或失败的算法 step 开始，顺序推进到 shot completed。`Run all` 从第一个未 completed 的 shot 开始逐 shot 执行 `Run Shot`。

项目级自动运行设置：

- `auto_run_step_review_delay_seconds`：网页端 `Run Shot` / `Run all` 在每个 step 完成后进入下一 step 前的检查等待时间，默认 10 秒，UI 选项为 0/10/30/60/120 秒。低成本 fake 与 dry-run 测试 runner 会跳过真实等待。
- `auto_reflect_visual_plan`：默认开启。开启后，网页端 `Run Shot` / `Run all` 在 Visual Elements Plan 初步结果生成后自动执行一次 Reflect。Reflect 不引入额外 Step 状态；成功后写回 Step2 表格并重置下游，失败只记录日志并继续自动推进。
- `require_human_confirmation_before_seedance`：开启后，自动运行到 `Seedance Video Generation` 前停止，必须用户手动点击该 step 的 `Run Step` 才会提交 Seedance。

### 13.1 状态机

Shot 的 `state.status` 直接表达当前推进位置：

```text
draft
visual_plan_running
visual_plan_completed
visual_plan_failed
reference_selection_running
reference_selection_completed
reference_selection_failed
seedance_prompt_running
seedance_prompt_completed
seedance_prompt_failed
seedance_generation_running
seedance_generation_completed
seedance_generation_failed
keyframe_maintaining_running
keyframe_maintaining_completed
keyframe_maintaining_failed
completed
interrupted
```

`completed` 等价于 `keyframe_maintaining_completed` 后的终态。项目级 `completed_prefix` 只统计连续 `completed` shot。后续 shot 仍只依赖前一个 shot 的完整完成状态，而不是依赖其中某个中间 step。

### 13.2 失效规则

重新运行中间 step 会自动失效其后续 step 和下游 shot：

- 重新运行 `Visual Elements Plan`：失效当前 shot 的 reference selection、Seedance generation、keyframe maintaining，以及所有后续 shot。
- 重新运行 `Historical Reference Selection`：失效当前 shot 的 Seedance generation、keyframe maintaining，以及所有后续 shot。
- 重新运行 `Seedance Video Generation`：失效当前 shot 的 keyframe maintaining，以及所有后续 shot。
- 重新运行 `Keyframe Maintaining`：失效所有后续 shot。

编辑 `Historical Reference Selection` 仅允许修改 `reference_guidance` 或删除参考帧，保存后按重新运行 Step 3 的语义失效下游。编辑 `Keyframe Maintaining` 仅允许修改 `holistic_description` 或删除关键帧，保存后当前 shot 保持 completed，并失效后续 shot。

项目级 `memory/visual_state.json` 只由连续 completed prefix 刷新。半完成或失败 shot 的中间视觉元素快照仅保存在当前 attempt 目录中，不污染后续 shot。

### 13.3 Algorithm Boundary

`VisualElementMemory` 后端应明确拆分为两个算法函数：

```python
plan_visual_elements_for_shot(...)
select_historical_references_for_shot(...)
```

`plan_visual_elements_for_shot` 负责 LLM 决策、插入新元素和产出视觉元素状态表；`select_historical_references_for_shot` 以前者的决策结果为输入，负责 retrieved/sink frame 选择、参考图 metadata 和 prompt context。CLI 与 Web UI 使用同一边界，避免 Web 专用临时逻辑。

Reference selection 当前支持五种项目级模式：

- `greedy_coverage`：默认完整方法，使用视觉元素状态、互补覆盖、质量权重和预算约束选帧；
- `static_top_k`：保留视觉元素规划与标注，但不做互补覆盖式贪心，用于 coverage retrieval 消融；
- `naive_top_k`：跳过 Visual Elements Plan，保持视觉元素集合为空；Step 3 使用当前 prompt 的 CLIP text embedding 对历史关键帧 image embedding 做相似度排序并取 top-k，image embedding 缓存在项目 `assets/embeddings/` 中；Step 6 仍抽关键帧入池，但跳过 VLM 标注并保持 annotation / holistic description 为空；
- `storymem_memory`：跳过 Visual Elements Plan，按 StoryMem 原始 `max_memory_size=10`、`fix=3` 的 Sink+Recent memory bank 规则选择历史关键帧；Step 6 使用 `storymem_original` keyframe profile 维护 keyframe library，抽帧入池但跳过 VLM 标注；历史相似去重仅对本 Shot Step 3 实际选中的 compact StoryMem memory bank 生效，不再扫描所有历史 keyframe；
- `none`：Step 3 跳过，不向 Seedance 提交历史参考图，后续步骤照常执行。

### 13.4 Shot Design 编辑

空项目应支持追加 shot，而不是固定只有一个 shot：

- `Add Shot` 默认追加到末尾；
- scene 默认沿用最后一个 shot 的 `scene_num`；
- `shot_num` 自动递增；
- prompt 使用简短示例；
- duration 默认使用项目设置；
- 第一 shot 默认 `cut=true` / `generation_mode=default`；
- 后续 shot 默认 `cut=false`，generation mode 使用项目设置 `generation.default_non_cut_mode`，初始值为 `smooth`。

添加新 shot 不会失效已有 completed prefix。修改已经产生算法 step 的 shot design 时，需要 reset 当前 shot 及其后续 shot。

### 13.5 Export Shot Design JSON

项目页应支持导出当前 Shot Design JSON。导出内容只包含可编辑剧本设计：

- story name / overview；
- scenes；
- `video_prompts`；
- `cut`；
- `generation_modes`；
- `durations`。

导出 JSON 不包含 attempt、assets、memory、Seedance/VLM 结果。导出的文件应可被当前 import 流程重新导入并恢复同样的 Shot Design。

## 14. 总结

`videogen_notebook` 应被设计为一个轻量、项目自包含、文件系统友好的 notebook 创作工具。它不以全局数据库为项目业务真相源，而是让每个项目自己保存完整状态。

这种设计更贴合当前的实验形态：

- 项目数量和规模有限；
- 主要耗时来自远端生成和本地视觉处理，而不是状态查询；
- 用户需要频繁复制、分叉、迁移和分析实验；
- 可人工检查和修复比复杂数据库关系更重要。

因此，新版的关键不是引入更多框架，而是把项目 bundle、asset manifest、attempt snapshot 和 fork/reset 语义一次设计清楚。这样后续无论继续改进 Visual Element Memory，还是加入新的生成模式和交互式分析，都能保持结构简单、可维护。
