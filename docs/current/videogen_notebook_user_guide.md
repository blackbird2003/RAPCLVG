# videogen_notebook 使用说明

`videogen_notebook` 是一个面向长视频生成实验的轻量 Web UI。它按
Notebook 的方式把一个长视频拆成多个 Shot，并围绕最新的 Visual Element
Memory pipeline 依次完成：视觉元素规划、历史参考帧选择、Seedance Prompt
组织、Seedance 视频生成、生成后关键帧维护和当前总视频拼接。

本文面向使用者，说明如何启动、创建项目、运行 Shot、检查和修改中间结果。

## 1. 启动与访问

在项目目录中执行：

```bash
cd /home/wxh/world_model_projects/StoryMem
./run_videogen_notebook.sh restart
```

常用命令：

```bash
./run_videogen_notebook.sh start
./run_videogen_notebook.sh stop
./run_videogen_notebook.sh restart
./run_videogen_notebook.sh status
./run_videogen_notebook.sh logs
./run_videogen_notebook.sh logs -f
```

默认地址：

```text
http://localhost:7870
```

如果通过 VS Code SSH 或本地浏览器访问远端服务器，通常需要把远端
`7870` 端口转发到本机，然后在笔记本浏览器中打开 `http://localhost:7870`。

默认运行模式是真实 runner，会实际调用付费 LLM/VLM/Seedance 服务。API key
和运行环境由 `storymem_env.sh` 及本机密钥文件提供。

## 2. 首页功能

首页用于管理项目：

- **New project**：创建一个空项目，默认包含一个 Shot。
- **Import JSON**：导入剧本 JSON，系统会解析为按顺序排列的 Shot。
- **Open**：进入项目 Notebook 页面。
- **Duplicate**：复制项目 bundle，用于复现实验或在已有结果上继续修改。
- **Delete**：删除项目 bundle。

项目数据默认保存在：

```text
.runtime/videogen_notebook/projects/
```

每个项目都是相对自包含的目录，包含剧本、设置、Shot 状态、Attempt、图片、
视频、日志和拼接结果。

## 3. 项目级设置

进入项目后，顶部区域可编辑项目名、保存全部修改、导出当前 Shot Design JSON，
并配置全局实验参数。

常用设置：

- **Sink frame count**：每个项目早期 Sink 参考帧数量，默认 `0`。
- **Max retrieved frames**：历史检索参考帧上限，默认 `4`。
- **Reference selection**：
  - `Greedy coverage`：按视觉元素覆盖与质量进行贪心选帧。
  - `Static top-k`：简单 top-k，用于消融实验。
  - `Naive top-k`：不维护视觉元素表，关键帧维护只抽帧入池、不做 VLM 标注；
    Step 3 用当前 prompt 的 CLIP text embedding 检索历史关键帧 image embedding，
    取 `Max retrieved frames` 张参考图。历史帧 image embedding 会缓存在项目的
    `assets/embeddings/` 下，后续复用。
  - `StoryMem memory`：不维护视觉元素表，按 StoryMem 原始 Sink+Recent 规则从历史
    keyframe library 中选择参考图；Step 6 使用 `storymem_original` keyframe profile
    维护关键帧库，抽帧入池但不做 VLM 标注。Step 6 的历史相似去重只比较本 Shot
    Step 3 实际选中的 compact StoryMem memory bank，而不是所有历史 keyframe。
  - `Disable reference images`：不使用历史参考图。Step 3 会直接完成并产出空参考图列表，
    后续 Prompt、Seedance 生成和关键帧维护仍照常执行。
- **Default duration**：新 Shot 默认时长。默认是 `自动 (-1)`，由 Seedance
  2.0 在有效范围内自主选择；也可手动指定 `4-15s` 的整数秒。
- **Default non-cut generation mode**：JSON 导入或新增 Shot 时，非 cut Shot 的默认
  Generation mode。默认 `Smooth`；可改为 `Default` 或 `Last frame only` 用于复现实验。
- **Quality / Aspect ratio**：Seedance 生成画质与宽高比，默认 `720p`
  和 `16:9`。
- **Generate audio**：是否要求 Seedance 生成有声视频。
- **Force animation**：开启时，在可编辑的 Seedance Prompt 中插入动画风格约束，
  默认开启。
- **Prompt modules**：控制 Seedance Prompt 的实验模块，默认全部开启：
  Full script context（当前及前序 Shot，不包含未来 Shot）、Visual element plan、
  Holistic description and reference guidance、Should reference constraints、
  Should exclude constraints。
- **Step review delay**：自动运行时，Step 完成后进入下一 Step 前等待多久。
  可选 `0s / 10s / 30s / 60s / 120s`。
- **Step max attempts**：算法 Step 的完整尝试次数，默认 `5`。当前作用于
  Visual Elements Plan、Historical Reference Selection、Seedance Video Generation
  和 Keyframe Maintaining；最大重试次数等于尝试次数减一。真实运行时每次完整
  Step 失败后会等待 `60s` 再重试，以避开短暂网络或服务端波动。
- **Smooth reference video**：Smooth 模式提交给 Seedance 的上一段尾部视频长度，
  默认 `2s`。
- **Seedance submit requires human confirmation**：开启后，自动运行到
  Seedance 视频生成前会停住，需要用户手动点击该 Step 的 Run。
- **Auto reflect visual plan**：自动运行时，在 Visual Elements Plan 初步完成后
  自动调用一次 Reflect 修正视觉元素表。
- **Auto Submit Eval**：开启后，项目进入 `completed` 状态时会在后台调用
  `tools/submit_videogen_eval_to_a6000.py` 将已完成项目提交到 A6000
  EntityBench 测评队列。Web UI 不读取或展示远端测评状态。
- **Advanced reference scoring**：调节历史参考帧选择的打分权重，包括人物、场景、
  物体、应排除项、reference quality 等。

注意：Seedance 静态参考图预算上限为 `9`。如果某个 Shot 设置了预定义参考图，
系统会先占用这些图片槽位，再按剩余槽位限制 `Sink frame count` 和
`Max retrieved frames`。

## 4. Shot 结构

每个 Shot 是一个 Notebook 块，包含输入、状态、步骤结果、日志和输出。

### Step 1 · Shot Design

这是人工输入区，不调用模型。字段包括：

- **Video prompt**：该 Shot 的视频生成描述。
- **Cut**：是否为转场镜头。
- **Generation mode**：
  - `Default`：普通图像参考生成。
  - `Last frame only`：只使用上一段 Seedance 原始 `last_frame_url` 作为首帧输入。
  - `Smooth`：非 Cut 默认模式，使用上一段原始视频尾部片段作为
    `reference_video`，再结合历史参考图。
- **Generation duration**：该 Shot 的生成时长。可选 `自动 (-1)` 或 `4-15s`。
  导入 JSON 时，如果某个 Shot 的时长不在该范围内，会自动设为 `自动 (-1)`。
- **Predefined References**：当前 Shot 的人工/剧本指定参考图。每行包含缩略图、
  Label 和 Guidance。可在页面中上传图片、编辑说明或取消 Keep 删除该行。

导入剧本时，首个 Shot 或 Cut Shot 默认使用 `Default`；非 Cut Shot 默认使用
`Smooth`。

导入 JSON 可在每个 scene 中提供 `predefined_references`，其长度与
`video_prompts` 对齐；每个 Shot 对应一个参考图列表。图片会复制进项目目录，
之后项目只依赖自己的 bundle。示例：

```json
{
  "scene_num": 1,
  "video_prompts": ["..."],
  "cut": [true],
  "predefined_references": [
    [
      {
        "image_path": "/absolute/path/to/role.jpg",
        "label": "角色参考",
        "guidance": "保持该角色的动画造型与服装。"
      }
    ]
  ]
}
```

编辑后可点击顶部 **Save all** 一次保存全部 Shot Design 修改。

### Step 2 · Visual Elements Plan

系统根据从第一个 Shot 到当前 Shot 的剧本上下文，维护当前 Shot 的视觉元素状态表。

表格主要字段：

- **Element / ID / Type**：视觉元素名称、内部 ID、类型。
- **Status**：应参考、应排除、不确定/可选、新元素等。
- **Introduced**：该元素首次出现的 Shot。
- **Notes / Status reason**：备注和状态原因。

可用操作：

- **Run Step**：重新运行该 Step。
- **Edit / Save**：手动新增、删除、修改元素行。
- **Reflect**：调用 LLM 检查并修正元素名称、备注和状态；不会阻断后续流程。

Details 中记录该 Step 的耗时、Prompt、返回内容和错误信息。

### Step 3 · Historical Reference Selection

系统从已有视觉记忆中选择当前 Shot 的历史参考帧。预定义参考图不属于历史检索
结果，但会占用 Seedance 静态图片预算；因此本 Step 的最多历史参考图数量会按
剩余槽位自动收紧。

表格主要字段：

- **Preview**：缩略图，点击可放大预览。
- **Source**：参考帧来自哪个 Shot。
- **Score**：客观打分。
- **Coverage**：覆盖了哪些应参考元素。
- **Exclude / optional visible**：图中出现但需排除或不确定的元素。
- **Holistic description**：关键帧整体画面描述。
- **Reference intent / Reference guidance**：该参考帧在当前 Shot 中的使用意图。

可用操作：

- **Run Step**：重新选择历史参考帧。
- **Edit / Save**：当前支持删除某些参考帧，以及修改 `reference guidance`。
- **Reflect**：按钮已预留，具体策略后续补充。

### Step 4 · Seedance Prompt Composition

系统根据 Shot Design、预定义参考图说明、视觉元素状态、历史参考帧说明和生成
模式组织完整 Seedance Prompt。Prompt 中 `[预定义参考图说明]` 位于
`[历史参考图说明]` 之前；提交给 Seedance 的静态图片顺序也遵循同样顺序。

可用操作：

- **Run Step**：重新组装 Prompt。
- **Edit / Save**：手动修改最终提交给 Seedance 的完整文本 Prompt。
- **Reflect**：按钮已预留。

当 `Default + non-cut` 且需要上一段尾帧连续性时，已有历史参考图会先提交，
上一段 Seedance 原始 `last_frame_url` 会作为最后一张参考图提交；Prompt 会明确
说明“最后一张参考图”是新视频第一帧约束。

### Step 5 · Seedance Video Generation

提交 Step 4 中保存的 Prompt 和参考媒体，调用 Seedance 生成当前 Shot 视频。

可查看：

- Seedance 输入内容摘要。
- Seedance request/response。
- 原始输出视频。
- 任务 ID、日志、错误信息和耗时。

如果开启了 **Seedance submit requires human confirmation**，自动运行到此处会等待
用户手动点击 **Run Step** 后才进行真实提交。

### Step 6 · Keyframe Maintaining

对当前 Shot 生成后的视频进行关键帧抽取、筛选、VLM 标注和视觉记忆更新。

表格主要字段：

- **Preview**：关键帧缩略图，点击可放大。
- **Visible tracked elements**：图中可见的已追踪元素。
- **Reference quality**：人物参考质量，`full / partial / weak`。
  `full` 要求人脸清晰可见；缺失时按 `full` 处理。
- **Holistic description**：该关键帧的整体画面描述。

可用操作：

- **Run Step**：重新执行关键帧维护。
- **Edit / Save**：当前支持删除关键帧，以及修改 `holistic description`。
- **Reflect**：按钮已预留。

## 5. 运行方式

有三层运行入口：

- **Run all**：从当前项目中未完成的位置开始，按 Shot 和 Step 顺序自动推进。
- **Run Shot**：运行当前 Shot 尚未完成的后续步骤。
- **Run Step**：只运行某个具体 Step。

自动运行时，每个 Step 完成后会按 **Step review delay** 倒计时再进入下一 Step。
如果用户进入 Edit 状态，自动推进会停止；保存后需要重新点击 Run all 或 Run Shot。

Visual Elements Plan、Historical Reference Selection、Seedance Video Generation
和 Keyframe Maintaining 会按 **Step max attempts** 自动重试完整 Step。默认最多尝试
5 次；真实运行时每次完整 Step 失败后等待 `60s` 再重试；全部失败后才进入对应的
failed 状态。LLM/VLM 返回 JSON 后的本地解析修正循环不属于完整 Step 重试，因此不会额外等待。

已完成的 Step 可以通过 Edit 修改结果。修改某个 Step 通常会使其后续 Step 结果失效，
需要重新运行后续流程。

## 6. 当前输出与拼接

页面底部显示 **Current output**，即当前连续完成前缀的拼接视频。

每个 Shot 生成成功后，系统会基于上一个 completed Shot 保存的 Current output
继续拼接，并将新的 Current output 记录到当前 Shot 下。因此如果修改了中间某个
Shot，重新生成后会从该位置继续构建新的当前总输出。

Smooth 模式下，最终拼接会对相邻片段边界做 RIFE 插帧替换；原始 Attempt 视频和
关键帧维护结果不会被覆盖。

## 7. 项目复制、分叉与 JSON

- **Duplicate**：复制整个项目，适合在已有项目上做实验对比。
- **Fork after Shot**：保留当前 Shot 及其之前的结果，将后续 Shot 重置为 Draft，
  用于从某个完成位置继续尝试新分支。
- **Export JSON**：导出当前 Shot Design，可用于保存或迁移剧本设计。
- **Import JSON**：从 JSON 剧本创建新项目。

第一版不提供项目 zip 导出。需要迁移时，可直接复制项目目录。

## 8. 常见注意事项

- 真实 runner 会产生付费 API 调用；测试前请确认当前项目和设置。
- Web UI 允许多个不同项目并行 Running，默认后台并行度为 `50`，可通过
  `VIDEOGEN_NOTEBOOK_MAX_WORKERS` 调整。
- Step 6 Keyframe Maintaining 会串行等待本地 GPU；多个项目同时到达 Step 6
  时，后到的项目会显示 `keyframe_maintaining_queued`。
- 如果页面不更新或代码修改后未生效，优先执行：

  ```bash
  ./run_videogen_notebook.sh restart
  ```

- 查看服务日志：

  ```bash
  ./run_videogen_notebook.sh logs -f
  ```

- 如果某个 Step 失败，先展开该 Step 的 Details，查看提交 Prompt、模型返回、
  request/response、错误类型和耗时。
- 不要把 `.runtime/`、生成媒体、模型缓存或 API key 提交到 Git。
