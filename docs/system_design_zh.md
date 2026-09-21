# RAPCLVG 系统设计说明

## 1. 目标与设计原则

RAPCLVG（Complementary Retrieval-Augmented Prompting for Consistent Long-Form Video Generation）是一个面向长视频创作的逐镜头生成系统。系统将一个长剧本拆分为有序 Shot，使用固定的短视频生成 API 逐段生成视频，并通过视觉元素规划、关键帧记忆与互补检索，将前序视觉信息以可解释的参考媒体和结构化提示词传递给后续 Shot。

系统的核心目标是：

1. 保持角色、场景、关键物体及整体视觉风格的跨 Shot 一致性。
2. 允许创作者在每个算法步骤后查看、编辑、停止、重跑或从已完成位置分叉。
3. 将项目状态、媒体、日志和历史 Attempt 放在同一个项目目录中，使项目可复制、迁移、备份和离线分析，而不依赖中心数据库。
4. 让 Web UI、命令行和自动化脚本共用同一套运行时逻辑，避免出现两套行为不一致的 pipeline。
5. 保持视频生成后端可替换。当前实际后端为 Seedance，但视觉记忆、检索、提示词组织和项目存储不应绑定某个生成模型。

当前实现的主流程标识为 `visual_element_memory_v1`，定义于 `pipeline/runtime/domain.py`。

## 2. 总体架构

```text
                +--------------------+
                |  Web UI :7880      |
                |  videogen_ui/      |
                +---------+----------+
                          |
                +---------v----------+
                | BackgroundJobManager|
                | ThreadPoolExecutor |
                +---------+----------+
                          |
                +---------v----------+
                | VideoGenRunner     |
                | pipeline/runtime/  |
                +----+----+----+-----+
                     |    |    |
       +-------------+    |    +---------------------+
       |                  |                          |
+------v-------+ +--------v---------+       +--------v---------+
| ProjectStore  | | Agentic Memory   |       | Real Backend      |
| JSON/project  | | LLM/VLM          |       | Seedance/API      |
+--------------+ +--------+---------+       +--------+---------+
                       |                            |
              +--------v---------+         +--------v---------+
              | Keyframes / GPU  |         | Media / RIFE /    |
              | HPSv3 + CLIP     |         | ref-video publish |
              +------------------+         +------------------+
```

依赖方向保持单向：

```text
videogen_ui ─┐
             ├─> pipeline.runtime ─> pipeline.agentic
CLI ─────────┘                    ├> pipeline.generators
                                  ├> pipeline.keyframes
                                  └> pipeline.media
```

### 2.1 接入层

- `videogen_ui/`：FastAPI 路由、Jinja 模板、静态资源和后台任务管理。默认监听 `7880`。
- `videogen_full_cli.py`：同一流程的命令行入口，适合批量实验、脚本化运行和无 UI 环境。
- `run_videogen_ui.sh`：启动、停止、重启和查看 UI 日志的运维入口。

Web UI 只负责请求校验、项目编辑和将长任务交给 `BackgroundJobManager`，不应直接实现 Agent、Seedance 或关键帧算法。

### 2.2 运行时编排层

`pipeline/runtime/runner.py` 中的 `VideoGenRunner` 是唯一的任务编排入口。它提供五类运行粒度：

- `run_all(project_id)`：从第一个未完成 Shot 开始，顺序推进整个项目。
- `run_shot(project_id, shot_id)`：推进一个 Shot 的所有未完成步骤。
- `run_step(project_id, shot_id, step)`：只执行指定步骤。
- `reflect_visual_plan(project_id, shot_id)`：对已有 Visual Plan 单独执行反思修正。
- `rerun_assembly(project_id)`：仅重建当前已完成前缀的视频拼接结果。

`create_runner()` 根据配置构建 `fake` 或 `real` 后端。`fake` 用于不产生网络/GPU费用的结构验证；`real` 默认开启真实 Seedance 提交，并接入 Agent、关键帧和后处理组件。

### 2.3 算法与媒体层

- `pipeline/agentic/`：视觉元素集合维护、关键帧 VLM 标注、Visual Plan Reflect 和剧本语言判定。
- `pipeline/runtime/runner_helpers/retrieval.py`：历史参考帧选择、Reference Guidance、提示词模块和多模态请求内容组织。
- `pipeline/keyframes/`：候选帧提取、HPSv3 质量评分、CLIP 相似度、Profile 配置。
- `pipeline/generators/seedance_client.py`：Seedance 异步创建、轮询、下载和错误解析。
- `pipeline/media/reference_video.py`：Smooth 模式的尾端参考视频提取和 Cloudflare Tunnel 公网发布。
- `pipeline/media/smooth_transition.py`：使用 Practical-RIFE 4.25 对 Smooth 边界做帧间插值，并拼接音频。

## 3. 项目包与数据持久化

### 3.1 自包含项目包

默认工作区为 `.runtime/videogen_ui`。每个项目是 `projects/<project_id>/` 下完整、自包含的目录，不使用中心数据库：

```text
.runtime/videogen_ui/
├── default_project_settings.json       # 仅影响以后创建/导入的项目
├── locks/                              # 每项目运行锁
└── projects/
    └── p_<timestamp>_<suffix>/
        ├── project.json                # 项目摘要、状态、成本、当前拼接产物
        ├── settings.json               # 该项目冻结后的全局设置
        ├── story.json                  # 标准化后的剧本及 source 原始 JSON
        ├── assets/
        │   ├── manifest.json           # asset_id 到相对路径/元数据的索引
        │   ├── images/
        │   ├── videos/
        │   └── predefined/             # 导入或上传的预定义媒体
        ├── memory/
        │   ├── visual_state.json       # 当前视觉元素集合快照
        │   └── lineage.json            # 已完成前缀等谱系信息
        ├── final/
        │   ├── assembly.json           # 各完成前缀的拼接记录
        │   └── prefix_<shot>_<attempt>.mp4
        └── shots/
            └── 0001/
                ├── shot.json           # Shot Design、状态、步骤状态、Attempt 引用
                └── attempts/
                    └── a001/
                        ├── attempt.json
                        ├── input_snapshot.json
                        ├── steps.json
                        ├── visual_element_status.json
                        ├── selected_references.json
                        ├── produced_visual_memory.json
                        ├── request.json
                        ├── response.json
                        ├── logs.jsonl
                        └── <step>/
                            ├── details.json
                            └── logs.jsonl
```

这种设计带来的直接性质：

- 复制项目目录即可复制全部已完成历史、原始视频、关键帧、检索结果和日志。
- 迁移时无需导出数据库；只要保留项目目录及所依赖的运行环境即可继续运行。
- 项目总览页通过扫描 `projects/*/project.json` 建立索引，因此创建时间排序不依赖数据库。
- 单个项目损坏不会影响其他项目；损坏的 JSON 会在项目列表读取时被跳过。

### 3.2 项目、Shot 与 Attempt 的职责

**`project.json`** 记录项目级摘要：项目名、创建/更新时间、`status`、`active_shot_id`、已完成连续前缀、当前拼接视频、费用汇总和 `run_state`。它不承载完整算法详情。

**`shot.json`** 记录一个 Shot 的可编辑设计与当前位置：

- `inputs`：`video_prompt`、`is_cut`、`generation_mode`、`duration_seconds`、`predefined_references`。
- `steps`：每个算法步骤的状态与时间。
- `state`：当前 Shot 状态、修订号和 `current_attempt_id`。
- `archived_attempt_ids`：保留的旧 Attempt，不默认作为 UI 主视图展示。

**Attempt** 是不可变执行快照。第一次运行某个 Shot 时，Runner 将 `shot.inputs` 与 `settings.json` 拷贝到 `input_snapshot.json`；此后同一 Attempt 的后续步骤只读取该快照，而不会读取用户正在编辑的新输入。这样可保证“视频、选择参考图与提交 Prompt”可追溯到同一份输入。

当修改已运行 Shot 或从某步重跑时，当前 Attempt 会归档，新 Attempt 会继承已完成且仍有效的上游步骤产物；下游 Attempt/Shot 则按依赖关系失效并回到 Draft。该机制保证交互编辑不会悄悄混用新旧结果。

### 3.3 资产管理

所有供 UI 和后端引用的持久化媒体使用 `asset_id` 记录在 `assets/manifest.json`。资产表保存相对路径、媒体类型、创建时间、来源和附加元数据；API 路由再通过项目目录安全地解析并提供预览。

原始 Seedance 视频、关键帧、标注可视化图、预定义媒体、尾帧以及各 Shot 完成时的前缀拼接视频都应注册为资产。运行时公网 URL（如 Cloudflare Tunnel 的 Smooth 参考视频 URL）是请求过程数据，不作为长期可靠资产地址。

## 4. 剧本模型与输入约束

### 4.1 标准剧本 JSON

剧本采用 `scenes[]` 容器，每个 Scene 可以有多个 `video_prompts`。Scene 是组织层级，实际生成顺序完全由展平后的 Shot 列表决定。

```json
{
  "story_name": "项目名称",
  "story_overview": "可选的故事总述",
  "scenes": [
    {
      "scene_num": 1,
      "video_prompts": ["Shot 1", "Shot 2"],
      "cut": [true, false],
      "durations": [-1, 8],
      "generation_modes": ["default", "smooth"],
      "predefined_references": [[...], []]
    }
  ]
}
```

导入时 `normalize_story()` 会校验每个可选数组与 `video_prompts` 等长，将所有 Shot 展平为稳定的 `0001`、`0002` 等 ID，并将原始 JSON 保存至 `story.json.source`。原始 JSON 的额外字段会保留，但 pipeline 只直接消费标准字段。

### 4.2 Shot Design

每个 Shot 的人工输入为：

| 字段 | 说明 |
| --- | --- |
| `video_prompt` | 当前视频的原始剧本/生成任务。 |
| `is_cut` | `true` 表示转场或独立镜头；`false` 表示与上一段连续。 |
| `generation_mode` | `default`、`last_frame_only` 或 `smooth`。 |
| `duration_seconds` | `-1` 表示由 Seedance 自动选择；显式值必须为 4–15 秒整数。 |
| `predefined_references` | 用户或数据集预先指定的图片、音频、视频与说明。 |

新建项目默认只有一个 Shot。后续手动添加 Shot 时，首 Shot 为 Cut/default；有前驱的新增 Shot 默认由项目设置 `default_non_cut_mode` 决定，当前默认是 `smooth`。

### 4.3 预定义参考媒体

每项预定义参考具有 `media_type`、本地路径、`label`、`guidance`。支持 image/audio/video；导入 JSON 时，媒体会复制进项目目录，避免依赖原始路径。UI 可上传、编辑说明、删除，并可将当前 Shot 的预定义参考复制到所有后续 Shot。

Seedance 的媒体上限为：图片最多 9 张、参考视频最多 3 段、音频最多 3 段。静态图片预算还需扣除 Sink、历史检索图片及 Smooth/Default 连续性图片；保存时会执行预算校验。

## 5. 六步 Shot 状态机

UI 将 Shot 展示为六个步骤。Step 1 是纯人工设计，其余步骤有独立日志、详情和可重跑入口。

| 步骤 | 内部 key | 主要输入 | 主要输出 |
| --- | --- | --- | --- |
| 1. Shot Design | 无算法 step | 剧本/用户输入 | 标准化 `inputs` |
| 2. Visual Elements Plan | `visual_plan` | 前序剧本、元素集合、当前 Shot | 元素状态表、集合增量、LLM 日志 |
| 3. Historical Reference Selection | `reference_selection` | 元素状态、历史关键帧、设置 | 已选参考帧、评分/Guidance、日志 |
| 4. Seedance Prompt Composition | `seedance_prompt` | 输入快照、参考图、Prompt 模块 | 可编辑的完整 Seedance Prompt、内容摘要 |
| 5. Seedance Video Generation | `seedance_generation` | Prompt、媒体、Seedance 设置 | 任务 ID、请求/响应、原始视频 |
| 6. Keyframe Maintaining | `keyframe_maintaining` | 原始视频、元素集合、关键帧设置 | 候选关键帧、VLM 标注、记忆资产 |

### 5.1 状态表示

Shot 的 `state.status` 直接表达其推进位置，例如：

```text
draft
visual_plan_running / visual_plan_completed / visual_plan_failed
reference_selection_running / reference_selection_completed / reference_selection_failed
seedance_prompt_completed
seedance_generation_running / seedance_generation_completed / seedance_generation_failed
keyframe_maintaining_queued / keyframe_maintaining_running /
keyframe_maintaining_completed
completed / interrupted
```

每一步也在 `shot.steps[step]` 中保存对应状态与起止时间。`completed` 是完整 Shot 的终态，等价于 `keyframe_maintaining_completed`。项目状态是面向总览的派生摘要，不应替代 Shot/Step 的事实状态。

### 5.2 依赖与失效规则

1. Shot 必须按顺序推进：当前 Shot 的前序 Shot 必须完整 `completed`。
2. 执行某一步之前，所有前置步骤必须完成。
3. 修改一个未完成 Shot 的 Shot Design 是直接保存。
4. 修改已运行 Shot，或从某一步重新执行，会归档当前 Attempt，并清空该步之后的产物。
5. 重新执行 Shot N 会使 Shot N+1 及之后的结果失效，因为它们的参考池、连续性媒体和拼接输出依赖前序视频。
6. Fork 仅允许从完整 Shot 后执行：新项目保留该 Shot 及之前的资源与状态，后续 Shot 保留剧本基本信息但回到 Draft。

### 5.3 自动运行、人工检查与编辑

`Run all` / `Run shot` 使用自动模式，会在每个完成步骤之间读取 `auto_run_step_review_delay_seconds`（可选 0/10/30/60/120 秒）等待。等待期间项目的 `run_state` 会显示下一步、截止时间和作用域。

若开启 `require_human_confirmation_before_seedance`，自动模式到达 Step 5 前暂停，必须由用户手动运行 Step 5；完成后不会自动恢复原先的高级别 Run，需要再次点击 Run all 或 Run shot。

当前可编辑能力包括：

- Visual Elements Plan：编辑、新增、删除元素；Type、Status、Introduced 为受控字段。
- Historical Reference Selection：修改 `reference_guidance`、删除已选参考帧。
- Seedance Prompt Composition：直接编辑装配后的 Prompt。
- Keyframe Maintaining：修改 `holistic_description`、删除关键帧。

Visual Plan 已实现 `Reflect`：LLM 基于当前与前序剧本审查现有元素的 Name/Type/Status/Notes/Reason，保留 element ID 和 introduced 信息。其他步骤的 Reflect 是预留扩展方向。

## 6. 视觉元素记忆与检索

### 6.1 视觉元素集合

Step 2 调用 LLM 维护跨 Shot 的元素集合。允许的元素类型固定为：

- `character`：需要保持身份、姿态或外观一致的人物实体。
- `scene`：可整体识别的地点、环境或宏观背景。
- `object`：对视觉叙事有意义、可被识别的物体。

对当前 Shot，每个已有元素必须被赋予一种状态：

- `should_reference`：应从历史图像中保持一致。
- `should_exclude`：当前镜头明确不应由历史图像引入。
- `optional_or_uncertain`：可能有帮助，但不是必须且未被明确禁止。
- `new`：当前 Shot 新出现的元素，随后加入集合。

集合属性（Name、Type、Notes）是跨 Shot 的稳定描述；姿态、动作、关系、镜头变化记录在 Shot 层 `shot_notes`，不污染集合定义。

Agent 的自然语言输出由 `pipeline/agentic/language.py` 判断剧本主语言：中文剧本输出中文，英文剧本输出英文。结构化 JSON 的固定枚举仍为英文，以保持接口稳定。

### 6.2 关键帧维护

Step 6 对原始 Seedance 视频执行：

1. 按 Keyframe Profile 抽取候选帧。
2. 使用 HPSv3 过滤质量不足的候选帧。
3. 使用 CLIP 相似度规则抑制相近帧；当前默认只与上一张候选帧比较，跨历史去重可通过 Profile 选项开启。
4. 调用 VLM 对保留帧标注可见元素的 bounding box。
5. 对人物输出 `reference_quality`：`full`、`partial`、`weak`。`full` 要求人物完整清晰且人脸可见；缺失该字段按 `full` 兼容处理。
6. 为每张帧生成 `holistic_description`，描述画面构成、氛围/动作/镜头状态及在剧情中的作用。
7. 生成包含元素名称、边框和质量信息的标注可视化图，并登记为项目资产。

关键帧与标注结果存入当前 Attempt 的 `produced_visual_memory.json`，同时作为后续 Shot 可检索的历史池。

本地模型包括 HPSv3、其依赖的 Qwen2-VL-7B、CLIP 和 Practical-RIFE；缓存通常位于 `.runtime/cache/huggingface`、`~/.cache/clip` 和 `.runtime/deps/Practical-RIFE`。

### 6.3 历史参考选择模式

`visual_element_memory.selection_mode` 支持：

| 模式 | 行为 |
| --- | --- |
| `greedy_coverage` | 默认完整方法。逐轮选择最能补齐 `should_reference` 元素且冲突较少的帧。 |
| `static_top_k` | 按静态元素分数一次排序，取 Top-K，不做逐轮互补覆盖。 |
| `naive_top_k` | 跳过视觉元素规划和 VLM 元素标注，使用缓存的 CLIP 图片 embedding 与当前文本语义的简单 Top-K。 |
| `sink_recent_memory` | 复现型策略：早期 Sink 加近期窗口记忆。 |
| `none` | 不选择历史参考图；关键帧维护仍可运行以供后续分析。 |

默认模式为 `greedy_coverage`，默认最大历史检索帧数为 4，默认 Sink 帧数为 0。

### 6.4 互补覆盖评分

完整策略为历史帧按当前元素状态计算分数。类型基础权重默认：人物 3.0、场景 2.0、物体 1.5；状态因素默认：首次覆盖 `should_reference` 为 1.0、重复覆盖为 0.2、optional 为 0.1、should_exclude 为 -0.1。

人物的 `reference_quality` 再乘以质量权重：full 1.0、partial 0.2、weak 0.1。只有 `full` 的人物帧才会将人物加入已覆盖集合。该规则避免“检测到人物但人脸缺失”的远景或背影帧被误认为已完成实体参考。

Greedy 选择每轮从可用候选中取最高分，`best_score` 以负无穷初始化，因此即使所有候选得分为零或负分，历史池非空时仍至少可保留一张风格参考帧。默认关闭“全部 required 元素已覆盖后提前停止”，从而与相同预算下的 Top-K 更可比较。

在元素级客观选择完成后，系统可调用 LLM 为每张有 `holistic_description` 的帧补充 `reference_guidance`。若选择帧没有覆盖任何必需元素，会追加固定的风格参考提醒。

## 7. Prompt 与生成媒体组织

### 7.1 Prompt 组成

Step 4 是本地瞬时文本装配，不调用 LLM。最终 Prompt 可按项目设置独立开关以下模块：

1. 当前 Shot 生成任务（始终存在）。
2. 前序完整剧本上下文（只到当前 Shot，不包括未来 Shot）。
3. Visual Element Plan。
4. 参考帧的 `holistic_description` 和 `reference_guidance`。
5. 应参考元素约束（should-reference）。
6. 不应引入元素约束（should-exclude）。

`force_animation_style` 默认开启时，动画与非写实人脸约束会插入“当前 Shot 生成任务”和“总体约束”开头。此文本在 Step 4 完成后可由用户编辑；Step 5 永远提交该 Attempt 中保存的最终 Prompt，而不是重新按规则装配。

### 7.2 参考媒体顺序

系统将预定义媒体、历史静态参考、连续性媒体分开组织，并在 Prompt 中说明引用方式：

- 预定义图片先出现，使用其 Label/Guidance 作为明确说明。
- 历史检索图按最终选择顺序编号，并附带来源 Shot、整体描述、Guidance 和元素级约束。
- `default` 非 Cut 时，上一段尾帧作为**最后一张静态图片**上传，并明确它是新视频第一帧约束；其他图片仍为视觉记忆。
- `last_frame_only` 时，仅使用上一段原始视频的 `last_frame_url` 作为首帧连续性输入。
- `smooth` 非 Cut 时，从上一段**原始视频**提取结尾一小段（默认 2 秒，合法范围 1.8–10 秒）并作为参考视频上传；静态历史图片仍可同时存在。Smooth 请求自动将 Seedance `ratio` 设为 `adaptive`，以满足视频延长类型约束。

参考视频需要公网 HTTPS URL。当前默认 `CloudflareTunnelPublisher`：本地临时 HTTP 服务提供文件，`cloudflared tunnel` 建立 `trycloudflare.com` 公网 URL，再交给 Seedance 拉取。其可用性依赖 `cloudflared`、外网和 Tunnel 生命周期。

### 7.3 Seedance 调用与容错

真实后端通过 ARK 的异步 Seedance API 创建任务、轮询任务状态、下载完成视频，并将任务 ID、请求摘要、响应、下载 URL、用量和本地文件路径写入 Attempt。

为便于排障，`request.json` 保存可读的内容摘要：图片 base64 不原样持久化，而表示为项目内本地路径；参考视频 URL、顺序及媒体调试信息会保留。若 Seedance 直接返回 `InputImageSensitiveContentDetected` 并给出 `content[n]`，系统会解析该编号、记录被丢弃的参考图，并移除该图后重新创建任务。

## 8. 视频拼接与连续性

每个 Shot 视频生成成功后，系统将从第一个已完成 Shot 到当前 Shot 重建一个前缀拼接视频，并保存到 `final/prefix_<shot>_<attempt>.mp4`。因此若用户修改中间 Shot，重新完成后只需读取前一个 Shot 的前缀输出并向后更新，不必依赖过期的项目最终视频。

拼接策略：

- 只有一个片段：直接拷贝/重封装。
- 普通片段串联：逐帧重编码，音频按原始片段顺序拼接。
- 后一段是 `smooth`：删除上段最后一帧和下段第一帧，以 A[-2] / B[1] 为锚点，Practical-RIFE 4.25 在 1/3、2/3 生成两张插值帧替换。总帧数和总视频时长保持不变。

RIFE 失败时系统不会静默降级成普通 FFmpeg 拼接；这使 Smooth 实验的连续性语义可验证。最终音频不参与插值，但按各原始视频的时间顺序复用，因此不会因拼接丢失声音。

## 9. 并发、锁与恢复

### 9.1 项目并发

`BackgroundJobManager` 使用 `ThreadPoolExecutor` 调度项目任务。默认代码值是 30；启动脚本目前导出 `VIDEOGEN_MAX_WORKERS=50`。每个项目只允许一个后台 Future，且 `GlobalRunLock` 在工作区 `locks/project_<id>.lock` 中记录 PID，防止同一项目被重复启动。

不同项目可以并行经历 LLM、Seedance 提交、轮询、下载和普通 CPU I/O。并发上限不是 Seedance 或 ARK 的服务端额度；外部 API 的 RPM、并发和账户配额仍需在客户端节流策略与重试策略中考虑。

### 9.2 单 GPU 关键帧队列

HPSv3、CLIP/GPU 配置以及 Smooth RIFE 都可能占用 CUDA。Step 6 通过进程内 `KeyframeGpuLock` 实现 FIFO 队列：Shot 先进入 `keyframe_maintaining_queued`，拿到 lease 后改为 running。这样多个项目可并行到 Step 5，但关键帧维护在一个 Web 进程内串行，避免高峰显存竞争。

该锁是**单 Python 进程内锁**。若未来启动多个 UI 进程或多个独立 CLI 进程，需要升级为文件锁/Redis 锁或独立 GPU Worker，才能实现跨进程的真实单 GPU 排队。

### 9.3 中断与启动恢复

`interrupt_project()` 设置取消事件，并将处于 running 的 Shot 标记为 `interrupted`。后端轮询、等待重试、GPU 排队均会周期检查取消事件。服务器启动时 `ProjectStore.recover_interrupted_runs()` 会扫描残留的运行状态，将死进程遗留的 running Shot 归为 interrupted，避免 UI 永久显示“正在运行”。

## 10. 重试、日志与可观测性

### 10.1 重试规则

步骤 2、3、5、6 具有完整步骤重试，默认最大尝试数 `algorithm_step_max_attempts=5`，即初次执行加最多 4 次重试。普通重试之间等待 60 秒，降低网络波动和服务端限流造成的连锁失败。

LLM/VLM 返回 HTTP 成功但 JSON 无法解析时，视觉元素模块会先在同一次算法执行内携带格式反馈重试（默认解析重试参数为 2）；这类格式修正不需要额外等待。外层步骤重试和内层 JSON 修正重试是两层不同机制。

当前 `doubao-seed-2-1-turbo` 的 429 为账户级 RPM 限流。现有代码将其视为普通可重试错误；未来可增加按模型/API key 共享的令牌桶或指数退避，以减少多个项目同时运行时的重复 429。

### 10.2 日志层级

日志以 JSONL 存储，保证长过程可以持续追加并被 UI 增量读取：

- `attempts/<id>/logs.jsonl`：整个 Attempt 的聚合日志。
- `attempts/<id>/<step>/logs.jsonl`：单步骤日志。
- `attempts/<id>/<step>/details.json`：该步骤的结构化输入、Prompt、原始模型输出、解析结果、评分细节或错误详情。
- `attempt.json`：最终结构化摘要与 terminal error。

当发生解析错误或远端错误时，Runner 会将异常类型、消息、可用错误详情、重试历史和 traceback 摘要写入对应 Step，UI 的 Details 面板应优先展示这里的数据，而不只显示最终错误字符串。

## 11. Web UI 结构与职责

主页提供项目列表、按创建日期折叠、创建、批量 JSON 导入、复制、删除和直接 Run；主页保存的 Default Settings 只影响之后新建或导入的项目，不回写已有项目。

项目页由以下区域组成：

1. 项目级设置：生成、记忆检索、Prompt 模块、关键帧 Profile 等。
2. 当前拼接输出和费用摘要。
3. 按 Shot 排列的 Notebook 区块。
4. 每个 Shot 的设计输入、状态、六步结果表格、媒体预览与折叠 Details/Logs。

三类核心表格对应算法流程的三份证据：

- Visual Elements Plan：元素集合及当前 Shot 状态。
- Historical Reference Selection：所选历史关键帧、来源、覆盖/排除信息、Guidance。
- Produced Visual Memory：本 Shot 产生的候选关键帧、元素标注、reference quality、整体描述。

缩略图可点击预览；表格保持固定列宽、竖分隔线和可控行高，以避免大量图片使页面不可用。

## 12. 配置分层

配置由三层组成：

1. **代码默认值**：`default_settings()` 与各模块常量，定义新环境的安全默认行为。
2. **工作区默认值**：`default_project_settings.json`，由主页 Default Settings 编辑，仅影响新项目。
3. **项目快照设置**：`projects/<id>/settings.json`，项目导入时复制并独立保存；运行 Attempt 时再冻结到 `input_snapshot.json`。

主要设置组：

| 路径 | 用途 |
| --- | --- |
| `generation.*` | 时长、默认非 Cut 模式、音频、人工确认、步骤检查时间、自动 Reflect、动画强制、算法重试次数、Smooth 尾段长度。 |
| `visual_element_memory.*` | Sink 数量、历史检索上限、选择模式、元素与质量评分权重。 |
| `seedance.*` | 模型、分辨率、画幅比例。 |
| `prompt_modules.*` | 五个结构化 Prompt 模块的消融开关。 |
| `keyframes.profile` | 候选关键帧抽取/质量/相似度策略 Profile。 |

`sink_frame_count + max_retrieved_frames <= 9` 是静态图片预算的基本约束；实际执行还会继续考虑预定义图片与连续性图片占用的槽位。

## 13. 扩展点与维护约定

### 13.1 新视频生成后端

新增后端应实现 `RunnerBackend` 或复用 `RealExecutionBackend` 的 Seedance 生成部分替换为新的 client。后端应返回统一的 `SeedanceGenerationResult` 等 contract，而不是让 UI 理解特定厂商响应。

需要明确新后端的：媒体 URL/上传方式、图片/视频/音频数量上限、引用媒体的编号语义、视频延长能力、时长/画幅约束、任务轮询和版权/敏感内容错误格式。

### 13.2 新的检索或记忆算法

优先在 `VisualElementMemoryAdapter`、`runner_helpers/retrieval.py` 和选择模式枚举中扩展。新算法应输出统一的 `selected_references` 行：来源资产、角色/用途元数据、分数、覆盖元素、冲突元素、整体描述和 Guidance。不要为每种算法另造一套 UI 数据格式。

### 13.3 新的 Reflect 功能

Reflect 不应改变主状态机的完成语义：它的前提是已有某一步完整产物，但其成功或失败不应阻塞后续人工编辑/视频生成。每种 Reflect 应记录 Prompt、原始响应、建议与应用后的差异，便于论文分析。

### 13.4 数据迁移原则

`schema_version` 已写入项目、资产、Attempt 等 JSON。新增字段时应：

1. 为读取旧项目提供缺省值。
2. 保持旧 Attempt 可读，不强制批量重写。
3. 将项目级数据放在项目目录，将短生命周期数据放在 Attempt 目录。
4. 不把绝对路径、临时公网 URL 或 API Key 当作可迁移的长期状态。

## 14. 当前已知边界

1. GPU 锁仅覆盖当前 UI Python 进程，不覆盖独立 CLI/多实例部署。
2. Seedance 与 VLM 的账户级 RPM/并发限制由外部服务控制；当前尚无全局跨项目限速器。
3. Cloudflare Quick Tunnel 适合实验参考视频发布，不是带鉴权、持久 URL 或 SLA 的生产对象存储方案。
4. VLM 结构化输出依赖 Prompt 与解析重试，超长或截断响应仍可能导致 Step 2/6 重试。
5. Smooth RIFE 拼接依赖 CUDA 与校验过的 Practical-RIFE 4.25 权重；故意不静默回退，以保障实验语义，但也提高了运行环境要求。
6. 项目复制与 Fork 在文件级可靠，但项目目录很大时会产生实际媒体复制成本；第一版不做内容寻址去重。

## 15. 常用定位入口

| 需求 | 主要文件 |
| --- | --- |
| Web 路由与页面动作 | `videogen_ui/app.py`、`videogen_ui/jobs.py`、`videogen_ui/templates/` |
| 项目/Shot/Attempt 文件格式 | `pipeline/runtime/project_store.py`、`pipeline/runtime/attempts.py` |
| 状态机、自动运行、重试 | `pipeline/runtime/step_flow.py`、`pipeline/runtime/constants.py` |
| 后端提交与请求内容 | `pipeline/runtime/backends/real.py`、`pipeline/generators/seedance_client.py` |
| 元素规划、VLM 标注、Reflect | `pipeline/agentic/visual_element_memory.py`、`pipeline/agentic/visual_plan_reflection.py` |
| 历史帧选择与 Prompt 模块 | `pipeline/runtime/runner_helpers/retrieval.py` |
| 关键帧 Profile | `pipeline/keyframes/extract.py`、`pipeline/keyframes/settings/` |
| Smooth 参考视频和拼接 | `pipeline/media/reference_video.py`、`pipeline/media/smooth_transition.py` |
| 系统/领域默认值 | `pipeline/runtime/domain.py`、`pipeline/models.py` |
