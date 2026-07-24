# Ours README: StoryMem + Prompt-Aware Retrieval Pipeline

本文档面向后续维护者或接手开发的 agent，说明当前项目在原始 StoryMem 基础上的研究目标、实现流程、关键文件和运行方式。

## 0. 当前项目的一句话理解

这个项目可以理解为一个 **长故事视频生成 AI agent 原型**：

```text
剧本/分镜文本
  -> 镜头级任务规划
  -> 历史视觉记忆维护
  -> prompt-aware 关键帧检索
  -> 增强多模态 prompt 组织
  -> Seedance/Wan/StoryMem 生成单段视频
  -> 抽取关键帧并更新 memory
  -> 拼接成完整视频并输出可分析日志
```

它的研究重点不是训练一个新的视频生成模型，而是在已有视频模型/API 外部构建一个可解释、可替换的记忆和决策层。当前最重要的模块是：

- **Memory policy**：决定哪些历史关键帧进入下一段视频生成。
- **Retrieval query**：决定当前镜头应该主动寻找什么历史视觉信息。
- **Reference-image prompt protocol**：明确告诉视频生成器每张参考图的角色和来源。
- **Analysis artifacts**：把 query、分数、选择结果和参考图整理成日志/Markdown，方便做实验对比。

## 1. 项目目标

我们的研究目标是：在不训练视频生成模型的前提下，改进长故事视频生成中的历史视觉记忆使用方式。

原始 StoryMem 已经会从已生成视频中保存少量 memory keyframes，并在后续镜头中作为参考图输入视频生成模型。我们当前关注的问题是：当故事发生场景回归、人物回归或重要物体回归时，默认记忆机制不一定会把最相关的历史帧放入下一段视频的参考图中。

因此当前实现的核心方向是 **Prompt-aware retrieval**：

- 对于转场镜头，根据当前 shot 的文本 prompt 或额外生成的 memory query，在历史 keyframes 中主动检索更相关的关键帧。
- 将检索到的历史帧与 StoryMem 默认保留的 sink frames、recent frames 组合成新的 memory bank。
- 将该 memory bank 作为多参考图输入 Seedance 视频生成 API。

当前方案是免训练的，主要使用：

- Seedance API：负责短视频生成。
- HPSv3 + Qwen2-VL：负责 StoryMem 关键帧质量评分。
- OpenAI CLIP：负责关键帧相似度、文本-图像检索和文本-文本相似度。
- DeepSeek 或兼容 OpenAI Chat Completions 的 LLM：可选，用于生成 memory retrieval query。

当前已经实现并验证的两个核心增强：

- **Prompt-aware retrieval**：根据当前 shot 的文本需求，从历史关键帧中主动检索最相关的帧。
- **Enhanced Seedance text prompt**：在调用 Seedance 时同时提供当前任务、完整剧本 shot plan、参考图角色和来源说明。

## 2. 当前主流程

入口脚本是 `seedance_pipeline.py`。它读取 story JSON，并按 scene/shot 顺序生成视频。

### 2.1 输入 story JSON

典型字段包括：

- `scenes`: 场景列表。
- `scene_num`: 场景编号。
- `video_prompts`: 每个 shot 的视频生成 prompt。
- `cut`: 每个 shot 是否为转场镜头。`true` 表示切换/转场，`false` 表示延续上一镜头。
- `first_frame_prompt`: 可选。原始 StoryMem 风格字段，也可作为 retrieval query 的 fallback。
- `memory_queries`: 可选。人工指定的历史记忆检索 query。

注意：

- `video_prompts` 会进入视频生成模型。
- `first_frame_prompt` 默认不会直接作为第二个视频 prompt 进入 Seedance；它主要用于补充 retrieval query 或 LLM query 生成。
- `memory_queries` 是我们新增/利用的检索查询字段，用于找历史关键帧。

### 2.2 每个 shot 的生成流程

对每个 shot：

1. 判断是否已有输出视频。若 `--resume` 且视频已存在，则跳过 Seedance 提交。
2. 若该视频尚无关键帧，则调用 `save_keyframes()` 补提关键帧。
3. 构建参考图列表：
   - 第一段视频：不使用参考图，纯文本生成。
   - 非转场镜头：优先加入上一段视频的 `last_frame.jpg`，保证连续性；再加入 memory keyframes。
   - 转场镜头：使用 memory bank 作为历史视觉记忆；若开启 `--prompt_retrieval`，会主动检索历史关键帧。
4. 组装 Seedance content：
   - 一个 text item：由 `_compose_prompt()` 生成，包含当前任务、故事 prompt、参考图使用说明、负面约束。
   - 若干 image_url items：本地参考图转 base64 data URL 传给 API。
5. 调用 Seedance 创建任务、轮询任务、下载 mp4。
6. 增量拼接当前已完成视频，输出最终 mp4。
7. 用 HPSv3/CLIP 从该段视频中保存关键帧，并更新 `last_frame.jpg` 和 `motion_frames.mp4`。
8. 将每段 prompt、参考帧角色、历史来源、检索分数和图片缩略图写入 `memory_report.md`。

## 3. 默认记忆机制与 Prompt-Aware Retrieval

### 3.1 默认 memory bank

实现位置：`prompt_retrieval.py::build_default_memory_bank`

默认策略是：

- 若历史 keyframes 数量不超过 `max_memory_size`，全部使用。
- 若超过上限，则保留：
  - 最早的 `fix` 张 sink frames。
  - 最近的若干 keyframes。

这模拟 StoryMem 中“早期稳定锚点 + 最近连续性”的思路。

### 3.2 Prompt-aware memory bank

实现位置：`prompt_retrieval.py::build_prompt_aware_memory_bank`

当 `--prompt_retrieval` 开启且当前 shot 是 `cut=true` 时，使用该策略：

1. 收集输出目录中所有 `*keyframe*.jpg`。
2. 保护默认记忆中的两类帧：
   - `sink`: 最早的 `fix` 张。
   - `recent`: 最近若干张。
3. 其余历史关键帧作为 candidates。
4. 对 candidates 计算检索分数：

```text
score = frame_weight * frame_score + video_weight * video_score
```

其中：

- `frame_score`: CLIP 文本-图像相似度，比较 `memory_query` 与候选关键帧。
- `video_score`: CLIP 文本-文本相似度，比较当前 prompt 与候选关键帧所属历史 shot 的 prompt。

5. 取前 `retrieval_top_k` 个候选帧。
6. 最终 memory bank 为：

```text
sink + retrieved + recent
```

该策略暂时不修改 keyframe 数据结构，而是按文件名解析 keyframe 所属的 scene/shot，再从 story JSON 中找到对应历史 prompt。

### 3.3 Memory query 来源

实现位置：

- `prompt_retrieval.py::get_memory_query`
- `memory_query_llm.py::MemoryQueryGenerator`

当前 memory query 的来源优先级：

1. story JSON 中当前 shot 的 `memory_queries`。
2. story JSON 中当前 shot 的 `retrieval_queries`。
3. story JSON 中当前 shot 的 `first_frame_prompt`。
4. 当前 `video_prompt`。

如果启动参数带 `--llm_memory_query`，则改为调用 LLM 生成 query，并缓存到：

```text
<output_dir>/llm_memory_queries.jsonl
```

LLM 生成的 query 要求是英文、短视觉描述，重点保留人物、服装、场景、物体、光照和氛围，去掉镜头运动和动作变化。

### 3.4 检索日志

每次 prompt-aware retrieval 会写入：

```text
<output_dir>/prompt_retrieval_log.jsonl
```

日志包含：

- 当前 prompt。
- memory query。
- 分数公式和权重。
- sink/recent/candidate/retrieved/final memory 列表。
- 每张历史关键帧的 `frame_score`、`video_score`、最终 `score` 和是否被选中。

这是后续分析“是否检索到了正确历史场景”的主要依据。

## 4. Seedance API 接入

实现位置：`seedance_client.py`

当前使用火山/Ark 风格的 Seedance task API：

- 默认 base URL: `https://ark.cn-beijing.volces.com/api/v3`
- 默认 model: `doubao-seedance-2-0-260128`
- API key 环境变量：`SEEDANCE_API_KEY` 或 `ARK_API_KEY`

调用方式：

1. `create_task()` 提交文本和参考图。
2. `wait_task()` 轮询任务状态。
3. `download_video()` 下载生成 mp4。

请求的图片参考图会被编码成 base64 data URL，因此 API 端可以直接读取本地图片内容。

任务日志写入：

```text
<output_dir>/seedance_tasks.jsonl
```

该文件记录 task id、prompt、参考图数量、响应摘要等。`--resume` 会利用它避免重复提交已确认创建的任务。

## 5. Prompt 组装与负面约束

实现位置：`seedance_pipeline.py::_compose_prompt`

当前 prompt 会根据镜头类型加不同前缀：

- 第一段：文本生成第一段。
- 转场镜头：参考图作为历史视觉记忆，不要求严格第一帧对齐。
- 非转场镜头：第一张参考图是上一段尾帧，用于时间连续性；其余图作为历史记忆。

统一加入的约束包括：

- 保持角色设计和视觉风格一致。
- 保持镜头尺度稳定。
- 不要拉远镜头、不要 zoom out、不要 fade out、不要结尾变暗。
- 不要字幕、文字叠加、水印。

### 5.1 Enhanced Seedance text prompt

启动参数：

```bash
--enhanced_text_prompt
```

或通过 `run_seedance_little_prince_smoke.sh` 的环境变量：

```bash
ENHANCED_TEXT_PROMPT=1
```

开启后，传给 Seedance 的 text item 会被组织成四个部分：

1. `CURRENT GENERATION TASK`
   - 当前要生成的 `Scene / Shot`。
   - 当前 shot 的 `video_prompt`。
   - 当前镜头是转场还是连续镜头。
   - 本地任务指导，例如“第一张参考图是上一段尾帧”或“参考图是历史视觉记忆”。
2. `FULL STORY SHOT PLAN FOR GLOBAL CONTEXT`
   - 将整个剧本中所有 shot 的 `video_prompt` 提供给 Seedance。
   - 目的只是帮助理解完整叙事、角色和场景回归，不允许模型跳到未来剧情。
3. `REFERENCE IMAGE GUIDE`
   - 按输入顺序说明每张参考图的角色。
   - 角色包括：
     - `previous ending frame`: 上一段尾帧，用于连续性。
     - `early sink memory`: 早期稳定锚点，用于角色/画风一致。
     - `prompt-retrieved historical memory`: 根据当前 prompt 主动检索到的历史帧。
     - `recent window memory`: 最近窗口帧，用于局部连续性。
     - `default historical memory`: 默认 StoryMem 记忆帧。
   - 同时说明每张图来自哪个历史 `Scene / Shot`，并附对应历史 shot prompt。
4. `VISUAL AND EDITING CONSTRAINTS`
   - 统一负面约束和镜头稳定约束。

这个增强的目标是让 Seedance 不只是“看几张图”，而是理解每张图在当前 generation task 中的用途。

## 6. 关键帧提取

实现位置：`extract_keyframes.py`

默认配置：`keyframe_settings/loose.json`。可用 `STORYMEM_KEYFRAME_PROFILE`
切换到 `default` / `strict`，或用 `STORYMEM_KEYFRAME_CONFIG` 指定 JSON。

`save_keyframes(video_path)` 会：

1. 用 decord 读取视频帧。
2. 用 HPSv3 判断低质量帧，跳过质量差的帧。
3. 用 CLIP/HSV 相似度筛选有代表性的关键帧，默认只与上一张已选候选帧比较。
4. 默认不与历史 memory keyframes 去重；如需恢复旧逻辑，可在 keyframe profile
   中设置 `compare_with_history: true`。
5. 保存：
   - `<scene>_<shot>_keyframe<i>.jpg`
   - `last_frame.jpg`
   - `motion_frames.mp4`

可通过 profile name 或 JSON 路径切换关键帧策略：

- `STORYMEM_KEYFRAME_PROFILE=default|strict|loose`
- `STORYMEM_KEYFRAME_CONFIG=/abs/path/custom.json`

注意：

- HPSv3 在 4090 单卡上可运行，但峰值显存约 22GB，几乎吃满一张 4090。
- 当前 Seedance pipeline 不在本地跑视频生成模型，因此 4090 显存主要消耗来自 HPSv3 关键帧评分。

## 7. 4090 服务器当前环境

项目路径：

```bash
/home/wxh/world_model_projects/StoryMem
```

统一使用 conda 环境：

```bash
/home/wxh/miniconda/envs/py311
```

环境脚本：

```bash
source storymem_env.sh
```

它会：

- 自动进入项目根目录。
- 在 4090 上激活 `py311`。
- 设置 HF cache 到项目内 `.runtime/cache/huggingface`。
- 从 `$HOME/.seedance_api_key` 和 `$HOME/.deepseek_api_key` 读取 API key。
- 取消 `TRANSFORMERS_CACHE`，避免 transformers 找不到已搬迁的 hub cache。

当前已搬到 4090 的本地缓存：

```text
.runtime/cache/huggingface/hub/models--MizzenAI--HPSv3
.runtime/cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct
/home/wxh/.cache/clip
```

Wan 权重未搬迁。

## 8. 常用运行命令

### 8.1 最小 smoke test

```bash
cd /home/wxh/world_model_projects/StoryMem

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
MAX_SHOTS=2 DURATION=4 PROMPT_RETRIEVAL=1 LLM_MEMORY_QUERY=0 RESUME=1 \
bash run_seedance_little_prince_smoke.sh results/little_prince_seedance_smoke_4090
```

已验证结果：

- 两段各约 4 秒。
- 最终拼接视频约 8 秒。
- 输出目录：

```text
results/little_prince_seedance_smoke_4090
```

### 8.2 使用 prompt-aware retrieval

```bash
PROMPT_RETRIEVAL=1 \
LLM_MEMORY_QUERY=0 \
RESUME=1 \
MAX_SHOTS=999 \
DURATION=10 \
bash run_seedance_little_prince_smoke.sh results/little_prince_seedance_prompt_retrieval_10s
```

### 8.3 使用 LLM 生成 memory query

```bash
PROMPT_RETRIEVAL=1 \
LLM_MEMORY_QUERY=1 \
RESUME=1 \
MAX_SHOTS=999 \
DURATION=10 \
bash run_seedance_little_prince_smoke.sh results/little_prince_seedance_prompt_retrieval_llm_10s
```

要求：

- `DEEPSEEK_API_KEY` 已设置，或 `$HOME/.deepseek_api_key` 存在。
- 默认模型为 `deepseek-chat`。

### 8.4 使用增强 Seedance prompt

推荐用于当前主实验。它会在每段生成时提供当前任务、完整 shot plan、参考图角色和来源说明：

```bash
cd /home/wxh/world_model_projects/StoryMem
source storymem_env.sh

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
PROMPT_RETRIEVAL=1 \
ENHANCED_TEXT_PROMPT=1 \
LLM_MEMORY_QUERY=0 \
RESUME=1 \
MAX_SHOTS=999 \
DURATION=10 \
bash run_seedance_little_prince_smoke.sh results/little_prince_seedance_prompt_retrieval_enhanced_10s
```

已完成的一次验证实验：

```text
results/little_prince_seedance_prompt_retrieval_enhanced_10s
```

验证结果：

- 12 个 shot。
- 每段 10 秒。
- 最终拼接视频约 120 秒。
- `memory_report.md` 已包含参考图缩略图、角色、来源镜头和 prompt。

### 8.5 用量统计

Seedance 用量汇总脚本：

```bash
python3 summarize_seedance_usage.py results/little_prince_seedance_prompt_retrieval_enhanced_10s
```

注意：

- `total_tokens` 和 `completion_tokens` 来自 Seedance API 返回的 `result_response.usage`。
- 金额是本地脚本按默认单价估算，不是 API 账单接口。
- 当前实验中，原始 prompt-aware 10s 与 enhanced prompt-aware 10s 的 usage 完全相同，说明本次增强文本输入没有增加 Seedance API token 计费。

### 8.6 关键参数

`seedance_pipeline.py` 支持的常用参数：

- `--story_script_path`: story JSON 路径。
- `--output_dir`: 输出目录。
- `--max_shots`: 最多生成多少个 shot。
- `--duration`: 每段视频时长。
- `--ratio`: 画幅比例，如 `16:9`。
- `--resolution`: 默认 `720p`。
- `--max_memory_size`: 最终传给视频模型的 memory keyframes 上限。
- `--fix`: 保留最早 sink frames 数量。
- `--prompt_retrieval`: 开启 prompt-aware retrieval。
- `--retrieval_top_k`: 主动检索帧数量。
- `--retrieval_frame_weight`: CLIP 文本-图像分数权重。
- `--retrieval_video_weight`: CLIP 文本-文本分数权重。
- `--retrieval_min_score`: 可选最低分阈值。
- `--llm_memory_query`: 调用 LLM 生成 memory query。
- `--enhanced_text_prompt`: 启用增强 Seedance text prompt。
- `--max_reference_images`: 最多给 Seedance 的参考图数量。
- `--skip_keyframes`: 跳过关键帧提取。
- `--resume`: 断点恢复。
- `--incremental_concat / --no-incremental_concat`: 每生成一段后是否立即更新最终拼接视频。

## 9. 输出目录结构

一次实验的输出目录通常包含：

```text
01_01.mp4
01_01_keyframe0.jpg
01_01_keyframe1.jpg
01_02.mp4
01_02_keyframe0.jpg
last_frame.jpg
motion_frames.mp4
<output_dir_name>.mp4
concat_list.txt
seedance_pipeline.log
seedance_tasks.jsonl
prompt_retrieval_log.jsonl
llm_memory_queries.jsonl
run_manifest.json
shots.jsonl
references.jsonl
memory_report.md
```

其中：

- `<output_dir_name>.mp4`: 当前已完成片段的拼接视频。
- `seedance_tasks.jsonl`: Seedance 任务与 resume 的依据。
- `run_manifest.json`: 当前 run 状态、当前 shot、配置、task id 和结构化错误，是网页端轮询的主入口。
- `shots.jsonl`: shot 级运行结果、状态、输出视频和实际参考图。
- `references.jsonl`: 实际发送给 Seedance 的逐张参考图、角色、来源和检索分数。
- `prompt_retrieval_log.jsonl`: Prompt-aware retrieval 分析的主要日志。
- `llm_memory_queries.jsonl`: LLM query 缓存，仅在开启 `--llm_memory_query` 时出现。
- `memory_report.md`: 面向人工分析的总览文档，包含每段 `video_prompt`、参考图缩略图、参考图角色、来源镜头、检索分数和来源 prompt。

### 9.1 `memory_report.md`

报告由 `storymem_seedance/reporting.py::write_memory_report` 基于结构化记录生成。它包含：

- 实验配置。
- Early sink frames 汇总。
- 每个 generated segment 的：
  - `Scene / Shot`。
  - 是否转场。
  - 输出视频文件。
  - 当前 `video_prompt`。
  - 参考图表格。

参考图表格中的 `Preview` 列直接插入同目录图片缩略图：

```html
<img src="01_01_keyframe0.jpg" alt="01_01_keyframe0.jpg" width="180">
```

这份报告是当前最方便的人工诊断入口：可以直接查看 prompt-aware retrieval 是否真的把需要回归的历史画面放进了 Seedance 输入。

## 10. 后续开发建议

### 10.1 更好的 memory query

当前 LLM query 只基于当前 shot prompt 和 first-frame prompt。后续可以加入：

- 当前 scene 的全局上下文。
- 人物表、地点表、物体表。
- 过去若干 shot 的摘要。
- 当前 shot 是否为“场景回归”的显式判断。

### 10.2 更细粒度的历史检索

当前检索对象是已保存的 StoryMem keyframes。后续可尝试：

- 先用文本匹配定位历史 shot，再在对应历史视频中密集采样检索。
- 保存每个 keyframe 的时间戳、原始视频路径、scene/shot 元数据。
- 引入多模态 embedding cache，避免每次重复编码所有历史帧。

### 10.3 Future-aware retention

当前只做 retrieval，不改 keyframe 保留策略。后续可尝试：

- 根据后续剧本预测哪些场景/角色/物体会回归。
- 在关键帧保存时提高这些内容的保留优先级。
- 将 memory capacity 分配给不同实体或场景。

### 10.4 评价

目前主要依赖：

- 人工观看对比。
- `prompt_retrieval_log.jsonl` 分析是否检索到目标历史帧。

后续可补：

- 角色一致性评分。
- 场景回归一致性评分。
- CLIP/DINO/face/object-level matching。
- 人工 pairwise preference 表格。

### 10.5 更像 agent 的闭环

当前 pipeline 仍然是“规则驱动 + 一次生成”。如果要继续做成更完整的视频生成 agent，可以加入：

- **Planner**：LLM 根据完整剧本决定 shot duration、是否 cut、是否需要回忆历史场景。
- **Retriever**：LLM 生成 memory query，再结合 CLIP 图文检索和文本-文本相似度选择历史帧。
- **Generator**：Seedance/Wan/其他视频模型负责生成当前片段。
- **Reviewer**：VLM 或规则检查生成结果是否满足当前 prompt、是否人物一致、是否出现字幕/拉远/淡出。
- **Repair loop**：若 reviewer 判定失败，自动调整 prompt、减少/增加参考图或重新检索后重试。
- **Memory manager**：根据未来剧情需求决定哪些关键帧长期保留，而不是只依赖 sink/recent。

## 11. 接手时优先检查

1. `git status` 是否干净。
2. `source storymem_env.sh` 是否能进入正确环境。
3. `SEEDANCE_API_KEY` 和 `DEEPSEEK_API_KEY` 是否存在。
4. `.runtime/cache/huggingface/hub` 下 HPSv3/Qwen 是否完整。
5. 先跑 `MAX_SHOTS=2 DURATION=4` smoke test，不要直接跑完整剧本。
6. 若 HPSv3 OOM，可先用 `--skip_keyframes` 验证 Seedance API，再单独排查关键帧模块。

## 12. A6000 迁移环境

当前可运行副本位于：

```text
Host: 10.130.128.150
User: lzg
Project: /home/lzg/wxh/world_model_projects/StoryMem
Runtime/cache: /data3/lzg/storymem_runtime
GPU: physical GPU 1, NVIDIA RTX A6000 48 GB
```

进入环境：

```bash
cd /home/lzg/wxh/world_model_projects/StoryMem
source ./a6000_env.sh
```

HPSv3、Qwen2-VL 和 CLIP 权重已经从源服务器迁移，远端启用了 Hugging Face offline mode，避免重复下载和 HEAD 请求超时。API key 保存在 `/home/lzg/.seedance_api_key` 和 `/home/lzg/.deepseek_api_key`，权限为 `600`；不要把 key 内容写进仓库。

A6000 已实际完成 Elon 第一段的关键帧提取，峰值显存约 23.5 GB，没有 OOM。完整 Elon memory run 在第二段提交时被 Seedance 的真人参考图策略拒绝：

```text
InputImageSensitiveContentDetected.PrivacyInformation
```

这属于外部 API 策略，不是迁移或 GPU 故障。完整部署、恢复命令、缓存路径和故障说明见：

```text
docs/seedance_pipeline_architecture.md
```

## 13. Notebook Web UI

4090 上的初版 Web UI 位于 `storymem_web/`。推荐使用后台启动脚本：

```bash
cd /home/wxh/world_model_projects/StoryMem
./storymem web start
```

统一运维命令如下：

```bash
./storymem status                    # Git/Web/GPU/项目摘要
./storymem status --project miss_d   # 指定项目的 Shot/Attempt 摘要
./storymem status --json             # 机器可读输出
./storymem web status                # 对比运行中与当前源码指纹
./storymem web restart               # 安全停止、重启并校验版本
./storymem web logs                  # 最近 100 行服务日志
./storymem web logs -f               # 持续跟踪日志
```

`start_storymem_web.sh` 作为兼容入口仍然保留。管理脚本会完成后台驻留、PID
归属检查、健康检查和源码指纹校验，并输出日志位置。默认监听
`0.0.0.0:7860`，工作区保存在
`${STORYMEM_DATA}/web`。可通过以下环境变量覆盖：

```text
STORYMEM_WEB_HOST
STORYMEM_WEB_PORT
STORYMEM_WEB_WORKSPACE
STORYMEM_WORKER_CPU_THREADS  # 默认 8，限制 HPSv3 worker 的 CPU 计算线程数
```

调试时如需前台运行，可以继续使用 `./run_storymem_web.sh`。

`GET /api/health` 返回启动时 Git commit、源码指纹、schema version、PID、
Python 版本和 workspace。页面或端口可访问并不代表代码是最新的；以
`./storymem web status` 的 `current` / `STALE` 判断为准。

当前页面支持：

- 新建单 Shot 空项目和导入现有 JSON 剧本。
- 在项目页直接重命名项目，并同步更新保存的 JSON 剧本名称。
- 编辑尚未提交或已失败/中断 Shot 的 `video_prompt` 与 `is_cut`。
- 每个 Shot 使用可扩展的 `Generation mode`，默认为 `Default`。非转场 Shot
  可选择 `Last frame only`：检索与记忆更新照常执行，但 Seedance 输入只保留
  上一 Shot 尾帧，并将其作为 `first_frame` 提交。该模式会通过上一 Attempt 的
  Seedance `task_id` 重新获取原始 `last_frame_url`，不会上传本地重新编码的
  `last_frame.jpg`。
- `Default` 模式下，非转场 Shot 若包含上一 Shot 尾帧，也使用同一原始
  `last_frame_url`，但仍以普通 `reference_image` 角色与其他记忆图片共同提交。
- 已确定但尚未实现的 `Smooth` 模式仅用于非转场 Shot：将上一段末尾 1 秒作为
  `reference_video`，在 Prompt 中要求 Seedance 从尾端运动继续且不复现输入段；
  最终拼接时删除上段末帧和下段首帧，以两侧相邻帧为锚点，用 Practical-RIFE
  4.25 在 `t=1/3,2/3` 生成两帧等长替换。原始 Shot 视频、检索和记忆更新保持不变。
  该方案将在独立工程线程实现，当前 Web UI 仍只支持前两种模式。
- Run All、单 Shot Run、合作式中断和从指定 Shot 重置。
- 对 Seedance 已成功但本地后处理失败的 Attempt 单独补跑关键帧，不重复生成视频或计费。
- 展示实际输入参考图、Memory 决策、分段视频和输出记忆候选帧。
- 只拼接当前有效 Attempt 的连续前缀。
- 区分当前版本费用与包含重试/废弃 Attempt 的累计估算费用。

运行测试：

```bash
python -m unittest discover -s tests -v
```

2026-06-21 的 4090 真实 smoke test 已跑通一段 8 秒增强 Prompt 视频，
Seedance 生成、下载、HPSv3 关键帧、最终拼接与费用汇总均成功。首次
HPSv3/Qwen 初始化约耗时 8 分钟，GPU 峰值约 23.6 GB；同一个 Run All
Worker 内后续 Shot 会复用模型。
