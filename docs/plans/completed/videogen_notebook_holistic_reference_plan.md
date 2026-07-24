# videogen_notebook 关键帧整体描述与参考理由计划

## 目标

在不改变现有客观检索/评分机制的前提下，为新版 `videogen_notebook`
补充两类文本记忆：

1. `holistic_description`：由 Step 6 Keyframe Maintaining 的 VLM 标注任务产出，描述关键帧的整体画面与剧情关系。
2. `reference_guidance`：由 Step 3 Historical Reference Selection 后的 LLM 增强任务产出，解释已选历史参考帧的主观选择理由与整体应用指引。

这些文本用于 Web 可视化、人工编辑，以及 Step 4 Seedance Prompt Composition
中的参考图片说明。第一版不把它们纳入客观评分，也不让 LLM/VLM 改变历史参考选择结果。

## 确认范围

- Step 6 关键帧标注新增 `holistic_description` 字段。
- Step 3 已选参考帧新增 `reference_guidance` 字段。
- Step 4 组装 Seedance prompt 时使用以上两个字段。
- Step 3 Edit 解锁，第一版只支持修改 `reference_guidance` 和删除参考帧。
- Step 6 Edit 解锁，第一版只支持修改 `holistic_description` 和删除关键帧。
- 缺失或空字段不视为错误。若 `holistic_description` 缺失或为空，后续 `reference_guidance` 也保持为空，Seedance prompt 组装时跳过对应文本。

## 非目标

- 不改变关键帧候选提取、质量评分、相似度去重和候选帧准入策略。
- 不把 `holistic_description` 纳入历史参考帧客观评分。
- 不让 LLM 在 Step 3 新增、替换或重排参考帧。
- 不实现 Step 3/6 的完整表格字段编辑。
- 不实现 `holistic_description` 或 `reference_guidance` 的 Reflect。
- 不修改 Seedance 真实提交、Smooth/RIFE、sink/recent/retrieve 的核心行为。

## 数据结构

### Produced Visual Memory

每个 Step 6 产出的关键帧记录新增：

```json
{
  "holistic_description": "画面整体描述..."
}
```

旧项目或旧 attempt 缺失该字段时，按空字符串处理。

### Selected Historical Reference

每个 Step 3 已选参考帧记录新增：

```json
{
  "reference_guidance": "选择理由与整体应用指引..."
}
```

该字段是解释性增强，不影响客观得分、coverage、exclude/optional visible 等原有字段。

## Step 6 Keyframe Maintaining 改动

### VLM 输入

当前 VLM 关键帧标注任务需要补充剧情上下文：

- 从 Shot 1 到当前 Shot 的所有 `video_prompt`；
- 当前 Shot 的 `video_prompt`；
- 当前 Visual Element Status；
- 当前待标注关键帧图片；
- 原有元素可见性标注要求。

### VLM 新增输出要求

在原有可见元素标注基础上，额外输出 `holistic_description`。描述应包含：

1. 画面中主要人物、物体、环境的构成；
2. 画面氛围、构图、动作或镜头状态；
3. 该画面与当前剧情的关系。

约束：

- 不编造画面中不可见的信息；
- 剧情关系可以基于给定的前序 Shot prompt 与当前 Shot prompt 判断；
- 若无法可靠描述，输出空字符串。

### 解析与存储

- 解析 VLM JSON 时读取 `holistic_description`；
- 写入 Produced Visual Memory artifact 与 attempt snapshot；
- 缺失或空值不导致 Step 6 失败；
- Details 中保留提交给 VLM 的 prompt、raw response、parsed result，便于诊断。

## Step 3 Historical Reference Selection 改动

### 执行顺序

Step 3 仍先执行现有客观选择流程：

1. 根据 Visual Element Status、记忆池与预算选择历史参考帧；
2. 产出原有 Selected Historical References 表格；
3. 对已选参考帧执行一次 LLM guidance 增强。

### LLM guidance 输入

LLM 只接收文本信息，不接收图片：

- 从 Shot 1 到当前 Shot 的所有 `video_prompt`；
- 当前 Shot 的 `video_prompt`；
- 当前 Visual Element Status；
- 已选参考帧列表：
  - reference id；
  - 来源 Scene / Shot；
  - `holistic_description`；
  - visible elements；
  - coverage；
  - exclude / optional visible；
  - objective score，如有。

若某条参考帧缺少 `holistic_description`，该条的 `reference_guidance`
应保持为空。

### LLM guidance 输出

输出结构：

```json
{
  "references": [
    {
      "reference_id": "ref-...",
      "reference_guidance": "这张图适合作为..."
    }
  ]
}
```

约束：

- 不新增参考帧；
- 不删除参考帧；
- 不替换参考帧；
- 不重排参考帧；
- 只为已有参考帧补充 `reference_guidance`；
- guidance 失败时记录日志，但保留客观选择结果。第一版可将缺失 guidance 置为空，不让 Step 3 因解释增强失败而整体失败。

## Step 4 Seedance Prompt Composition 改动

当某张历史参考图被提交给 Seedance 时，参考图片段落按以下顺序组织：

```text
参考图片 Image 1 来自 Scene 1 / Shot 1：
[holistic_description，如非空]
[reference_guidance，如非空]

# 客观参考约束
应参考其中的：...
不应引入其中的：...
```

规则：

- `holistic_description` 为空时跳过该段；
- `reference_guidance` 为空时跳过该段；
- 原有客观 coverage / exclude / optional visible 文本保留；
- 不改变参考图片数量预算与媒体提交顺序。

## Web UI 改动

### Step 3 表格

Selected Historical References 表格新增或展示：

- `Holistic Description` 列，只读；
- `Reference Guidance` 列，Edit模式下可修改；
- 已有 preview、source、coverage、exclude/optional visible、score 等列保持，且不支持修改。

Step 3 Details 中展示：

- 客观选择日志；
- guidance LLM prompt；
- raw response；
- parsed guidance；
- guidance 失败日志。

### Step 3 Edit

解锁 `Step 3 · Historical Reference Selection` 的 Edit 按钮。

第一版编辑能力：

- 修改每条参考帧的 `reference_guidance`；
- 删除参考帧；
- 保存后写回 Step 3 artifact 与 attempt snapshot；
- 保存后重置 Step 4、Step 5、Step 6 及后续 Shot；
- 不支持新增参考帧、替换参考帧、改 coverage、改 score、改 source。

### Step 6 表格

Produced Visual Memory 表格新增或展示：

- `Holistic Description` 列。

建议第一版在表格中显示可换行短文本，完整内容仍可在 Details JSON 中查看。

### Step 6 Edit

解锁 `Step 6 · Keyframe Maintaining` 的 Edit 按钮。

第一版编辑能力：

- 修改每条关键帧的 `holistic_description`；
- 删除关键帧；
- 保存后写回 Step 6 artifact 与 attempt snapshot；
- 保存后重置后续 Shot；
- 不支持新增关键帧、替换图片、改 visible elements、改 VLM annotation。

## 状态与失效语义

- Step 3 guidance 是 Step 3 的产出增强，不新增独立 Step 状态。
- Step 6 holistic description 是 Step 6 的产出字段，不新增独立 Step 状态。
- Step 3 Edit 保存后，当前 Shot 回到 `reference_selection_completed`，下游 Step 重置。
- Step 6 Edit 保存后，当前 Shot 保持 `completed` 或 `keyframe_maintaining_completed` 语义，但所有后续 Shot 重置。
- Edit 需要遵循现有锁定规则：运行中的项目不能保存编辑，除非已有明确的 stop/reset 流程。

## 测试计划

低成本测试优先：

1. fake VLM 返回 `holistic_description`，检查 Step 6 artifact、attempt snapshot、Produced Visual Memory 表格均包含该字段。
2. fake LLM 返回 `reference_guidance`，检查 Step 3 selected references 写入该字段。
3. `holistic_description` 缺失或为空时，guidance 为空且流程不失败。
4. Seedance prompt 组装包含：
   - 图片来源；
   - 非空 `holistic_description`；
   - 非空 `reference_guidance`；
   - 原有客观约束。
5. Step 3 Edit 可修改 guidance、删除参考帧，并重置下游。
6. Step 6 Edit 可修改 holistic description、删除关键帧，并重置后续 Shot。
7. 旧数据缺字段时项目页面渲染不报错。

不需要执行真实 Seedance 调用。真实 LLM/VLM 调用可在 fake 测试通过后做一次小样本 smoke。

## 建议实现顺序

1. 扩展 Produced Visual Memory 与 Selected Historical Reference 的字段读写兼容。
2. 修改 Step 6 VLM prompt、parser 和 artifact 写入，加入 `holistic_description`。
3. 修改 Step 6 表格展示与 Details。
4. 在 Step 3 客观选择完成后增加 LLM guidance 调用，写入 `reference_guidance`。
5. 修改 Step 3 表格展示与 Details。
6. 修改 Step 4 Seedance prompt composition。
7. 实现 Step 3 Edit：只改 guidance、删除 reference。
8. 实现 Step 6 Edit：只改 holistic description、删除 produced memory。
9. 补充 fake tests、模板渲染检查与 `git diff --check`。

