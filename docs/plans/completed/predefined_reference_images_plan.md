# Predefined Reference Images Plan

## 背景与目标

为了在 `videogen_notebook` 中支持 MSVBench 以及普通创作者手动指定参考图，新增 Shot 级 **Predefined Reference Images** 功能。

该功能属于 Step 1 `Shot Design` 的一部分。它不改变 Visual Element Plan、历史关键帧维护、记忆池结构，也不把这些参考图写入历史记忆。预定义参考图只作为当前 Shot 的固定 Seedance 输入媒体，并在 Prompt 中提供用户可编辑的图片参考说明。

MSVBench 不需要专用导入入口。推荐先将 MSVBench 转换为当前可导入的 JSON 剧本格式，并在每个 Shot 中增加预定义参考图字段。导入后 Web UI 自动复制图片到项目目录，后续项目状态只引用项目内本地资源。

## 剧本 JSON 扩展

现有 Shot Design 在每个 shot 上增加可选字段：

```json
{
  "video_prompt": "...",
  "cut": true,
  "generation_mode": "default",
  "duration": -1,
  "predefined_references": [
    {
      "image_path": "/absolute/or/relative/path/to/reference.jpg",
      "label": "Little Brown Rabbit",
      "guidance": "参考该图中 Little Brown Rabbit 的外观、毛色、体型、面部特征和整体动画风格。"
    }
  ]
}
```

兼容命名建议：

- `image_path` 为必需字段。
- `label` 可为空，导入时可回退到文件名。
- `guidance` 可为空。
- 后续可扩展 `source`、`enabled`、`role`，初版不要求。

MSVBench 转换脚本应：

- 使用 `Dataset/Dataset/prompt/<story_id>.txt` 每行作为 `video_prompt`；
- 将 `cut` 统一设为 `true`；
- 将 `duration` 设为 `-1`；
- 为每个 Shot 添加：
  - 1 张 shot reference image；
  - 每个出场角色 1 张 `merge.jpg`；
  - 对应 `label` 与 `guidance`。

## 项目内持久化

导入 JSON 时，所有 `predefined_references[].image_path` 指向的图片复制到项目目录，例如：

```text
projects/<project_id>/
  assets/
    predefined_references/
      shot-0001_pref-0001.jpg
      shot-0001_pref-0002.jpg
```

Shot 状态只保存项目内相对路径：

```json
"inputs": {
  "predefined_references": [
    {
      "id": "pref-0001",
      "image_path": "assets/predefined_references/shot-0001_pref-0001.jpg",
      "label": "Shot reference",
      "guidance": "参考该图的画面构图、角色位置和整体动画风格，但不要机械复刻。"
    }
  ]
}
```

原则：

- 原始外部路径不作为运行时依赖；
- 项目复制、fork、迁移时只依赖项目目录；
- `image_path` 不进入全局 asset manifest 也可以，但如果需要复用 `/projects/<id>/assets/<asset_id>` 路由，可同时登记 asset manifest。初版优先采用项目相对路径加专用静态读取逻辑或导入时登记 asset，两者选更贴近现有实现者。

## Step 1 UI

在 `Step 1 · Shot Design` 中新增 `Predefined References` 表格，样式参考 Step 3 表格，但只保留三列：

| Preview | Label | Guidance |
| --- | --- | --- |

行为：

- 默认展开；
- Preview 显示缩略图，点击可复用现有图片预览弹窗；
- Label 为文本输入框；
- Guidance 为多行文本框；
- 保存由现有 `Save all` / Shot Design 保存逻辑处理；
- 修改、添加、删除预定义参考图视为 Step 1 修改，应重置当前 Shot 下游步骤以及后续 Shot，遵循现有 Shot Design 编辑语义。

手动编辑：

- 支持删除一行；
- 支持添加图片：用户选择本地图片后，后端复制到当前项目目录并新增一条 `predefined_references`；
- 初版不要求拖拽排序。如需稳定顺序，按表格顺序提交。后续可增加上移/下移按钮。

Seedance 预算校验：

- 单个 Shot 启用的预定义参考图数量不得超过 9；
- 保存或运行前发现超过 9 时应报错；
- 图片尺寸检查可先沿用 Seedance 提交失败反馈，后续再加入本地宽高比/大小预检查。

## Seedance Prompt 组装

在 `Seedance Prompt Composition` 中新增独立块：

```text
[预定义参考图说明]
Image 1: Shot reference
参考该图的画面构图、角色位置和整体动画风格，但不要机械复刻。

Image 2: Little Brown Rabbit
参考该图中 Little Brown Rabbit 的外观、毛色、体型、面部特征和整体动画风格。
```

该块放置在 `[历史参考图说明]` 之前。

图片提交顺序必须与 Prompt 中的 `Image N` 一致：

1. 预定义参考图；
2. Step 3 选择的历史参考图；
3. 其他静态参考图如未来继续保留的 sink 图。

如果没有预定义参考图，不输出该块。

## Step 3 历史参考图预算

Seedance 静态参考图总上限为 9。预定义参考图占用预算后，Step 3 的实际历史检索数量应动态收缩：

```python
predefined_count = len(enabled_predefined_references)
remaining_slots = max(0, 9 - predefined_count)
effective_max_retrieved_frames = min(
    project.settings.visual.max_retrieved_historical_frames,
    remaining_slots,
)
```

注意这里是 `min`，不是 `max`。

当 `remaining_slots == 0` 时：

- Step 3 可以完成；
- 不选历史参考图；
- Details/Logs 中记录原因：预定义参考图已占满 Seedance 静态图片预算。

当 `predefined_count > 9` 时：

- 当前 Shot 不应进入 Step 3/Step 5；
- 保存或运行时返回明确错误。

## Attempt Snapshot

预定义参考图属于 Shot Design 输入，运行时必须冻结进 Attempt snapshot：

- `input_snapshot.predefined_references`
- `seedance_prompt.request_content_summary`
- `seedance_generation.request.json`

这样用户后续修改 Step 1 不会影响已有 Attempt 的可追溯性。

## Export JSON

项目导出 Shot Design JSON 时应包含 `predefined_references`。

导出路径可优先使用项目内相对路径或复制后的可解析路径。为保证重新导入可用，导出时可写项目内绝对路径，或者在导出包能力出现前说明导出的 JSON 依赖当前项目目录中的图片。

## 实现步骤

1. 扩展 Shot 输入数据结构与校验逻辑，加入 `predefined_references`。
2. 修改 JSON 导入流程：解析字段、复制图片到项目目录、保存项目相对路径。
3. 修改 Shot Design 保存逻辑：支持保存 Label/Guidance、删除条目、添加上传图片。
4. 修改项目页模板与 CSS：在 Step 1 加三列表格。
5. 修改 Step 3：按预定义参考图数量计算 `effective_max_retrieved_frames`，并写入 details/logs。
6. 修改 Seedance Prompt Composition：在历史参考图说明前插入 `[预定义参考图说明]`，并确保媒体顺序一致。
7. 修改 Seedance generation 请求组装：先提交预定义图片，再提交历史参考图。
8. 修改 Export JSON：包含预定义参考图字段。
9. 增加聚焦测试：
   - JSON 导入会复制图片并保存项目相对路径；
   - Step 1 保存 Label/Guidance 后重置下游；
   - 预定义参考图超过 9 张时报错；
   - Step 3 使用 `min(max_retrieved, 9 - predefined_count)`；
   - Prompt 中 `[预定义参考图说明]` 位于 `[历史参考图说明]` 前；
   - Seedance request 图片顺序与 Prompt 编号一致。

## 仍需确认

1. 手动上传初版是否必须实现多选上传，还是单张上传即可。
单张上传即可。
2. 导出 JSON 中的图片路径应写项目内绝对路径，还是继续写项目相对路径并要求从项目目录解析。
绝对路径。
3. 预定义参考图是否需要 `enabled` 开关。当前计划初版不加，仅支持删除；如希望临时关闭而不删除，可补充 `enabled`。
不需要，有没有预定义参考图由用户/json是否输入决定。
