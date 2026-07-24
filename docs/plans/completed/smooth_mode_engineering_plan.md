# Smooth Mode Engineering Plan

## Status And Scope

**Status: implemented. This document records the intended behavior and the
original handoff; later commits made Smooth the default for new/imported
continuous shots and added independent Sink/Retrieve/Recent memory controls.
The as-built memory-input correction below takes precedence over the historical
tail-video-only wording.**

Add `smooth` as the third non-cut Web generation mode beside `default` and
`last_frame_only`. Implement only this continuity feature. Preserve all current
memory selection, window-memory, retrieval, keyframe extraction, CLI defaults,
and existing generation-mode behavior. Retrieval-first architecture work is a
separate research task.

## Required Behavior

### Validation And UI

- Add `smooth` to the central generation-mode domain, API schema, repository
  validation, Attempt snapshots, status output, and Web selector.
- Permit it only when `is_cut=false` and a completed predecessor exists, using
  the same edit/reset semantics as `last_frame_only`.
- New/imported non-cut shots with a predecessor default to `smooth`; first shots
  and cut shots default to `default`. Enabling Cut resets an incompatible mode
  to `default`.

### Seedance Generation

For a Smooth shot:

1. Read the previous current Attempt's immutable raw output video.
2. Extract its final 1.0 second at the source frame rate and resolution, without
   audio, into the current Attempt directory.
3. Publish the clip to a transient public HTTPS URL through a small injectable
   publisher abstraction. Tests must use a fake publisher. The initial research
   provider may implement the proven tmpfiles multipart flow behind explicit
   configuration, with bounded timeout/retry and URL-redacted logs.
4. Submit Seedance content in this order: full text prompt, then one `video_url`
   item with role `reference_video`, followed by Default's remaining selected
   memory images in their existing order. Smooth replaces only Default's
   previous-ending tail-frame image; it is not a tail-video-only mode.
5. Prefix the actual submitted prompt with an explicit instruction equivalent
   to: continue after Video 1's final image, character/object motion, and camera
   motion; do not rewind, repeat, pause, or include/replay the input segment.
6. Keep the downloaded Seedance video as the raw Attempt output used by the shot
   player, keyframe extraction, and memory updates. Never persist the public
   upload URL in SQLite, reports, logs, or committed files.

### RIFE Transition And Final Assembly

After the current raw video is downloaded:

- Remove the previous raw segment's final frame `A[-1]` and the current raw
  segment's first frame `B[0]` from the project-level assembly.
- Use `A[-2]` and `B[1]` as Practical-RIFE 4.25 anchors.
- Generate exactly two frames with `Model.inference(..., timestep=t, scale=1.0)`
  at `t=1/3` and `t=2/3`.
- Assemble the boundary as `A[:-1] + [I.33, I.67] + B[1:]`. This replaces two
  frames with two frames and preserves project duration and frame count.
- Store the derived transition under the current Attempt, including source
  Attempt IDs/hashes, model revision/version, anchors, timesteps, metadata, and
  any error. Do not overwrite either raw video.
- A shot's mode describes the boundary before that shot. The assembler must
  support multiple Smooth boundaries: a middle raw clip may lose its first frame
  because its own mode is Smooth and its last frame because the next shot's mode
  is Smooth.
- RIFE failure is explicit and must not silently use FFmpeg. Preserve the raw
  Seedance output and make smoothing idempotent so postprocessing can be retried
  without another paid generation.

## Proven RIFE 4.25 Experience

Reuse or extract the validated implementation from
`tools/rife_seam_interpolation_experiment.py`; do not import a tool script from
production modules.

- Practical-RIFE repository:
  `https://github.com/hzwer/Practical-RIFE.git`
- Validated revision: `17d8c7a1005b37f4c97bfee04e316aaec7fdc536`
- Model: official Practical-RIFE 4.25 release, `RIFEv4.25_0919.zip`
- Archive SHA-256:
  `e63d481b7ae5d4a4e6ad7ac5b410ff78f3bf7be3b51b2e38ca8152747abde5b4`
- `flownet.pkl` SHA-256:
  `6615790efd627772917205db291f51cd392528a157ecbb2ecaeec3bff8eb6de2`
- Existing 4090 cache: `.runtime/deps/Practical-RIFE/`
- Do not modify shared Python environments globally. Load weights lazily and
  cache one model instance per persistent worker. Respect configured
  `CUDA_VISIBLE_DEVICES`; do not hardcode physical GPU indices.

Important 720p lesson: Practical-RIFE 4.25 expects dimensions padded to a
multiple of `max(128, int(128/scale))`. At scale 1.0, pad 1280x720 anchors to
1280x768 before inference, then crop each result back to 1280x720. Omitting this
caused the observed `Expected size 768 but got size 720` tensor error.

The validated two-frame experiment used `t=1/3,2/3`, took about 0.301 seconds of
RIFE inference on a 4090, and produced a 386-frame, 24 fps, 1280x720 output with
no audio. Human review preferred RIFE to FFmpeg because RIFE was smoother and
FFmpeg retained visible seam ghosting.

## Suggested Ownership

Expected primary files include:

```text
storymem_web/domain.py
storymem_web/schemas.py
storymem_web/repository.py
storymem_web/project_runner.py
storymem_web/templates/project.html
storymem_seedance/executor.py
storymem_seedance/prompting.py
storymem_seedance/artifacts.py or a new transition assembler module
tests/test_web_repository.py
tests/test_web_app.py
tests/test_project_runner.py
tests/test_seedance_executor.py
```

Prefer small reusable modules for reference-video publishing, tail extraction,
RIFE loading/inference, and mode-aware assembly. Keep FastAPI free of direct
CUDA work; RIFE runs in the persistent project worker.

## Acceptance Criteria

1. Existing Default and Last frame only tests and behavior remain unchanged.
2. Smooth is editable/selectable only for eligible non-cut shots and is frozen
   in immutable Attempt input snapshots.
3. A fake-client test verifies exact Seedance content ordering, role, public URL
   provenance, extension prompt, replacement of the previous-ending image, and
   retention of the remaining selected memory images.
4. Public URLs and API keys are absent from SQLite, reports, logs, and Git.
5. Synthetic-video tests verify exact tail extraction and constant-frame-count
   assembly for one and multiple Smooth boundaries.
6. Unit tests cover 720-to-768 padding and crop-back without loading CUDA.
7. A local RIFE smoke test reuses cached weights and validates two intermediate
   frames plus output metadata; do not make a paid Seedance call unless the user
   explicitly approves it.
8. Smooth errors preserve raw output, are visible, and can retry local
   postprocessing without resubmitting Seedance.
9. Focused tests, Python compilation, template rendering/JavaScript syntax, and
   `git diff --check` pass. Record the result in `../../engineering/task_report.md` and commit
   only task-owned changes.

## Non-Goals

- Do not change sink/recent window memory, prompt-aware retrieval, selector
  weights, candidate-pool policy, or the retrieval-first TODO.
- Do not remove or redefine existing generation modes.
- Do not add frame-search, SSIM/motion seam selection, variable interpolation
  count, or automatic FFmpeg fallback.
- Do not redesign unrelated Web UI or CLI behavior.

## Historical Engineering Thread Handoff Prompt

The prompt below is retained for provenance. Its tail-video-only instruction was
superseded during implementation: Smooth retains Default's selected memory
images after replacing only the previous-ending image.

```text
请在 /home/wxh/world_model_projects/StoryMem 中实现 Smooth generation mode。

开始前阅读 AGENTS.md 和 docs/plans/completed/smooth_mode_engineering_plan.md，并严格以该计划
的 Scope、Required Behavior、RIFE 经验、Acceptance Criteria 和 Non-Goals 为准。

目标：在 Web notebook 中为非 Cut Shot 增加第三种 generation mode `smooth`。
它使用上一段原始视频的末尾 1 秒作为 Seedance reference_video，并在实际提交
Prompt 中明确要求从尾端动作与运镜继续、不得回退或复现输入段。生成后保持原始
Attempt 视频和记忆提取不变；最终项目拼接删除上段末帧与下段首帧，以 A[-2]/B[1]
为锚点，用 Practical-RIFE 4.25 在 t=1/3、2/3 生成两帧等长替换。

这是当前最高优先级工程任务，但范围仅限连续性机制。不要修改 StoryMem
sink/recent 窗口记忆、prompt-aware retrieval、候选池、selector、Default、
Last frame only 或其他无关行为。不得静默回退 FFmpeg，不得发起付费 Seedance
真实调用；网络发布与 Seedance 请求使用可注入 fake 完成自动测试，本地只做已有
视频与缓存权重的 RIFE smoke test。

先阅读现有代码和测试，按现有 Attempt/versioning/worker 模式实现，避免大范围重构。
完成端到端实现、聚焦测试、编译/模板/JS/diff 检查；将简洁结果写入
docs/engineering/task_report.md，提交所有 task-owned changes。最终汇报：行为、主要文件、
测试结果、未运行的付费/网络验证、运行或部署注意事项、commit hash。
```
