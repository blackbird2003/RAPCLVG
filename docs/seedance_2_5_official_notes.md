# Seedance 2.5 Official API Notes

Last reviewed: 2026-08-18

This note records the Seedance 2.5 behavior relevant to RAPCLVG. It is a
design/reference note. RAPCLVG defaults to Seedance 2.5; existing project
settings remain immutable unless explicitly edited.

## Official sources

- [Create a video generation task](https://docs.volcengine.com/docs/82379/1520757?lang=zh)
- [Doubao Seedance 2.5 tutorial](https://docs.volcengine.com/docs/82379/2607688?lang=zh#2.5_compatibility)
- [Doubao Seedance 2.5 prompting guide](https://docs.volcengine.com/docs/82379/2607689?lang=zh)

## Availability and model identity

- The official task-creation reference states that Seedance 2.5 is publicly
  available through Volcano Ark API and the online experience.
- Model ID: `doubao-seedance-2-5-260628`.
- The API path is unchanged from the current backend:
  `POST https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks`.
- Generation is asynchronous. Create a task, poll
  `GET /contents/generations/tasks/{task_id}`, then download the returned
  `video_url` after `succeeded`.

## Capability delta from Seedance 2.0

| Item | Seedance 2.0 | Seedance 2.5 |
| --- | --- | --- |
| Maximum output duration | 15 s | 30 s |
| Reference images | Up to 9 | Up to 30 |
| Reference videos | Up to 3; total <= 15 s | Up to 10; total <= 30 s |
| Reference audio clips | Up to 3 | Up to 10 |
| Total multimodal references | 15 slots in the documented per-type limits | 50 slots (30 images + 10 videos + 10 audio clips) |
| Task types | Text/image/multimodal generation, with limited continuity use | Reference generation, video editing, and video extension are explicitly modeled |
| Output format | Existing MP4 workflow | `mp4` or `mov`; MOV is recommended for editing/extension |

The task API accepts text, image URLs/Base64/material IDs, video URLs/material
IDs, and audio URLs. Images, videos, and audio are numbered by upload order
for prompting (`Image 1`, `Video 1`, `Audio 1`; official examples also use
`@image1`, `@video1`, and `@audio1`). The prompting guide recommends explicit
text bindings for every input rather than encoding identity only inside an
image.

## 2.5 task-type constraints

For Seedance 2.5 multimodal reference requests, `omni_reference_task_type`
may be set to `auto` (default), `reference`, `edit`, or `extend`.

- `reference`: generate a new video with references. `ratio` and `duration`
  remain controllable within normal 2.5 limits.
- `edit`: must include `reference_video`; the edited video must be 4-30 s;
  `ratio` must be `adaptive`; `duration` must be `-1`.
- `extend`: must include `reference_video`; `ratio` must be `adaptive`.
  Output `duration` may be 4-30 s or `-1`.
- `auto`: the model infers task type from media and text. Incorrect inferred
  compatibility can fail asynchronously with
  `InvalidParameter.TaskTypeConstraint`.
- Explicit `edit` or `extend` moves part of this validation to submission
  time, but prompt intent must still match the selected type or the task can
  fail with `InvalidParameter.TaskTypeMismatch`.

For extension prompts, the official guide requires wording such as
"extend forward", "extend backward", "continue", or "continue writing" and
uses explicit video references (for example, `Extend Video 1 forward ...`).

## Parameter and media limits

- `duration`: `4-30` seconds or `-1` (automatic length). Video editing only
  supports `-1`.
- `ratio`: normal reference generation can use `16:9`, `4:3`, `1:1`, `3:4`,
  `9:16`, `21:9`, or `adaptive`. Editing, extension, first-frame, and
  first/last-frame tasks require `adaptive`.
- `resolution`: `480p`, `720p`, `1080p`, or `4k` where supported by the model
  and task. Seedance 2.5 1080p output uses 10-bit H.265/HEVC, which may not
  play in every browser or system player.
- `output_format`: `mp4` by default; use `mov` for video extension/editing
  where color, brightness, and audio continuity matter.
- Images: JPEG/PNG/WEBP/BMP/TIFF/GIF/HEIC/HEIF, aspect ratio `[0.4, 2.5]`,
  width and height `[300, 6000]`, less than 30 MB each; request body <= 64 MB.
- Videos: MP4/MOV, 2-30 s for non-edit reference use and 4-30 s for editing;
  each <= 200 MB; 24-60 FPS; all reference videos total <= 30 s.
- `return_last_frame` is available and returns a watermark-free PNG tail frame.
- `generate_audio` defaults to `true`; generated audio is mono.

First-frame / first-last-frame generation and multimodal reference generation
are mutually exclusive API scenarios. For multimodal generation, a prompt can
ask a reference image to function as a first/last-frame-like constraint, but
the strict first/last-frame API remains the stronger guarantee.

## Implications for RAPCLVG

The current generic request path, asynchronous task polling, image/video/audio
content representation, and prompt composition are compatible foundations.
However, a proper Seedance 2.5 backend should be an explicit provider/model
branch rather than merely swapping the model ID.

1. Add a 2.5 capability profile to validate the larger 30/10/10 media budget
   and `4-30` duration range. Do not relax the existing 2.0 nine-image budget
   globally.
2. Add optional `omni_reference_task_type` to the request client. Use
   `reference` for ordinary multimodal retrieval; use `extend` for the
   non-cut Smooth mode when a tail reference video is submitted.
3. In Smooth/extension mode, set `ratio="adaptive"`, preserve a reference
   tail video in `content`, and emit the documented explicit extension
   instruction. The current 2-second tail is valid as a non-edit video input,
   though the tutorial recommends 4-30 s reference video for generic `auto`
   multimodal usage.
4. Treat the 2.5 30-second maximum as a project/shot setting, not a new
   default. Existing project JSON and 2.0 experiments must retain their
   `4-15` validation and current model behavior.
5. Retain explicit per-reference textual guidance. With more than a handful
   of media inputs, prompt-side image/video/audio numbering and binding become
   more important, not less.
6. Consider a MOV-aware download/preview/assembly path before enabling 2.5
   extension or editing in the UI. Current browser previews and FFmpeg/RIFE
   tooling are MP4-oriented.

## Pricing note

The task reference documents a temporary Seedance 2.5 promotion: from
2026-08-14 14:00 to 2026-09-17 14:00 (UTC+8), 1080p is charged at 72% of list
price, stated as "from approximately CNY 2.7 per second". The same document
does not provide a complete public 2.5 rate table for all resolutions, so
RAPCLVG should continue to record provider-reported usage/cost rather than
derive estimates from this promotion.
