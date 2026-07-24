# Project History

This file summarizes completed decisions and implementation milestones. Use Git
commits and `../engineering/task_report.md` for detailed diffs, tests, and issue
notes.

## 2026-07-12

### videogen_notebook as the primary Web UI

The current Web UI for experiments is `videogen_notebook` on port 7870. It is a
lightweight, project-bundle-based notebook system rather than the older SQLite
`storymem_web` UI on port 7860. Each project owns its scripts, settings, Shot
state, Attempts, assets, logs, and current-output prefixes under
`.runtime/videogen_notebook/projects`.

The active Shot pipeline has six visible steps:

```text
Shot Design
Visual Elements Plan
Historical Reference Selection
Seedance Prompt Composition
Seedance Video Generation
Keyframe Maintaining
```

Prompt composition is its own editable step. The current Shot task appears
before the full-script context, and Force animation is enabled by default for
new/imported projects while remaining editable before Seedance submission.

### Current continuity and generation defaults

Imported non-cut Shots default to Smooth, while first Shots and cut Shots
default to Default. Smooth submits the previous raw Shot's configurable tail
segment as Seedance `reference_video`; the default tail length is 2 seconds.
Default non-cut continuity uses the previous Seedance original `last_frame_url`
as the final static image reference. Last-frame-only mode submits only that
original `last_frame_url`.

Per-Shot generation duration now follows Seedance 2.0 behavior: `-1` means
automatic duration, and explicit user choices are limited to integer seconds in
the `4-15` range. Seedance quality and ratio are project settings, defaulting to
`720p` and `16:9`.

### Current Visual Element Memory baseline

The active memory path is Visual Element Memory rather than the older
Sink/Retrieve/Recent per-Shot switch UI. Step 2 maintains a visual-element state
table from story context. Step 3 selects historical references from existing
visual memory and can enrich selected frames with `reference_guidance`. Step 6
extracts produced keyframes, annotates visible elements and reference quality,
and stores a `holistic_description` for future reference prompting.

Reference selection supports Greedy Coverage, Static Top-K, and Disable
reference images for ablation. Greedy now attempts to select up to
`max_retrieved_frames` without early-stopping when all required elements are
covered, so it can still provide style anchors. Candidate reference quality
affects scoring: `full`, `partial`, and `weak` references are weighted
differently, and only full matches count as truly covering required elements.

### Current execution baseline

`videogen_notebook` uses the real paid runner by default. The background job
manager allows multiple projects to run concurrently, with a default worker
limit of 30 via `VIDEOGEN_NOTEBOOK_MAX_WORKERS`. Local GPU-heavy Step 6 work is
serialized by a GPU lock.

## 2026-06-25

### Keyframe JSON profiles

Representative-keyframe extraction constants were moved into
`keyframe_settings/*.json`. The `loose` profile now loads by default, and
experiments can switch presets with `STORYMEM_KEYFRAME_PROFILE` or an explicit
config path. Historical-memory deduplication is an opt-in
`compare_with_history` setting, so generated candidates are normally filtered
only against the previous selected candidate frame inside the same shot.

### Audio-enabled generation and Smooth final audio

Seedance requests now generate audio-enabled videos by default. Smooth final
assembly also preserves concatenated clip audio when source videos contain
audio, instead of producing a silent derived final video.

### Prompt optimization context

Seedance prompt guidance and project notes were added to support later prompt
template experiments. The active research question is no longer only how to
write a stronger global prompt, but how to bind each reference image to an
explicit use and avoid leaking irrelevant visual elements.

## 2026-06-24

### Smooth continuity mode

Smooth was implemented as a third non-cut generation mode beside `default` and
`last_frame_only`. It submits the previous raw shot's final one-second video as
Seedance `reference_video`, asks Seedance to continue after the ending motion,
and keeps selected memory images independent from the continuity input.

Raw Attempt videos remain immutable. The derived final project video handles
Smooth boundaries by removing the previous segment's last frame and the current
segment's first frame, then inserting two Practical-RIFE 4.25 frames between the
neighboring anchors. RIFE errors are explicit and retryable; FFmpeg is not used
as a silent fallback.

### Smooth default for continuous shots

New or imported non-cut shots now initialize as Smooth when they have a
predecessor. First shots and cut shots continue to initialize as Default.
Existing projects and explicit user selections are preserved.

### Per-shot memory source controls

The Web UI added independent `Sink`, `Retrieve`, and `Recent` switches for each
shot. New shots default to `on / on / off`. Existing databases migrate to the
legacy policy:

```text
cut:     Sink=on, Retrieve=on,  Recent=on
non-cut: Sink=on, Retrieve=off, Recent=on
```

Retrieve now runs for cut and non-cut shots when enabled. Continuity mode and
memory policy are independent axes: Smooth changes the continuity media, while
the switches decide which historical memory images are submitted.

## 2026-06-23

### Per-shot duration and Save all

Generation duration became a per-shot input with an 8-second default and an
Attempt snapshot. The Web UI also gained `Save all`, letting multiple edited
shots be persisted before `Run all`.

### Continuity experiments

Several temporary tools were added to study video-extension and seam handling on
the Little Prince sample. Findings:

- Default prompt constraints were not enough to force a stable first frame.
- Last-frame-only generation preserved the first frame but not motion state.
- Plain tail-video extension improved perceived motion continuity.
- Retained-input extension was marked maybe not effective because Seedance
  re-rendered the input tail and introduced appearance drift.
- RIFE seam interpolation looked smoother than FFmpeg interpolation in human
  review and became the provisional Smooth transition backend.

## Current Research Baseline

The project now has enough infrastructure to treat continuity, historical
reference selection, keyframe maintenance, and agentic reflection as independent
experiment axes:

```text
continuity: default | last_frame_only | smooth
reference selection: greedy coverage | static top-k | future reflect/rerank
keyframe maintenance: profile, quality threshold, reference-quality weighting
reflection: off | visual-plan reflect | future step-specific reflect
```

The active research tracks are:

1. precise reference prompting and visual-memory preprocessing, especially
   holistic frame descriptions and reference guidance;
2. historical frame retrieval algorithm improvement, including style anchors,
   quality-aware coverage, and stronger ablation support;
3. more complete evaluation, including VBench-style consistency checks and
   single-video memory-strategy ablations;
4. agentic reflection beyond Visual Elements Plan, especially reference
   selection and keyframe maintenance.
