# Seedance Pipeline Architecture

This document describes the lightweight Seedance-only pipeline used by the
current StoryMem agent experiments. It records stable implementation behavior.
Use `history.md` for completed milestone context and `research_plan.md` for
active research directions.

## Scope

On this server we focus on the Seedance pipeline rather than local Wan/StoryMem
model inference. The intended system shape is:

```text
story JSON
  -> shot iteration
  -> memory selection / prompt-aware retrieval
  -> Seedance 10s short-video task
  -> video download
  -> keyframe extraction
  -> structured artifacts + Markdown report
  -> notebook Web UI / future agent evaluation loop
```

## Module Layout

The CLI entry remains `seedance_pipeline.py` for backwards compatibility with
existing shell scripts. It now only parses arguments, builds a `RunConfig`, and
calls the reusable runner.

```text
seedance_pipeline.py
  Thin CLI wrapper.

storymem_seedance/models.py
  Dataclasses for stable internal contracts:
  RunConfig, ShotSpec, ReferenceImage, ShotRunRecord.

storymem_seedance/artifacts.py
  File-backed run state and artifact conventions:
  run_manifest.json, shots.jsonl, references.jsonl, seedance_tasks.jsonl,
  video paths, keyframe checks, URL redaction, and video concatenation.

storymem_seedance/prompting.py
  Seedance text prompt composition plus reference-image data URL packaging.

storymem_seedance/reporting.py
  Markdown memory report generation from structured records.

storymem_seedance/runner.py
  The orchestration layer:
  load story, iterate shots, select memory, submit/wait/download Seedance tasks,
  extract keyframes, update artifacts, and refresh the report.

storymem_seedance/visual_element_memory.py
  Visual-element registry maintenance, historical keyframe scoring, prompt
  context construction, and generated-keyframe VLM annotation. The CLI and Web
  runner share this implementation.
```

Existing algorithm modules remain reusable:

```text
prompt_retrieval.py
  Default memory and CLIP-based prompt-aware retrieval.

memory_query_llm.py
  Optional LLM memory query generation and caching.

extract_keyframes.py
  HPSv3/CLIP keyframe extraction plus last_frame and motion_frames outputs.
keyframe_settings/*.json
  named presets for extraction thresholds and limits; `loose` loads when no
  profile or explicit config path is provided. Historical-memory deduplication
  is controlled by `compare_with_history` and is off by default. The 7860
  Classic StoryMem baseline overrides this with `storymem_original`, which
  restores the original-style settings: max 3 keyframes, HPSv3 threshold 3.0,
  frame-similarity threshold 0.9, and historical memory comparison enabled.

seedance_client.py
  Seedance task API client.
```

## Output Artifacts

Each run writes to `output_dir`. Existing files are preserved, and new
structured files are added for future UI/API consumers.

```text
run_manifest.json
  Current run status, config, story path, active shot, task id, and final video.

seedance_tasks.jsonl
  Backwards-compatible task log. Each task creation and completion appends a
  record. External URLs are redacted after download.

shots.jsonl
  Shot-level execution records: scene/shot id, prompt, cut flag, output video,
  references, status, and task id.

references.jsonl
  One record per reference image actually sent to Seedance, including source
  shot, roles, scores, and file paths.

prompt_retrieval_log.jsonl
  Written by `prompt_retrieval.py`; contains query, candidate rankings, scores,
  selected frames, and final memory bank.

memory_report.md
  Human-readable report generated from structured shot/reference records.

XX_XX.mp4
XX_XX_keyframe*.jpg
last_frame.jpg
motion_frames.mp4
<output_dir>.mp4
  Media outputs used by the next shot and by analysis/UI.
```

## Run State

`run_manifest.json` is the primary lightweight state file for a UI to poll. The
runner updates `status` with values such as:

```text
initialized
running
submitting
waiting_for_seedance
downloading
concatenating
extracting_keyframes
failed
completed
```

When a shot fails during task creation, polling, download, or keyframe
extraction, the runner marks the manifest as `failed`, records the current shot,
stores the task id when available, and writes a structured `error` object into
`seedance_tasks.jsonl` and `shots.jsonl`.

This is intentionally file-backed for now. A future FastAPI service can expose
the same data directly, and a later database migration can mirror these files
without changing the pipeline core.

## Extension Points

### Memory policy

Current policies are implemented in `runner.select_memory()` using helpers from
`prompt_retrieval.py`:

```text
per-Shot Sink / Retrieve / Recent source controls
prompt-aware CLIP retrieval for Cut and non-Cut shots when Retrieve is enabled
optional LLM-generated memory query
```

In the 7860 `storymem_web` baseline, new Classic Web Shots default to Sink on,
Retrieve off, Recent on, matching the original StoryMem-style passive memory
policy while using Seedance as the generator. The memory bank follows the
original `max_memory_size=10, fix=3` default and keyframe maintenance uses the
`storymem_original` profile. Because Seedance accepts at most 9 image
references and non-cut shots reserve one slot for the previous ending frame,
the runner trims submitted memory after selection, preserving sink frames first
and then the most recent recent-window frames. The controlled selector
allocates the existing memory budget in
Sink, Retrieve, Recent order, deduplicates paths, and preserves submission
ordering. The final Web reference budget may further truncate the tail of that
ordered memory list after reserving continuity media.

Representative-keyframe extraction remains shared code, but runner-level
profiles keep the baselines separate: 7860 Classic uses original-style
keyframe maintenance; 7870 visual-element experiments use their project-level
profile and may keep newer, looser candidate-admission settings.

Future policies should return the same reference metadata fields:

```text
source_path
file
roles
source_scene_num
source_shot_num
source_prompt
score / frame_score / video_score
```

This keeps the web UI and Markdown report independent of the policy internals.

### Visual element memory

`visual_element_v1` remains the current research pipeline in the newer
`videogen_notebook` UI on port 7870. In the older 7860 `storymem_web` UI,
newly created or imported projects now default to `classic` so the port can be
used as a StoryMem-style Seedance baseline.

For each Web Attempt, `ProjectRunner` restores the visual-element registry and
annotated historical keyframes from the previous completed Attempt's frozen
`memory_selection.visual_element.state_after`. It then:

```text
plan current-shot element states
-> select visual-element historical references
-> append configured visual sink anchors
-> compose the full Seedance prompt override
-> submit/wait/download Seedance
-> extract ordinary keyframe assets
-> annotate produced keyframes against the current registry
-> freeze state_after for the next Shot
```

The project-level settings are:

```text
pipeline_version: visual_element_v1 | classic
visual_element_sink_frame_count: default 0
visual_element_max_retrieved_frames: default 4
```

The settings validator enforces `sink + max_retrieved <= 9`, matching
Seedance's static reference-image limit. For Default non-cut shots, the Web
runner reserves one static image slot for the previous ending image and reduces
visual retrieval if necessary. Smooth uses a `reference_video`, so that
continuity media does not count against the image budget. Last frame only still
suppresses selected memory and submits only the previous ending image.

### Prompting

Seedance text construction lives in `storymem_seedance/prompting.py`. A future
prompt template experiment should add a new function or template selector there
rather than modifying `runner.py`.

### Agent evaluation loop

The recommended next layer is an evaluator that reads `shots.jsonl`,
`references.jsonl`, `prompt_retrieval_log.jsonl`, and generated media, then
writes:

```text
evals.jsonl
```

Possible evaluators:

```text
manual web rating
CLIP / DINO consistency metrics
VLM judge for character and scene consistency
retry policy that changes memory query or reference set
```

## Web UI Integration Plan

The detailed product design, shot state machine, execution semantics, cost
summary, persistence model, API surface, and implementation phases are defined
in [`../plans/completed/notebook_web_ui_design.md`](../plans/completed/notebook_web_ui_design.md).

The initial implementation is available under `storymem_web/` and can be started
on the 4090 with:

```bash
./storymem web start
```

It provides:

```text
JSON import and empty projects
editable prompt, cut, duration, and generation-mode inputs
project-level classic/visual-element pipeline settings
extensible per-shot generation modes, including last-frame-only and Smooth
Run All and per-shot execution
cooperative interruption and downstream reset
classic structured input references and produced-memory candidates
visual-element status, selected historical references, and produced visual memory
attempt-versioned media and explicit valid-prefix concatenation
project-level token and estimated CNY cost summaries
```

For `last_frame_only`, the worker re-queries the previous current Attempt by
its Seedance task ID and submits the returned original `last_frame_url` with
role `first_frame`. The local ending-frame asset remains available for UI and
memory bookkeeping only; signed Ark URLs remain transient and redacted.
Default non-cut shots use the same original URL for their previous-ending
`reference_image`, while retaining the rest of the selected memory inputs.

### Smooth mode

`smooth` is configurable for any non-cut shot with a predecessor. At execution,
that predecessor must have a completed raw Attempt. Smooth
uses a configurable tail segment from the previous raw shot as a dedicated continuity
input with Seedance role `reference_video`, plus an explicit prompt to generate
content after that video's ending motion. It must not ask Seedance to reproduce
the input segment. Because Seedance requires a public URL, the executor needs a
transient upload/materialization service. The notebook stores the submitted URL
and media preflight result in per-Attempt debug files so `Invalid video_url`
failures can be diagnosed.

Smooth replaces Default's previous-ending image with this reference video and
retains the remaining memory images enabled by Sink/Retrieve/Recent in order. Continuity mode and
memory policy remain independent axes; selector behavior, reports, candidate
admission, and raw-video keyframe extraction are unchanged.

Memory selection, memory reports, and keyframe extraction continue against raw
shot outputs. Smooth affects only Seedance generation media and the derived
project assembly. For each Smooth boundary, keep raw Attempt videos immutable,
drop `A[-1]` and `B[0]`, generate two Practical-RIFE 4.25 frames between anchors
`A[-2]` and `B[1]` at `t=1/3,2/3`, and assemble:

```text
A[:-1] + [I.33, I.67] + B[1:]
```

The derived artifact records source Attempt IDs, source hashes, RIFE model
version, anchor indices, timesteps, output metadata, and errors. Final-prefix
assembly reuses matching transition artifacts without replacing either raw
video. RIFE errors are explicit and retryable; there is no silent interpolation
fallback.

FastAPI remains separate from CUDA work. A persistent subprocess executes a
project sequentially so HPSv3/Qwen model state is reused across shots, while
SQLite stores transactional UI and job state. The existing `run_story()` CLI
entry remains supported through the same lower-level Seedance executor.

Operational diagnostics are intentionally centralized:

```text
./storymem status [--project NAME] [--json]
./storymem web status|start|stop|restart|logs
GET /api/health
```

The health payload captures both the Git commit and a deterministic fingerprint
of the Python execution surface at process startup. Management commands compare
that fingerprint with the current checkout, so uncommitted Python edits and
stale Uvicorn processes are visible without reading logs.

## A6000 Deployment Runbook

The current migrated deployment is configured as follows:

```text
Host: 10.130.128.150
User: lzg
Project: /home/lzg/wxh/world_model_projects/StoryMem
Conda env: /home/lzg/miniconda3/envs/py311
Runtime/cache: /data3/lzg/storymem_runtime
GPU: physical GPU 1, NVIDIA RTX A6000 48 GB
```

Secrets are stored outside the repository with mode `600`:

```text
/home/lzg/.seedance_api_key
/home/lzg/.deepseek_api_key
```

Do not add key contents, passwords, or private URLs to the repository or run
reports.

The remote environment entry point is:

```bash
cd /home/lzg/wxh/world_model_projects/StoryMem
source ./a6000_env.sh
```

It selects the `py311` environment, sets `CUDA_VISIBLE_DEVICES=1`, redirects
Hugging Face and pip caches to `/data3`, and enables Hugging Face offline mode.
Offline mode is important because the HPSv3 and Qwen2-VL caches were migrated
from the source server and the remote host may time out on Hugging Face HEAD
requests.

Migrated model caches:

```text
/data3/lzg/storymem_runtime/cache/huggingface/hub/models--MizzenAI--HPSv3
/data3/lzg/storymem_runtime/cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct
/home/lzg/.cache/clip/ViT-B-32.pt
```

### Verification

Check the environment and GPU:

```bash
source ./a6000_env.sh
python -c 'import torch, hpsv3, clip; print(torch.cuda.get_device_name(0))'
nvidia-smi
```

Verify the exact keyframe path that previously failed on a 24 GB GPU:

```bash
python -c 'from extract_keyframes import save_keyframes; save_keyframes("results/elon_seedance_enhanced_8s/01_01.mp4")'
```

On the A6000 this path uses approximately 23.5 GB of GPU memory and completes
without OOM.

### Resume a run

```bash
python seedance_pipeline.py \
  --story_script_path ./story/elon.json \
  --output_dir results/elon_seedance_enhanced_8s \
  --max_shots 999 --duration 8 --ratio 16:9 \
  --max_memory_size 10 --fix 3 --retrieval_top_k 2 \
  --prompt_retrieval --enhanced_text_prompt --resume
```

`--resume` refreshes the manifest config, records `resumed_at`, reloads prior
successful shot records for `memory_report.md`, fills missing keyframes, and
continues from the next incomplete shot.

### Known Seedance policy failures

These failures are external API policy decisions, not GPU or environment
errors:

```text
OutputVideoSensitiveContentDetected.PolicyViolation
  Output-side copyright policy; may be returned only after a long generation.

InputImageSensitiveContentDetected.PrivacyInformation
  Reference image is classified as containing a real person. The task is
  rejected during create_task before a task id is assigned.
```

The Elon experiment currently stops at Scene 1 / Shot 2 because the generated
Elon reference frames trigger `InputImageSensitiveContentDetected`. Removing
the references would no longer test the intended memory-conditioned pipeline;
use a fictional character story for a full memory experiment instead of trying
to bypass the provider policy.

### Operational files

For a background run, inspect:

```bash
cat results/<run>/run_manifest.json
tail -f results/<run>/seedance_pipeline.log
tail -f results/<run>/a6000_run.stdout.log
```

The manifest is authoritative for `submitting`, `waiting_for_seedance`,
`extracting_keyframes`, `failed`, and `completed` states.

## Compatibility Notes

- Existing shell scripts can continue invoking `python seedance_pipeline.py ...`.
- `memory_report.md` is still generated for human inspection.
- The older `seedance_tasks.jsonl` format remains available.
- The new structured artifacts are additive and intended as stable UI inputs.
