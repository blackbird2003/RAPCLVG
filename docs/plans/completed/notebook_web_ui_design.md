# StoryMem Notebook Web UI Design and Implementation Plan

## 1. Document Status

- Status: completed legacy baseline; see `../../current/history.md` for
  completed milestones and `../../current/research_plan.md` for current
  research tracks.
- Scope: Seedance-based long-video generation only
- Primary deployment: the 4090 server during development
- A6000 deployment: synchronize source through Git when needed; keep secrets,
  generated media, model weights, and runtime state outside Git
- Compatibility requirement: retain the existing CLI, story JSON, result files, and `memory_report.md`
- Current default for new Web projects: project-level `visual_element_v1`.
  Existing projects are migrated as `classic` for reproducibility.

Implemented modules:

```text
storymem_web/domain.py            story normalization and shot states
storymem_web/repository.py        SQLite project/attempt/job control state
storymem_web/project_runner.py    persistent sequential shot runner
storymem_web/process_manager.py   worker launch, cancellation, and child reaping
storymem_web/app.py               FastAPI application
storymem_web/templates/           notebook UI and project summary
```

The first real 4090 web smoke test completed on 2026-06-21 with one enhanced
8-second Seedance shot. It verified submission, polling, download, usage capture,
HPSv3 extraction, three retrieval candidates, ending-frame persistence, explicit
concatenation, and a CNY 7.9902 estimated cost. First-process HPSv3 initialization
took about eight minutes and peaked near 23.6 GB GPU memory; subsequent shots in
the same Run All worker reuse the loaded model.

## 2. Product Definition

The first web UI is a linear, notebook-like workspace for building a long video
from short Seedance generations. It is not a general DAG editor and does not try
to reproduce ComfyUI.

The primary mental model is:

```text
Shot 1: video prompt + cut switch + reference images -> generated video
Shot 2: video prompt + cut switch + reference images -> generated video
...
Valid completed shot prefix -> concatenated project video
```

`scene` remains in the imported story format and in display labels, but the
execution unit is always one ordered shot. In the first version, a scene is only
an optional human-authored shot group and does not impose execution semantics.

## 3. Goals

The first version must support:

1. Creating an empty video project containing only Scene 1 / Shot 1.
2. Importing and validating an existing StoryMem JSON story.
3. Editing each unsubmitted shot's video prompt and cut switch.
4. Displaying the automatically selected memory and reference images for each shot.
5. Running one shot or running all shots sequentially.
6. Interrupting execution and rerunning from a selected shot with notebook-like downstream invalidation.
7. Displaying each shot's generated video directly below its inputs.
8. Displaying the concatenated valid output and project-level cost information below the notebook.
9. Recovering the visible project state after a browser refresh or web-server restart.

## 4. Non-goals for Version 1

- Arbitrary node graphs or workflow plugins
- Parallel shot generation
- Multi-user accounts and permissions
- Reference-image upload, deletion, reordering, or manual role assignment
- Shot insertion, deletion, drag-and-drop reordering, or scene editing
- Automated quality evaluation and retry agents
- A database for video or image bytes
- Replacing the existing CLI and file artifacts

The data model should leave room for these capabilities without exposing them in
the first UI.

## 5. Core Domain Model

### 5.1 Project

A project owns an ordered list of shots, shared generation settings, execution
state, a current final video, and aggregate usage.

Important project fields:

```text
project_id
name
source_story_path / imported_story_json
status
run_mode
active_shot_id
created_at / updated_at
generation_config
  pipeline_version: visual_element_v1 | classic
  visual_element_sink_frame_count: integer, default 0
  visual_element_max_retrieved_frames: integer, default 4
current_final_video
```

`visual_element_v1` replaces the old Shot-level memory-source controls in the
main notebook layout. It freezes the project-level visual settings into each
Attempt through `generation_config` and `memory_selection`. `classic` keeps the
existing Sink/Retrieve/Recent switches and memory-decision display.

### 5.2 Shot

A shot is the editable notebook cell. It has a stable ID independent of its
display position.

```text
shot_id
order_index
scene_num
shot_num
video_prompt
is_cut
generation_mode: default | last_frame_only | smooth
duration_seconds: integer, 2-15
memory_sink: boolean
memory_retrieve: boolean
memory_recent: boolean
first_frame_prompt (preserved but not exposed initially)
memory_query override (preserved but not exposed initially)
state
revision
current_attempt_id
```

Existing nested `scenes[].video_prompts[]` stories are normalized into an
ordered shot list on import. Scene and shot numbers are retained for labels and
backwards-compatible artifact names.

`generation_mode` is an extensible web experiment selector. It defaults to
`default` for Cut/first shots and `smooth` for subsequent non-Cut shots. The
`last_frame_only` mode is available only when `is_cut=false` and
a predecessor exists. It does not alter retrieval or memory updates. It limits
the submitted image list to the previous shot's ending frame and sends that
image with Seedance role `first_frame`; enabling `is_cut` restores `default`.
The image URL is fetched transiently from the previous Seedance task's original
`last_frame_url`. It is never replaced by the locally extracted/re-encoded
`last_frame.jpg`, and the signed URL is not persisted in reports or SQLite.
The same original URL replacement applies to the previous-ending reference in
`default` mode; it remains a `reference_image` alongside selected memory images.

The `smooth` mode is configurable when `is_cut=false` and a predecessor exists;
the runner requires that predecessor to be completed before execution. It
extracts a configurable tail segment from the predecessor's raw
video, exposes it through a transient public URL, and sends it as a Seedance
`reference_video` with an explicit extension prompt. The prompt requests content
after the input video's ending motion and must not request that the input video
be included in the output. Smooth replaces only Default's previous-ending image;
the remaining selected memory images are submitted in their original order.
Retrieval, memory reports, candidate admission, and raw-output memory extraction
remain unchanged.

After generation, `smooth` keeps the raw shot output immutable. Project-video
assembly removes the predecessor's last frame and the current shot's first
frame, then inserts two Practical-RIFE 4.25 frames between the predecessor's
penultimate frame and the current shot's second frame. The derived transition
preserves total duration. The Attempt records raw and derived paths, RIFE
version, anchor indices, and interpolation timesteps. Signed upload URLs remain
transient and redacted. Failed local smoothing is explicitly retryable without
another Seedance submission.

`duration_seconds` is a per-shot generation input because prompt density varies
across a story. Imported stories may provide a parallel `scenes[].durations[]`
array; otherwise each shot inherits the project's existing duration (8 seconds
for new Web projects). The value is frozen in the Attempt input snapshot.

In `classic`, the three memory-source switches are independent of generation
mode. New and imported Shots default to Sink on, Retrieve on, Recent off.
Existing databases migrate to the historical behavior (Cut on/on/on; non-Cut
on/off/on) so old experiments remain reproducible. Retrieve runs prompt-aware
candidate retrieval for both Cut and non-Cut Shots. The enabled policy and
actual submitted-memory counts are frozen in the Attempt snapshot and
memory-selection report.

In `visual_element_v1`, Shot-level Sink/Retrieve/Recent controls are hidden.
The project-level visual settings control early sink anchors and the maximum
number of element-selected historical frames. `sink + max_retrieved <= 9` is
validated against Seedance's static image budget. Continuity media remains
controlled only by `generation_mode`: Default may add the previous ending image,
Last frame only submits only that previous ending image, and Smooth replaces
the previous ending image with the previous raw video's configurable tail segment as
`reference_video`.

### 5.3 Attempt

Every Seedance submission is an immutable attempt. Editing a previously run shot
creates a new revision and a new attempt instead of overwriting the old one.

```text
attempt_id
shot_id
revision
status
input_snapshot
memory_selection
submitted_prompt
task_id
timestamps
usage
error
output_video
produced_memory_assets
```

The input snapshot contains the exact video prompt, cut value, generation config,
and ordered reference list used for that API request.

### 5.4 Reference Image

The reference model must already support future user-provided images even though
the first UI only displays automatic references.

```text
reference_id
source_type: auto | user
roles: previous_last_frame | early_sink_memory |
       prompt_retrieved_memory | recent_window_memory
source_shot_id
source_path
score / frame_score / video_score
order_index
```

### 5.5 Produced Memory Asset

After a shot completes, postprocessing produces visual assets that may affect
later shots. These are outputs of the current shot, not reference inputs to the
current shot.

```text
memory_asset_id
source_shot_id
source_attempt_id
asset_type: retrieval_keyframe | ending_frame | motion_preview
source_path
rank
active_in_memory_pool
created_at
```

The current pipeline gives these asset types different semantics:

```text
retrieval_keyframe
  HPSv3-selected XX_XX_keyframe*.jpg; enters the candidate pool for later shots

ending_frame
  the shot's final frame; used only as direct continuity input for the next
  shot when that next shot has is_cut=false

motion_preview
  diagnostic motion_frames.mp4; displayed optionally but never retrieved
```

Assets from stale or superseded attempts remain in history but must have
`active_in_memory_pool=false` so they cannot leak into a regenerated chain.

## 6. Project Creation and Story Import

### 6.1 Empty project

Creating a project opens the notebook immediately with one editable shot:

```text
Scene 1 / Shot 1
video_prompt: "A traveler opens the door of a quiet workshop at sunrise..."
is_cut: true
references: empty
output: empty
```

The example prompt is ordinary editable content and may be replaced directly.

### 6.2 Import JSON

Importing a JSON file creates a new project rather than replacing an existing
project. The importer validates:

- top-level `scenes`
- each `scene_num`
- non-empty `video_prompts`
- optional `cut` length and boolean values
- optional `first_frame_prompt`, `memory_queries`, and `retrieval_queries` lengths

Missing `cut` entries default to `true`, matching the current runner. Unknown
fields are preserved in the imported source JSON so future schema extensions do
not destroy data.

## 7. Notebook Workspace

### 7.1 Project toolbar

The compact toolbar contains:

- project name
- overall status and progress, for example `3 / 9 completed`
- New Project
- Import JSON
- Run All
- Interrupt
- project settings in a small drawer

There is no marketing header or dashboard-style hero. This is a desktop-first
research tool with a quiet, dense layout.

### 7.2 Shot block

Shots appear from top to bottom in execution order. Each shot is a full-width
section separated by a divider, not a stack of nested cards.

For `classic`, the section order is:

1. Shot header and execution gutter
2. Video prompt textarea
3. Cut / continuous toggle
4. Generation mode selector
5. Generation duration input
6. Sink / Retrieve / Recent switches
7. Memory-decision summary
8. Ordered reference-image strip
9. Output video
10. Produced memory-candidate strip
11. Timing, usage, and error details

For `visual_element_v1`, the per-Shot inputs stay the same except that
Sink/Retrieve/Recent switches are replaced by project-level visual settings.
The runtime section is:

1. Shot status and action buttons
2. Visual Element Status list plus collapsed Visual Element Details
3. Selected Historical References list plus collapsed Reference Selection Details
4. Input Prompt
5. Output Video plus collapsed Seedance Attempt Details
6. Produced Visual Memory list plus collapsed Memory Extraction Details

The three visual-element lists are expanded by default and collapsible. They are
rendered from `attempt.memory_selection.visual_element`, not by parsing
`memory_report.md`.

The execution gutter follows notebook conventions:

- Run icon when the shot can start
- Stop icon only for the active shot
- status indicator
- attempt or revision number
- elapsed time

Completed sections may be collapsed to keep long projects scannable. The active
shot is always expanded.

### 7.3 Shot inputs

The prompt uses a resizable multiline textarea. The cut control is a binary
toggle with two explicit states:

```text
Cut: on       scene transition; use historical memory retrieval
Cut: off      continuous shot; include the previous ending frame
```

Inputs remain editable while a shot is `draft`, `queued`, `stale`, `failed`, or
`interrupted`. They lock when the runner begins `preparing`, because that is
when the immutable input snapshot is created. Completed shots require Reset
From Here before editing.

In `classic`, Sink, Retrieve, and Recent use the same lock and Save / Save all semantics. A
disabled source contributes no role or reference image. Enabled sources are
deduplicated and ordered Sink, retrieved (temporal path order), then Recent,
within the existing reference limit. Continuity media remains independent:
Last frame only still submits only the previous original frame, while Smooth
replaces Default's previous-ending image with its tail video and retains the
enabled historical images.

In `visual_element_v1`, the four visible Shot inputs are Video prompt, Cut,
Generation mode, and Generation duration. Visual-memory selection is controlled
by project settings: pipeline version, sink frame count, and maximum retrieved
historical frames.

### 7.4 Memory decision

The UI must render structured memory data rather than parse
`memory_report.md`. The section appears after preparation and remains visible
for completed attempts.

It shows:

- policy name
- memory query
- sink, recent, candidate, retrieved, and final-memory counts
- retrieval score formula and weights when applicable
- source shots and selected roles

`prompt_retrieval_log.jsonl` currently lacks an explicit target shot ID. The
pipeline must add `target_shot_id`, `scene_num`, and `shot_num` before the UI is
implemented.

### 7.5 Reference strip

References are displayed in actual submission order as stable-size thumbnails
in a horizontally scrollable strip. Each thumbnail shows its role, source shot,
and score when available.

The first version is read-only. The component contract nevertheless includes an
optional user source so upload, deletion, pinning, and reordering can be added
later without changing attempt records.

### 7.6 Output

Before generation, the output region is an empty 16:9 placeholder. During
generation it displays the current phase. After download it contains the video
player, and keyframe extraction continues to report progress independently.

An output is considered valid only when its attempt is the shot's current
attempt and every preceding shot also has a valid completed attempt.

### 7.7 Produced memory candidates

After the output video, the shot displays the assets created by keyframe
postprocessing. This closes the per-shot data-flow loop:

```text
input references -> generated video -> memory candidates for future shots
```

Retrieval keyframes are shown as a horizontally scrollable thumbnail strip in
rank order. Each thumbnail is labelled `candidate pool` and becomes available
only to later shots. The ending frame is shown separately with a `next-shot
continuity` label. `motion_frames.mp4` may be exposed behind a small diagnostic
control and is not presented as memory.

The section is empty before postprocessing, shows extraction progress while the
shot is in `extracting_keyframes`, and becomes immutable when the attempt
completes. A future analysis view may also show which later shots selected each
candidate, but that reverse-link visualization is not required in version 1.

## 8. Execution Semantics

### 8.1 Shot state machine

```text
draft -> queued -> preparing -> submitting -> waiting_for_seedance
      -> downloading -> extracting_keyframes -> smoothing_transition -> completed

preparing/submitting/waiting/downloading/extracting -> interrupted
any execution state -> failed
completed -> stale (when an upstream shot is reset)
failed/interrupted/stale -> draft (through reset)
```

The `smooth` implementation adds a transition-smoothing phase before
the new valid prefix is published. RIFE failure leaves the raw Attempt available
for diagnosis but must not silently fall back to FFmpeg or publish an
unsmoothed boundary as a successful Smooth result.

Editing rules:

| State | Prompt / cut editable |
| --- | --- |
| `draft` | yes |
| `queued` | yes, until the runner reaches it |
| `preparing` and later active states | no |
| `completed` | no |
| `failed`, `interrupted`, `stale` | yes; the next run creates a new attempt |

### 8.2 Run All

Run All launches one persistent project-runner subprocess. It processes shots
sequentially and reads each shot's latest editable values only when that shot
enters `preparing`.

This allows the user to modify later queued shots while an earlier shot is
generating. Only one shot is active at a time.

The runner remains alive across shots so HPSv3, Qwen, and CLIP model caches are
not reloaded for every output.

### 8.3 Run one shot

The per-shot Run action executes only that shot and stops after its postprocessing
finishes. It is enabled only when all preceding shots have valid completed
attempts. This preserves deterministic memory and final-video ordering.

### 8.4 Interrupt

Cancellation is cooperative where possible:

- before Seedance submission, the shot can stop without a remote task
- while polling, the runner records the task ID and stops local waiting
- during local postprocessing, the worker stops at the next safe checkpoint

Seedance may continue a submitted remote task because the current client has no
remote cancellation API. An interrupted attempt retains its task ID. If inputs
are unchanged, a resume operation may reconnect to it instead of creating a
duplicate paid task.

### 8.5 Reset From Here

Resetting Shot N creates notebook-like downstream invalidation:

```text
shots before N: retain current valid attempts
shot N: increment revision and return to draft
shots after N: mark current attempts stale and return to draft
final video: immediately falls back to the valid prefix before N
```

Old attempts and media remain available for auditing and future comparison but
are excluded from the active project version.

## 9. Project Summary Below the Notebook

The project summary is outside all shot blocks and has two sections.

### 9.1 Concatenated video

The final player contains only the contiguous prefix of valid current attempts.
It updates after each completed shot.

For a `smooth` boundary, the final player uses the derived RIFE transition while
the per-shot output player continues to expose the immutable raw Seedance clip.

Concatenation must receive an explicit ordered path list. The current glob-based
`concat_videos()` cannot distinguish current, stale, and superseded attempts and
must not be used unchanged for notebook projects.

The summary displays:

- valid segments / total segments
- total generated duration
- final resolution and ratio
- updated time
- final video player

### 9.2 Cost and usage

Cost is project-level information rather than part of an individual notebook
input/output block. The summary displays:

- submitted task count
- successful, failed, interrupted, and superseded task counts
- generated seconds
- total and completion tokens
- current-version estimated cost
- all-attempt estimated incurred cost
- pricing rule and currency

The two cost totals have different meanings:

```text
current-version cost
  usage from current valid attempts only

all-attempt incurred cost
  usage from every task response, including retries and superseded attempts
```

The current estimator uses Seedance response `usage.total_tokens` and defaults
from `summarize_seedance_usage.py`:

```text
text/image references: CNY 46 per million tokens
video references:      CNY 28 per million tokens
```

Pricing must be stored as project configuration and labelled as an estimate,
not hard-coded into templates. Tasks without returned usage are shown separately
as `usage pending/unknown`; they are not silently treated as free.

## 10. Persistence Layout

SQLite stores the small transactional control plane. Generated media and
research artifacts remain files.

Suggested tables:

```text
projects
shots
attempts
references
memory_assets
jobs
```

Suggested workspace:

```text
workspace/<project_id>/
  source_story.json
  project.log
  shots/<shot_id>/attempts/<attempt_id>/
    input.json
    memory.json
    seedance_task.json
    output.mp4
    memory_assets.json
    keyframes/
  final/current.mp4
  exports/
    run_manifest.json
    shots.jsonl
    references.jsonl
    prompt_retrieval_log.jsonl
    memory_report.md
```

SQLite is used for atomic editing, job ownership, revision invalidation, and
restart recovery. It does not store videos, images, API keys, or raw private
URLs.

## 11. Runtime Architecture

```text
Browser
  -> FastAPI pages and JSON endpoints
  -> Jinja2 + HTMX fragments
  -> SQLite control state

FastAPI process manager
  -> one persistent project-runner subprocess
  -> existing Seedance client and memory algorithms
  -> project artifact directory
```

The web process must not run HPSv3/Qwen inference in-process. A CUDA OOM or
worker crash must not take down the UI.

Version 1 uses one global execution lease on the 4090. A second project may be
edited while one project runs, but it cannot start GPU work until the lease is
released. Redis and Celery are unnecessary at this stage.

## 12. Backend Refactoring

The current `run_story()` loop should become a compatibility wrapper around
shot-level operations:

```text
normalize_story()
prepare_shot()
execute_shot()
postprocess_shot()
concat_valid_attempts()
run_project()
```

Required contract changes:

1. Add stable `project_id`, `shot_id`, `attempt_id`, and `revision` identifiers.
2. Return a structured `MemorySelection` from memory selection.
3. Return structured produced-memory metadata from keyframe extraction.
4. Add target shot identifiers to retrieval logs.
5. Persist the actual submitted enhanced prompt.
6. Record stage timestamps and elapsed seconds.
7. Make manifest writes atomic with temporary files and `os.replace()`.
8. Add a cancellation callback to Seedance polling and safe worker checkpoints.
9. Concatenate explicit valid attempt paths instead of globbing the directory.

The CLI continues to build a project in memory and call `run_project()` with
Run All semantics.

## 13. Web Module Layout

```text
storymem_web/
  app.py
  schemas.py
  repository.py
  process_manager.py
  routes/
    pages.py
    api.py
    media.py
  templates/
    base.html
    projects.html
    project.html
    fragments/
      project_toolbar.html
      shot.html
      memory.html
      references.html
      produced_memory.html
      project_summary.html
  static/
    app.css
    htmx.min.js
    lucide.min.js
```

FastAPI serves both HTML and a small stable JSON API. HTMX polls only active
fragments every two seconds. Completed shots do not continue polling.

Media routes must resolve paths against the project's workspace root and reject
path traversal. API keys and provider URLs are never sent to the browser.

## 14. API Surface

Minimum endpoints:

```text
POST  /api/projects
POST  /api/projects/import
GET   /api/projects/{project_id}
PATCH /api/projects/{project_id}/shots/{shot_id}
POST  /api/projects/{project_id}/run
POST  /api/projects/{project_id}/interrupt
POST  /api/projects/{project_id}/shots/{shot_id}/run
POST  /api/projects/{project_id}/shots/{shot_id}/interrupt
POST  /api/projects/{project_id}/shots/{shot_id}/reset-from-here
GET   /api/projects/{project_id}/summary
GET   /media/{project_id}/{path}
```

Shot updates use an expected revision number so stale browser tabs cannot
silently overwrite newer edits.

## 15. Implementation Phases

### Phase 1: domain and persistence

- Normalize old stories into ordered shots.
- Add Project, Shot, Attempt, Reference, and Job repositories.
- Add produced-memory assets and active-pool membership.
- Add attempt-versioned media paths.
- Add explicit final-video concatenation.
- Add cost aggregation from task usage.

### Phase 2: shot executor

- Extract `prepare_shot()` and `execute_shot()` from `run_story()`.
- Persist input snapshots and structured memory selections.
- Add cancellation checks and stage events.
- Preserve the current CLI as a regression path.

### Phase 3: read/edit UI

- Create and import projects.
- Render the notebook and project summary.
- Edit draft and queued prompts and cut switches.
- Display structured memory, references, existing output, and costs.

### Phase 4: execution controls

- Implement Run All and per-shot Run.
- Implement Interrupt, resume, and Reset From Here.
- Add local polling, errors, progress, and restart reconciliation.

### Phase 5: verification and deployment

- Run unit tests for normalization, state transitions, invalidation, and costs.
- Run a short mocked API integration project.
- Run a real two-shot Seedance smoke test on the 4090.
- Verify browser refresh and process restart recovery.
- Synchronize the completed implementation and environment changes to A6000.

## 16. Acceptance Criteria

Version 1 is complete when all of the following hold:

1. A user can create a one-shot empty project without editing JSON.
2. Existing nested StoryMem JSON files import with the same shot order and cut values.
3. A queued future shot remains editable while the current shot runs.
4. A submitted shot cannot be edited without Reset From Here.
5. Run All executes exactly one shot at a time in order.
6. Per-shot Run stops after that shot and respects upstream prerequisites.
7. Interrupt leaves enough task state to avoid accidental duplicate submission.
8. Reset From Here invalidates all downstream current attempts without deleting history.
9. Each completed shot displays its memory decision, ordered input references, and video.
10. Each completed shot displays the keyframes it contributes to the future retrieval pool.
11. Ending frames and diagnostic motion previews are not mislabelled as retrieval candidates.
12. Stale or superseded attempts cannot contribute active memory candidates.
13. The final player contains only the contiguous valid current attempt prefix.
14. Cost summary distinguishes current-version and all-attempt incurred estimates.
15. Refreshing the page or restarting FastAPI does not lose project state.
16. Existing CLI experiments still run and produce compatible reports.
17. No API key, private download URL, or unrestricted filesystem path reaches the browser.
