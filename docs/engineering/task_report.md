# Engineering Task Report

This is a compact handoff log for completed engineering tasks. Keep each entry
short and factual. Record implementation, verification, and meaningful issues;
use Git for detailed history and never paste full command output here.

## 2026-07-05 - Holistic Reference Fields

- Result: Implemented keyframe `holistic_description`, reference
  `reference_guidance`, prompt composition usage, and first-pass Step 3/Step 6
  Web editing for guidance/description plus row deletion.
- Scope: Visual element VLM annotation schema, reference guidance LLM
  enrichment, prompt context composition, `videogen_notebook` store/routes/UI,
  focused tests, and design docs.
- Verification: Python compilation passed; `tests.test_visual_element_memory`
  passed; focused `tests.test_videogen_notebook` store/app tests for guidance,
  prompt context, fake output, and edit/reset behavior passed; `git diff
  --check` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-05 - Holistic Reference Plan

- Result: Added an implementation plan for keyframe `holistic_description`,
  historical-reference `reference_guidance`, Seedance prompt composition usage,
  and first-pass Step 3/Step 6 edit support.
- Scope: Documentation only.
- Verification: `git diff --check` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-05 - Visual Element Type Convergence

- Result: Confirmed the visual element pipeline is still designed around the
  three core types `character`, `scene`, and `object`. Removed expanded type
  choices from the Web editor and Visual Plan reflection prompt, and normalized
  legacy `location` / `environment` rows to `scene` at Web save and runtime
  snapshot loading boundaries to prevent `KeyError: 'location'` during
  keyframe annotation visualization.
- Scope: Visual element memory type normalization, Visual Plan reflection type
  validation, Web Visual Plan editing options, focused regression tests.
- Verification: Python compilation, `git diff --check`, and two focused
  regression tests passed. The broader notebook test file was not used as a
  gate because existing route tests now wait on the intentional default 10s
  step review delay and timed out while still in `running`.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-05 - Notebook Small UI Defaults

- Result: Added a 0s Step review delay option, changed the default review
  delay to 10s, made Auto reflect visual plan enabled by default, and moved Add
  Shot from the top toolbar to below the shot list.
- Scope: `videogen_notebook` defaults, project template, and design docs.
- Verification: Lightweight Python compilation and `git diff --check` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-05 - Web Visual Plan Reflect

- Result: Connected Visual Elements Plan Reflect to the Web UI. The Step2
  Reflect button now runs a background job, applies validated reflection edits
  to the visual plan rows, records reflection details, and keeps the visual_plan
  step completed. Added a project-level `auto_reflect_visual_plan` switch so
  automatic Run Shot / Run all can reflect once after the initial Step2 result.
- Scope: `videogen_notebook` settings, runner, project store, jobs, FastAPI
  routes, project template, focused tests, and design docs.
- Verification: Python compilation passed; `tests.test_visual_plan_reflection`
  and `tests.test_videogen_notebook` passed with fake reflection coverage for
  manual route, direct runner call, and auto-run behavior.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run in
  this implementation pass.
- Commit: this commit; final hash reported in handoff.

## 2026-07-05 - Visual Plan Reflection Prototype

- Result: Added a Chinese prompt/function prototype for Visual Elements Plan
  reflection. It asks the LLM to review each current element row, keep
  `element_id` and `introduced_at` fixed, and suggest no-change or field-level
  edits for name/type/status/notes/reason.
- Scope: New `storymem_seedance.visual_plan_reflection` module with prompt
  construction, Ark LLM call, JSON parsing/validation, apply helper, and a
  small CLI for historical project samples.
- Verification: Python compilation and `tests.test_visual_plan_reflection`
  passed. A real Seed2.1 Turbo call on project
  `p_20260704_033206_e7ab0077`, Shot `0005`, Attempt `a003` produced 37
  validated row decisions and 2 warnings.
- Safety: No Seedance video generation, upload, VLM, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-04 - videogen_notebook Shot Current Output Cache

- Result: Fixed Current output reuse after modifying completed prefixes. Each
  completed shot now records its own `current_final_video_asset_id`, and the next
  shot assembles from the previous shot's saved Current output plus the current
  raw video.
- Scope: `videogen_notebook` final assembly, reset semantics, focused tests, and
  design docs.
- Verification: Python compilation and `tests.test_videogen_notebook` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-04 - videogen_notebook Prompt Step Implementation

- Result: Split Seedance Prompt Composition into its own `seedance_prompt` step,
  added prompt persistence/edit/save, kept Seedance Video Generation reading the
  saved prompt, and added project settings for step review delay plus optional
  human confirmation before Seedance submission.
- Scope: `videogen_notebook` domain, runner, project store, FastAPI routes,
  project template/CSS, focused tests, and design docs.
- Verification: Python compilation passed; `tests.test_videogen_notebook` passed
  with fake, dry-run, injected-client, route, edit, and confirmation-gate
  coverage.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.
- Commit: this commit; final hash reported in handoff.

## 2026-07-04 - Step Review Delay Decisions

- Result: Updated the step-state design with confirmed decisions: no `Never`
  delay option, a separate Seedance human-confirmation switch, no auto-resume
  after editing, prompt editing in scope, and reference/keyframe edits plus all
  Reflect actions reserved for later.
- Scope: Documentation only.
- Verification: Text reviewed locally; no code or runtime behavior changed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-04 - Seedance Prompt Step Clarification

- Result: Corrected the step-state design document so Edit/Save/Reflect apply
  to `seedance_prompt` rather than `seedance_generation`; added implementation
  clarification questions.
- Scope: Documentation only.
- Verification: Text reviewed locally; no code or runtime behavior changed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-04 - videogen_notebook Step State Machine Plan

- Result: Added a design/implementation plan for Step-local states,
  project-level auto-run review gates, Seedance prompt composition splitting,
  and future Edit/Reflect interfaces.
- Scope: Documentation only.
- Verification: Markdown reviewed locally; no code or runtime behavior changed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-02 - Visual Element LLM Output Limit

- Result: Raised Visual Elements Plan and keyframe VLM Ark chat `max_tokens`
  from small hard-coded caps to 128K tokens (`131072`) for
  `doubao-seed-2-1-turbo-260628`.
- Scope: `storymem_seedance.visual_element_memory` and focused unit coverage.
- Verification: Python compilation, focused visual-element-memory tests, and
  `git diff --check` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-02 - videogen_notebook Visual Plan Editing

- Result: Added post-AI editing for Step 2 Visual Elements Plan. Users can enter
  edit mode, add/delete rows, modify fields, and save the table. Saving updates
  the attempt visual rows, rebuilds the minimal visual-element record used by
  reference selection, and resets downstream steps/results.
- Scope: `videogen_notebook` project route/store/template/CSS/JS and focused
  regression coverage.
- Verification: Python compilation passed; `node --check` for the frontend JS
  passed; focused web tests for edit/save flow, project rendering, and run-all
  rendering passed. `git diff --check` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-02 - videogen_notebook Diagnostic Step Logs

- Result: Step Details logs now include diagnostic LLM/VLM attempt entries on
  failure, including submitted prompt, raw response, parse error, and failure
  point. Long log messages render as scrollable preformatted text.
- Scope: `videogen_notebook` runner terminal error logging, log filtering,
  project template/CSS, and focused regression tests.
- Verification: Python compilation passed; focused unittest coverage for visual
  plan failure logs and status placement passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-02 - videogen_notebook Failure Details Cleanup

- Result: Moved shot status/actions above Step 1, rendered failed step bodies as
  explicit error panels instead of placeholders, and preserved LLM/VLM failure
  context so Details show prompts, raw responses, parse errors, and readable
  Chinese text.
- Scope: `videogen_notebook` project template/CSS/runner failure persistence,
  Visual Element Memory exception payloads, and focused web regression tests.
- Verification: Python compilation passed; focused unittest coverage for status
  placement, visual-plan failure context rendering, and VLM attempt recording
  passed. `git diff --check` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU work was run.

## 2026-07-02 - videogen_notebook Staged Attempt Details

- Result: Replaced the single generic shot `Attempt Details` block with three
  default-collapsed staged panels: collection maintenance, Seedance generation,
  and keyframe/VLM postprocess. Panels show filtered logs, request/response
  summaries, submitted prompts, errors, visual rows, selected references, and
  produced-memory rows.
- Scope: `videogen_notebook` template/CSS/app filters/project-store hydration,
  real runner attempt snapshots, and Visual Element Memory attempt recording.
  Real visual runs now persist per-attempt `visual_element_details.json`; VLM
  annotations record prompt/raw response/parse errors without storing base64
  image payloads in metadata.
- Verification: `python -m compileall -q videogen_notebook storymem_seedance
  tests/test_videogen_notebook.py tests/test_visual_element_memory.py` and
  `python -m unittest -q tests.test_visual_element_memory
  tests.test_videogen_notebook` passed.
- Safety: No paid Seedance request, real LLM/VLM call, or GPU keyframe
  extraction was run.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Postprocess Boundary

- Result: Split Seedance video submission/download from produced-memory
  postprocessing in `videogen_notebook`; real runner now accepts an injected
  `VideoPostprocessor`.
- Scope: Added a placeholder postprocessor for dry-run/default real paths and a
  non-default `RealKeyframePostprocessor` boundary for later keyframe/VLM
  integration.
- Verification: Python compilation and the focused `tests.test_videogen_notebook`
  suite passed with injected fake Seedance client and fake postprocessor.
- Safety: No paid Seedance request, network upload, VLM call, or GPU keyframe
  extraction was run.

## 2026-07-01 - videogen_notebook Visual Planner Boundary

- Result: Added an injectable `VisualMemoryPlanner` boundary for the real
  runner. The default planner preserves the current dry-run placeholder visual
  element rows and current-lineage reference selection.
- Scope: Planner output now controls visual-element status, selected
  references, optional full prompt override, and planner logs before Seedance
  submission.
- Verification: Python compilation and `tests.test_videogen_notebook` passed
  with a fake planner exercising prompt override and UI table payloads.
- Safety: No paid Seedance request, network upload, LLM/VLM call, or GPU work
  was run.

## 2026-07-01 - videogen_notebook Memory State Snapshot

- Result: Completed shots now refresh project-local `memory/visual_state.json`
  and `memory/lineage.json` with the current completed prefix, element rows,
  selected references, produced memory records, and raw output video ids.
- Scope: Snapshot data is derived from immutable current attempts inside the
  project bundle; reset/fork refreshes truncate the snapshot to the retained
  completed prefix while preserving reset/fork markers.
- Verification: Python compilation and `tests.test_videogen_notebook` passed,
  including run-all and reset snapshot assertions.
- Safety: No paid Seedance request, network upload, LLM/VLM call, or GPU work
  was run.

## 2026-07-01 - videogen_notebook Real Runner Completion

- Result: Completed the `real` runner path behind explicit submit opt-in. It
  now wires the notebook runner to Visual Element Memory planning, precise
  selected references, Seedance submission/download, Smooth reference-video
  publication, original `last_frame_url` continuity inputs, keyframe extraction,
  VLM annotation, produced-memory records, and URL-redacted request/response
  persistence.
- Scope: Default server startup now selects `real` runner, while actual paid
  submission remains gated by `VIDEOGEN_NOTEBOOK_REAL_SUBMIT=1`; dry-run remains
  safe for normal UI startup and tests.
- Verification: Python compilation, focused `tests.test_videogen_notebook`, and
  `git diff --check` passed with injected fake Seedance/planner/postprocessor
  paths.
- Safety: No paid Seedance request, Smooth upload, real LLM/VLM call, or GPU
  keyframe extraction was run during verification.

## 2026-07-01 - videogen_notebook Final Prefix Assembly

- Result: Completed shots now create project-local final prefix video assets
  such as `vid_final_0003`, with `project.json.current_final_video_asset_id`
  pointing to the current completed prefix rather than the latest raw shot.
- Scope: Valid non-Smooth videos use FFmpeg concat, Smooth prefixes use the
  existing RIFE assembler, and fake/dry-run placeholder media write a traceable
  placeholder final asset for low-cost UI tests.
- Verification: Python compilation, focused `tests.test_videogen_notebook`, and
  `git diff --check` passed. Tests cover run-all and reset/fork prefix asset
  semantics with placeholder media.
- Safety: No paid Seedance request, RIFE/GPU assembly, or real media concat was
  run during verification.

## 2026-07-01 - videogen_notebook Background Runs

- Result: Web Run Shot and Run All now start a lightweight background job and
  immediately redirect, so long Seedance/LLM/VLM waits no longer block the page
  request. Stop sets the runner cancel event and marks active shots/project as
  interrupted.
- Scope: Added a single-worker background job manager, wired route handlers to
  it, passed cancellation into Seedance `wait_task`, and made project reads
  tolerate the short running window before `attempt.json` exists.
- Verification: Python compilation, focused `tests.test_videogen_notebook`, and
  `git diff --check` passed with route tests polling project status.
- Safety: No paid Seedance request, real network upload, LLM/VLM call, or GPU
  work was run.

## 2026-07-01 - videogen_notebook Media Visualization

- Result: The project page now renders image thumbnails for selected references
  and produced visual memory, bbox rows from VLM annotations, playable raw/final
  video previews, assembly logs, and active failed/interrupted/running shot
  summaries.
- Scope: UI/template/CSS and bundle-loading only; no memory selection,
  generation, or postprocessing behavior changed.
- Verification: Python compilation, focused `tests.test_videogen_notebook`, and
  `git diff --check` passed with route assertions for thumbnail/video/assembly
  UI markers.
- Safety: No paid Seedance request, real network upload, LLM/VLM call, or GPU
  work was run.

## 2026-07-01 - videogen_notebook Shot Controls

- Result: Added read-only Seedance model/resolution/ratio settings to the
  project page and exposed a per-shot `Stop Shot` action while a shot is
  running.
- Scope: UI/template/CSS only; stop still uses the existing project cancel path
  because the first version runs one active shot at a time.
- Verification: Python compilation and focused `tests.test_videogen_notebook`
  passed with template assertions for Seedance settings and running-shot stop.

## 2026-07-01 - videogen_notebook Project Thumbnails

- Result: Produced visual-memory images now update `project.thumbnail_asset_id`,
  and the workspace project list renders fixed-size thumbnails plus active-shot
  hints for running projects.
- Scope: Project metadata, homepage template/CSS, and focused assertions only;
  no generation, memory selection, or asset-copy behavior changed.
- Verification: Python compilation and focused `tests.test_videogen_notebook`
  passed with assertions for thumbnail persistence and homepage rendering.

## 2026-07-01 - videogen_notebook Running Duplicate Cleanup

- Result: Duplicating a running project now converts the copied project to
  `interrupted`, clears `active_shot_id`, and converts copied running shots to
  interrupted with their transient current attempt archived.
- Scope: Project bundle copy semantics only; the original running project and
  its worker are untouched.
- Verification: Python compilation and focused `tests.test_videogen_notebook`
  passed with a running-project duplicate regression test.

## 2026-07-01 - videogen_notebook Startup Recovery

- Result: App startup now recovers orphan `running` projects from a previous
  server process by marking the project and running shots as `interrupted`,
  clearing active/current attempt pointers, and archiving transient attempt ids.
- Scope: Startup/project-store recovery only; no attempt files are deleted and
  no active worker in the current process is affected.
- Verification: Python compilation and focused `tests.test_videogen_notebook`
  passed with store-level and app-startup recovery tests.

## 2026-06-25 - Keyframe JSON Profiles

- Result: Moved the representative-keyframe constants into
  `keyframe_settings/default.json`, added named `strict` and `loose` presets,
  and made `save_keyframes()` load `default` automatically.
- Scope: Added a lightweight JSON preset loader, optional profile/path
  overrides for `save_keyframes()` and `extract_shot_memory()`, plus docs.
- Verification: Focused loader tests, Python compilation, and `git diff --check`
  passed. No GPU keyframe extraction or paid Seedance request was run.
- Commit: Keyframe JSON profile presets (see Git history).

## 2026-06-25 - Final Video Audio Preservation

- Result: Smooth final assembly now preserves concatenated clip audio when the
  source videos contain audio, instead of always writing a silent final video.
- Scope: Added post-assembly audio muxing for Smooth outputs, a focused unit
  test, and a temporary script to add audio back onto an existing final video.
- Verification: `tests.test_smooth_transition`, Python compilation, and manual
  audio verification on the latest completed `锣老师别这样：咖啡店杯型风波`
  project passed.
- Commit: Smooth final audio preservation (see Git history).

## 2026-06-24 - Shot Memory Source Controls

- Result: Added editable per-Shot Sink, Retrieve, and Recent switches with new
  defaults on/on/off, Save/Save all support, persistence, and Attempt snapshots.
- Compatibility: Existing databases migrate to Cut on/on/on and non-Cut
  on/off/on. Retrieve now runs for Cut and non-Cut Shots when enabled.
- Selection: Existing role metadata drives filtering, budget allocation,
  deduplication, ordering, and submitted-memory counts; continuity modes remain
  independent.
- Verification: Focused repository, runner, runtime, selector, template, JS,
  compile, and diff checks passed. No paid Seedance request was made.
- Issue: Representative-keyframe coupling to Recent and candidate admission is
  intentionally deferred.
- Commit: Shot memory controls implementation (see Git history).

## 2026-06-24 - Smooth Default For Continuous Shots

- Result: Newly created or imported non-Cut shots now initialize with Smooth;
  first shots and Cut shots continue to initialize with Default.
- Compatibility: Existing projects and explicit user selections are unchanged.
- Verification: Focused repository and runner tests passed.
- Commit: Smooth-default change (see Git history).

## 2026-06-24 - Smooth Preconfiguration

- Result: Smooth can now be selected on imported draft shots before their
  predecessors run, allowing complete scripts to be configured before Run all.
- Safety: Execution still requires the preceding raw Attempt to be completed.
- Verification: Focused repository/runner tests and template rendering passed.
- Commit: Smooth preconfiguration fix (see Git history).

## 2026-06-24 - Smooth Continuity Mode

- Result: Added non-cut `smooth` generation using the previous raw video's final
  second as `reference_video`, followed by existing selected memory images, plus
  explicit endpoint-motion continuation prompting.
- Assembly: Raw Attempt videos remain unchanged. Project output replaces
  `A[-1]/B[0]` with Practical-RIFE 4.25 frames at `t=1/3,2/3`; transition
  metadata is hash-keyed, supports multiple boundaries, and is retryable locally.
- Verification: 28 focused pipeline/repository tests and 5 synthetic-video
  tests passed. Cached RIFE weights produced two 1280x720 frames; full assembly
  preserved 386 frames at 24 fps and 1280x720.
- Issue: The engineering plan's tail-video-only input was superseded by the
  user's instruction to retain Default's selected memory images. No paid
  Seedance call or real public upload was performed.
- Commit: Smooth mode implementation commit (see Git history).

## 2026-06-23 - Save All Shot Edits

- Result: Added a `Save all` action before `Run all` that persists every dirty
  Shot through the existing PATCH API and refreshes once after completion.
- Scope: Frontend template and JavaScript only; no database or runner changes.
- Verification: JavaScript syntax and direct template rendering passed.
- Commit: `1a3fdfc`

## 2026-06-23 - Per-Shot Generation Duration

- Result: Added editable `duration_seconds` per Shot, defaulting to 8 seconds,
  with a 2-15 second range and Seedance request propagation.
- Compatibility: Existing projects migrated from their project-level duration;
  Attempt snapshots preserve the submitted value.
- Verification: 33 focused backend tests, template rendering, JavaScript syntax,
  Python compilation, and diff checks passed.
- Issue: The full Web TestClient suite stalled because of the existing
  Starlette/httpx compatibility problem; direct template checks were used.
- Commit: `57c596a`

## 2026-06-23 - Experimental Video-Extension Seam Tools

- Result: Added SSIM edge-frame matching, retained-input extension prompting,
  and sequence-level overlap alignment for Little Prince continuity studies.
- Scope: Standalone experiment tools and research plan only; production Web
  generation behavior is unchanged.
- Verification: Python compilation, FFmpeg smoke outputs using existing videos,
  and diff checks passed. A retained-input Seedance task was submitted and left
  running remotely for later resume.
- Commit: `4975c67`

## 2026-06-23 - Retained-Input Extension Analysis

- Result: Completed the retained-input Little Prince experiment and added
  same-index 24-frame retention scoring plus a localized 12-by-12 SSIM splice
  search around the expected one-second boundary.
- Verification: The paid task completed with 303,300 tokens; analysis produced
  a matrix report, frame-pair preview, and playable 24 fps concatenation.
- Finding: The generated prefix was re-rendered rather than copied exactly
  (mean same-index SSIM 0.736665); the best requested-window seam was shot 1
  frame 192 to generated frame 18.
- Commit: `020f370`
- Follow-up: A 12-by-24 search selected generated frame 12 at the search
  boundary; dimension-specific artifact names preserve both comparisons.
  Commit: `08fe52b`.

## 2026-06-23 - Delta-Consistent Seam Experiment

- Result: Added a 12-by-12 seam scorer combining normalized appearance,
  previous-to-cross motion, and cross-to-next motion with 0.4/0.3/0.3 weights.
  Reports preserve raw camera/local flow components and every weighted term.
- Verification: OpenCV DIS/RANSAC analysis completed over 144 candidates and
  produced a playable 24 fps concatenation, CSV, JSON report, and frame preview.
- Finding: The combined score selected shot 1 frame 192 to shot 2 frame 2,
  matching the earlier SSIM-only choice while adding motion evidence.
- Commit: `cc19f66`

## 2026-06-24 - FFmpeg Seam Interpolation Experiment

- Result: Replaced `A192/B0/B1` with three FFmpeg motion-compensated frames
  between anchors `A191/B2`, preserving the original 386-frame duration.
- Scope: Standalone experiment script and ignored runtime outputs only;
  production generation and Web behavior are unchanged.
- Verification: Output is 1280x720, 24 fps, 386 frames, and audio-free. The
  sampled seam strip has mean SSIM 0.927012 and no obvious contact-sheet
  ghosting. RIFE comparison was deferred after an initial padding mismatch.
- Commit: `c2adb8f`

## 2026-06-24 - RIFE Seam Interpolation Experiment

- Result: Deployed Practical-RIFE 4.25 in ignored runtime storage and generated
  three constant-duration seam frames between `A191/B2` on GPU 0.
- Compatibility: 720p anchors are padded to 768 pixels for inference and
  cropped back afterward; no shared Python environment was modified.
- Verification: Output is 1280x720, 24 fps, 386 frames, and audio-free. RIFE
  inference took 0.366 seconds; sampled mean SSIM is 0.883113 and mean DIS-flow
  magnitude is 1.810760.
- Commit: `69bee0b`

## 2026-06-24 - Memory Retrieval Overview Slide

- Result: Added a one-slide visual overview of historical keyframe admission,
  cut-triggered two-stage retrieval, deduplication, temporal ordering, and final
  sink/retrieved/recent memory composition using real project keyframes.
- Verification: PPTX structure opens correctly, contains one 16:9 slide with no
  out-of-bounds shapes, and passed LibreOffice PDF/PNG visual inspection.
- Commit: `109cc6e`

## 2026-06-26 - Visual Element Memory Text and Keyframe Experiment

- Result: Extended the standalone visual-element experiment to build the
  prompt-derived registry, annotate each current Attempt keyframe once with
  Seed2.1 Turbo, reclassify historical frames from stored annotations only, and
  render current-annotation and historical-evaluation bounding-box images.
- Scope: Experimental script and research documentation only; production Web
  retrieval and generation behavior are unchanged.
- Verification: Python compilation, current-Attempt discovery (19 Little
  Prince keyframes), closed-set bbox parsing, full-image scene-box checks, and
  a completed 12-Shot / 19-frame paid experiment passed. One VLM response was
  retried for formatting; all final annotations passed validation. Offline log
  rendering produced 19 current annotation images and 131 history evaluation
  images with Chinese labels and role-colored boxes.
- Commit: recorded in Git history with this task.

## 2026-06-27 - Visual Element Memory Methodology

- Result: Added a Chinese methodology document that consolidates the recent
  visual-element registry, element state management, closed-set VLM annotation,
  historical-frame scoring, greedy coverage selection, and precise reference
  prompting design. Revised the selector formulation to a unified three-term
  score with optional greedy coverage via uncovered/covered reference weights,
  and expanded the final Seedance prompt into a five-part structure with full
  script context, current-shot task, shot-level element plan, per-image
  references, and global constraints.
- Scope: Research documentation only; no runtime pipeline behavior changed.
- Verification: Markdown content review and `git diff --check` passed.

## 2026-06-28 - Visual Element Memory CLI Pipeline

- Result: Added an opt-in CLI path that maintains a visual-element registry,
  selects historical keyframes with static top-k or greedy coverage scoring,
  writes per-reference colored-box visualizations to `memory_report.md`, and
  annotates generated keyframes for later shots. Follow-up: visual-element
  prompts now include the stable-shot-scale / no-unrequested-ending-transition
  constraint, and `memory_report.md` records the full submitted Seedance prompt
  per Shot.
- Scope: Seedance CLI only; Web UI behavior is unchanged.
- Verification: Python compilation, focused executor tests, CLI help, markdown
  diff check, local Hugging Face cache resolution, and a live Little Prince
  visual-element run startup passed. The live run is continuing in the
  background.

## 2026-06-28 - Smooth Mode For Visual Element CLI

- Result: Added opt-in CLI `--smooth_non_cut` support. Non-cut shots now submit
  the previous raw video's final tail as a transient `reference_video`, keep
  visual-element memory images afterward, and use the existing RIFE Smooth
  assembler for incremental/final output when any smooth boundary exists.
  Follow-up: Smooth prompts now explicitly tell Seedance to extend Video 1 as
  the previous-shot tail prefix for motion continuity and reserve Image 1/Image
  2 numbering for static visual-memory references.
- Scope: Seedance CLI only; Web Smooth behavior is unchanged.
- Verification: Python compilation, focused executor and smooth-transition
  tests, CLI help, and `git diff --check` passed. Live experiment restart uses
  `STORYMEM_REFERENCE_VIDEO_PUBLISHER=tmpfiles`.

## 2026-06-28 - Dog Anime Script Experiment

- Result: Added `story/dog_anime.json` by adapting the dog script with explicit
  anime/cartoon-style and non-realistic generation constraints. Started a live
  Seedance CLI experiment with visual-element memory, greedy coverage reference
  selection, and Smooth mode for non-cut shots.
- Scope: Added story script only; pipeline code is unchanged.
- Verification: JSON validation passed; the live run created the first
  Seedance task successfully.

## 2026-06-28 - Stronger Smooth Extension Prompt

- Result: Verified that CLI Smooth uses the same `reference_video` content shape
  as the earlier local Seedance extension experiment. Strengthened Smooth prompts
  to describe the task as extending Video 1 forward from its final frame, added
  the same instruction inside the current-shot section, and recorded a sanitized
  input-media summary in task logs for future request inspection.
- Scope: Prompting, CLI diagnostics, methodology docs, and focused tests.
- Verification: Python compilation, focused project-runner / executor /
  smooth-transition tests, and `git diff --check` passed.

## 2026-06-28 - CLI Per-Shot Durations

- Result: Added CLI support for story-level `durations` arrays. When a shot has
  a duration in the script, Seedance submission uses that value; otherwise it
  falls back to the global `--duration` argument. Memory reports and task/shot
  records now include the effective shot duration.
- Scope: Seedance CLI story parsing, submission, reporting, and focused tests.
- Verification: Python compilation, executor tests, `git diff --check`, and
  direct parsing of `story/luo_teacher_coffee.json` durations passed.

## 2026-06-28 - Weekly Research Report

- Result: Added a concise Chinese weekly report summarizing the continuity
  experiments, provisional Smooth/RIFE solution, Visual Element Memory selector
  design, and next-week research plan.
- Scope: Documentation only.
- Verification: Markdown content review and `git diff --check` passed.

## 2026-06-30 - Visual Element Web UI Plan

- Result: Added a Web implementation plan for project-level
  `visual_element_v1`, including the revised Shot layout, three expandable
  evidence lists, attempt artifact layout, runner integration, state/versioning
  semantics, and focused test plan. Follow-up: added global visual-memory
  settings for optional sink frame retention and maximum visual-element
  retrieved historical frames. Follow-up: implemented the matching CLI
  visual-element sink options, with element-retrieved frames first and sink
  anchors appended last.
- Scope: Documentation and Seedance CLI visual-element memory only; no Web
  runtime behavior changed.
- Verification: Focused visual-element unit test, CLI help check, Python
  compilation, and `git diff --check` passed.

## 2026-06-30 - Visual Element Web UI Runtime

- Result: Implemented project-level `visual_element_v1` as the default for new
  and imported Web projects while migrating existing projects to `classic`.
  Added project visual-memory settings, visual-element runner integration,
  Attempt-frozen visual decisions/references/annotations, and the new Shot
  runtime layout with Visual Element Status, Selected Historical References,
  and Produced Visual Memory sections.
- Scope: Web repository/settings/routes/views/templates/static assets,
  ProjectRunner visual-element orchestration, shared visual-memory snapshot
  support, focused tests, and affected docs.
- Verification: Python compilation, Web repository/app tests, and a fake
  visual-memory ProjectRunner test passed. No paid Seedance call or real LLM/VLM
  call was executed.
- Commit: this task commit

## 2026-06-30 - Visual Element Web UI Tables And Sink Default

- Result: Changed new Web project visual-element sink frame default from 3 to
  0. Reworked Visual Element Status, Selected Historical References, and
  Produced Visual Memory from list/card-like markup into real table layouts
  with stable headers, preview columns, and horizontal overflow.
- Scope: Web defaults, visual-element runtime template/CSS, focused rendering
  tests, and affected docs.
- Verification: Web repository/app tests, Python compilation, and task-owned
  `git diff --check` passed.
- Commit: this task commit

## 2026-06-30 - Visual Reference Table Missing Score Fix

- Result: Fixed an Internal Server Error on running visual-element projects
  whose selected historical references include sink anchors without `score` or
  `visual_element_selection` fields.
- Scope: Visual-element runtime template and focused Web rendering regression
  test.
- Verification: Focused Web app test, direct TestClient request for the
  affected running project, Python compilation, and task-owned `git diff
  --check` passed.
- Commit: this task commit

## 2026-06-30 - Compact Visual Element Tables

- Result: Reduced visual table image previews to fixed thumbnails, split the
  Visual Element Status element column into Name / ID / Type columns, and
  strengthened vertical table separators.
- Scope: Visual-element runtime template, CSS, and focused rendering test.
- Verification: Focused Web app table-rendering test, direct TestClient request
  for the active project, and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Phase 0/1 Scaffold

- Result: Added the first lightweight `videogen_notebook` scaffold with a
  self-contained project bundle store, workspace project scanning, empty
  project creation, JSON story import, project page rendering, Save all,
  Duplicate, Delete, and `Fork after this shot` semantics. Fork preserves the
  selected prefix and resets later shots to draft while retaining their saved
  imported inputs and archiving old attempts by default.
- Scope: New `videogen_notebook` package, startup wrapper, focused tests, and
  updated design documentation. No old Web UI behavior or real generation
  pipeline was changed.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, `python -m videogen_notebook --help`, and
  task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Fake Runner

- Result: Added a low-cost fake notebook runner for `videogen_notebook` with
  Run Shot, Run All, Stop, a workspace-level global run lock, immutable attempt
  folders, fake visual-element status, fake historical references, fake
  produced visual memory, asset manifest updates, and project completed-prefix
  updates. The project page now renders generated attempt tables and fake output
  asset IDs. Save all now permits unchanged completed shots while still
  rejecting edits to locked shots.
- Scope: New runner/lock modules plus route, store, template, CSS, and focused
  tests. No real Seedance, LLM, VLM, keyframe extraction, or GPU work was run.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Reset And Assets

- Result: Added `Reset from this shot` semantics and route for
  `videogen_notebook`. Reset archives the current attempts for the selected
  shot and downstream shots, preserves their saved inputs, returns them to
  draft, and rolls the completed prefix/final asset back to the previous shot.
  Added manifest-backed project asset serving and linked fake output assets from
  the notebook UI.
- Scope: Project store reset/asset helpers, FastAPI asset/reset routes,
  notebook template links/actions, fake assembly metadata, and focused tests.
  No real generation, LLM/VLM, or GPU work was run.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Runner Backend Boundary

- Result: Split `videogen_notebook` execution into a reusable `NotebookRunner`
  orchestration layer and an injectable runner backend. The existing fake
  behavior now lives in `FakeExecutionBackend`, while the runner owns locking,
  shot state transitions, immutable attempt writes, asset manifest updates, and
  completed-prefix updates. Attempts now persist backend metadata and
  `logs.jsonl`, and the notebook UI shows attempt details/logs.
- Scope: Runner abstraction, JSONL helpers, attempt loading, notebook template,
  and focused tests. No real Seedance, LLM/VLM, keyframe extraction, or GPU work
  was run.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Backend Selection Scaffold

- Result: Added runner backend selection for `videogen_notebook` through
  `VIDEOGEN_NOTEBOOK_RUNNER` and `python -m videogen_notebook start --runner`.
  The default remains the fake backend. A non-submitting `real` backend scaffold
  now records a failed immutable Attempt with backend metadata, error details,
  request/response placeholders, and logs instead of leaving the Shot running.
  The notebook UI displays failed Attempt errors in Attempt Details.
- Scope: Backend factory, CLI/app runner selection, failed-attempt persistence,
  error display, and focused tests. No real Seedance, LLM/VLM, keyframe
  extraction, or GPU work was run.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, `python -m videogen_notebook start --help`,
  and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Real Backend Dry Run

- Result: Replaced the `real` backend placeholder failure with a non-submitting
  dry-run path. The backend now builds a Seedance-shaped request from the saved
  Shot inputs and project settings, composes the full prompt through the shared
  prompt helper, selects deterministic current-lineage reference placeholders,
  registers dry-run output/keyframe assets, and writes a completed immutable
  Attempt marked with `dry_run` metadata. No external task is submitted.
- Scope: `videogen_notebook` runner backend implementation and focused tests.
  No real Seedance, LLM/VLM, keyframe extraction, or GPU work was run.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Injectable Seedance Path

- Result: Extended the `real` backend with an explicit opt-in submit path while
  keeping dry-run as the default. `VIDEOGEN_NOTEBOOK_REAL_SUBMIT=1` or
  `create_runner(..., real_submit=True)` enables submission. The submit path
  accepts an injected Seedance-like client, converts project asset references
  into Seedance image content, creates/waits/downloads the task, registers the
  downloaded video asset, and keeps keyframe/VLM extraction as a placeholder.
- Scope: `videogen_notebook` runner backend and focused fake-client tests. No
  real Seedance, LLM/VLM, keyframe extraction, or GPU work was run.
- Verification: `python -m compileall -q videogen_notebook
  tests/test_videogen_notebook.py`, `python -m unittest -q
  tests.test_videogen_notebook`, and task-owned `git diff --check` passed.
- Commit: this task commit


## 2026-07-01 - videogen_notebook Real Runner Smoke Validation

- Result: Ran the new `videogen_notebook` real runner end to end on a one-shot
  low-risk smoke project. The flow completed real visual-element LLM planning,
  real Seedance task submission/download, local keyframe extraction, real VLM
  annotation, produced visual memory registration, and final prefix assembly.
  The live project was `p_20260701_131943_dddd14ca` in
  `.runtime/videogen_notebook_live`; Seedance task id was
  `cgt-20260702011949-vv6nb`.
- Scope: Fixed the new notebook default Seedance model to reuse
  `seedance_client.DEFAULT_MODEL` instead of the unavailable pro endpoint, and
  set HuggingFace/Transformers offline mode in `storymem_env.sh` so cached
  HPS/Qwen models are used instead of silently downloading.
- Verification: `python -m compileall -q videogen_notebook`,
  `python -m unittest -q tests.test_videogen_notebook`, the paid/live smoke
  run above, and task-owned `git diff --check` passed.
- Issue: The first live submit attempt failed before generation because
  `doubao-seedance-2-0-pro` is not available for the configured account; the
  default now tracks the known working client default `doubao-seedance-2-0-260128`.
- Commit: this task commit


## 2026-07-01 - videogen_notebook Real Submit Default

- Result: Changed the `videogen_notebook` real backend so real paid submission is
  the default. `VIDEOGEN_NOTEBOOK_REAL_SUBMIT=0` now explicitly selects dry-run
  mode for debugging/tests. The CLI help and design doc were updated to match.
- Scope: Runner backend selection, CLI help text, focused tests, and docs.
- Verification: `python -m compileall -q videogen_notebook`,
  `python -m unittest -q tests.test_videogen_notebook`, and task-owned
  `git diff --check` passed. Submit-default coverage uses an injected fake
  Seedance client and does not issue a real paid request.
- Commit: this task commit

## 2026-07-01 - videogen_notebook Restart Command

- Result: Extended `run_videogen_notebook.sh` with simple `start`, `stop`,
  `restart`, `status`, and `logs` actions. The common restart command is now
  `./run_videogen_notebook.sh restart`; it loads `storymem_env.sh`, uses port
  `7870`, starts the real runner by default, writes a PID file, and logs to
  `.runtime/videogen_notebook/server.log`.
- Scope: Shell wrapper only.
- Verification: `bash -n run_videogen_notebook.sh`,
  `./run_videogen_notebook.sh restart`, `./run_videogen_notebook.sh status`,
  and task-owned `git diff --check` passed.
- Commit: this task commit

## 2026-07-02 - videogen_notebook StoryMem Theme And Visual Tables

- Result: Restyled `videogen_notebook` to match the existing StoryMem Web visual
  theme: dark green header, compact typography, green status palette, notebook
  shot layout, and StoryMem-style visual data tables. The three visual tables now
  follow the old Web UI structure: Visual Element Status, Selected Historical
  References, and Produced Visual Memory. Selected/produced image previews now
  prefer VLM/selection visualization images and fall back to project assets.
- Scope: New restricted project-file media route for project-local
  visualization images, attempt media URL hydration, project template, CSS, and
  focused tests. No generation, LLM/VLM, or GPU work was run.
- Verification: `python -m compileall -q videogen_notebook`,
  `python -m unittest -q tests.test_videogen_notebook`, a TestClient render
  probe against an existing real project including visualization image fetch,
  and `git diff --check` passed.
- Commit: this task commit

## 2026-07-02 - videogen_notebook Image Lightbox

- Result: Added click-to-preview behavior for visual table thumbnails. Clicking
  a selected-reference or produced-memory thumbnail opens a centered large image
  preview; clicking the backdrop, close button, or pressing Escape closes it.
- Scope: `videogen_notebook` base template, CSS, and a small static JS file.
- Verification: `python -m compileall -q videogen_notebook`,
  `python -m unittest -q tests.test_videogen_notebook`, `node --check
  videogen_notebook/static/app.js`, a TestClient static/page probe, and
  `git diff --check` passed.
- Commit: this task commit

## 2026-07-02 - Keyframe Candidate Defaults

- Result: Changed generated-video keyframe extraction defaults so candidate
  frames are filtered only against the previous selected candidate frame inside
  the same shot. Historical-memory deduplication is now controlled by the
  `compare_with_history` keyframe setting and defaults to off. The default
  keyframe profile is now `loose`, with at most 5 keyframes, HPSv3 quality
  threshold 2.5, and frame similarity threshold 0.95.
- Scope: Keyframe settings loader/profiles, `save_keyframes()` history-dedup
  option, Web default keyframe profile, focused tests, and docs.
- Verification: `python -m compileall -q extract_keyframes.py
  keyframe_settings videogen_notebook storymem_seedance`, `python -m unittest
  -q tests.test_keyframe_settings tests.test_videogen_notebook`, and
  `git diff --check` passed. No GPU keyframe extraction or paid Seedance request
  was run.
- Commit: this task commit

## 2026-07-02 - Selected Reference Coverage Names

- Result: Updated the `Selected Historical References` table so the Coverage
  column displays visual element names instead of raw `element-xxxx` ids. It
  uses the selected reference's visual-element metadata and falls back to ids
  only when a name is unavailable.
- Scope: `videogen_notebook` template display helper and focused tests.
- Verification: `python -m compileall -q videogen_notebook`, `python -m
  unittest -q tests.test_videogen_notebook`, a TestClient render probe against
  an existing real project, and `git diff --check` passed.
- Commit: this task commit

## 2026-07-02 - Visual Element Status Metadata Hydration

- Result: Fixed `Visual Element Status` rows for referenced historical elements:
  `Introduced` now resolves to the original introduction shot instead of the
  element id, and `Notes` is backfilled from the element registry when LLM state
  rows omit it. Existing project pages are hydrated on load, and future attempts
  are written with the same metadata.
- Scope: `videogen_notebook` visual status row conversion, project bundle
  hydration, and focused tests.
- Verification: `python -m compileall -q videogen_notebook`, `python -m
  unittest -q tests.test_videogen_notebook`, a live project metadata probe, and
  `git diff --check` passed.
- Commit: this task commit

## 2026-07-02 - videogen_notebook Step Pipeline

- Result: Split the visual-element backend into explicit Visual Elements Plan
  and Historical Reference Selection functions, then rebuilt `videogen_notebook`
  execution around four algorithm steps: visual plan, reference selection,
  Seedance generation, and keyframe maintaining. Shot status now reports the
  active/completed/failed step, Run Step / Run Shot / Run All share the same
  step runner, and project pages expose Add Shot plus exportable Shot Design
  JSON.
- Scope: Visual Element Memory algorithm boundary, CLI call path, new
  self-contained notebook step artifacts, project/shot state management,
  FastAPI routes, background jobs, project template/CSS, tests, and design docs.
  Existing old Web UI files were not intentionally modified.
- Verification: `python -m py_compile videogen_notebook/*.py
  storymem_seedance/*.py`, `git diff --check`, `python -m unittest -q
  tests.test_videogen_notebook tests.test_visual_element_memory` in the default
  shell, and the same focused tests under `source ./storymem_env.sh` / py311
  passed. A real paid smoke project `p_20260702_045343_c561082e` completed:
  Seedance task `cgt-20260702165351-r6txj`, raw asset `vid_0001_a001`, and 3
  produced visual-memory rows.
- Issue: The first smoke run used bare Python 3.13, where `hpsv3` is not
  installed, so keyframe maintaining failed after the paid Seedance video had
  already completed. Re-running only the failed step under `storymem_env.sh`
  py311 reused the existing video and completed without another Seedance
  submission.
- Commit: 8de1e2e

## 2026-07-02 - videogen_notebook Step Detail Separation

- Result: Made the five notebook steps visually distinct with solid step cards
  and dark step headers. Moved Visual Elements Plan details into Step 2 and
  Historical Reference Selection details into Step 3. Added readable
  `ensure_ascii=False` JSON rendering for detailed LLM/VLM records so prompts
  and raw responses are visible in expanded details. Keyframe Maintaining now
  archives and hydrates VLM details separately, and standalone keyframe reruns
  reconstruct the current shot visual registry from Step 2 artifacts.
- Scope: `videogen_notebook` template/CSS, log grouping, attempt detail
  hydration, keyframe detail persistence, Visual Element Memory adapter
  postprocess state reconstruction, and focused tests.
- Verification: `source ./storymem_env.sh && python -m py_compile
  videogen_notebook/app.py videogen_notebook/project_store.py
  videogen_notebook/runner.py`, `source ./storymem_env.sh && python -m unittest
  -q tests.test_videogen_notebook tests.test_visual_element_memory`, `git diff
  --check`, and a TestClient render probe against real smoke project
  `p_20260702_045343_c561082e` confirmed separate details plus visible LLM and
  VLM prompt text.
- Issue: An extra optional Step 5 rerun was interrupted while loading HPSv3 to
  avoid waiting longer; the smoke project metadata was restored to the prior
  completed state using the existing successful artifacts.
- Commit: this task commit

## 2026-07-05 - VBench-Long Manual Evaluation Config

- Result: Configured the existing conda env `vbench` to run VBench-Long on
  arbitrary local videos, keeping downloaded model caches under the user's home
  directory. Scored `/home/wxh/world_model_projects/VBench/luo.mp4` as a smoke
  evaluation. Added wrapper script
  `/home/wxh/world_model_projects/VBench/scripts/evaluate_long_video.sh` so a
  user only needs to pass a video path; output defaults to
  `evaluation_results/<video_stem>` and summary scores are printed to the
  terminal.
- Environment:
  ```bash
  source /home/wxh/miniconda/etc/profile.d/conda.sh
  conda activate vbench
  export PYTHONNOUSERSITE=1
  cd /home/wxh/world_model_projects/VBench
  ```
- Input preparation:
  ```bash
  VIDEO=/abs/path/to/video.mp4
  NAME=my_video
  mkdir -p evaluation_inputs/$NAME evaluation_results/$NAME
  ln -sf "$VIDEO" evaluation_inputs/$NAME/$(basename "$VIDEO")
  ```
- Consistency evaluation:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python vbench2_beta_long/eval_long.py \
    --videos_path evaluation_inputs/$NAME \
    --dimension subject_consistency background_consistency \
    --mode long_custom_input \
    --dev_flag \
    --sb_clip2clip_feat_extractor dino \
    --bg_clip2clip_feat_extractor clip \
    --w_inclip 0.5 \
    --w_clip2clip 0.5 \
    --output_path evaluation_results/$NAME/consistency
  ```
- Quality/motion evaluation:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python vbench2_beta_long/eval_long.py \
    --videos_path evaluation_inputs/$NAME \
    --dimension motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    --mode long_custom_input \
    --dev_flag \
    --output_path evaluation_results/$NAME/quality_motion
  ```
- Result extraction:
  ```bash
  python - <<'PY'
  import json, glob
  for path in sorted(glob.glob('evaluation_results/*/*/*_eval_results.json')):
      print('\n', path)
      data = json.load(open(path, encoding='utf-8'))
      for k, v in data.items():
          print(f'{k}: {v[0]:.6f}')
  PY
  ```
- Cache policy: VBench weights are expected in home caches such as
  `/home/wxh/.cache/vbench` and `/home/wxh/.cache/torch/hub`, not in project
  `.runtime` folders.
- Verification: `luo.mp4` completed 6 VBench-Long metrics on GPU1. Results
  were written to
  `/home/wxh/world_model_projects/VBench/evaluation_results/luo/consistency`
  and
  `/home/wxh/world_model_projects/VBench/evaluation_results/luo/quality_motion`.
- Commit: not committed

## 2026-07-06 - Frame-Accurate Prefix Assembly Fix

- Result: Investigated project `p_20260705_231020_16b45a49` where the final
  assembled video appeared to lose or skip frames even though the last raw
  Seedance clip was normal. Root cause was concat-copying an already assembled
  prefix video with a new raw clip; after prior RIFE/re-encode steps this caused
  fps/timestamp drift in the MP4 container. Future notebook assemblies now use
  frame-accurate re-encoding whenever an assembled prefix participates in the
  next prefix output, and Smooth assembly rounds near-integer fps back to the
  stable nominal fps.
- Scope: `videogen_notebook` assembly routing and
  `storymem_web.smooth_transition` output fps selection. No Seedance calls or
  keyframe extraction were rerun.
- Verification: `python -m py_compile storymem_web/smooth_transition.py
  videogen_notebook/runner.py` and `git diff --check` passed. Rebuilt
  `/home/wxh/world_model_projects/StoryMem/.runtime/videogen_notebook/projects/p_20260705_231020_16b45a49/final/prefix_0011_a001.mp4`;
  the old file was backed up as
  `prefix_0011_a001.mp4.before_frame_accurate_fix`. The repaired final is 1379
  frames at 24 fps / 57.46 s; the broken backup had the same frame count but an
  abnormal 26.69 fps / 51.67 s container interpretation.
- Commit: not committed

## 2026-07-06 - Raw Full-Rebuild Assembly and UI Rerun Button

- Result: Changed `videogen_notebook` current-output assembly to rebuild each
  completed prefix from raw shot videos instead of incrementally reusing the
  previous assembled prefix. Multi-clip outputs now use frame-accurate
  re-encoding, while Smooth transitions still insert the cached or generated
  RIFE seam frames. Added a `Rerun Assembly` button beside `Current output`;
  it queues a background job that rebuilds prefix outputs for all completed
  shots without calling Seedance or rerunning keyframe maintenance.
- Scope: `videogen_notebook` runner assembly routing, background job route,
  project template/CSS, and Smooth output fps stabilization. Removed obsolete
  incremental-prefix helper code to avoid accidentally reintroducing
  concat-copy timestamp drift.
- Verification: Rebuilt project
  `p_20260705_231020_16b45a49` from raw clips. Every prefix now has exact
  expected frame counts and stable 24 fps, including final `prefix_0011_a001`
  at 1379 frames / 57.46 s. `python -m py_compile
  videogen_notebook/app.py videogen_notebook/jobs.py videogen_notebook/runner.py
  storymem_web/smooth_transition.py`, `git diff --check`, and an HTTP page
  probe for the new button passed. The web UI service on port 7870 was restarted
  with the updated code.
- Commit: not committed

## 2026-07-06 - VBench Input Symlink Guard

- Result: Diagnosed VBench `subject_consistency` `ZeroDivisionError` on the
  rebuilt `luo_2.mp4`. The rebuilt video was valid, but
  `evaluation_inputs/luo_2/luo_2.mp4` had become a self-referential symlink
  (`./luo_2.mp4`), so VBench produced empty metadata and divided by zero. Updated
  `/home/wxh/world_model_projects/VBench/scripts/evaluate_long_video.sh` to
  resolve the input video to an absolute real path before creating the
  evaluation symlink, then repaired the broken input symlink. The same wrapper
  now runs the full six-metric VBench-Long suite and writes both
  `summary.json` and `summary.md` under `evaluation_results/<video_stem>`.
- Verification: `load_video` now reads 1379 frames from
  `evaluation_inputs/luo_2/luo_2.mp4`; VBench-Long consistency rerun completed
  with `subject_consistency=0.935663` and
  `background_consistency=0.958371`. The summary writer was validated against
  existing `luo_2` results and produced all six metric rows.
- Commit: not committed

## 2026-07-06 - Character Reference Quality for Visual Memory

- Result: Added first-pass character `reference_quality` support to VLM
  keyframe annotations (`full` / `partial` / `weak`). Character `full` now
  explicitly requires a clear visible face. Historical reference scoring applies
  1.0 / 0.4 / 0.1 quality weights, and only `full` should-reference elements
  enter the covered set used by greedy coverage and Web prompt constraints.
- Scope: Visual Element Memory annotation schema/prompt/parser/scoring, notebook
  selected-reference coverage mapping, produced-memory hydration, Produced
  Visual Memory table display, and focused unit coverage.
- Verification: `python -m compileall -q storymem_seedance/visual_element_memory.py
  videogen_notebook/runner.py videogen_notebook/project_store.py`,
  `python -m unittest -q tests.test_visual_element_memory`, selected
  `tests.test_videogen_notebook` store/app tests, and `git diff --check` passed.
  Full `tests.test_videogen_notebook` was also tried but three existing async
  route tests timed out waiting for background project completion and remained
  at `running`.
- Commit: not committed

## 2026-07-06 - Visual Reference Selection Settings

- Result: Added project-level notebook controls for Visual Element Memory
  reference selection mode (`greedy_coverage` or `static_top_k`) and advanced
  scoring weights. Updated defaults so `exclude_weight=-0.1` and
  `quality_partial_weight=0.2`, while preserving existing type, optional,
  covered-decay, full, and weak defaults. The runner now passes these settings
  into `RunConfig`, and the visual-memory scorer reads all weights from config.
- Scope: Notebook settings schema/validation, save route, project template/CSS,
  RunConfig, VEM scoring, and focused tests.
- Verification: `python -m compileall -q storymem_seedance/visual_element_memory.py
  storymem_seedance/models.py videogen_notebook/domain.py videogen_notebook/app.py
  videogen_notebook/runner.py`, focused `tests.test_visual_element_memory` and
  notebook store/app tests, plus `git diff --check` passed. No paid Seedance,
  real LLM/VLM, or GPU keyframe extraction was run.
- Commit: not committed

## 2026-07-07 - Keyframe Maintaining Retry Defaults

- Result: Raised Visual Element Memory parse-feedback retries from 1 to 2,
  giving VLM JSON/coordinate correction up to three attempts per frame. Added
  two full Keyframe Maintaining step retries around postprocess execution, so
  transient extraction, API, timeout, or parse-exhaustion failures can rerun the
  complete keyframe extraction and VLM annotation step before the shot is marked
  failed. User cancellation is not retried.
- Scope: `RunConfig` defaults, notebook Keyframe Maintaining step execution,
  retry diagnostics, and focused tests.
- Verification: `python -m compileall -q storymem_seedance/models.py
  videogen_notebook/runner.py tests/test_videogen_notebook.py
  tests/test_visual_element_memory.py`, `python -m unittest -q
  tests.test_visual_element_memory
  tests.test_videogen_notebook.VideogenNotebookStoreTests.test_keyframe_maintaining_retries_full_step_twice_before_success`,
  and `git diff --check` passed. No paid Seedance, real LLM/VLM, or GPU
  keyframe extraction was run.
- Commit: not committed

## 2026-07-07 - Notebook Step Elapsed Display

- Result: Added `started_at` tracking for notebook algorithm steps and a compact
  per-step elapsed display inside each Step Details panel, formatted as
  `in Xm Ys` or `in Ys`. Step 1 remains excluded because it has no algorithm
  run state; old shots without `started_at` simply omit the elapsed text.
- Scope: Step state persistence in runner/store paths, project page template,
  elapsed-time formatting filter, and focused tests.
- Verification: `python -m compileall -q videogen_notebook/app.py
  videogen_notebook/runner.py videogen_notebook/project_store.py
  tests/test_videogen_notebook.py`, focused notebook store/app tests, and
  `git diff --check` passed.
- Commit: not committed

## 2026-07-07 - Default Non-Cut Last-Frame Reference Ordering

- Result: Updated Default generation prompt/media handling so Default+cut no
  longer emits the old Image 1 previous-ending-frame instruction, while
  Default+non-cut submits existing memory reference images first and appends the
  previous Seedance last-frame URL as the final reference image. The Seedance
  prompt now explicitly describes that final image as the first-frame
  continuity constraint.
- Scope: Notebook real runner request construction, prompt-mode wrapping,
  generic Seedance prompt composition, and focused fake-client tests.
- Verification: `python -m compileall -q storymem_seedance/prompting.py
  videogen_notebook/runner.py tests/test_videogen_notebook.py`, three focused
  notebook submit/prompt tests, and `git diff --check` passed. No paid Seedance,
  real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-07 - videogen_notebook User Guide

- Result: Added a concise Chinese user guide for the current
  `videogen_notebook` Web UI, covering startup, project management, global
  settings, six-step Shot workflow, editing/reflect hooks, run modes, Current
  output assembly, duplication/fork/export behavior, and common operational
  notes.
- Scope: Documentation only.
- Verification: `git diff --check -- docs/videogen_notebook_user_guide.md
  docs/task_report.md` passed.
- Commit: not committed

## 2026-07-07 - Seedance Auto Duration Selection

- Result: Updated `videogen_notebook` duration handling to match Seedance 2.0
  semantics: default duration is now `-1` (auto), valid manual Shot durations
  are `4-15` seconds, and invalid imported JSON durations normalize to `-1`.
  The Web UI now uses select controls showing `自动` plus `4s-15s`, and the
  reusable CLI defaults/report labels were aligned with auto duration.
- Scope: Notebook domain validation/defaults, save form handling, project
  template/CSS, CLI defaults, memory-report duration display, user guide, and
  focused tests.
- Verification: `python -m compileall -q` on touched Python files, focused
  `tests.test_videogen_notebook` store/submit tests, project-page App tests
  with `-k project`, `git diff --check`, and `./run_videogen_notebook.sh
  restart` passed. No paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-08 - Configurable Algorithm Step Retries

- Result: Added project-level `Step max attempts` setting, defaulting to `3`,
  and applied complete-step retry to Visual Elements Plan, Historical Reference
  Selection, Seedance Video Generation, and Keyframe Maintaining. The setting
  defines total attempts, so retry count is attempts minus one. Retry summaries
  are written into Step logs/details, and final failed attempts still enter the
  existing `step_failed` terminal state.
- Scope: Notebook settings schema/save form/template, runner retry orchestration,
  Keyframe Maintaining retry parameterization, user guide, and focused tests.
- Verification: `python -m compileall -q` on touched notebook files, focused
  retry/store/page tests in `tests.test_videogen_notebook`, and `git diff
  --check` passed. No paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-09 - Seedance Request Debug Preservation

- Result: Updated `videogen_notebook` real Seedance submission paths to persist
  the actual request content before `create_task`, then update it after task
  creation and completion. Reference-video submissions now also write
  `media_debug.json` with the exact submitted URL, redirect/final URL,
  content-type, content-length, first bytes, and a probable-MP4 flag, so
  `Invalid video_url` failures can be diagnosed from Attempt files.
- Scope: Notebook real runner request persistence, URL preservation in request
  summaries/responses, and focused media-debug tests.
- Verification: `python -m compileall -q videogen_notebook/runner.py
  tests/test_videogen_notebook.py`, two focused notebook tests, and `git diff
  --check` passed. No paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-09 - Seedance Media Debug Display

- Result: Added Seedance generation details hydration and a dedicated Step 5
  `Media debug` block so the Web UI shows reference-video URL diagnostics
  separately from the large full request payload.
- Scope: Notebook project loading and project template display.
- Verification: `python -m compileall -q videogen_notebook/project_store.py`,
  `git diff --check`, service restart, and a no-proxy page fetch confirmed the
  latest failed attempt renders `Media debug`.
- Commit: not committed

## 2026-07-09 - Seedance Request Image Display Cleanup

- Result: Replaced image data URLs in Seedance request details with local
  source paths for display, while preserving real HTTP URLs for reference
  videos and Seedance last-frame images. New request summaries no longer store
  embedded image base64 for local reference images.
- Scope: Notebook request summary storage, attempt hydration for legacy request
  display, and a focused fake-submit regression test.
- Verification: compile checks, one focused notebook submit test, `git diff
  --check`, service restart, and a no-proxy page fetch confirmed the project
  page no longer contains `data:image/` in request details.
- Commit: not committed

## 2026-07-09 - Cloudflare Tunnel Reference Video Publisher

- Result: Switched the notebook startup default reference-video publisher from
  `tmpfiles` to `cloudflare_tunnel`, added a reusable Cloudflare Quick Tunnel
  publisher for Smooth reference videos, and added a project-level `Smooth
  reference video` setting defaulting to `2s`. Smooth now extracts the configured
  previous-tail duration, records that duration in reference metadata, and no
  longer describes the reference segment as a fixed 1s clip in current docs.
- Scope: Reference-video publishing helper, notebook settings schema/save
  form/template, Smooth runner configuration, shared CLI default, startup
  script default, current user/architecture docs, and focused tests.
- Verification: `python -m compileall -q` on touched Python files, three
  focused notebook tests, and `git diff --check` passed. A prior manual paid
  Seedance probe confirmed a Cloudflare Tunnel URL can be accepted as
  `reference_video`; no additional paid generation was run for this patch.
- Commit: not committed

## 2026-07-09 - Notebook Seedance Quality Settings

- Result: Added editable global Seedance `Quality` and `Aspect ratio` settings
  to the notebook UI. New projects now default to `720p` and `16:9`, invalid
  stored values normalize back to those defaults, and the real runner submits
  the saved values to Seedance.
- Scope: Notebook settings defaults/validation, save form/template, runner
  fallback, user guide, and focused tests.
- Verification: `python -m compileall -q` on touched notebook files, two
  focused notebook store/config tests, and `git diff --check` passed. No paid
  Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-09 - Notebook Parallel Execution Plan

- Result: Added an implementation plan for allowing multiple
  `videogen_notebook` projects to run concurrently while serializing Step 6
  Keyframe Maintaining behind a single GPU lock/queue.
- Scope: Planning document only; no runtime code was changed.
- Verification: Documentation-only change reviewed against current
  `jobs.py`, runner Step6 flow, and existing project self-contained state
  design. No tests were run.
- Commit: not committed

## 2026-07-09 - Notebook Parallel Execution Runtime

- Result: Implemented multi-project background execution for
  `videogen_notebook` with `VIDEOGEN_NOTEBOOK_MAX_WORKERS` defaulting to `10`.
  Different projects can now run concurrently, while duplicate jobs for the same
  project are rejected. Step 6 Keyframe Maintaining now waits behind a single
  process-local GPU queue and exposes `keyframe_maintaining_queued` while
  waiting; queued/running states are recovered as interrupted on restart.
- Scope: Background job manager, project run lock semantics, Step6 GPU queue
  lock, runner Step6 status/log integration, running-like state handling,
  project store recovery, Web template/CSS queued display, user/design docs, and
  focused tests.
- Verification: `python -m compileall -q` on touched notebook files,
  `python -m unittest -q tests.test_videogen_notebook` (48 tests), and
  `git diff --check` passed. No paid Seedance, real LLM/VLM, or GPU work was
  run.
- Commit: not committed

## 2026-07-09 - Notebook Interrupted Resume Artifact Inheritance

- Result: Fixed `videogen_notebook` resumed runs after service restart or
  interruption so a new Attempt inherits completed prior Step artifacts from
  the latest archived Attempt before continuing. This preserves Visual Plan,
  Reference Selection, composed prompt, and Seedance output state when resuming
  from later steps.
- Scope: Runner Attempt creation/resume path and focused notebook regression
  tests for Seedance-generation and Keyframe-maintaining resume cases.
- Verification: Two new focused resume tests plus existing interrupted recovery
  tests passed; `python -m compileall -q` on touched files and
  `git diff --check -- videogen_notebook/runner.py tests/test_videogen_notebook.py`
  passed. No paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-09 - Notebook Home Batch Import And Run Action

- Result: Updated the `videogen_notebook` home page so Story JSON import accepts
  multiple JSON files and always returns to the project list after import.
  Added a home-page `Run` action for each project that starts the existing Run
  All job.
- Scope: FastAPI import route, home-page template, and focused app tests.
- Verification: Two focused app tests passed; `python -m compileall -q` on
  touched files and `git diff --check -- videogen_notebook/app.py
  videogen_notebook/templates/projects.html tests/test_videogen_notebook.py`
  passed. No paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-09 - Notebook Default Settings And Textarea Newline Save Fix

- Result: Fixed project save false positives caused by browser textarea CRLF
  normalization on locked completed Shots. Added home-page default global
  settings; saved defaults are stored in the notebook workspace and applied only
  to future new/imported projects.
- Scope: Project store settings defaults, project save normalization, home-page
  settings form, FastAPI settings route, and focused regression tests.
- Verification: Three focused notebook tests passed; `python -m compileall -q`
  on touched files and `git diff --check -- videogen_notebook/app.py
  videogen_notebook/project_store.py videogen_notebook/templates/projects.html
  tests/test_videogen_notebook.py` passed. No paid Seedance, real LLM/VLM, or
  GPU work was run.
- Commit: not committed

## 2026-07-10 - Notebook Force Animation Prompt Setting

- Result: Added a `Force animation` generation setting, defaulting on for new
  and imported projects. During Seedance Prompt Composition, enabled projects
  prepend the Chinese animation-style instruction to the assembled prompt.
  Manual edits remain authoritative: if the user deletes the instruction from
  the composed prompt and saves, Seedance Video Generation submits the edited
  text as-is.
- Scope: Notebook settings defaults/validation, project and home settings UI,
  settings form parsing, prompt composition, and focused tests.
- Verification: Three focused notebook tests passed; `python -m compileall -q`
  on touched files and `git diff --check` for touched notebook files passed. No
  paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-10 - Visual Reference Selection Allows Style Anchors

- Result: Updated Visual Element Memory reference selection so each greedy
  round chooses the best available candidate even when its score is zero or
  negative, while preserving the existing stop condition once all required
  elements are covered. Selected references that cover no needed elements now
  receive a fixed `reference_guidance` note telling Seedance to use them as
  visual style references.
- Scope: Core visual element selector, notebook reference guidance postprocess,
  and focused regression tests.
- Verification: Focused tests for negative-score selection and style-reference
  guidance passed; `python -m compileall -q` on touched files and
  `git diff --check` passed. No paid Seedance, real LLM/VLM, or GPU work was
  run.
- Commit: not committed

## 2026-07-10 - Visual Reference Selection No Early Stop

- Result: Removed the Greedy reference selection early-stop condition that
  stopped once all required elements were covered. Greedy and Static Top-K now
  both attempt to select up to `max_retrieved_frames`, while Greedy still
  downweights already covered reference elements.
- Scope: Core visual element selector and focused regression test.
- Verification: Two focused visual element selection tests passed;
  `python -m compileall -q` on touched files and `git diff --check` passed. No
  paid Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-11 - Notebook Prompt Ordering And Worker Limit

- Result: Reordered structured Seedance prompt composition so the current shot
  task appears before the full script context. When Force animation is enabled,
  the animation constraint is inserted in both the current-shot task section and
  the global constraints section. Raised the default `videogen_notebook`
  background worker limit from 10 to 20.
- Scope: Prompt composition, background job default, user/parallelism docs, and
  focused prompt regression test.
- Verification: Focused Force animation prompt test passed; `python -m
  compileall -q` on touched Python files and `git diff --check` passed. No paid
  Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-12 - Conditional Reference Video Prompt Instruction

- Result: Updated structured Seedance prompt composition so the reference-video
  extension instruction is included only when the shot actually uses Smooth
  reference video input. The wording now uses direct phrasing instead of the
  conditional "if/then" sentence.
- Scope: Notebook prompt composition and focused prompt regression tests.
- Verification: Two focused notebook prompt tests passed; `python -m compileall
  -q` on touched Python files and `git diff --check` passed. No paid Seedance,
  real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-12 - Agent And History Documentation Refresh

- Result: Reviewed the main engineering documentation after recent
  `videogen_notebook` changes. Updated `AGENTS.md` so future engineering work
  defaults to the current 7870 `videogen_notebook` UI, filesystem project
  bundles, six-step pipeline, current continuity semantics, worker defaults, and
  focused-test workflow. Updated `docs/history.md` with the current July 2026
  research/implementation baseline.
- Scope: Documentation only.
- Verification: Targeted `rg` checks for stale old-UI/default references and
  `git diff --check` on touched docs passed.
- Commit: not committed

## 2026-07-12 - Step Retry Delay And Default Attempts

- Result: Added a cancel-aware 60 second wait between full Step retry attempts
  for retryable algorithm Steps in the real notebook runner. JSON parse
  correction loops inside LLM/VLM calls are unchanged and do not wait. Raised
  the default `Step max attempts` from 3 to 5 in code and the current runtime
  default settings file.
- Scope: Notebook runner retry flow, default settings, settings form fallback,
  user guide, and focused retry tests.
- Verification: Four focused notebook retry/default tests passed; `python -m
  compileall -q` on touched Python files, targeted stale-default search, and
  `git diff --check` passed. No paid Seedance, real LLM/VLM, or GPU work was
  run.
- Commit: not committed

## 2026-07-12 - Disable Reference Images Ablation Mode

- Result: Added a third Reference selection mode, `Disable reference images`.
  In this mode Step 3 completes with an empty reference list and skips the
  reference-selection backend, while downstream Prompt composition, Seedance
  generation, and Keyframe Maintaining continue normally. Raised the default
  `videogen_notebook` background worker limit from 20 to 30.
- Scope: Notebook settings validation, project/home UI, Step 3 runner behavior,
  worker default, docs, and focused tests.
- Verification: Two focused notebook tests passed; `python -m compileall -q` on
  touched Python files and `git diff --check` passed. No paid Seedance, real
  LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-13 - Seedance Prompt Module Switches

- Result: Added five project/default settings switches for Seedance Prompt
  ablations: full script context, visual element plan, holistic guidance,
  should-reference constraints, and should-exclude constraints. Prompt section
  headings now omit numeric prefixes.
- Scope: Notebook settings validation, project/home settings forms, Seedance
  Prompt composition, runtime default settings, user guide, and focused tests.
- Verification: Three focused notebook prompt tests passed; `python -m
  compileall -q videogen_notebook` and `git diff --check` passed. No paid
  Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-14 - Home Project Date Grouping

- Result: The `videogen_notebook` home page now groups projects by creation
  date in default-open collapsible sections. Project listing order is fixed to
  `created_at` descending instead of `updated_at` descending.
- Scope: Project listing order, home route grouping, home template, CSS, and
  focused tests.
- Verification: Two focused notebook home/listing tests passed; `python -m
  compileall -q videogen_notebook` and `git diff --check` passed. No paid
  Seedance, real LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-14 - Current Web UI Agent Guidance

- Result: Clarified `AGENTS.md` so future engineering work treats
  `videogen_notebook` on port 7870 as the current Web project, and treats
  `storymem_web` on port 7860 as legacy-only. The guide now warns that
  `./storymem status` may show the legacy 7860 Web line and should not drive
  current UI diagnosis.
- Scope: Agent guidance documentation only.
- Verification: `git diff --check` passed. No runtime, paid Seedance, real
  LLM/VLM, or GPU work was run.
- Commit: not committed

## 2026-07-14 - Auto Submit EntityBench Eval

- Result: Added an `Auto Submit Eval` project/default setting for
  `videogen_notebook`. When enabled, a project that reaches `completed` starts
  `tools/submit_videogen_eval_to_a6000.py --project-dir <project>` in a
  background subprocess. Web UI does not read or display A6000 eval status.
- Scope: Settings validation, project/home settings forms, runner completion
  hook, runtime default settings, user docs, A6000 eval workflow docs, and
  focused tests.
- Verification: Three focused notebook tests passed; `python -m compileall -q
  videogen_notebook tools/submit_videogen_eval_to_a6000.py` and `git diff
  --check` passed. No paid Seedance, real LLM/VLM, GPU work, SSH, rsync, or
  real A6000 eval submission was run.
- Commit: not committed

## 2026-07-16 - Shot Predefined Reference Images

- Result: Added Shot-level predefined reference images to `videogen_notebook`.
  Imported JSON references and Web uploads are copied into the project bundle,
  shown in Step 1 with Preview/Label/Guidance controls, saved via Save all, and
  frozen into Attempt snapshots. Seedance Prompt Composition now emits a
  `[预定义参考图说明]` block before historical references, and Seedance static
  image submission uses the same order. Step 3 now reserves static image slots
  for predefined references before applying sink/retrieved frame budgets.
- Scope: Story normalization, project persistence/assets/export, Web routes and
  template/CSS, fake and real runner request assembly, user/design docs, and
  focused tests.
- Verification: Four predefined-reference focused notebook tests passed;
  `python -m py_compile` on touched Python files and `git diff --check` passed.
  No paid Seedance, real LLM/VLM, GPU work, or service restart was run.
- Issue: A broader `tests.test_videogen_notebook` run still has three unrelated
  stale failures around current Force Animation prompt expectations, the old
  prompt section title, and existing reset/completed-prefix display semantics.
- Commit: ce43c43

## 2026-07-16 - MSVBench Videogen JSON Export

- Result: Added an independent MSVBench-to-`videogen_notebook` conversion script
  and exported three 20-story JSON sets: no predefined references, character
  reference images only, and character plus shot reference images. All shots use
  prompt-file lines as `video_prompt`, `cut=true`, and `duration=-1`.
- Scope: Conversion utility and generated JSON experiment inputs under
  `data/msvbench_videogen`.
- Verification: Ran the converter, py-compiled it, normalized all 60 generated
  story JSON files with `videogen_notebook.domain.normalize_story`, verified
  referenced image paths exist, and ran `git diff --check`.
- Commit: 1aaa0c4

## 2026-07-16 - MSVBench Export File Naming

- Result: Updated MSVBench exported story JSON names to include their ablation
  suffixes and set each `story_name` to the matching file stem.
- Scope: MSVBench conversion script and generated JSON experiment inputs.
- Verification: Re-ran the converter, verified all three directories contain
  20 suffix-named JSON files, normalized all generated stories, checked image
  paths, py-compiled the converter, and ran `git diff --check`.
- Commit: f8922af

## 2026-07-16 - Documentation Directory Cleanup

- Result: Reorganized accumulated documentation into current, engineering,
  completed-plan, report, experiment, evaluation, and reference subdirectories.
  Added a docs index and updated AGENTS.md so future work focuses on the
  current videogen_notebook docs instead of archived plans or large references.
- Scope: Documentation layout and navigation only.
- Verification: Checked the docs tree, scanned key references with `rg`, and
  ran `git diff --check`.
- Commit: 7ff6930

## 2026-07-16 - Notebook Table Width Cleanup

- Result: Updated `videogen_notebook` step table styling so visual plan,
  reference selection, predefined reference, produced memory, and attempt log
  tables use fixed 100% layouts with compact proportional columns instead of
  forcing horizontal scrolling.
- Scope: Web template classes for edit-mode tables and shared notebook CSS.
- Verification: Checked for remaining table min-width rules and ran
  `git diff --check` on the touched Web files.
- Commit: 73058ea

## 2026-07-19 - Naive Top-K Reference Selection Ablation

- Result: Added `Naive top-k` as a `videogen_notebook` Reference selection
  mode. It skips visual element planning, selects prior produced keyframes by a
  lightweight prompt/text score, and keeps Step 6 keyframe annotations empty
  while still extracting keyframes into the memory pool.
- Scope: Current Web UI settings, runner step branches, real postprocessor
  naive behavior, focused tests, and current user/design docs.
- Verification: Ran focused `tests.test_videogen_notebook` cases for settings,
  disabled references, naive selection, and naive keyframe maintaining; ran
  `python -m compileall -q videogen_notebook tests/test_videogen_notebook.py`;
  ran `git diff --check` on task-owned files. Full `git diff --check` remains
  blocked by unrelated pre-existing whitespace in paper files.
- Commit: 8c506f4

## 2026-07-20 - Cached CLIP Naive Top-K

- Result: Replaced the initial lightweight text-overlap naive retrieval with
  CLIP text-image scoring. Historical keyframe image embeddings are cached under
  each project bundle's `assets/embeddings/` directory and reused on later
  reference selection runs.
- Scope: `videogen_notebook` naive reference scoring, project asset directory
  initialization, focused tests, and current docs.
- Verification: Ran a minimal notebook runner script with patched CLIP
  embeddings, a cache write/hit script with patched image embedding, py-compiled
  `videogen_notebook` and the focused test file, and ran `git diff --check` on
  task-owned files. The unittest module import path was not used because it
  currently scans the large real runtime workspace during import.
- Commit: 59705b0

## 2026-07-21 - Sponsored Video API Survey

- Result: Added a sanitized reference document summarizing the sponsor-provided
  Wan R2V and Veo 3.1 proxy video generation APIs, their request shapes,
  publicly documented underlying model capabilities, StoryMem fit, and
  recommended smoke-test sequence.
- Scope: Documentation only; no API calls were made and no sponsor AK values
  were copied into docs.
- Verification: Reviewed `new_api.txt`, checked official Alibaba Cloud and
  Google Veo documentation, updated the docs index, and ran diff checks on the
  touched documentation files.
- Commit: 4f9d287

## 2026-07-21 - Wan2.7 R2V Usage Notes

- Result: Added a focused Wan2.7 R2V usage document for StoryMem experiments,
  emphasizing first-frame plus multi-reference-image non-cut generation,
  prompt reference numbering, duration defaults, media budgets, and smoke-test
  priorities.
- Scope: Documentation only; no API calls or runtime code changes.
- Verification: Reviewed the official Alibaba Cloud Wan2.7 R2V user guide and
  API reference, updated the docs index, and ran diff checks on touched docs.
- Commit: e8cfd1e

## 2026-07-21 - Wan2.7 R2V Sponsor Proxy Smoke Script

- Result: Added a local CLI smoke-test script that converts the first three
  shots of `Selected105_Mid_29` into compact Wan R2V requests: short current
  shot prompts, per-image guidance, historical reference images, and previous
  last-frame `first_frame` inputs for non-cut shots.
- Scope: `tools/wan27_r2v_smoke.py` only; runtime smoke manifests are written
  under `.runtime/wan27_r2v_smoke/` and remain untracked.
- Verification: Ran py-compile and dry-run request assembly. Real sponsor proxy
  attempts reached connection-layer SSL/read-timeout failures on both documented
  proxy hosts before any Wan task response was returned.
- Issue: No local DashScope API key or Alibaba workspace id was found, so the
  official asynchronous Wan2.7 endpoint could not be tested from this server.
- Commit: d873c2a

## 2026-07-21 - ViMax Benchmark Videogen Conversion

- Result: Added a converter for `/home/wxh/world_model_projects/ViMax/vimax_benchmark`
  and generated 35 `videogen_notebook`-importable story JSON files under
  `data/vimax_benchmark_videogen/`. Each converted shot uses the original
  video prompt followed by the first-frame description, `cut=true`, and
  `duration=-1`.
- Scope: Conversion tool and generated ViMax-Bench story JSON artifacts only.
- Verification: Ran the converter, py-compiled it, and validated all 35 output
  stories with `videogen_notebook.domain.normalize_story` for 437 total shots.
- Commit: 8c893d8

## 2026-07-21 - ViMax Duration Update

- Result: Updated all generated ViMax-Bench videogen JSON files to use
  `duration=8` for every shot, and changed the converter default so future
  regeneration keeps the same setting.
- Scope: ViMax conversion output and converter default duration only.
- Verification: Py-compiled the converter and validated all 35 generated
  stories with `videogen_notebook.domain.normalize_story` for 437 shots.
- Commit: f602931

## 2026-07-21 - Videogen Notebook Worker Limit Restart

- Result: Restarted the unresponsive 7870 `videogen_notebook` service and
  raised the default background worker limit from 30 to 50.
- Scope: Startup script default plus current operator/user documentation.
- Verification: Confirmed the restarted process listens on port 7870, homepage
  returns HTTP 200 locally, and the process environment contains
  `VIDEOGEN_NOTEBOOK_MAX_WORKERS=50`.
- Commit: ddd28e0

## 2026-07-22 - Previous Script Context Prompt

- Result: Changed Seedance Prompt Composition to render `[前序完整剧本]`
  instead of `[完整剧本上下文]`, and to include only the current Shot plus earlier
  Shots in that section.
- Scope: `videogen_notebook` prompt assembly, focused prompt tests, and current
  user/methodology documentation.
- Verification: Ran py-compile, two focused prompt unit tests, and
  `git diff --check` for task-owned files.
- Commit: ba7e8dd

## 2026-07-22 - StoryMem Web Baseline Defaults

- Result: Reconfigured the 7860 `storymem_web` UI as a StoryMem-style Classic
  baseline with Seedance: default Classic pipeline, Sink+Recent memory on,
  Retrieve off, original `max_memory_size=10` and `fix=3`, plain video-prompt
  submission, and minimal previous-last-frame continuity text for non-cut
  shots.
- Scope: 7860 `storymem_web` defaults, Classic runner prompt/reference-budget
  handling, focused docs, and legacy Web tests.
- Verification: Py-compiled touched legacy modules, ran 42 legacy Web/runner
  unit tests, and ran `git diff --check` for task-owned files.
- Commit: a83145f

## 2026-07-22 - StoryMem Original Keyframe Profile

- Result: Added a dedicated `storymem_original` keyframe extraction profile and
  wired the 7860 Classic baseline to use it, keeping original-style max-3
  keyframe maintenance and historical memory comparison separate from 7870
  visual-element experiments.
- Scope: Keyframe profile data, RunConfig/CLI profile plumbing, 7860
  post-generation memory extraction, documentation, and focused tests.
- Verification: Py-compiled touched modules, ran 46 keyframe/legacy Web/runner
  unit tests, and ran `git diff --check` for task-owned files.
- Commit: 15e3594

## 2026-07-22 - Videogen Notebook StoryMem Memory Mode

- Result: Added a 7870 Reference selection mode named `StoryMem memory` that
  skips visual element planning, selects historical keyframes with original
  Sink+Recent memory rules (`max_memory_size=10`, `fix=3`), and maintains the
  keyframe library with the `storymem_original` profile while skipping VLM
  annotation. Added project/default setting `Default non-cut generation mode`
  for import and Add Shot defaults, with `Smooth` preserved as the default.
- Scope: 7870 domain/settings validation, project import/Add Shot defaults,
  Web settings templates, step runner branches, current docs, AGENTS guide, and
  focused unit tests.
- Verification: Py-compiled touched modules, ran 7 focused
  videogen_notebook tests covering settings, StoryMem memory selection, no-ref
  and naive modes, and StoryMem keyframe maintenance, then ran
  `git diff --check` for task-owned files.
- Commit: b8d28b3

## 2026-07-22 - StoryMem Memory Bank Keyframe Dedup Scope

- Result: Changed 7870 `storymem_memory` Step 6 history similarity comparison
  to use the Shot's selected compact StoryMem memory bank instead of all
  historical keyframes, matching the paper-level memory-bank description more
  closely.
- Scope: `videogen_notebook` keyframe postprocess path for StoryMem memory
  mode, focused tests, AGENTS guide, and current UI/user docs.
- Verification: Py-compiled touched Python files, ran 2 focused
  videogen_notebook tests, and ran `git diff --check` for task-owned files.
- Issue: Full-repo `git diff --check` is still blocked by unrelated existing
  whitespace in `paper/*`.
- Commit: 37d13e0

## 2026-07-22 - HPSv3 Inference Mode Guard

- Result: Wrapped HPSv3 keyframe quality scoring in `torch.inference_mode()`
  and froze the cached HPSv3 model parameters after loading, reducing Step 6
  autograd memory pressure without changing the keyframe scoring algorithm.
- Scope: `extract_keyframes.py` HPSv3 quality model setup/scoring and focused
  unit coverage.
- Verification: Ran py-compile for touched Python files, ran
  `tests.test_extract_keyframes`, and ran `git diff --check` for task-owned
  files.
- Commit: 8fbab54

## 2026-07-22 - Run State For Queued Keyframe Steps

- Result: Fixed `videogen_notebook` run-state reporting so starting each Step
  records the current Step/Shot/Attempt, Step 6 queueing reports
  `keyframe_maintaining_queued`, and GPU lock acquisition reports
  `keyframe_maintaining_running` instead of leaving stale previous-Shot
  `completed/run_all` state.
- Scope: 7870 runner project `run_state` updates and focused queue-state test.
- Verification: Py-compiled touched Python files, ran the focused
  keyframe-GPU-queue unit test, and ran `git diff --check` for task-owned
  files.
- Commit: 8e4f202

## 2026-07-23 - Seedance Sensitive Image Retry Filter

- Result: Added a Step 5 Seedance submission fallback that parses
  `InputImageSensitiveContentDetected` `content[N]` errors, drops only that
  image from the current retry submission, and records the dropped image in the
  Attempt request/debug data without changing prior reference-selection state.
- Scope: 7870 real Seedance submit paths and focused fake-client coverage.
- Verification: Py-compiled touched Python files, ran 2 focused
  videogen_notebook Seedance submit tests, and ran `git diff --check` for
  task-owned files.
- Commit: 5df92af

## 2026-07-23 - Seedance Indexed Image Rejection Filter

- Result: Broadened the Step 5 Seedance image-filter retry path so any
  Seedance error that explicitly identifies a rejected `content[N]` is eligible
  for one-shot image removal when that indexed item is an image, covering
  policy-suffixed image rejection codes.
- Scope: 7870 Seedance submit error parsing and focused fake-client coverage.
- Verification: Py-compiled touched Python files and ran the focused sensitive
  image retry unit test.
- Commit: 4e913a1

## Entry Template

```text
## YYYY-MM-DD - Task Name

- Result:
- Scope:
- Verification:
- Issue: (omit when none)
- Commit:
```
