# StoryMem Agent Guide

## Scope

This server work focuses on the Seedance pipeline and the lightweight
`videogen_notebook` Web UI. The goal is long-video generation from mature
short-video APIs using visual-element memory, retrieval, evaluation, reflection,
and shot-by-shot orchestration. The `storymem_web` UI on port 7860 is maintained
only when explicitly targeted; it is the StoryMem-style Sink+Recent baseline
with a Seedance backend.

## Read Order

1. Read this file.
2. Run `./storymem status` before diagnosing runtime state, but treat its
   `Web` line as legacy `storymem_web`/7860 status only. It does not describe
   the current `videogen_notebook` service.
3. Read only the relevant source module.
4. Use `docs/README.md` only as a document map when you are unsure where a
   specific kind of context lives.
5. Use `docs/current/videogen_notebook_user_guide.md` for current user-facing
   behavior.
6. Use `docs/current/videogen_notebook_design.md` for current UI/data design
   and state semantics.
7. Use `docs/current/seedance_pipeline_architecture.md` for deeper
   Seedance/CLI architecture.
8. Use `docs/current/research_plan.md` for active research directions and
   `docs/current/history.md` for completed milestone context when a plan
   document looks older than the implementation.
9. Read archived plans, weekly reports, papers, PDFs, or large external
   references only when the task specifically needs that historical or external
   context.

After completing an engineering task, append a concise entry to
`docs/engineering/task_report.md`. Record the result, scope, verification,
meaningful issues if any, and commit. Do not paste logs or duplicate the full
Git diff.

## Document Layout

- `docs/current/` contains the small set of current behavior, architecture,
  research, and milestone documents that should guide new work.
- `docs/engineering/` contains task handoff logs and operational workflows.
- `docs/plans/completed/` contains completed or superseded implementation
  plans. Treat these as historical context, not source-of-truth behavior.
- `docs/reports/` contains weekly reports and meeting materials.
- `docs/experiments/` and `docs/evaluation/` contain experiment notes,
  benchmark instructions, metrics, and results.
- `docs/references/` contains external API docs, papers, slides, and large
  reference files. Open them only when needed.

Do not read large result logs or every JSON artifact by default. For current
work, prefer `./run_videogen_notebook.sh status`, project-level
`.runtime/videogen_notebook/projects/<project_id>/project.json`, and the latest
Attempt files for one Shot. For the legacy 7860 UI, SQLite state may still be
relevant.

## Collaboration Workflow

- Keep research, architecture discussion, decisions, and milestone planning in
  the long-lived research thread. Maintain the relevant decision and plan docs
  there as the source of truth for what to build and why. Summarize completed
  milestones in `docs/current/history.md` when they change the research
  baseline.
- Delegate each concrete, independently verifiable implementation task to a
  short-lived engineering thread. Give it a clear goal, relevant file context,
  constraints, and completion criteria.
- Engineering threads must not silently change frozen decisions. Return genuine
  design ambiguities to the research thread instead of guessing.
- Git and focused tests provide implementation evidence. On completion, the
  engineering thread records only the incremental result in
  `docs/engineering/task_report.md`; avoid full logs, repeated history, or
  copied diffs.
- Avoid concurrent edits to the same files. Use separate worktrees for truly
  parallel tasks, otherwise complete and commit one dependent task at a time.

## Core Map

```text
seedance_pipeline.py             CLI entry
seedance_client.py               Ark task API client
storymem_seedance/               reusable pipeline, prompting, memory, reports
videogen_notebook/               current lightweight Web UI and project runner
videogen_notebook/runner.py      step runner, real/fake backend, assembly
videogen_notebook/project_store.py
                                  filesystem project bundle persistence
videogen_notebook/jobs.py        background job manager
videogen_notebook/templates/     server-rendered notebook UI
storymem_web/                    7860 StoryMem-style baseline plus shared helpers
extract_keyframes.py             HPSv3/CLIP post-processing
keyframe_settings/*.json        keyframe extraction presets
story/                           input scripts
.runtime/videogen_notebook/      current Web state; never commit
.runtime/web/                    legacy Web state; never commit
```

## Standard Commands

```bash
./storymem status
./storymem status --project miss_d
./storymem status --json

./run_videogen_notebook.sh status
./run_videogen_notebook.sh start
./run_videogen_notebook.sh restart
./run_videogen_notebook.sh logs -f

./storymem web status     # legacy storymem_web / port 7860 only
./storymem web start      # legacy storymem_web / port 7860 only
./storymem web restart    # legacy storymem_web / port 7860 only
./storymem web logs       # legacy storymem_web / port 7860 only

python -m unittest -q \
  tests.test_videogen_notebook tests.test_visual_element_memory
```

The current Web project is `videogen_notebook`, served on port 7870 with state
under `.runtime/videogen_notebook`. The `storymem_web` UI is served on port
7860 with state under `${STORYMEM_DATA}/web`; use it only when the task
explicitly asks for the StoryMem-style baseline. Use `http://localhost:7870`
for current visual-element work unless the task explicitly says otherwise.

## State Semantics

- Projects are filesystem bundles under `.runtime/videogen_notebook/projects`;
  `project.json` is the project truth source. Avoid introducing unnecessary
  global database coupling.
- A Shot has editable Shot Design plus immutable Attempt outputs for algorithm
  steps. Resetting a Shot invalidates that Shot and all downstream Shots.
- The current Step pipeline is: Shot Design, Visual Elements Plan, Historical
  Reference Selection, Seedance Prompt Composition, Seedance Video Generation,
  and Keyframe Maintaining.
- Only the current contiguous completed prefix is concatenated. Each completed
  Shot stores its derived current output so downstream regeneration can resume
  from the previous completed prefix.
- Generation duration is stored per Shot. `-1` means Seedance auto duration;
  explicit values must be integer seconds in the Seedance 2.0 range `4-15`.
- Imported or newly added non-cut Shots use the project/default setting
  `generation.default_non_cut_mode`, which defaults to `Smooth`; first Shots
  and cut Shots default to `Default`. Explicit user selections are preserved.
- `Default + non-cut` appends the previous Seedance original `last_frame_url` as
  the final image reference. `Last frame only` submits only that original
  `last_frame_url` as first-frame input.
- `Smooth` submits the previous raw Shot's configurable tail segment as
  Seedance `reference_video` and keeps static visual-memory references
  independent. The default tail length is 2 seconds.
- Smooth/RIFE final assembly never overwrites raw Attempt videos. Reference
  video publication currently uses the configured Cloudflare Tunnel publisher.
- Visual Element Memory is the current default memory path: Step 2 plans visual
  element states, Step 3 selects historical references, Step 6 maintains
  produced keyframes with VLM annotations and holistic descriptions.
- Reference selection supports Greedy Coverage, Static Top-K, Naive Top-K,
  StoryMem memory, and Disable reference images. Greedy attempts to select up to
  `max_retrieved_frames` without early stopping after required elements are
  covered. Naive Top-K is an ablation mode: it skips visual element planning,
  leaves keyframe annotations empty, and selects prior keyframes by cached CLIP
  image embeddings against the current prompt text embedding.
- StoryMem memory is a baseline mode: it skips visual element planning, selects
  historical keyframes with original-style Sink+Recent memory
  (`max_memory_size=10`, `fix=3`), and maintains the keyframe library with the
  `storymem_original` profile while skipping VLM annotation. During Step 6,
  historical similarity de-duplication compares new keyframes against this
  Shot's selected compact StoryMem memory bank, not every historical keyframe.
- Force animation is enabled by default for new/imported projects. It is added
  during Seedance Prompt Composition and remains user-editable before submission.

## Operations

- Check `./run_videogen_notebook.sh status` for current UI work. Use
  `./storymem web status` only for the legacy 7860 UI. If `./storymem status`
  reports `Web: ... port=7860`, do not chase that process for current
  `videogen_notebook` issues; check port 7870 and
  `.runtime/videogen_notebook/server.log` instead.
- A matching port is insufficient: compare the running source fingerprint when
  diagnosing stale behavior.
- Before restarting `videogen_notebook`, check for active project jobs. Avoid
  interrupting paid Seedance polling, LLM/VLM calls, or Step 6 GPU work unless
  the user asked for it.
- `videogen_notebook` allows multiple projects to run concurrently. The default
  background worker limit is 50 and can be overridden with
  `VIDEOGEN_NOTEBOOK_MAX_WORKERS`; Step 6 local GPU work is serialized by a GPU
  lock.
- Do not expose signed Ark URLs in SQLite, reports, logs, or commits.
- Do not print or commit API keys. Keys live in `$HOME/.seedance_api_key` and
  `$HOME/.deepseek_api_key` or environment variables.
- Do not commit generated media, `.runtime`, model weights, caches, or user PDFs.
- Preserve unrelated user changes in a dirty worktree.

## Servers

```text
4090 workspace: /home/wxh/world_model_projects/StoryMem
A6000 workspace: /home/lzg/wxh/world_model_projects/StoryMem
```

Use Git for source synchronization. Keep secrets and model caches outside Git.
Do not copy large models unless deployment work explicitly requires it.

## Testing Policy

- Run focused tests for the touched layer first.
- Real Seedance calls are opt-in because they cost money and may trigger policy.
- GPU keyframe extraction is a separate smoke test; unit tests must not load it.
- Keyframe settings default to the `loose` profile unless a runner overrides
  them. The 7860 `storymem_web` Classic baseline explicitly uses
  `keyframe_settings/storymem_original.json`; the 7870 `videogen_notebook`
  keeps its project-level keyframe profile. Override ad hoc CLI experiments
  with `STORYMEM_KEYFRAME_PROFILE` or `STORYMEM_KEYFRAME_CONFIG`.
- Keep fake Seedance/LLM/VLM assertions on exact content ordering, role, prompt
  structure, and URL provenance when changing generation inputs.
- Prefer focused tests and compile/diff checks. Avoid full slow suites unless
  the change touches broad runner state or the user explicitly asks for it.
- Run `git diff --check` and syntax/compile checks before committing.

## Done Checklist

- Behavior implemented end to end, not only in the template.
- Existing SQLite workspaces migrate safely.
- Runtime health reports the new source fingerprint after restart.
- Focused tests pass; any unrun integration or paid test is reported.
- Documentation describes current behavior, not the historical implementation.
- Commit only task-owned files with one clear commit message.
