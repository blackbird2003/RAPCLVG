# Visual Element Memory Web UI Implementation Plan

## 1. Status and Intent

- Status: implementation plan, not yet active Web behavior
- Scope: Seedance notebook Web UI only
- Default target for new projects: `visual_element_v1`
- Compatibility target: keep the current Web/CLI memory flow available as
  `classic`

The visual-element method changes the primary memory question from whole-frame
similarity to an explainable element workflow:

```text
maintain visual element set
-> decide current Shot element states
-> evaluate annotated historical keyframes
-> select reference frames by element coverage/exclusion
-> build precise Seedance prompt references
-> annotate the current Shot's produced keyframes for later Shots
```

Because this is larger than a small retriever option, it should be introduced as
a project-level pipeline version instead of being forced into the old
Sink/Retrieve/Recent layout. The old behavior remains useful for legacy
experiments and ablations.

## 2. Pipeline Version Model

Add a project-level pipeline selector:

```text
pipeline_version: visual_element_v1 | classic
```

Recommended migration behavior:

- New empty projects default to `visual_element_v1`.
- Newly imported stories default to `visual_element_v1`.
- Existing projects keep their current behavior as `classic`, unless the user
  explicitly switches or recreates the project.
- The current generation mode remains independent of pipeline version:
  `default`, `last_frame_only`, and `smooth` continue to control only short-term
  continuity media.

`visual_element_v1` should replace the old Shot-level memory-source controls in
the main UI. The old `memory_sink`, `memory_retrieve`, and `memory_recent`
fields may remain in the database for `classic` and for reproducibility.

### 2.1 Project-Level Visual Memory Settings

The first Web implementation should expose a small global settings block near
the project pipeline selector:

```text
Pipeline version: Visual Element Memory v1 | Classic
Sink frame count: integer, default 0
Max retrieved historical frames: integer, default 4
```

These settings apply to the whole project and are frozen into every Attempt
snapshot when a Shot runs.

`Sink frame count` preserves the useful part of the classic early-anchor
heuristic without reviving the old Shot-level Sink/Retrieve/Recent switches.
In `visual_element_v1`, sink frames mean the earliest active historical
retrieval keyframes from the current completed prefix. They should be included
after element-selected retrieved frames as early reference anchors. A value of
`0` disables sink frames.

`Max retrieved historical frames` is the maximum number of non-sink historical
keyframes selected by the visual-element scoring/coverage algorithm. This is
separate from continuity media such as previous ending image, last-frame-only
input, or Smooth `reference_video`.

The combined static reference order should be:

```text
continuity input, when applicable
-> visual-element retrieved frames
-> visual sink frames
```

For Smooth, the continuity input is a `reference_video` and is not counted as a
static historical keyframe. The Seedance image budget remains at most 9 static
images, so `Sink frame count + Max retrieved historical frames <= 9`. For Last
frame only, selected memory images are still suppressed by the generation mode,
matching current semantics.

The CLI visual-element path supports the same budget semantics: visual-element
retrieval first, visual sink anchors last, and
`visual_element_sink_frame_count + visual_element_max_retrieved_frames <= 9`.
This affects selector and reporting logic, but not the Seedance client,
keyframe extraction, Smooth assembly, or VLM annotation model.

## 3. New Shot Layout

Each Shot remains the notebook execution unit. The new UI should emphasize the
three visual-element method steps and their evidence trails.

```text
Shot N
├─ Inputs
│  ├─ Video prompt
│  ├─ Cut
│  ├─ Generation mode
│  └─ Generation duration
├─ Shot Status
├─ Visual Element Status
│  ├─ element list
│  └─ Visual Element Details
├─ Selected Historical References
│  ├─ selected reference list
│  └─ Reference Selection Details
├─ Input Prompt
├─ Output Video
│  └─ Seedance Attempt Details
└─ Produced Visual Memory
   ├─ produced memory list
   └─ Memory Extraction Details
```

### 3.1 Inputs

Inputs are editable before submission and are frozen in the immutable Attempt
snapshot when the Shot runs:

```text
Video prompt
Cut switch
Generation mode
Generation duration
```

Values can be imported from story JSON. `Generation mode` is derived from the
Cut state by default:

- first Shot or Cut on: `default`
- non-Cut with predecessor: `smooth` by default

Manual mode selection remains available for non-Cut Shots.

### 3.2 Shot Status

The status header should remain compact and visible near the Shot actions.

Add visual-element-specific intermediate states so users can tell which slow
step is running:

```text
planning_visual_elements
selecting_visual_references
submitting
waiting_for_seedance
downloading
extracting_keyframes
annotating_visual_memory
smoothing_transition
completed
failed
interrupted
```

The exact state names can be adjusted to fit the current enum, but the Web UI
should distinguish LLM planning, reference selection, Seedance waiting, and VLM
annotation.

### 3.3 Visual Element Status

Purpose: show the result of element-set maintenance and Shot-level element
state decisions.

Display as a list that is expanded by default and collapsible by the user.
Each row should include:

```text
element id
name
type: character | scene | object
introduced_at
notes
state: should_reference | should_exclude | optional_or_uncertain | new
state_reason
```

The state is decided by the LLM in the first implementation. The UI and data
shape should leave room for future manual overrides, but manual editing is not
part of the first implementation.

Follow the list with a collapsed log panel named `Visual Element Details`.
It should show:

- model name
- prompt sent to the LLM
- raw response
- parse attempts and parse errors
- registry before and after this Shot
- inserted new elements

### 3.4 Selected Historical References

Purpose: show how historical keyframes were judged and why the final reference
images were selected.

Display as a list that is expanded by default and collapsible by the user.
Each row should include:

```text
annotated keyframe visualization
source scene / shot
reference index
score
should-reference elements visible in this frame
newly covered elements
already covered elements
should-exclude elements visible in this frame
optional elements visible in this frame
reference intent text
```

Do not include a separate `reason` column in the main list. Exclusion and
reference reasons already belong to `Visual Element Status`; repeating them in
the reference list would make the table noisy.

Follow the list with a collapsed log panel named `Reference Selection Details`.
It should show:

- selection mode: `static_top_k` or `greedy_coverage`
- sink frame count
- max retrieved historical frames
- effective static reference budget after continuity inputs
- scoring weights
- candidate keyframe count
- sink frames appended after visual-element retrieval
- per-candidate score details
- greedy rounds, selected frames, and covered element set after each round
- skipped candidates with non-positive score if applicable

### 3.5 Input Prompt

Keep the existing full submitted prompt section. It can remain collapsed by
default or preserve the user's expanded state.

For `visual_element_v1`, the prompt should include:

- complete story context
- current Shot task
- Smooth continuation instruction when a `reference_video` is present
- Shot-level visual element plan
- per-reference image instructions
- global constraint that only explicitly named elements should be referenced

### 3.6 Output Video

Keep the generated Shot video section.

Follow it with a collapsed panel named `Seedance Attempt Details`, showing:

- task ID
- submitted media summary
- redacted create/result responses
- usage
- error if present

This replaces the ambiguous generic `Attempt Details` name for the Seedance
generation step.

### 3.7 Produced Visual Memory

Purpose: show what the current Shot contributes to later Shots.

Display as a list that is expanded by default and collapsible by the user.
Each row should include:

```text
keyframe thumbnail
annotation visualization
rank / selection order
visible known elements
active_in_memory_pool
source asset path or media URL
```

This section corresponds to post-generation keyframe selection and VLM
annotation. It should appear after `Output Video` because it describes the
memory produced by the generated result.

Follow the list with a collapsed log panel named `Memory Extraction Details`.
It should show:

- keyframe extraction settings/profile
- selected keyframe paths
- dropped/duplicate frame information when available
- VLM annotation prompts
- raw VLM responses
- parse attempts and parse errors
- annotation visualization paths

## 4. Data and Artifact Plan

The first implementation should avoid a large schema rewrite. Store complete
visual-element artifacts in each Attempt directory and persist only the summary
needed for fast page rendering in SQLite JSON fields.

Recommended Attempt artifact layout:

```text
attempt_dir/
  input.json
  memory.json
  seedance_task.json
  visual_element/
    plan.json
    llm_calls.jsonl
    selected_references.json
    reference_selection_details.json
    prompt_context.txt
    produced_annotations.json
    vlm_calls.jsonl
    visualizations/
```

Recommended `attempts.memory_selection_json` summary:

```json
{
  "policy": "visual_element_v1",
  "visual_element": {
    "enabled": true,
    "selection_mode": "greedy_coverage",
    "sink_frame_count": 3,
    "max_retrieved_historical_frames": 4,
    "plan": {
      "should_reference": [],
      "should_exclude": [],
      "optional_or_uncertain": [],
      "new_elements": []
    },
    "sink_references": [],
    "selected_references": [],
    "produced_annotations": []
  },
  "references": []
}
```

The full LLM/VLM logs should stay in artifacts, not in the primary page payload.
The view layer can expose artifact summaries and media URLs.

## 5. State and Versioning Semantics

The visual-element registry may be project-level conceptually, but every Attempt
must freeze the state it used:

```text
registry_before
LLM decision
inserted elements
registry_after
selected references
submitted prompt
produced annotations
```

This keeps Shots independently inspectable and makes reset/rerun behavior
reproducible.

When a Shot is reset, current Attempts and memory assets from that Shot onward
are already invalidated by the Web pipeline. `visual_element_v1` should follow
the same rule:

- downstream visual-element artifacts remain on disk for history/debugging
  but are no longer current
- downstream keyframe annotations must not participate in new reference
  selection
- rerunning a Shot reconstructs the active registry and annotations from the
  current completed prefix only

Future manual overrides should be represented as Shot revision inputs, not as
edits to historical Attempt results:

```json
{
  "element_state_overrides": {
    "element-0003": {
      "state": "should_reference",
      "reason": "manual override"
    }
  }
}
```

Manual override UI is a future feature.

## 6. Runner Integration Plan

Add a Web adapter around the existing CLI visual-element logic instead of
copying the algorithm into templates or the runner.

Suggested module:

```text
storymem_web/visual_element_session.py
```

Responsibilities:

1. Load the active completed prefix for a project.
2. Reconstruct the visual-element registry from current Attempt artifacts.
3. Reconstruct historical keyframe annotations from current produced memory.
4. Plan the current Shot with the LLM.
5. Select historical keyframes using `static_top_k` or
   `greedy_coverage`, capped by `max_retrieved_historical_frames` and the
   Seedance static image budget.
6. Append visual sink frames according to the project setting.
7. Return:
   - Seedance reference media items
   - memory-selection summary
   - prompt override/context
   - artifact paths for logs and visualizations
8. After video generation and keyframe extraction, annotate produced keyframes
   and persist their summaries for downstream Shots.

`ProjectRunner._prepare_references()` should become the main branch point:

```text
if pipeline_version == "visual_element_v1":
    prepare visual-element references
else:
    prepare classic references
```

The existing continuity handling remains after reference preparation:

- `default`: previous ending image plus selected memory images
- `last_frame_only`: only previous original `last_frame_url`
- `smooth`: previous raw tail `reference_video` plus selected memory images

`smooth` should not change visual-element selection. It only replaces the
previous-ending image/video continuity input.

Selector implementation detail:

- Sink frames do not participate in element coverage or visual-element scoring.
- Sink frames do not require annotation. If annotations already exist, they may
  be displayed later, but the prompt should describe them simply as early
  reference anchors.
- The visual-element retrieved-frame selector should avoid selecting duplicate
  paths that will also be used as sink.
- The final reference list should preserve source roles such as
  `visual_sink_memory` and `visual_element_memory` for reporting and UI badges.
- The static image budget is limited to 9 images. Enforce
  `visual_element_sink_frame_count + visual_element_max_retrieved_frames <= 9`.

## 7. Web View and Template Plan

Extend `project_view()` to prepare these fields for each Shot:

```text
shot.visual_element_plan
shot.visual_reference_selection
shot.visual_produced_memory
shot.visual_element_detail_artifacts
shot.reference_selection_detail_artifacts
shot.memory_extraction_detail_artifacts
```

Media paths in visualizations should be converted through the existing
`media_url()` mechanism and must remain inside the project directory.

Template changes:

- Hide `Memory sources` when `pipeline_version == visual_element_v1`.
- Replace the old `Memory decision` section with `Visual Element Status`.
- Replace or enrich `Input references` with `Selected Historical References`.
- Replace `Produced memory candidates` with `Produced Visual Memory`.
- Rename the generic attempt details panels according to their stage.

The three main lists should be expanded by default and user-collapsible.
Implementation can use native `<details open>` for lightweight behavior.

## 8. Testing Plan

Use fake LLM/VLM and fake Seedance clients. Do not make paid Seedance calls in
automated tests.

Focused tests should cover:

1. New projects default to `visual_element_v1`; existing projects preserve
   legacy behavior.
2. Project settings persist `sink_frame_count=0` and
   `max_retrieved_historical_frames=4` by default and freeze into Attempt
   snapshots.
3. Visual-element planning freezes registry before/after and element states in
   the Attempt artifact.
4. Historical reference selection appends sink frames after visual-element
   retrieval, avoids duplicate paths, and respects the 9-image budget.
5. Historical reference selection records selected frames, scores, coverage, and
   excluded visible elements.
6. Generated Seedance prompt contains the visual-element plan and per-reference
   instructions.
7. Smooth mode prepends `reference_video` while preserving selected static
   visual-element references.
8. Produced keyframes are annotated and made available only from current
   completed Attempts.
9. Reset from a Shot prevents stale downstream annotations from being used.
10. Web rendering shows all three lists, sink/retrieval counts, and the renamed
    detail panels.

Low-cost verification:

```bash
python -m unittest -q \
  tests.test_project_runner \
  tests.test_web_repository \
  tests.test_web_app

git diff --check
```

Add narrower visual-element unit tests as the adapter module is introduced.

## 9. Non-goals for First Implementation

- Manual element editing
- Manual bbox editing
- Open-set image discovery of new elements
- Cross-project visual-element memory
- Advanced weight tuning UI
- Replacing Smooth, Last frame only, or classic pipeline behavior
- Parallel project execution

These are intentionally left for later once the automatic evidence chain is
visible and stable.
