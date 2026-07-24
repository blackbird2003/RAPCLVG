# Research Plan

This document tracks active research hypotheses, decisions, and experimental
TODOs for the Seedance long-video pipeline. Completed implementation milestones
are summarized in `history.md`; detailed engineering handoffs live in
`../engineering/task_report.md`.

## Current System Baseline

The Web pipeline now has the infrastructure needed for controlled continuity
and memory ablations:

- Continuity/generation mode is an explicit per-shot input. New imported or
  created non-cut shots default to `smooth`; `default` and `last_frame_only`
  remain available as ablations.
- Smooth mode uses the previous raw shot's configurable tail segment as
  `reference_video`, asks Seedance to continue after the ending motion, keeps
  selected memory images independent from the continuity input, and assembles
  the final project video with a two-frame Practical-RIFE transition.
- Memory source policy is an independent per-shot input with `Sink / Retrieve /
  Recent` switches. New shots default to `on / on / off`; existing projects
  migrate to the legacy cut/non-cut policies.
- Prompt-aware retrieval can run for both cut and non-cut shots when Retrieve is
  enabled. The old window-memory behavior is preserved through switches rather
  than removed.
- Keyframe extraction settings are now JSON profiles, enabling lighter
  experiment control without editing code.

The main research work should therefore avoid changing generation-mode code
unless the experiment specifically concerns continuity. Memory and retrieval
experiments should use the existing switches and Attempt snapshots for
reproducible comparisons.

## Active Research Tracks

### V1 Design: Visual Element Memory

`../plans/completed/visual_element_memory_v1.md` records the initial conceptual direction for
precise reference prompting and historical retrieval: a small prompt-derived
set of characters, scenes, and objects; per-Shot reference/exclusion/optional/
new roles; compatibility-aware frame selection; and image-specific reference
instructions. It deliberately does not prescribe an implementation or
experiment protocol. Frame-derived element discovery remains future work.

### 1. Precise Reference Prompting

Problem: prompt-enhanced generation currently uploads selected historical
keyframes together with the raw video prompt and expects Seedance to infer what
to use. In practice, whole-frame references can leak irrelevant elements into
the new shot. For example, a frame containing the Little Prince and the rose may
help preserve character style, but may also cause the rose to appear on another
planet where it should be absent.

Hypothesis: wrong-reference leakage can be reduced by converting historical
frames from whole-frame visual signals into explicitly described, bounded, and
role-labeled reference assets.

Proposed direction:

1. Build a small wrong-reference benchmark from existing projects. For each
   case, record required entities, forbidden entities, submitted reference
   frames, and whether the generated video leaked an irrelevant object.
2. Add an offline Memory Inspector for keyframes. For each frame, store scene
   caption, visible entities, attributes, likely character identities,
   background/setting, region crops if available, and possible contaminants.
3. Decompose each shot prompt into reference needs before retrieval:
   characters, clothing, scene, objects, style, motion continuity, and explicit
   exclusions.
4. Prefer precise assets over whole frames when possible: crops, masks, or
   region-level descriptions such as "use only the boy's outfit from Image 1"
   rather than "use Image 1".
5. Compose reference-bound prompts that name each submitted reference and state
   both allowed use and forbidden leakage.
6. Add a post-generation VLM/manual evaluation loop later: required elements,
   forbidden elements, consistency, and retry reason.

Minimal first experiment:

- keep the current retrieval candidate pool;
- run VLM/frame annotation offline on selected references;
- manually or semi-automatically select one to three precise reference assets;
- compare current whole-frame prompt, whole-frame plus precise text, and
  crop/region plus precise text on the same wrong-reference cases.

Success criteria:

- fewer forbidden-object leaks;
- no obvious loss of character/scene consistency;
- lower need for manual prompt rewriting;
- diagnostic records that explain why each reference was allowed.

### 2. Historical Frame Retrieval Algorithm

Problem: with Recent disabled by default, long-range consistency depends on
retrieving the right historical frames for both cut and non-cut shots. Retrieval
must improve recall for needed history while avoiding irrelevant frames that
introduce wrong objects, people, or settings.

Hypothesis: retrieval should be precision-first and conflict-aware, not just
semantic similarity over whole-frame CLIP features.

Proposed direction:

1. Define query construction by role: identity, clothing, persistent object,
   current setting, returning location, style, and story state. Separate
   persistent appearance from transient motion.
2. Keep an explicit reference budget. Use retrieval only when the frame has a
   clear role; low-confidence references should be omitted.
3. Score candidates with positive evidence and penalties:

   ```text
   score = semantic_relevance
         + entity_match
         + identity/scene consistency
         + useful_recency
         + quality
         - contaminant_risk
         - prompt_conflict
         - redundancy
   ```

4. Use diversity selection so multiple references do not repeat the same visual
   evidence unless that repetition is intentional.
5. Add negative/exclusion evidence when the prompt implies that a previously
   seen entity should not appear.
6. Evaluate retrieval separately from generation: recall of needed references,
   precision of submitted references, forbidden-entity risk, downstream video
   consistency, and token/cost impact.

Minimal first experiment:

- run the current `Sink=on, Retrieve=on, Recent=off` policy as the baseline;
- manually label a small set of shots with required and forbidden memory;
- compare CLIP/prompt-aware retrieval with a VLM-reranked or entity-filtered
  version;
- keep Smooth/RIFE continuity fixed so retrieval results are not confounded
  with non-cut motion handling.

## Archived Continuity Findings

These findings explain why Smooth is the current default for continuous shots.
They are retained for ablations but are not the main research focus now.

- Prompt-constrained default generation does not strictly preserve the first
  frame and can introduce an immediate visual jump.
- First-frame generation preserves the boundary frame but does not preserve the
  incoming motion state.
- Plain extension from the previous shot's final one-second video improved
  perceived motion continuity more than asking Seedance to retain the input
  segment.
- Retained-overlap extension was marked maybe not effective: Seedance re-rendered
  the uploaded tail instead of copying it and introduced unwanted appearance
  drift.
- SSIM-only, motion-only, and combined delta-consistent seam scoring selected
  nearby splice points and looked broadly similar on the Little Prince sample.
- RIFE interpolation around the seam was preferred over FFmpeg interpolation by
  human review, so Smooth currently removes `A[-1]` and `B[0]` and inserts two
  RIFE frames between `A[-2]` and `B[1]`.

Future continuity experiments can revisit:

- delta-consistent splice selection with optical flow and camera-motion terms;
- subject-aware motion masks;
- learned seam quality metrics;
- different numbers of removed or inserted RIFE frames.
