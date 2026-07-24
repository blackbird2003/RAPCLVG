# Complementary Retrieval-Augmented Prompting for Consistent Long-Form Video Generation

This repository contains a lightweight long-form video generation pipeline built
around mature short-video generation APIs. The system generates a story
shot-by-shot, maintains visual memory from completed clips, retrieves
complementary historical references, and composes structured multimodal prompts
to improve entity and style consistency across long videos.

## Overview

The current implementation focuses on the `videogen_notebook` Web UI and the
Seedance backend. A project is represented as a sequence of shots. Each shot
contains a video prompt, cut flag, generation mode, duration, optional
predefined references, and step-level outputs.

The main notebook pipeline is:

1. Shot Design
2. Visual Elements Plan
3. Historical Reference Selection
4. Seedance Prompt Composition
5. Seedance Video Generation
6. Keyframe Maintaining

Generated clips are concatenated into a current output video as the completed
prefix grows. Produced keyframes are added to the project memory and can be used
by later shots.

## Key Features

- Visual-element memory planning with LLM/VLM support.
- Complementary historical reference retrieval for multi-shot consistency.
- Structured prompt grounding for selected reference images.
- Editable notebook-style Web workflow for inspection and ablation.
- Support for predefined per-shot reference images.
- Multiple reference-selection modes, including greedy coverage, static top-k,
  naive top-k, disabled references, and a StoryMem-style baseline mode.
- Smooth non-cut continuation through short reference-video conditioning.

## Web UI

Start the current Web UI:

```bash
./run_videogen_notebook.sh start
```

Restart it:

```bash
./run_videogen_notebook.sh restart
```

Open:

```text
http://localhost:7870
```

Runtime project data is stored under:

```text
.runtime/videogen_notebook/
```

This directory is ignored by Git and should not be committed.

## CLI And Scripts

Useful entry points include:

```text
seedance_pipeline.py                      Seedance CLI pipeline
videogen_notebook/                        Current notebook Web UI
storymem_seedance/                        Shared prompting, memory, and runner logic
extract_keyframes.py                      Keyframe extraction and scoring
tools/convert_msvbench_to_videogen_json.py
tools/convert_vimax_benchmark_to_videogen_json.py
tools/submit_videogen_eval_to_a6000.py
```

Current documentation is organized under:

```text
docs/current/
docs/engineering/
docs/evaluation/
docs/references/
```

See `AGENTS.md` for engineering workflow notes and the current source map.

## Secrets

API keys are expected to be provided through environment variables or private
files outside the repository, for example:

```text
$HOME/.seedance_api_key
$HOME/.deepseek_api_key
```

Do not commit API keys, runtime outputs, generated media, model weights, or
cache directories.
