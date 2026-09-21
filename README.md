# Complementary Retrieval-Augmented Prompting for Consistent Long-Form Video Generation

> A training-free, interpretable workflow for keeping characters, scenes, and objects consistent across generated video shots.

This repository contains the research implementation and a stepwise creation UI. Visit the [project page](https://blackbird2003.github.io/RAPCLVG_Page/) for generated-story demos, method visualizations, and qualitative results. The paper preprint link and citation will be added when they are publicly available.

<p align="center">
  <img src="https://blackbird2003.github.io/RAPCLVG_Page/assets/figures/pipeline.png" alt="Overview of Complementary Retrieval-Augmented Prompting" width="100%">
</p>

This project is a lightweight, migratable pipeline for long-form video generation. It generates a story shot by shot, maintains a visual-element memory across shots, retrieves complementary historical references, and composes structured multimodal prompts to keep characters, scenes, and objects consistent over time.

The project is designed to be API-agnostic at the generation layer. The current implementation uses Seedance as the video backend, but the pipeline core does not depend on it.

## About

This repository is the implementation of Complementary Retrieval-Augmented Prompting for Consistent Long-Form Video Generation. Instead of relying on recency-biased memory or manually curated references, the system treats long-form generation as an interpretable retrieval and prompting problem.

The pipeline maintains two evolving evidence structures:

- A text-grounded visual element registry that tracks characters, objects, and scenes introduced by the story, together with shot-level states such as should-reference, should-exclude, optional, and newly introduced.
- A VLM-annotated keyframe library that records where these elements appear in previously generated shots, including element-level annotations and holistic frame descriptions.

For each new shot, an agent retrieves a compact set of complementary historical keyframes that jointly cover the required visual elements while suppressing conflicting or redundant content. The selected references, structured element states, and grounding instructions are then assembled into a unified multimodal prompt for a frozen short-video generator.

This formulation is training-free, backend-agnostic, and supports incremental and interactive creation. It also makes reference decisions explainable: every selected frame can be traced to the visual elements it covers and the historical content it may introduce.

## Project Page

The public [RAP-CLVG project page](https://blackbird2003.github.io/RAPCLVG_Page/) is maintained in the separate [`RAPCLVG_Page`](https://github.com/blackbird2003/RAPCLVG_Page) repository. It includes generated-story demos, the pipeline overview, and qualitative comparisons without duplicating large media files in this code repository.

## Highlights

- Shot-by-shot video generation from a story JSON.
- Visual element planning with LLM maintenance and optional reflection.
- Historical reference selection with multiple retrieval strategies.
- Structured Seedance prompt composition with modular toggles.
- Seedance video generation with smooth non-cut transitions.
- Keyframe extraction, VLM annotation, and visual-memory maintenance.
- A Web UI for project management and manual step control.
- A CLI that runs the same pipeline without the Web UI.
- Self-contained project folders; no central database is required.
- Local GPU lock so keyframe maintenance can share one GPU safely.

## Repository Layout

```text
pipeline/       Core agentic planning, retrieval, prompting, media, and runtime modules.
videogen_ui/    Flask-based stepwise Web UI.
videogen_full_cli.py
                CLI entry point for end-to-end generation.
story/          Example StoryMem and EntityBench-format scripts.
tests/          Focused unit and integration tests.
tools/          Dataset conversion, setup, and evaluation utilities.
docs/           Engineering and API notes.
```

Generated project state, models, API credentials, unpublished paper sources, and demo media are intentionally excluded from version control. See [`.gitignore`](.gitignore) for the full policy.

## Requirements

- Linux or a compatible Unix-like environment.
- Python 3.11.
- CUDA GPU for HPSv3 quality scoring and Smooth RIFE interpolation.
- A Seedance/ARK API key for real video generation and LLM/VLM calls.
- Network access on first setup to download dependencies and model weights.
- Roughly 32 GB of Hugging Face model cache plus disk space for generated videos and project assets.

## Installation

Run the setup script from the project root:

```bash
./setup_env.sh
```

The script:

1. Sources `videogen_env.sh` to configure cache paths and the Python environment.
2. Installs `requirements.txt` with pip.
3. Installs all direct Python dependencies, including `gdown`.
4. Downloads all local model weights required by the pipeline.

You can skip parts of the setup:

```bash
VIDEOGEN_SKIP_PIP=1 ./setup_env.sh            # skip pip install
./setup_env.sh --skip-hf                     # skip Hugging Face models
./setup_env.sh --skip-clip                   # skip CLIP checkpoint
./setup_env.sh --skip-rife                   # skip Practical-RIFE
```

After setup, runtime processes use Hugging Face in offline mode and read from `.runtime/cache/huggingface`.

## API Keys

The pipeline needs an API key for Seedance and for LLM/VLM calls. The supported environment variables are:

```bash
export SEEDANCE_API_KEY="your-seedance-key"
# or
export ARK_API_KEY="your-ark-key"
```

`videogen_env.sh` also reads `~/.seedance_api_key` when the environment variable is empty.

## License

See [LICENSE.md](LICENSE.md) for the current non-commercial redistribution terms.

## Story JSON Format

A story is a single JSON file with a `scenes` array. Each scene contains one or more `video_prompts`, and optional arrays describe cut flags, durations, generation modes, and predefined references. Optional arrays must have the same length as `video_prompts` for that scene.

```json
{
  "story_name": "The Little Prince",
  "story_overview": "A gentle and poetic story of a young prince...",
  "scenes": [
    {
      "scene_num": 1,
      "video_prompts": [
        "Wide shot of the Little Prince on his tiny planet at sunrise.",
        "Close-up of the Little Prince watering his rose."
      ],
      "cut": [true, false],
      "durations": [-1, 8],
      "generation_modes": ["default", "smooth"],
      "predefined_references": [
        [
          {
            "media_type": "image",
            "image_path": "assets/character_sheet.png",
            "label": "Little Prince",
            "guidance": "Use this image as the character identity reference."
          }
        ],
        []
      ]
    }
  ]
}
```

Field notes:

- `scene_num`: scene index; it should be unique.
- `video_prompts`: one text prompt per shot.
- `cut`: optional boolean per shot. `true` means the shot is a scene transition and does not continue from the previous shot; `false` means continuous non-cut generation.
- `durations`: optional integer per shot. `-1` means automatic duration; an explicit value must be between 4 and 15 seconds.
- `generation_modes`: optional per-shot mode. Supported values are `default`, `last_frame_only`, and `smooth`. When omitted, the project default is used.
- `predefined_references`: optional list aligned with shots. Each entry is a list of reference objects. Supported media types are `image`, `audio`, and `video`. Image paths are copied into the project directory on import.

### Bundled Story Scripts

The repository includes two example script sets:

- `story/entitybench_1of4_scripts/`: 35 JSON scripts (Easy 1-20, Mid 1-10, Hard 1-5) from the EntityBench subset used in the paper.
- `story/ST_Bench/`: 30 example story scripts in the StoryMem benchmark format.

You can run any of these directly:

```bash
python videogen_full_cli.py story/ST_Bench/little_prince.json --dry-run
```

## Quick Start (CLI)

Import a story JSON and run the full pipeline:

```bash
python videogen_full_cli.py path/to/story.json
```

Real Seedance submission is enabled by default. Use a dry run to validate the pipeline without LLM, VLM, Seedance, or GPU work:

```bash
python videogen_full_cli.py path/to/story.json --dry-run
```

Other useful options:

```text
--settings settings.json   Override project settings.
--workspace /path/to/ws    Use a custom workspace.
--name "My project"        Name a newly imported project.
--project-id PROJECT_ID    Resume an existing project.
--yes                      Auto-approve Seedance when confirmation is required.
--poll-seconds 0.5         Progress display refresh interval.
```

Example with a partial settings file:

```json
{
  "generation": {
    "default_duration_seconds": -1,
    "default_non_cut_mode": "smooth"
  },
  "visual_element_memory": {
    "selection_mode": "greedy_coverage",
    "sink_frame_count": 0,
    "max_retrieved_frames": 4
  },
  "seedance": {
    "resolution": "720p",
    "ratio": "16:9"
  }
}
```

## Web UI

Start the Web UI:

```bash
./run_videogen_ui.sh start
```

Open `http://localhost:7880`. The script also supports:

```bash
./run_videogen_ui.sh restart
./run_videogen_ui.sh stop
./run_videogen_ui.sh status
./run_videogen_ui.sh logs
./run_videogen_ui.sh logs -f
```

Use a custom port with `VIDEOGEN_PORT`:

```bash
VIDEOGEN_PORT=7880 ./run_videogen_ui.sh start
```

### Web UI Features

- Create new projects and import one or more story JSON files at once.
- Edit global default settings from the home page.
- Duplicate and delete projects.
- Edit Shot 1 design inputs such as video prompt, cut flag, generation mode, duration, and predefined references.
- Add shots manually or import them from JSON.
- Run the whole project, a single shot, or one algorithm step.
- Edit Visual Elements Plan, Historical Reference Selection, Seedance Prompt, and Keyframe Maintaining results where supported.
- Trigger Visual Elements Plan reflection.
- Apply predefined references to all subsequent shots.
- Reset or fork a project from a completed shot.
- View assets, attempt details, logs, prompts, and media previews.

## Pipeline Overview

Each shot moves through the following steps:

| Step | Name | Description |
| --- | --- | --- |
| 1 | Shot Design | Video prompt, cut flag, generation mode, duration, predefined references. Manual or JSON-imported; no algorithm step. |
| 2 | Visual Elements Plan | LLM maintains the visual-element registry and assigns reference/exclude/optional/new states. |
| 3 | Historical Reference Selection | Selects historical frames using the configured retrieval strategy. |
| 4 | Seedance Prompt Composition | Assembles the final multimodal prompt and request content. |
| 5 | Seedance Video Generation | Submits the request, polls the task, and downloads the video. |
| 6 | Keyframe Maintaining | Extracts keyframes, runs VLM annotation, and updates visual memory. |

### Run Levels

- **Run all**: automatically advances through every incomplete shot and step.
- **Run shot**: runs all remaining steps for one shot.
- **Run step**: runs one algorithm step for one shot.

Steps have states such as `draft`, `running`, `completed`, `failed`, and `queued`. Keyframe maintenance also queues on a process-local GPU lock before running.

### Retry Behavior

Algorithm steps 2, 3, 5, and 6 retry after failure. The maximum number of attempts is configurable through `algorithm_step_max_attempts` (default 5, meaning up to 4 retries). Retries wait 60 seconds between attempts.

## Project Data

Projects are self-contained folders under the workspace:

```text
.runtime/videogen_ui/projects/
  <project_id>/
    project.json
    story.json
    settings.json
    shots/
      <shot_id>/
        shot.json
        attempts/
          <attempt_id>/
    memory/
    assets/
    final/
```

Project state, attempts, media, memory, and logs live inside this folder, so a project can be copied or migrated without a central database.

## Models

The setup script prepares the following local assets:

| Model | Purpose | Location |
| --- | --- | --- |
| `MizzenAI/HPSv3` | Keyframe quality scoring | `.runtime/cache/huggingface` |
| `Qwen/Qwen2-VL-7B-Instruct` | Base model required by HPSv3 | `.runtime/cache/huggingface` |
| OpenAI CLIP `ViT-B/32` | Frame similarity and embeddings | `~/.cache/clip` |
| Practical-RIFE 4.25 | Smooth frame interpolation | `.runtime/deps/Practical-RIFE` |

## Configuration

### Generation Modes

- `default`: uses the previous ending frame as continuity constraints for seedance.
- `last_frame_only`: uses only the previous last frame, no additional references are supported, and the first frame constraints for the new video are stronger.
- `smooth`: uses the tail of the previous raw video as a reference video, which could preserve the motion compared with default, and later interpolates the transition with RIFE.

### Reference Selection Modes

- `greedy_coverage`: complementary coverage selection using visual element states and scoring weights.
- `static_top_k`: fixed Top-K selection without greedy coverage.
- `naive_top_k`: simple CLIP embedding-based Top-K without visual planning.
- `sink_recent_memory`: early-sink plus recent-window memory selection.
- `none`: disables historical reference images.

### Prompt Modules

The prompt assembler supports independent toggles:

- Full script context
- Visual element plan
- Holistic description and reference guidance
- Should-reference element constraints
- Should-exclude element constraints

### Reference Image Budget

`sink_frame_count + max_retrieved_frames` must not exceed 9 reference images. Predefined references also count toward the Seedance reference limit.

### Keyframe Profiles

Keyframe settings are selected from `pipeline/keyframes/settings/`. Available profiles include `default`, `loose`, `strict`, and `sink_recent_original`.

## Environment Variables

| Variable | Default | Purpose | Network |
| --- | --- | --- | --- |
| `SEEDANCE_API_KEY` / `ARK_API_KEY` | empty | Required for real Seedance and LLM/VLM calls. | Yes |
| `SEEDANCE_MODEL` | `doubao-seedance-2-5-260628` | Seedance model name. | Yes |
| `SEEDANCE_API_BASE` | `https://ark.cn-beijing.volces.com/api/v3` | Seedance API base URL. | Yes |
| `SEEDANCE_RESOLUTION` | `720p` | Default generation resolution. | No |
| `SEEDANCE_CALLBACK_URL` | empty | Optional task callback URL. | Yes |
| `VISUAL_ELEMENT_MODEL` | `doubao-seed-2-1-turbo-260628` | LLM/VLM model for element planning and annotation. | Yes |
| `VIDEOGEN_ROOT` | project root | Project root. | No |
| `VIDEOGEN_DATA` | `$VIDEOGEN_ROOT/.runtime` | Cache and runtime data root. | No |
| `VIDEOGEN_EXTERNAL_DATA_DIR` | empty | Optional external data directory. | No |
| `VIDEOGEN_VENV` | empty | Optional Python virtual environment. | No |
| `VIDEOGEN_WORKSPACE` | `.runtime/videogen_ui` | Project workspace. | No |
| `VIDEOGEN_PORT` | `7880` | Web UI port. | No |
| `VIDEOGEN_HOST` | `127.0.0.1` | Web UI bind host. | No |
| `VIDEOGEN_RUNNER` | `real` | Execution backend. | No |
| `VIDEOGEN_MAX_WORKERS` | `30` | Concurrent project workers (the run script exports 50). | No |
| `VIDEOGEN_REAL_SUBMIT` | `1` | Enable real Seedance submission. | Yes |
| `VIDEOGEN_SKIP_PIP` | `0` | Skip pip install in `setup_env.sh`. | No |
| `VIDEOGEN_KEYFRAME_PROFILE` | `loose` | Default keyframe profile name. | No |
| `VIDEOGEN_KEYFRAME_CONFIG` | empty | Path to a custom keyframe JSON. | No |
| `VIDEOGEN_REFERENCE_VIDEO_PUBLISHER` | `cloudflare_tunnel` | Reference video publisher. | Yes |
| `VIDEOGEN_REFERENCE_VIDEO_SERVE_DIR` | temp `videogen_reference_video_tunnel` | Local reference video serve directory. | No |
| `VIDEOGEN_CLOUDFLARED_BIN` | auto | Path to the `cloudflared` binary. | No |
| `VIDEOGEN_CLOUDFLARE_TUNNEL_TIMEOUT` | `60` | Cloudflare tunnel startup timeout. | Yes |
| `VIDEOGEN_REFERENCE_VIDEO_TIMEOUT` | `60` | Reference video upload timeout. | Yes |
| `VIDEOGEN_REFERENCE_VIDEO_RETRIES` | `10` | Reference video upload retries. | Yes |
| `VIDEOGEN_CLOUDFLARED_DOWNLOAD_URL` | GitHub latest release | URL used to download `cloudflared`. | Yes |
| `VIDEOGEN_RIFE_DIR` | `.runtime/deps/Practical-RIFE` | Custom Practical-RIFE path. | No |
| `VIDEOGEN_NAIVE_CLIP_DEVICE` | `cpu` | CLIP device for naive retrieval. | No |
| `VIDEOGEN_NAIVE_CLIP_HALF` | `0` | Use half precision for naive CLIP. | No |

## Development

Run basic checks:

```bash
python -m compileall pipeline videogen_ui videogen_full_cli.py
python videogen_full_cli.py story.json --dry-run
```

The source layout separates the Web shell from the pipeline core:

```text
videogen_ui/          Web UI routes, templates, static files, job manager.
pipeline/             API-agnostic pipeline core.
pipeline/runtime/     Persistence, runner, backends, step flow.
pipeline/agentic/     Visual element memory and reflection.
pipeline/generators/  Video API adapters.
pipeline/keyframes/   Keyframe extraction and profiles.
pipeline/media/       Reference video publishing and RIFE assembly.
videogen_full_cli.py  CLI entry point.
```

The dependency direction is:

```text
videogen_ui -> pipeline.runtime -> pipeline.{agentic,generators,media,keyframes}
videogen_full_cli.py -> pipeline.runtime
```

## Notes

- Real Seedance calls are paid. Use `--dry-run` for local pipeline testing.
- Keyframe maintenance and Smooth assembly require a CUDA GPU.
- Hugging Face models are used offline after setup; run `./setup_env.sh` again to refresh or repair the cache.
- Projects are independent and can be copied, deleted, or migrated by moving their folder under the workspace.
