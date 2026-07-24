# Seam Interpolation Experiment

## Goal

Test whether frame interpolation removes the brief pause at the Little Prince
plain-extension seam. Compare FFmpeg motion-compensated interpolation with RIFE
under one fixed, constant-duration protocol.

## Status

- FFmpeg: completed. The constant-duration result is a promising candidate;
  sampled seam frames show no obvious ghosting or hard pause.
- RIFE: completed after padding 1280x720 anchors to 1280x768 for model inference
  and cropping generated frames back to 1280x720.

Both outputs contain 386 frames at 24 fps. FFmpeg produced higher adjacent-frame
SSIM (0.927012 versus 0.883113) but substantially lower flow magnitude through
the inserted frames. RIFE kept inserted-frame DIS flow near the incoming motion
speed (roughly 1.80-1.98 versus 1.92), at the cost of mildly softened character
and rose details. Human review found RIFE smoother and found visible ghosting at
the FFmpeg seam. Use RIFE as the provisional default and retain FFmpeg as the
low-dependency fallback.

Practical-RIFE 4.25 accepts arbitrary timesteps. For `n` intermediate frames,
generate `t = k/(n+1)` for `k=1..n`; runtime grows approximately linearly. Keep
the default transition short (currently three frames) because excessive frames
create slow motion and may accumulate softness without adding motion evidence.

Follow-up human review selected RIFE over FFmpeg because the FFmpeg seam retained
visible ghosting. A two-frame RIFE replacement also looked acceptable and is the
provisional implementation choice: remove `A[-1]` and `B[0]`, use `A[-2]/B[1]`
as anchors, and generate `t=1/3,2/3`. This simpler rule preserves duration and
does not depend on a frame-search algorithm.

## Fixed Inputs

- First clip: shot 1 referenced by
  `.runtime/experiments/little_prince_video_extension/experiment_report.json`.
- Second clip: `shot2_video_extension.mp4` from the same experiment directory.
- Both clips: 1280x720, 24 fps, 193 frames, no audio in the comparison output.
- Left anchor: shot 1 frame `A191`.
- Right anchor: shot 2 frame `B2`.

## Replacement Protocol

Replace the original interior frames `A192, B0, B1` with exactly three
interpolated frames at `t = 0.25, 0.50, 0.75` between `A191` and `B2`:

```text
Before: ..., A190, A191, A192, B0, B1, B2, B3, ...
After:  ..., A190, A191, I.25, I.50, I.75, B2, B3, ...
```

The result must remain 386 frames at 24 fps (about 16.083 seconds). Do not
duplicate either anchor and do not interpolate the rest of either source clip.

## Experiment A: FFmpeg

- Use the bundled FFmpeg `minterpolate` filter.
- Prefer motion-compensated interpolation with bidirectional estimation,
  adaptive overlapped block compensation, EPZS search, and variable-size block
  compensation when supported.
- Keep all implementation and outputs independent from the RIFE experiment.

## Experiment B: RIFE

- Deploy Practical-RIFE outside tracked source and without modifying the shared
  project environment globally.
- Prefer the official recommended general model; record the exact repository
  revision and model version actually used.
- Use GPU 0 and 4x interpolation to obtain the three intermediate frames.

## Required Outputs

Each experiment must provide:

- a playable constant-duration MP4;
- the three interpolated PNG frames and a contact sheet;
- a JSON report with commands, versions, runtime, output metadata, and errors;
- frame-to-frame SSIM and mean DIS-flow magnitude for
  `A190->A191->I.25->I.50->I.75->B2->B3`;
- concise qualitative notes on pause reduction, warping, ghosting, character
  deformation, rose deformation, and background flicker.

Do not modify the production pipeline or Web UI. Generated assets belong under
`.runtime/experiments/little_prince_video_extension/interpolation_<method>/`.
