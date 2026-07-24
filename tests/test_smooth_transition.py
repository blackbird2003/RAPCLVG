import tempfile
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

from storymem_web.reference_video import extract_video_tail
from storymem_web.smooth_transition import (
    SmoothTransitionError,
    SmoothVideoAssembler,
    _mux_concatenated_audio,
    crop_to_shape,
    padded_size,
)


class FakeInterpolator:
    metadata = {"implementation": "fake-rife", "model_version": "test"}

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def interpolate(self, first, second, timesteps):
        self.calls.append((first.copy(), second.copy(), tuple(timesteps)))
        if self.fail:
            raise RuntimeError("synthetic RIFE failure")
        return [
            np.full_like(first, 90),
            np.full_like(first, 170),
        ]


def write_video(path, values, *, fps=6.0, size=(64, 48)):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV test video writer unavailable")
    try:
        for value in values:
            writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    finally:
        writer.release()


def metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        return (
            float(capture.get(cv2.CAP_PROP_FPS)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def frame_means(path):
    capture = cv2.VideoCapture(str(path))
    values = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            values.append(float(frame.mean()))
    finally:
        capture.release()
    return values


class SmoothTransitionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def clip(self, name, values, mode, attempt_id):
        path = self.root / f"{name}.mp4"
        write_video(path, values)
        attempt_dir = self.root / f"attempt-{attempt_id}"
        attempt_dir.mkdir()
        return {
            "attempt_id": attempt_id,
            "output_video": str(path),
            "generation_mode": mode,
            "attempt_dir": str(attempt_dir),
        }

    def test_tail_extraction_preserves_source_rate_size_and_one_second(self):
        source = self.root / "source.mp4"
        output = self.root / "tail.mp4"
        write_video(source, range(0, 240, 20), fps=6.0)

        extract_video_tail(source, output)

        self.assertEqual(metadata(output), (6.0, 6, 64, 48))
        values = frame_means(output)
        self.assertLess(abs(values[0] - 120), 8)
        self.assertLess(abs(values[-1] - 220), 8)

    def test_padding_and_crop_for_720p(self):
        self.assertEqual(padded_size(720, 1280), (768, 1280))
        padded = np.zeros((768, 1280, 3), dtype=np.uint8)
        self.assertEqual(crop_to_shape(padded, 720, 1280).shape, (720, 1280, 3))

    def test_one_smooth_boundary_preserves_total_frame_count_and_is_idempotent(self):
        clips = [
            self.clip("a", [10, 20, 30, 40, 50], "default", "a"),
            self.clip("b", [60, 70, 80, 90, 100], "smooth", "b"),
        ]
        interpolator = FakeInterpolator()
        assembler = SmoothVideoAssembler(interpolator)
        output = self.root / "one-boundary.mp4"

        assembler(clips, str(output))
        assembler(clips, str(self.root / "one-boundary-retry.mp4"))

        self.assertEqual(metadata(output)[1], 10)
        self.assertEqual(len(interpolator.calls), 1)
        left, right, timesteps = interpolator.calls[0]
        self.assertLess(abs(float(left.mean()) - 40), 8)
        self.assertLess(abs(float(right.mean()) - 70), 8)
        self.assertEqual(timesteps, (1.0 / 3.0, 2.0 / 3.0))
        report = self.root / "attempt-b" / "smooth_transition" / "metadata.json"
        self.assertIn('"status": "completed"', report.read_text())

    def test_multiple_smooth_boundaries_trim_both_ends_of_middle_clip(self):
        clips = [
            self.clip("a", [10, 20, 30, 40, 50], "default", "a"),
            self.clip("b", [60, 70, 80, 90, 100], "smooth", "b"),
            self.clip("c", [110, 120, 130, 140, 150], "smooth", "c"),
        ]
        interpolator = FakeInterpolator()
        output = self.root / "two-boundaries.mp4"

        SmoothVideoAssembler(interpolator)(clips, str(output))

        self.assertEqual(metadata(output)[1], 15)
        self.assertEqual(len(interpolator.calls), 2)
        self.assertLess(abs(float(interpolator.calls[1][0].mean()) - 90), 8)
        self.assertLess(abs(float(interpolator.calls[1][1].mean()) - 120), 8)

    def test_rife_failure_records_error_and_preserves_raw_clips(self):
        clips = [
            self.clip("a", [10, 20, 30], "default", "a"),
            self.clip("b", [40, 50, 60], "smooth", "b"),
        ]
        source_bytes = [Path(item["output_video"]).read_bytes() for item in clips]

        with self.assertRaises(SmoothTransitionError):
            SmoothVideoAssembler(FakeInterpolator(fail=True))(
                clips, str(self.root / "failed.mp4")
            )

        self.assertEqual(
            [Path(item["output_video"]).read_bytes() for item in clips], source_bytes
        )
        report = self.root / "attempt-b" / "smooth_transition" / "metadata.json"
        self.assertIn('"status": "failed"', report.read_text())

    def test_audio_mux_replaces_encoded_video_when_audio_exists(self):
        encoded = self.root / "encoded.mp4"
        source_a = self.root / "source-a.mp4"
        source_b = self.root / "source-b.mp4"
        encoded.write_bytes(b"video-only")
        source_a.write_bytes(b"a")
        source_b.write_bytes(b"b")
        muxed = encoded.with_suffix(encoded.suffix + ".muxed.mp4")

        def fake_run(command, check=False, capture_output=False, text=False):
            muxed.write_bytes(b"video-with-audio")
            return mock.Mock(returncode=0, stderr="", stdout="")

        with mock.patch(
            "storymem_web.smooth_transition._video_has_audio_stream",
            return_value=True,
        ), mock.patch(
            "storymem_web.smooth_transition._ffmpeg_executable",
            return_value="/tmp/fake-ffmpeg",
        ), mock.patch(
            "storymem_web.smooth_transition.subprocess.run",
            side_effect=fake_run,
        ):
            _mux_concatenated_audio(encoded, [source_a, source_b])

        self.assertEqual(encoded.read_bytes(), b"video-with-audio")
        self.assertFalse(muxed.exists())


if __name__ == "__main__":
    unittest.main()
