from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from keyframe_settings import available_profiles, load_keyframe_settings


class KeyframeSettingsTests(unittest.TestCase):
    def test_default_profile_loads_expected_values(self):
        settings = load_keyframe_settings()
        self.assertEqual(settings["image_factor"], 28)
        self.assertEqual(settings["max_keyframe_num"], 6)
        self.assertEqual(settings["min_frame_similarity"], 0.95)
        self.assertEqual(settings["hpsv3_quality_threshold"], 2.5)
        self.assertFalse(settings["compare_with_history"])

    def test_named_profile_override_is_available(self):
        self.assertEqual(available_profiles(), ["default", "loose", "storymem_original", "strict"])
        settings = load_keyframe_settings(profile="strict")
        self.assertEqual(settings["max_keyframe_num"], 2)
        self.assertEqual(settings["hpsv3_quality_threshold"], 3.2)
        self.assertFalse(settings["compare_with_history"])

    def test_storymem_original_profile_matches_baseline_keyframe_policy(self):
        settings = load_keyframe_settings(profile="storymem_original")
        self.assertEqual(settings["max_keyframe_num"], 3)
        self.assertEqual(settings["min_frame_similarity"], 0.9)
        self.assertEqual(settings["hpsv3_quality_threshold"], 3.0)
        self.assertEqual(settings["video_max_pixels"], 256 * 28 * 28)
        self.assertTrue(settings["compare_with_history"])

    def test_explicit_json_path_merges_with_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "custom.json"
            path.write_text(
                json.dumps(
                    {
                        "min_frame_similarity": 0.91,
                        "max_keyframe_num": 5,
                        "compare_with_history": True,
                    }
                ),
                encoding="utf-8",
            )
            settings = load_keyframe_settings(config_path=str(path))
        self.assertEqual(settings["max_keyframe_num"], 5)
        self.assertEqual(settings["min_frame_similarity"], 0.91)
        self.assertEqual(settings["image_factor"], 28)
        self.assertTrue(settings["compare_with_history"])


if __name__ == "__main__":
    unittest.main()
