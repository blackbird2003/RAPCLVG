import unittest

from storymem_web.domain import StoryValidationError, empty_story, normalize_story


class NormalizeStoryTests(unittest.TestCase):
    def test_empty_story_has_one_editable_shot(self):
        normalized = normalize_story(empty_story())

        self.assertEqual(normalized.story_name, "Untitled video project")
        self.assertEqual(len(normalized.shots), 1)
        self.assertEqual(normalized.shots[0].shot_id, "shot-0001")
        self.assertTrue(normalized.shots[0].is_cut)
        self.assertTrue(normalized.shots[0].video_prompt)

    def test_nested_story_is_flattened_in_order(self):
        story = {
            "story_name": "Test",
            "scenes": [
                {
                    "scene_num": 4,
                    "video_prompts": ["one", "two"],
                    "cut": [True, False],
                    "durations": [6, 12],
                    "first_frame_prompt": ["frame one", "frame two"],
                },
                {"scene_num": 8, "video_prompts": ["three"]},
            ],
        }

        shots = normalize_story(story).shots

        self.assertEqual([(shot.scene_num, shot.shot_num) for shot in shots], [(4, 1), (4, 2), (8, 1)])
        self.assertEqual([shot.is_cut for shot in shots], [True, False, True])
        self.assertEqual([shot.duration_seconds for shot in shots], [6, 12, None])
        self.assertEqual(shots[1].memory_query, "frame two")

    def test_invalid_parallel_array_length_is_rejected(self):
        story = {
            "scenes": [
                {"scene_num": 1, "video_prompts": ["one", "two"], "cut": [True]}
            ]
        }

        with self.assertRaisesRegex(StoryValidationError, "expected 2"):
            normalize_story(story)

    def test_non_boolean_cut_is_rejected(self):
        story = {"scenes": [{"scene_num": 1, "video_prompts": ["one"], "cut": [1]}]}

        with self.assertRaisesRegex(StoryValidationError, "true or false"):
            normalize_story(story)

    def test_duration_outside_seedance_range_is_rejected(self):
        story = {
            "scenes": [
                {"scene_num": 1, "video_prompts": ["one"], "durations": [16]}
            ]
        }

        with self.assertRaisesRegex(ValueError, "between 2 and 15"):
            normalize_story(story)


if __name__ == "__main__":
    unittest.main()
