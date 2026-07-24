import tempfile
import unittest
from unittest.mock import patch

from prompt_retrieval import RetrievalHit, build_prompt_aware_memory_bank
from storymem_seedance.models import RunConfig, ShotSpec
from storymem_seedance.runner import select_memory


class MemoryStore:
    def __init__(self, paths):
        self.paths = paths

    def memory_keyframes(self):
        return list(self.paths)


def hit(path, score):
    return RetrievalHit(
        path=path,
        frame_score=score,
        video_score=score,
        score=score,
        source_prompt="source",
    )


class MemoryPolicyTests(unittest.TestCase):
    def test_controlled_budget_deduplicates_and_orders_sink_retrieved_recent(self):
        paths = [f"/tmp/01_{index:02d}_keyframe0.jpg" for index in range(10)]
        ranked = [hit(paths[5], 0.9), hit(paths[3], 0.8), hit(paths[4], 0.7)]

        with tempfile.TemporaryDirectory() as output, patch(
            "prompt_retrieval.rank_prompt_aware_keyframes", return_value=ranked
        ):
            bank, selected = build_prompt_aware_memory_bank(
                all_memory=paths,
                current_prompt="current",
                memory_query="query",
                story_script={"scenes": []},
                max_memory_size=5,
                fix=2,
                retrieval_top_k=2,
                retrieval_log_path=f"{output}/retrieval.jsonl",
                explicit_policy=True,
                include_sink=True,
                include_recent=True,
            )

        self.assertEqual(bank, [paths[0], paths[1], paths[3], paths[5], paths[9]])
        self.assertEqual([item.path for item in selected], [paths[5], paths[3]])
        self.assertEqual(len(bank), len(set(bank)))

    def test_retrieve_runs_for_non_cut_shot_and_assigns_retrieved_role(self):
        paths = [f"/tmp/01_0{index + 1}_keyframe0.jpg" for index in range(4)]
        selected = hit(paths[3], 0.9)
        shot = ShotSpec(
            scene={"scene_num": 1, "video_prompts": ["continue"]},
            scene_num=1,
            shot_num=1,
            prompt="continue",
            is_cut=False,
        )
        config = RunConfig(output_dir="/tmp", max_memory_size=4, fix=1, retrieval_top_k=2)

        with patch(
            "storymem_seedance.runner.build_prompt_aware_memory_bank",
            return_value=([paths[0], paths[3]], [selected]),
        ) as build:
            records = select_memory(
                config=config,
                store=MemoryStore(paths),
                story_script={"scenes": []},
                shot=shot,
                memory_query_generator=None,
                memory_policy={"sink": True, "retrieve": True, "recent": False},
            )

        self.assertTrue(build.called)
        self.assertTrue(build.call_args.kwargs["explicit_policy"])
        self.assertFalse(build.call_args.kwargs["include_recent"])
        self.assertEqual(records[0]["roles"], ["early_sink_memory"])
        self.assertEqual(records[1]["roles"], ["prompt_retrieved_memory"])

    def test_disabled_sources_are_absent(self):
        paths = [f"/tmp/01_0{index + 1}_keyframe0.jpg" for index in range(4)]
        shot = ShotSpec(
            scene={"scene_num": 1, "video_prompts": ["continue"]},
            scene_num=1,
            shot_num=1,
            prompt="continue",
            is_cut=True,
        )
        config = RunConfig(output_dir="/tmp", max_memory_size=3, fix=1)

        with patch("storymem_seedance.runner.build_prompt_aware_memory_bank") as build:
            records = select_memory(
                config=config,
                store=MemoryStore(paths),
                story_script={"scenes": []},
                shot=shot,
                memory_query_generator=None,
                memory_policy={"sink": False, "retrieve": False, "recent": True},
            )

        build.assert_not_called()
        self.assertTrue(records)
        self.assertTrue(
            all(item["roles"] == ["recent_window_memory"] for item in records)
        )


if __name__ == "__main__":
    unittest.main()
