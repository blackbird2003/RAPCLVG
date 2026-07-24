import argparse
import os

from storymem_seedance.models import RunConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run StoryMem memory policy with Seedance2 API video generation.")
    parser.add_argument("--story_script_path", type=str, default="./story/little_prince.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_shots", type=int, default=2)
    parser.add_argument("--duration", type=int, default=-1)
    parser.add_argument("--ratio", type=str, default="16:9")
    parser.add_argument("--resolution", type=str, default=os.getenv("SEEDANCE_RESOLUTION", "720p"))
    parser.add_argument("--generate_audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--return_last_frame", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--execution_expires_after", type=int, default=86400)
    parser.add_argument("--callback_url", type=str, default=os.getenv("SEEDANCE_CALLBACK_URL"))
    parser.add_argument("--max_memory_size", type=int, default=8)
    parser.add_argument("--fix", type=int, default=3)
    parser.add_argument("--prompt_retrieval", action="store_true")
    parser.add_argument("--retrieval_top_k", type=int, default=2)
    parser.add_argument("--retrieval_frame_weight", type=float, default=0.7)
    parser.add_argument("--retrieval_video_weight", type=float, default=0.3)
    parser.add_argument("--retrieval_min_score", type=float, default=None)
    parser.add_argument("--retrieval_clip_device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--llm_memory_query", action="store_true")
    parser.add_argument("--llm_memory_query_model", type=str, default=os.getenv("MEMORY_QUERY_LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--llm_memory_query_base_url", type=str, default=os.getenv("MEMORY_QUERY_LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--enhanced_text_prompt", action="store_true")
    parser.add_argument("--seedance_model", type=str, default=os.getenv("SEEDANCE_MODEL", "doubao-seedance-2-0-260128"))
    parser.add_argument("--seedance_api_base", type=str, default=os.getenv("SEEDANCE_API_BASE", "https://ark.cn-beijing.volces.com/api/v3"))
    parser.add_argument("--poll_interval", type=int, default=10)
    parser.add_argument("--max_wait_seconds", type=int, default=1800)
    parser.add_argument("--max_reference_images", type=int, default=9)
    parser.add_argument("--skip_keyframes", action="store_true")
    parser.add_argument("--keyframe_profile", type=str, default=os.getenv("STORYMEM_KEYFRAME_PROFILE"))
    parser.add_argument("--keyframe_config_path", type=str, default=os.getenv("STORYMEM_KEYFRAME_CONFIG"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--incremental_concat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual_element_memory", action="store_true")
    parser.add_argument("--visual_element_model", type=str, default=os.getenv("VISUAL_ELEMENT_MODEL", "doubao-seed-2-1-turbo-260628"))
    parser.add_argument("--visual_element_timeout", type=int, default=90)
    parser.add_argument("--visual_element_max_parse_retries", type=int, default=1)
    parser.add_argument(
        "--visual_element_selection_mode",
        choices=["static_top_k", "greedy_coverage"],
        default="greedy_coverage",
    )
    parser.add_argument("--visual_element_sink_frame_count", type=int, default=3)
    parser.add_argument("--visual_element_max_retrieved_frames", type=int, default=4)
    parser.add_argument("--smooth_non_cut", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smooth_reference_seconds", type=float, default=1.0)
    return parser.parse_args()


def main(args=None) -> None:
    namespace = args if args is not None else parse_args()
    from storymem_seedance.runner import run_story

    run_story(RunConfig.from_namespace(namespace))


if __name__ == "__main__":
    main()
