from pipeline.agentic.language import detect_story_language
from pipeline.agentic.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    build_visual_plan_reflection_prompt,
)


def test_detect_story_language_handles_english_proper_names_in_chinese_script() -> None:
    assert detect_story_language(["Mei 和 Arthur 在昏暗的书房里交谈。镜头缓慢推进。"] ) == "Chinese"
    assert detect_story_language(["Mei and Arthur speak in the dim study."]) == "English"


def test_visual_plan_prompt_requests_chinese_for_chinese_script() -> None:
    prompt = build_visual_plan_reflection_prompt(
        VisualPlanReflectionRequest(
            story_title="测试", current_shot_id="shot-0001", current_visual_plan=[],
            shots_until_current=[{"shot_id": "shot-0001", "video_prompt": "一名女子走进书房。"}],
        )
    )
    assert "detected primary script language is Chinese" in prompt
