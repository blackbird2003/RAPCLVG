from __future__ import annotations

import json
import unittest

from storymem_seedance.visual_plan_reflection import (
    VisualPlanReflectionRequest,
    VisualElementMemoryError,
    apply_visual_plan_reflection,
    build_visual_plan_reflection_prompt,
    parse_visual_plan_reflection,
)


class VisualPlanReflectionTests(unittest.TestCase):
    def test_prompt_mentions_fixed_id_and_introduced_constraints(self):
        request = VisualPlanReflectionRequest(
            story_title="Demo",
            current_shot_id="0002",
            shots_until_current=[
                {
                    "shot_id": "0001",
                    "scene_num": 1,
                    "shot_num": 1,
                    "video_prompt": "小猫坐在客厅。",
                    "is_cut": True,
                    "generation_mode": "default",
                },
                {
                    "shot_id": "0002",
                    "scene_num": 1,
                    "shot_num": 2,
                    "video_prompt": "小猫走到餐桌旁。",
                    "is_cut": False,
                    "generation_mode": "smooth",
                },
            ],
            current_visual_plan=[
                {
                    "element_id": "element-0001",
                    "name": "小猫",
                    "type": "character",
                    "introduced_at": "0001",
                    "notes": "",
                    "status": "should_reference",
                    "reason": "核心角色",
                }
            ],
        )

        prompt = build_visual_plan_reflection_prompt(request)

        self.assertIn("element_id 是固定信息，绝对不能修改", prompt)
        self.assertIn("introduced_at 是固定信息，绝对不能修改", prompt)
        self.assertIn("从第一个 Shot 到当前 Shot 的剧本", prompt)
        self.assertIn("小猫走到餐桌旁", prompt)

    def test_parse_and_apply_reflection_preserves_fixed_fields(self):
        source = [
            {
                "element_id": "element-0001",
                "name": "人物",
                "type": "character",
                "introduced_at": "0001",
                "notes": "",
                "status": "optional_or_uncertain",
                "reason": "不确定",
            }
        ]
        raw = json.dumps(
            {
                "summary": "建议细化名称和状态。",
                "rows": [
                    {
                        "element_id": "element-0001",
                        "action": "modify",
                        "change_fields": ["name", "status", "notes", "reason"],
                        "suggested": {
                            "name": "小猫阿虎",
                            "type": "character",
                            "status": "should_reference",
                            "notes": "虎皮猫，圆脸，粗尾巴。",
                            "reason": "当前镜头核心角色，需要保持形象一致。",
                        },
                        "rationale": "原名称过泛。",
                    }
                ],
                "warnings": [],
            },
            ensure_ascii=False,
        )

        parsed = parse_visual_plan_reflection(raw, source)
        revised = apply_visual_plan_reflection(source, parsed["rows"])

        self.assertEqual(parsed["rows"][0]["change_fields"], ["name", "status", "notes", "reason"])
        self.assertEqual(revised[0]["element_id"], "element-0001")
        self.assertEqual(revised[0]["introduced_at"], "0001")
        self.assertEqual(revised[0]["name"], "小猫阿虎")
        self.assertEqual(revised[0]["status"], "should_reference")

    def test_parse_rejects_non_core_type(self):
        source = [
            {
                "element_id": "element-0001",
                "name": "客厅",
                "type": "scene",
                "introduced_at": "0001",
                "notes": "",
                "status": "should_reference",
                "reason": "当前地点",
            }
        ]
        raw = json.dumps(
            {
                "summary": "bad type",
                "rows": [
                    {
                        "element_id": "element-0001",
                        "action": "modify",
                        "change_fields": ["type"],
                        "suggested": {
                            "name": "客厅",
                            "type": "location",
                            "status": "should_reference",
                            "notes": "",
                            "reason": "当前地点",
                        },
                        "rationale": "bad",
                    }
                ],
                "warnings": [],
            }
        )

        with self.assertRaises(VisualElementMemoryError):
            parse_visual_plan_reflection(raw, source)


if __name__ == "__main__":
    unittest.main()
