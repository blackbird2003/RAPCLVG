EntityBench 论文中总共是 **51 个细分指标**：

| Pillar | 含义 | 指标数 |
|---|---|---:|
| Pillar 1 | Intra-shot quality / 单镜头视频质量 | 6 |
| Pillar 2 | Intra-shot prompt following / 单镜头 prompt 对齐 | 24 |
| Pillar 3 | Cross-shot consistency / 跨镜头一致性 | 21 |
| 总计 |  | 51 |

**完整 51 指标构成**

Pillar 1：6 个 VBench 指标  
`subject_consistency`, `temporal_flickering`, `aesthetic_quality`, `imaging_quality`, `motion_smoothness`, `dynamic_degree`

Pillar 2：24 个 prompt-following 指标  
- Presence 3 个：`intra_character_presence`, `intra_object_presence`, `intra_location_presence`
- Character fidelity 5 个：`intra_face_fidelity`, `intra_face_face`, `intra_face_hair`, `intra_face_clothing`, `intra_face_build`
- Object fidelity 5 个：`intra_object_fidelity`, `intra_object_shape`, `intra_object_color_texture`, `intra_object_proportions`, `intra_object_details`
- Location fidelity 5 个：`intra_location_fidelity`, `intra_location_layout`, `intra_location_color_mood`, `intra_location_landmarks`, `intra_location_perspective`
- Action fidelity 6 个：`intra_action_overall`, `intra_action_depicted`, `intra_action_subject_identity`, `intra_action_subject_action`, `intra_action_object_interaction`, `intra_action_motion_quality`

Pillar 3：21 个跨镜头一致性指标  
- DINOv2 embedding 3 个：`cs_face`, `cs_object`, `cs_transition_boundary`
- LLM characters 6 个：`llm_face_accuracy`, `llm_face_mean_score`, `llm_face_face`, `llm_face_hair`, `llm_face_clothing`, `llm_face_build`
- LLM objects 6 个：`llm_object_accuracy`, `llm_object_mean_score`, `llm_object_shape`, `llm_object_color_texture`, `llm_object_proportions`, `llm_object_details`
- LLM scenes/locations 6 个：`llm_scene_accuracy`, `llm_scene_mean_score`, `llm_scene_layout`, `llm_scene_color_mood`, `llm_scene_landmarks`, `llm_scene_perspective`

**论文中主表详细展示了哪些**

主文 **Table 4** 不是展示全部 51 个，而是展示代表性指标，约 24 个：

- P1 展示：`imaging_quality`, `aesthetic_quality`, `motion_smoothness`
- P2 展示：
  - `char_presence`, `obj_presence`, `loc_presence`
  - `face_fidelity`, `object_fidelity`, `location_fidelity`
  - `action_overall`, `action_subject`, `action_interaction`
- P3 展示：
  - `cs_face`, `cs_object`, `cs_transition_boundary`
  - `llm_face_accuracy`, `llm_face_mean_score`, `llm_face_face`
  - `llm_object_accuracy`, `llm_object_mean_score`, `llm_object_shape`
  - `llm_scene_accuracy`, `llm_scene_mean_score`, `llm_scene_layout`

论文正文说明：**完整 51-metric results 在 Appendix F.1**。也就是说，主文 Table 4 用代表性指标讲主要结论，附录 F.1 才是完整细分指标表。