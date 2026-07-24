# 组会讨论提纲：Retrieval-Augmented Multimodal Prompting

## 1. 当前论文思路

### 暂定标题

**Retrieval-Augmented Multimodal Prompting for Consistent Long-Form Video Generation**

### 当前 Abstract 核心

短视频生成模型已经能生成质量较高的单段视频，但将多个短片段组合成长视频时，仍然容易出现跨镜头角色、物体、场景不一致，以及错误引用历史元素的问题。

我们提出一种 **retrieval-augmented multimodal prompting** 框架，不训练或修改底层视频生成模型，而是通过 agent 动态维护结构化视觉证据库，为每个 shot 检索互补的历史参考帧，并将文本、图像、视频参考组织为具有明确指代关系的多模态 prompt。

核心思想是：不再把历史帧当作被动 memory buffer，而是像 RAG 一样构建可检索、可过滤、可解释的视觉证据库。

### 论文结构

当前 `paper.tex` 中的主要结构为：

1. **Introduction**  
   TODO：需要补充问题背景、现有 memory 方法局限、RAG-style reference construction 的动机和贡献。

2. **Related Work**  
   已有初稿，覆盖：
   - reference-conditioned / multimodal video generators；
   - multi-shot / long-form video generation；
   - visual memory and retrieval；
   - agentic planning and context management；
   - multi-shot consistency benchmarks。

3. **Methods**  
   已有初稿，主要包括：
   - Problem Setup；
   - Pipeline Overview；
   - Text-Grounded Visual Element Registry；
   - VLM-Annotated Keyframe Library；
   - Target-Covered Reference Retrieval；
   - Grounded Multimodal Prompt Construction；
   - Generation, Assembly, and Memory Update；
   - Human and Agentic Quality Control。

4. **Experiments**  
   已有骨架和当前阶段性结果表，主要基于 EntityBench。

5. **Conclusion**  
   TODO。

### 主要方法逻辑

方法可以概括为一个 RAG-style 长视频生成循环：

1. **视觉元素集合维护**  
   LLM 根据剧本和当前 shot prompt，维护角色、场景、物体三类视觉元素，并为当前 shot 判断每个元素状态：
   - should-reference；
   - should-exclude；
   - optional-or-uncertain；
   - new。

2. **关键帧库构建**  
   每个已生成 shot 会抽取代表性关键帧，并由 VLM 对当前视觉元素集合中的元素进行闭集标注，得到元素是否出现、bbox、reference quality、holistic description 等信息。

3. **目标覆盖式参考检索**  
   对当前 shot，系统从历史关键帧库中选择有限数量参考帧。Greedy Coverage 优先选择能覆盖当前 should-reference 元素、彼此互补、且尽量少引入 should-exclude 元素的帧。

4. **可解释多模态 prompt 组织**  
   选中的参考帧会被写入 prompt，并明确说明：
   - 该参考图来自哪个 shot；
   - 应参考其中哪些角色/物体/场景；
   - 不应引入其中哪些历史元素；
   - 该图在当前生成中的作用。

5. **短视频生成与记忆更新**  
   Seedance 生成当前 shot 视频；生成后再次抽帧、VLM 标注，并将新关键帧加入后续检索库。

核心贡献点可以表述为：

- 将 long-form video generation 转化为 retrieval-augmented multimodal prompting；
- 构建文本侧视觉元素集合 + 视觉侧 VLM 标注关键帧库；
- 提出目标覆盖式参考帧选择，使参考帧选择更互补、更可解释；
- 用 grounded prompt 明确每张参考图的使用方式，减少错误历史元素泄漏。

## 2. 当前实验方法与结果

### Benchmark

当前主要使用 **EntityBench** 进行评测。EntityBench 面向多镜头长视频生成，包含角色、物体、地点等实体一致性评测。

当前使用 1/10 子集：

- Easy: 8 episodes
- Medium: 4 episodes
- Hard: 2 episodes
- 合计 14 episodes / 243 shots

由于 Seedance API 版权与真人审核限制，我们对可测剧本做了筛选，并在生成 prompt 中加入动画风格约束。

### 指标

EntityBench 包含三个 Pillar：

1. **Pillar 1**：单 shot 基础视频质量；
2. **Pillar 2**：shot 内 prompt/entity/action 对齐；
3. **Pillar 3**：跨 shot 实体一致性。

我们的核心关注是 **Pillar 3**，因为它最直接对应长视频跨镜头一致性。

### 当前主结果

当前 `Ours current 14` 在 14-episode 子集上的 P3 结果为：

| Metric | Score |
|---|---:|
| CS Face | 0.8151 |
| CS Object | 0.8830 |
| CS Boundary | 0.8004 |
| LLM Face Acc | 0.9001 |
| LLM Face Score | 0.8443 |
| LLM Object Acc | 0.9506 |
| LLM Object Score | 0.8617 |
| LLM Scene Acc | 0.9833 |
| LLM Scene Score | 0.8900 |

阶段性看，P3 结果较突出，说明“视觉元素集合 + VLM 标注关键帧库 + 覆盖式检索 + 指代 prompt”的路线对跨镜头实体一致性有帮助。

### 当前消融结果

在 Easy8 + Medium4 上，三种设置均已完成，范围一致：

- Greedy：目标覆盖式选帧；
- TopK：静态 Top-K 选帧；
- NoRef：不使用历史参考图。

平均结果：

| Method | Scope | P1 Avg | P2 Avg | P3 Avg | Overall Avg |
|---|---:|---:|---:|---:|---:|
| Greedy | Easy8+Mid4 | 0.7848 | 0.7523 | 0.8906 | 0.8092 |
| TopK | Easy8+Mid4 | 0.7830 | 0.7466 | 0.8675 | 0.7990 |
| NoRef | Easy8+Mid4 | 0.7819 | 0.7544 | 0.7509 | 0.7624 |

主要观察：

- Greedy 总体平均最高，优势主要来自 P3；
- TopK 接近 Greedy，但 P3 稍低；
- NoRef 在部分单 shot 对齐指标上不差，但跨镜头实体一致性明显下降；
- 这支持“检索参考帧与 grounded prompt 主要改善跨 shot 一致性，而不一定显著提升单 shot 质量”的判断。

## 3. 需要讨论的问题

### 3.1 实验设置与公平性问题

#### 1. Seedance API 审核限制

当前实验受到 Seedance API 版权审核与真人/人脸审核限制：

- EntityBench 中只有部分剧本能稳定生成；
- 可生成剧本也通常需要加入强制动画风格 prompt；
- 这使我们的实验分布与原始 EntityBench 设置存在差异。

需要讨论：

- 论文中是否应明确限定为 animation-style long-form generation？
- 是否需要单独说明这是为了规避商业 API safety filter，而非方法本身限制？
- 是否需要补充少量非 EntityBench 自建剧本作为 qualitative demo？

#### 2. VLM Judge 不一致

EntityBench 论文使用 Gemini 2.5 Pro 作为 VLM judge，而我们当前评测使用 Seed2.1 Turbo。

需要讨论：

- 是否必须尽量复现 Gemini 2.5 Pro judge？
- 如果无法使用 Gemini，论文中应如何表述这一差异？
- 是否需要对少量样本做 Gemini / Seed2.1 Turbo 的 judge agreement 检查？

#### 3. EntityBench 的人工实体计划表问题

EntityBench 本身包含人工标注的 entity schedule，并且原始 pipeline 会将其转化为输入 prompt。相比之下，我们的方法没有使用这部分人工实体计划表，而是由算法自己从 prompt 中维护视觉元素集合与状态表，并以此作为检索基础。

这带来一个复杂点：

- 如果不用 EntityBench entity schedule，我们的方法更符合真实自动 pipeline，但在 presence / fidelity 上可能吃亏；
- 如果使用 entity schedule，则可以更公平地比较生成器能力，但会削弱“算法自动维护视觉元素集合”的贡献；
- baseline 如果使用人工 entity schedule，而我们不用，实验对我们不利；
- 但如果我们也用，方法贡献边界会变得不清楚。

需要讨论：

- 主实验是否坚持不使用 EntityBench entity schedule，只把它作为 evaluation ground truth？
- 是否增加一个 oracle setting：使用 EntityBench entity schedule 作为上界或辅助对照？
- 论文中如何清楚说明：我们维护的 visual element registry 是方法的一部分，不是 benchmark 提供的输入？

### 3.2 Demo 制作细节

#### 1. 声音一致性

当前方法主要关注视觉一致性，但如果要制作更完整的 Demo，有声视频会引入新的跨镜头一致性问题：

- 同一角色跨 shot 的声线是否一致；
- 旁白或角色对白是否连续；
- 背景音乐/环境音是否稳定；
- Smooth 或拼接边界处音频是否自然。

可能需要扩展“记忆”概念到音频侧：

- 为角色维护 voice identity / voice profile；
- 记录每个角色的声线、语速、情绪、口音；
- 生成 prompt 中显式说明角色声线保持一致；
- 必要时使用外部 TTS / voice cloning 管线，而不是完全依赖 Seedance 内置音频。

需要讨论：

- 当前投稿是否只关注视觉一致性，将声音一致性作为 limitation / future work？
- Demo 是否需要关闭角色语音，只保留背景音或旁白？
- 如果保留声音，是否需要设计最小版 voice memory？

## 4. 希望组会上形成的决策

1. 论文标题和主叙事是否采用当前版本：  
   **Retrieval-Augmented Multimodal Prompting for Consistent Long-Form Video Generation**

2. 实验主线是否以 EntityBench P3 为核心，并将 P1/P2 作为辅助分析。

3. 是否接受当前动画风格约束，还是需要补充其他类型剧本。

4. VLM judge 是否继续使用 Seed2.1 Turbo，或尝试补充 Gemini 2.5 Pro 小规模对照。

5. EntityBench entity schedule 是否完全不用、作为 oracle ablation 使用，或部分用于辅助分析。

6. Demo 是否只展示视觉一致性，声音一致性暂列 future work。
