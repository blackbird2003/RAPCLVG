# Visual Element Memory: Initial Idea

## Purpose

This note records an initial research idea for making visual memory explicit in
the Seedance long-video pipeline. Instead of treating every historical
keyframe as an undifferentiated image embedding, maintain a shared set of
visual elements that can explain what a new Shot needs from history and what
history must not leak into the new image.

The idea is intended to improve both historical-frame retrieval and precise
reference prompting. It is not yet an implementation, data model, or experiment
protocol.

## A Small Visual Vocabulary

Keep the element categories deliberately simple:

1. **Character**: an entity that has meaningful movement, pose, expression, or
   dialogue changes.
2. **Scene**: the macro background or setting that is judged from the overall
   image.
3. **Object**: a visually detectable or segmentable item in the image.

The goal is not to construct a complete visual ontology. The categories only
need to be sufficient for deciding what should be inherited from history and
what should be prevented from appearing again.

## Managing Elements

Visual elements should be named concretely enough to identify a particular
visual referent, rather than a broad class. For example, "a deep-red thorny
rose growing on the Little Prince's small asteroid" is preferable to "flower";
"a worn dark-blue canvas backpack with a brass clasp" is preferable to
"backpack". This reduces collisions between different entities that share a
generic category.

When a new video prompt is considered, an LLM should compare its described
elements with the existing set and decide whether it introduces a genuinely new
element or refers to one that is already known. The model should be conservative
about additions: same class does not imply same entity, but generic incidental
details should not create uncontrolled new memory either.

At this initial stage, the element set is derived from prompts. In the future,
VLM analysis of generated frames may identify continuity-relevant elements that
were not stated in the prompt and propose them for inclusion. That extension is
deferred because it changes the question from prompt interpretation to
open-ended visual discovery.

## Availability in a New Shot

For each new Shot, the current element set and the Shot prompt can be used to
decide an element's role:

- **Should reference**: it is needed to preserve intended visual continuity.
- **Should exclude**: the prompt makes it clear that the element should not be
  introduced into this Shot.
- **Optional or uncertain**: it may help, but the prompt establishes neither a
  requirement nor a prohibition.
- **New**: the prompt establishes a new visual element.

Exclusion should be used only with positive evidence. An element being absent
from the prompt is not enough to decide that it must not appear. This preserves
the difference between a clearly incompatible old scene or character and a
detail that is simply not mentioned.

Short-lived poses, actions, and relations should remain Shot-specific context,
not global elements. A previous frame should not be rejected merely because a
character was once watering a rose or sitting under a tree when the character
itself is still the desired reference.

## Consequences for Retrieval

This shared element view makes retrieval a compatibility question rather than
only an image/text similarity question. A historical frame is useful when it
contains elements that should be referenced, and risky when it contains
elements that should be excluded.

When reference capacity is ample, frames that provide useful required elements
can all be retained. When capacity is scarce, frames should be preferred when
they cover more needed elements and contain fewer elements that conflict with
the next Shot. Image quality, redundancy, and prompt-level semantic relevance
remain useful secondary signals.

This means that a frame containing both the Little Prince and a rose can still
be useful for character continuity, while being recognized as a poor reference
for a later scene where the rose or its original planet would be misleading.

## Consequences for Precise Reference Prompting

The same element reasoning can turn selected frames into explicit instructions
for Seedance. Rather than issuing a generic request to reference an image, the
prompt can identify which parts of a particular reference should be reused and
which visible parts should not be brought forward.

Conceptually:

```text
Reference image x comes from Shot y. Its scene a, character b, and object c
should be used as reference. Character e and object f visible in that image
should not be introduced into the new Shot.
```

This provides provenance and a concrete intended role for every submitted
reference image. It also makes failures easier to interpret: a later error may
come from selecting the wrong evidence, assigning the wrong element role, or
the video model not following an otherwise precise instruction.

## VLM Role

When analyzing a historical keyframe, a VLM should first consider the known
element set instead of performing unconstrained object detection. Its primary
question becomes: which established characters, scenes, and objects are visible
in this frame, and which of them can the frame reliably support as evidence?

That keeps labels comparable across frames and gives the retrieval mechanism a
shared vocabulary. It also leaves room for later discovery of genuinely novel
visual elements without making unbounded detection the default behavior.
