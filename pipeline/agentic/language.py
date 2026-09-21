from __future__ import annotations

from collections.abc import Iterable


def detect_story_language(texts: Iterable[object]) -> str:
    """Return a concise primary-language label for agent natural-language output."""
    text = "\n".join(str(item or "") for item in texts)
    han_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin_count = sum(("a" <= char.lower() <= "z") for char in text)

    # Chinese scripts commonly retain English proper names. Require enough Han
    # characters that those names do not incorrectly flip the output language.
    if han_count >= 3 and han_count >= latin_count * 0.15:
        return "Chinese"
    return "English"


def output_language_instruction(language: str) -> str:
    if language == "Chinese":
        return "The detected primary script language is Chinese. Write every human-readable output field in Chinese."
    return "The detected primary script language is English. Write every human-readable output field in English."
