import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import requests


DEFAULT_LLM_BASE_URL = "https://api.deepseek.com"
DEFAULT_LLM_MODEL = "deepseek-chat"


SYSTEM_PROMPT = """You generate memory retrieval queries for long-form story video generation.
The query will be used to retrieve historical keyframes, not to generate the video directly.

Return only JSON:
{"memory_query": "..."}

The memory_query must:
- Be in English.
- Be one concise visual description, 25-60 words.
- Preserve stable visual anchors: character identity, clothing, location, key objects, lighting, atmosphere.
- Remove camera motion, editing instructions, fade/zoom/pan, and newly changing actions.
- Avoid abstract plot language unless it is visually grounded."""


def _clean_api_key(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return value.strip().lstrip("\ufeff")


def _read_key_file() -> Optional[str]:
    for path in (
        os.getenv("MEMORY_QUERY_LLM_API_KEY_FILE"),
        os.getenv("DEEPSEEK_API_KEY_FILE"),
        "/root/.deepseek_api_key",
        "/root/.llm_api_key",
    ):
        if not path:
            continue
        key_path = Path(path)
        if key_path.exists():
            return key_path.read_text(encoding="utf-8-sig")
    return None


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


class MemoryQueryGenerator:
    def __init__(
        self,
        cache_path: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.api_key = _clean_api_key(
            api_key
            or os.getenv("MEMORY_QUERY_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or _read_key_file()
        )
        self.base_url = (base_url or os.getenv("MEMORY_QUERY_LLM_BASE_URL") or os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
        self.model = model or os.getenv("MEMORY_QUERY_LLM_MODEL") or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL
        self.timeout = timeout
        self.cache = self._load_cache()

    def _load_cache(self) -> dict[tuple[int, int], dict]:
        if not self.cache_path.exists():
            return {}
        cache = {}
        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (int(record["scene_num"]), int(record["shot_num"]))
                cache[key] = record
        return cache

    def get_or_generate(
        self,
        scene_num: int,
        shot_num: int,
        video_prompt: str,
        first_frame_prompt: str = "",
    ) -> str:
        key = (scene_num, shot_num)
        if key in self.cache:
            return str(self.cache[key]["memory_query"])
        if not self.api_key:
            raise RuntimeError("LLM memory query generation requires MEMORY_QUERY_LLM_API_KEY, DEEPSEEK_API_KEY, LLM_API_KEY, or OPENAI_API_KEY.")

        user_prompt = (
            "Generate a memory retrieval query for this story video shot.\n\n"
            f"Video prompt:\n{video_prompt}\n\n"
            f"First-frame prompt, if available:\n{first_frame_prompt or '(none)'}\n"
        )
        response = self._chat_completion(user_prompt)
        data = _extract_json(response)
        memory_query = str(data["memory_query"]).strip()
        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scene_num": scene_num,
            "shot_num": shot_num,
            "model": self.model,
            "video_prompt": video_prompt,
            "first_frame_prompt": first_frame_prompt,
            "memory_query": memory_query,
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.cache[key] = record
        return memory_query

    def _chat_completion(self, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        last_error = None
        for attempt in range(1, 4):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code < 400:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            except Exception as exc:
                last_error = exc
            if attempt < 3:
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"LLM memory query generation failed: {last_error}") from last_error
