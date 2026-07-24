#!/usr/bin/env python3
"""Smoke-test the sponsor Wan2.7 R2V proxy with videogen_notebook assets."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ID = "p_20260715_033550_6ee54e8a"
DEFAULT_ENDPOINT = "https://aidp.bytedance.net/api/modelhub/online/multimodal/crawl"
DEFAULT_MODEL = "wan2.7-r2v-2026-06-12"
DEFAULT_SHOTS = ("0001", "0002", "0003")
MAX_PROMPT_CHARS = 4800


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_wan_ak(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    match = re.search(r"modelhub/online/multimodal/crawl\?ak=([^\s'\"\\]+)", text)
    if not match:
        raise RuntimeError(f"Wan AK not found in {path}")
    return match.group(1)


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def sanitize_request(request: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(request, ensure_ascii=False))
    for item in sanitized.get("input", {}).get("media", []):
        url = item.get("url")
        if isinstance(url, str) and url.startswith("data:"):
            item["url"] = f"<base64:{item.get('_source_path', 'unknown')}>"
    return sanitized


def sanitize_response(response: Any) -> Any:
    if not isinstance(response, dict):
        return response
    sanitized = json.loads(json.dumps(response, ensure_ascii=False))

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ("<omitted>" if k.lower() in {"ak", "api_key", "authorization"} else walk(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(sanitized)


def asset_path(project_dir: Path, assets: dict[str, Any], asset_id: str) -> Path:
    record = assets.get(asset_id) or {}
    rel = record.get("path")
    if not rel:
        raise RuntimeError(f"Asset not found in manifest: {asset_id}")
    path = (project_dir / rel).resolve()
    if not path.exists():
        raise RuntimeError(f"Asset file missing: {asset_id} -> {path}")
    return path


def first_produced_keyframe(shot: dict[str, Any]) -> Path | None:
    memory = (shot.get("attempt") or {}).get("produced_visual_memory") or []
    for item in memory:
        path = item.get("source_path") or item.get("media_path")
        if path and Path(path).exists():
            return Path(path)
        frame_path = (item.get("frame_annotation") or {}).get("frame_path")
        if frame_path and Path(frame_path).exists():
            return Path(frame_path)
        media_url = item.get("media_url")
        if isinstance(media_url, str) and media_url.startswith("/projects/"):
            continue
    return None


def compact_guidance(ref: dict[str, Any]) -> str:
    guidance = (ref.get("reference_guidance") or "").strip()
    if guidance:
        return " ".join(guidance.split())
    holistic = (ref.get("holistic_description") or "").strip()
    if holistic:
        return " ".join(holistic.split())
    selection = ref.get("visual_element_selection") or {}
    should = [x.get("name") for x in selection.get("should_reference") or [] if x.get("name")]
    exclude = [x.get("name") for x in selection.get("should_exclude") or [] if x.get("name")]
    parts = []
    if should:
        parts.append("参考：" + "、".join(should[:6]))
    if exclude:
        parts.append("不要引入：" + "、".join(exclude[:4]))
    return "；".join(parts) or "作为整体画面风格和视觉一致性参考。"


def build_prompt(shot: dict[str, Any], refs: list[dict[str, Any]], has_first_frame: bool) -> str:
    task = " ".join((shot.get("inputs") or {}).get("video_prompt", "").split())
    parts = [
        "请生成动画视频，不要生成真实感人脸或写实人物。",
        "",
        "当前 Shot 任务：",
        task,
    ]
    if has_first_frame:
        parts.extend(
            [
                "",
                "连续性要求：输入的 first_frame 是上一段视频的尾帧，请让新视频从这张首帧自然开始，保持场景、构图、人物位置和运动方向连续。",
            ]
        )
    if refs:
        parts.append("")
        parts.append("参考图指引：")
        for index, ref in enumerate(refs, 1):
            parts.append(f"- Image {index}: {compact_guidance(ref)}")
    prompt = "\n".join(parts)
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[: MAX_PROMPT_CHARS - 80].rstrip() + "\n（已为 Wan 5000 字符限制截断，仅保留核心任务与参考图指引。）"
    return prompt


def build_shot_request(
    project_dir: Path,
    assets: dict[str, Any],
    shot_id: str,
    previous_last_frame: Path | None,
    model: str,
    duration: int,
    resolution: str,
    ratio: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shot = read_json(project_dir / "shots" / shot_id / "shot.json")
    inputs = shot.get("inputs") or {}
    is_cut = bool(inputs.get("is_cut", True))
    selected_refs = list((shot.get("attempt") or {}).get("selected_references") or [])[:5]
    ref_records: list[dict[str, Any]] = []
    media: list[dict[str, Any]] = []

    if not is_cut and previous_last_frame:
        media.append({"type": "first_frame", "url": data_url(previous_last_frame), "_source_path": str(previous_last_frame)})

    for ref in selected_refs:
        path = Path(ref.get("source_path") or "")
        if not path.exists():
            continue
        media.append({"type": "reference_image", "url": data_url(path), "_source_path": str(path)})
        ref_records.append(ref)
        if len(ref_records) >= 5:
            break

    if not ref_records:
        fallback = first_produced_keyframe(shot)
        if not fallback:
            raw_id = (shot.get("attempt") or {}).get("outputs", {}).get("raw_video_asset_id")
            raise RuntimeError(f"Shot {shot_id} has no reference image; raw video was {raw_id}")
        ref_records.append(
            {
                "file": fallback.name,
                "source_path": str(fallback),
                "reference_guidance": "该图来自同一 Shot 的历史输出，仅作为动画风格、场景氛围和画面构成参考；请以当前 Shot 任务为准生成新视频。",
            }
        )
        media.append({"type": "reference_image", "url": data_url(fallback), "_source_path": str(fallback)})

    prompt = build_prompt(shot, ref_records, has_first_frame=(not is_cut and previous_last_frame is not None))
    request = {
        "model": model,
        "input": {"prompt": prompt, "media": media},
        "parameters": {
            "resolution": resolution,
            "ratio": ratio,
            "duration": duration,
            "prompt_extend": False,
            "watermark": False,
        },
    }
    debug = {
        "shot_id": shot_id,
        "is_cut": is_cut,
        "prompt_chars": len(prompt),
        "first_frame": str(previous_last_frame) if (not is_cut and previous_last_frame) else None,
        "reference_images": [
            {
                "image_index": i,
                "source_path": ref.get("source_path"),
                "file": ref.get("file"),
                "guidance": compact_guidance(ref),
            }
            for i, ref in enumerate(ref_records, 1)
        ],
    }
    return request, debug


def post_request(endpoint: str, ak: str, request: dict[str, Any], timeout: int) -> tuple[int, Any, str]:
    url = endpoint + "?" + urlencode({"ak": ak})
    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json", "X-TT-LOGID": f"storymem-wan27-{int(time.time())}"},
            data=json.dumps(request, ensure_ascii=False).encode("utf-8"),
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Wan proxy request failed for {endpoint}: {exc.__class__.__name__}") from None
    text = response.text
    try:
        payload = response.json()
    except ValueError:
        payload = {"raw_text": text}
    return response.status_code, payload, text


def download_video(response_payload: Any, output_path: Path) -> str | None:
    if not isinstance(response_payload, dict):
        return None

    def find_video_url(value: Any) -> str | None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"video_url", "url"} and isinstance(child, str) and child.startswith("http"):
                    return child
                found = find_video_url(child)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = find_video_url(child)
                if found:
                    return found
        return None

    video_url = find_video_url(response_payload)
    if not video_url:
        return None
    with requests.get(video_url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return video_url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", default=DEFAULT_PROJECT_ID)
    parser.add_argument("--shots", nargs="+", default=list(DEFAULT_SHOTS))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--resolution", default="720P")
    parser.add_argument("--ratio", default="16:9")
    parser.add_argument("--ak-file", type=Path, default=ROOT / "new_api.txt")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_dir = ROOT / ".runtime" / "videogen_notebook" / "projects" / args.project_id
    if not project_dir.exists():
        raise RuntimeError(f"Project not found: {project_dir}")
    assets = read_json(project_dir / "assets" / "manifest.json").get("assets", {})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / ".runtime" / "wan27_r2v_smoke" / f"{stamp}_{args.project_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ak = "" if args.dry_run else load_wan_ak(args.ak_file)
    manifest: dict[str, Any] = {
        "project_id": args.project_id,
        "shots": [],
        "endpoint": args.endpoint,
        "model": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
    }
    previous_last_frame: Path | None = None
    for shot_id in args.shots:
        request, debug = build_shot_request(
            project_dir,
            assets,
            shot_id,
            previous_last_frame,
            args.model,
            args.duration,
            args.resolution,
            args.ratio,
        )
        shot_dir = out_dir / f"shot_{shot_id}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        write_json(shot_dir / "request.sanitized.json", sanitize_request(request))
        write_json(shot_dir / "media_debug.json", debug)
        (shot_dir / "prompt.txt").write_text(request["input"]["prompt"], encoding="utf-8")
        print(f"shot {shot_id}: prompt_chars={debug['prompt_chars']} media={len(request['input']['media'])} first_frame={bool(debug['first_frame'])}")
        record = {"shot_id": shot_id, **debug, "request_path": str(shot_dir / "request.sanitized.json")}
        if not args.dry_run:
            try:
                status, payload, text = post_request(args.endpoint, ak, request, args.timeout)
            except RuntimeError as exc:
                record["error"] = str(exc)
                manifest["shots"].append(record)
                write_json(out_dir / "manifest.json", manifest)
                print(f"shot {shot_id}: {exc}")
                return 2
            write_json(shot_dir / "response.sanitized.json", sanitize_response(payload))
            (shot_dir / "response.raw.txt").write_text(text, encoding="utf-8")
            record["http_status"] = status
            record["response_path"] = str(shot_dir / "response.sanitized.json")
            if status >= 400:
                print(f"shot {shot_id}: HTTP {status}, stopping; see {shot_dir / 'response.sanitized.json'}")
                manifest["shots"].append(record)
                write_json(out_dir / "manifest.json", manifest)
                return 2
            video_url = download_video(payload, shot_dir / f"wan27_{shot_id}.mp4")
            record["downloaded_video"] = str(shot_dir / f"wan27_{shot_id}.mp4") if video_url else None
            record["video_url_present"] = bool(video_url)
            print(f"shot {shot_id}: HTTP {status}, video_url_present={bool(video_url)}")
        last_asset_id = f"img_{int(shot_id):04d}_a001_last"
        previous_last_frame = asset_path(project_dir, assets, last_asset_id)
        manifest["shots"].append(record)
        write_json(out_dir / "manifest.json", manifest)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
