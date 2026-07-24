#!/usr/bin/env python3
"""Create merged MSVBench character reference images.

For every character directory under MSVBench's characters folder, this script
creates ``merge.jpg`` from the original reference images in that directory.
The source images are not modified.  Each image is fitted into a fixed
``maxw`` x ``maxh`` cell with aspect ratio preserved, then centered on a white
canvas.  Layout is chosen from the number of images: 3 -> 2+1, 4 -> 2x2,
5 -> 3+2, 6 -> 3x2, and larger counts use a near-square grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


DEFAULT_CHARACTERS_ROOT = Path(
    "/home/wxh/world_model_projects/MSVBench/Dataset/Dataset/characters"
)
DEFAULT_REPORT_PATH = Path(
    "/home/wxh/world_model_projects/StoryMem/.runtime/msvbench_character_merge_report.json"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}


def layout_for_count(count: int) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return 1, 1
    if count == 2:
        return 2, 1
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return cols, rows


def source_images(character_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in character_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name.lower() != "merge.jpg"
    )


def fit_into_cell(image: Image.Image, maxw: int, maxh: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = min(maxw / width, maxh / height, 1.0)
    if scale >= 1.0:
        return image.copy()
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def merge_character_dir(character_dir: Path, maxw: int, maxh: int, quality: int) -> dict[str, Any]:
    images = source_images(character_dir)
    if not images:
        return {
            "character_dir": str(character_dir),
            "status": "skipped",
            "reason": "no source images",
            "source_count": 0,
        }

    cols, rows = layout_for_count(len(images))
    canvas = Image.new("RGB", (cols * maxw, rows * maxh), (255, 255, 255))
    source_sizes: list[dict[str, Any]] = []
    placed_sizes: list[dict[str, Any]] = []

    for index, path in enumerate(images):
        with Image.open(path) as raw:
            source_sizes.append({"path": str(path), "width": raw.width, "height": raw.height})
            fitted = fit_into_cell(raw, maxw, maxh)
        col = index % cols
        row = index // cols
        x = col * maxw + (maxw - fitted.width) // 2
        y = row * maxh + (maxh - fitted.height) // 2
        canvas.paste(fitted, (x, y))
        placed_sizes.append(
            {
                "path": str(path),
                "width": fitted.width,
                "height": fitted.height,
                "cell_col": col,
                "cell_row": row,
            }
        )

    output = character_dir / "merge.jpg"
    canvas.save(output, format="JPEG", quality=quality, optimize=True, progressive=True)
    return {
        "character_dir": str(character_dir),
        "status": "written",
        "output": str(output),
        "source_count": len(images),
        "layout": {"cols": cols, "rows": rows, "cell_width": maxw, "cell_height": maxh},
        "output_width": canvas.width,
        "output_height": canvas.height,
        "output_bytes": output.stat().st_size,
        "source_sizes": source_sizes,
        "placed_sizes": placed_sizes,
        "seedance_check": seedance_image_check(output, canvas.width, canvas.height),
    }


def seedance_image_check(path: Path, width: int, height: int) -> dict[str, Any]:
    ratio = width / height if height else 0.0
    size_bytes = path.stat().st_size if path.exists() else 0
    checks = {
        "format_ok": path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif", ".heic", ".heif"},
        "width_ok": 300 <= width <= 6000,
        "height_ok": 300 <= height <= 6000,
        "ratio_ok": 0.4 <= ratio <= 2.5,
        "file_size_ok": size_bytes < 30 * 1024 * 1024,
    }
    return {
        **checks,
        "ratio": ratio,
        "size_mb": size_bytes / 1024 / 1024,
        "ok": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--characters-root", type=Path, default=DEFAULT_CHARACTERS_ROOT)
    parser.add_argument("--maxw", type=int, default=1024)
    parser.add_argument("--maxh", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    if args.maxw < 300 or args.maxh < 300:
        raise ValueError("--maxw and --maxh should be at least 300 for Seedance compatibility")
    if not args.characters_root.exists():
        raise FileNotFoundError(args.characters_root)

    character_dirs = sorted(path for path in args.characters_root.glob("*/*") if path.is_dir())
    records = [
        merge_character_dir(character_dir, args.maxw, args.maxh, args.quality)
        for character_dir in character_dirs
    ]
    invalid = [
        item for item in records
        if item.get("status") == "written" and not item.get("seedance_check", {}).get("ok")
    ]
    summary = {
        "characters_root": str(args.characters_root),
        "maxw": args.maxw,
        "maxh": args.maxh,
        "quality": args.quality,
        "total_character_dirs": len(character_dirs),
        "written": sum(1 for item in records if item.get("status") == "written"),
        "skipped": sum(1 for item in records if item.get("status") == "skipped"),
        "seedance_invalid": len(invalid),
        "invalid_outputs": invalid,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"character dirs: {summary['total_character_dirs']}")
    print(f"written merge.jpg: {summary['written']}")
    print(f"skipped: {summary['skipped']}")
    print(f"seedance invalid: {summary['seedance_invalid']}")
    print(f"report: {args.report}")
    if invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
