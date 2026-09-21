from hpsv3 import HPSv3RewardInferencer
import os
import time
import glob
import io
import math
import cv2
import decord
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from typing import Any, Dict, Optional, Tuple, List
# from utils.image_quality_score import is_low_quality

import logging
from .settings import load_keyframe_settings

logger = logging.getLogger()

DEFAULT_KEYFRAME_SETTINGS = load_keyframe_settings()
IMAGE_FACTOR = DEFAULT_KEYFRAME_SETTINGS["image_factor"]
MIN_TOKENS = DEFAULT_KEYFRAME_SETTINGS["min_tokens"]
MAX_TOKENS = DEFAULT_KEYFRAME_SETTINGS["max_tokens"]
MIN_PIXELS = DEFAULT_KEYFRAME_SETTINGS["min_pixels"]
MAX_PIXELS = DEFAULT_KEYFRAME_SETTINGS["max_pixels"]
MAX_RATIO = DEFAULT_KEYFRAME_SETTINGS["max_ratio"]
VIDEO_MIN_TOKENS = DEFAULT_KEYFRAME_SETTINGS["video_min_tokens"]
VIDEO_MAX_TOKENS = DEFAULT_KEYFRAME_SETTINGS["video_max_tokens"]
VIDEO_MIN_PIXELS = DEFAULT_KEYFRAME_SETTINGS["video_min_pixels"]
VIDEO_MAX_PIXELS = DEFAULT_KEYFRAME_SETTINGS["video_max_pixels"]
VIDEO_TOTAL_PIXELS = DEFAULT_KEYFRAME_SETTINGS["video_total_pixels"]
MIN_FRAME_SIMILARITY = DEFAULT_KEYFRAME_SETTINGS["min_frame_similarity"]
MAX_KEYFRAME_NUM = DEFAULT_KEYFRAME_SETTINGS["max_keyframe_num"]
ADAPTIVE_ALPHA = DEFAULT_KEYFRAME_SETTINGS["adaptive_alpha"]
HPSV3_QUALITY_THRESHOLD = DEFAULT_KEYFRAME_SETTINGS["hpsv3_quality_threshold"]

def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def smart_resize(
        height: int, width: int,
        factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        max_ratio: int = MAX_RATIO) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return max(h_bar, factor), max(w_bar, factor)

def get_frame_sim(frame1, frame2,
                  patch_size: int=28,
                  threshold: float = 0.7,
                  epsilon: float=1e-8):
    assert frame1.dim() == 3 and frame2.dim() == 3, "Input must be a 3-D tensor [C, H, W]"

    def to_numpy_cvt(tensor):
        tensor = tensor.cpu().permute(1, 2, 0).numpy()
        if tensor.dtype == np.float32 or tensor.dtype == np.float64:
            tensor = (tensor).astype(np.uint8)
        return cv2.cvtColor(tensor, cv2.COLOR_RGB2HSV)

    frame1_hsv = to_numpy_cvt(frame1)
    frame2_hsv = to_numpy_cvt(frame2)

    frame1_tensor = torch.from_numpy(frame1_hsv).permute(2, 0, 1).to(frame1.device).float()
    frame2_tensor = torch.from_numpy(frame2_hsv).permute(2, 0, 1).to(frame2.device).float()

    patch1 = rearrange(
        frame1_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()
    patch2 = rearrange(
        frame2_tensor, "c (h p1) (w p2) -> h w (c p1 p2)", p1=patch_size, p2=patch_size).float()

    norm1 = torch.norm(patch1, p=2, dim=-1, keepdim=True) + epsilon
    norm2 = torch.norm(patch2, p=2, dim=-1, keepdim=True) + epsilon

    normalized1 = patch1 / norm1
    normalized2 = patch2 / norm2
    cos_sim = (normalized1 * normalized2).sum(dim=-1)

    zero_vector_mask = (norm1.squeeze() < 0.01) & (norm2.squeeze() < 0.01)

    similar = torch.ones_like(cos_sim)

    non_zero_mask = ~zero_vector_mask
    similar[non_zero_mask] = (cos_sim[non_zero_mask] > threshold).float()

    return similar[non_zero_mask].float().mean().item()

# Singleton cache
class _CLIPCtx:
    model = None
    device = None
    dtype = torch.float32

def _get_clip_model(device="cpu", use_half=False):
    if _CLIPCtx.model is not None:
        return _CLIPCtx.model, _CLIPCtx.device, _CLIPCtx.dtype

    dev = device # or ("cuda:0" if torch.cuda.is_available() else "cpu")
    import clip  # from openai/CLIP

    model, _ = clip.load("ViT-B/32", device=dev, jit=False)  # 224 input
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dt = torch.float16 if (use_half and dev != "cpu") else torch.float32
    if dt == torch.float16:
        model = model.half()

    _CLIPCtx.model, _CLIPCtx.device, _CLIPCtx.dtype = model, dev, dt
    return model, dev, dt

class _HPSv3Ctx:
    model = None
    device = None

def _get_quality_model(device="cuda"):
    if _HPSv3Ctx.model is not None and _HPSv3Ctx.device == device:
        return _HPSv3Ctx.model

    model = HPSv3RewardInferencer(device=device)
    if hasattr(model, "model"):
        model.model.eval()
        for param in model.model.parameters():
            param.requires_grad_(False)
    _HPSv3Ctx.model = model
    _HPSv3Ctx.device = device
    return model

def is_low_quality(frame, quality_model, threshold=HPSV3_QUALITY_THRESHOLD):
    frame = frame.permute(1, 2, 0).numpy().astype(np.uint8).clip(0, 255)
    with torch.inference_mode():
        rewards = quality_model.reward(image_paths=[Image.fromarray(frame)], prompts=[""])
        score = rewards[0][0].item()
    return score < threshold

def _ensure_rgb01(chw: torch.Tensor):
    assert chw.dim() == 3 and chw.shape[0] in (1,3), f"expect (C,H,W), got {tuple(chw.shape)}"
    x = chw
    if not torch.is_floating_point(x):
        x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    if x.shape[0] == 1:
        x = x.repeat(3,1,1)
    return x.clamp(0.0, 1.0)

def _clip_preprocess_tensor(x_chw: torch.Tensor, size=224):
    x = x_chw.unsqueeze(0)  # (1,3,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
    x = (x - mean) / std
    return x

@torch.no_grad()
def get_frame_sim_clip(frame1: torch.Tensor, frame2: torch.Tensor, device=None, use_half=False) -> float:
    model, dev, dt = _get_clip_model()

    f1 = _ensure_rgb01(frame1).to(dev)
    f2 = _ensure_rgb01(frame2).to(dev)

    x1 = _clip_preprocess_tensor(f1).to(dev, dtype=dt)
    x2 = _clip_preprocess_tensor(f2).to(dev, dtype=dt)

    z1 = model.encode_image(x1)
    z2 = model.encode_image(x2)

    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    cos = (z1 * z2).sum(dim=-1)  # (1,)
    return cos.item()

def extract_keyframe_indices(
    frames,
    quality_model,
    threshold=MIN_FRAME_SIMILARITY,
    frame_sim_func=get_frame_sim_clip,
    *,
    image_factor: int = IMAGE_FACTOR,
    video_min_pixels: int = VIDEO_MIN_PIXELS,
    video_max_pixels: int = VIDEO_MAX_PIXELS,
    max_ratio: int = MAX_RATIO,
    quality_threshold: float = HPSV3_QUALITY_THRESHOLD,
):
    assert frames.dim() == 4, "Input must be a 4-D tensor [N, C, H, W]"

    num_frames, _, height, width = frames.shape
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=image_factor,
        min_pixels=video_min_pixels,
        max_pixels=video_max_pixels,
        max_ratio=max_ratio,
    )

    resized_frames = nn.functional.interpolate(
        frames,
        [resized_height, resized_width],
        mode="bilinear",
        antialias=True,
    ).float()

    first_keyframe_indice = 0
    while is_low_quality(resized_frames[first_keyframe_indice], quality_model, quality_threshold):
        first_keyframe_indice += 1
        if first_keyframe_indice >= num_frames:
            return []
    keyframe_indices = [first_keyframe_indice]
    last_keyframe = resized_frames[first_keyframe_indice]
    for i in range(2, resized_frames.size(0)):
        current_frame = resized_frames[i]
        sim = frame_sim_func(last_keyframe, current_frame)
        if sim < threshold and not is_low_quality(current_frame, quality_model, quality_threshold):
            keyframe_indices.append(i)
            last_keyframe = current_frame

    return keyframe_indices

def read_video(
        video: str | bytes,
) -> Tuple[torch.Tensor, List[int]]:
    if isinstance(video, bytes):
        fp = io.BytesIO(video)
        vr = decord.VideoReader(fp)
    else:
        vr = decord.VideoReader(video)
    nframes, video_fps = len(vr), vr.get_avg_fps()
    timestamps = torch.FloatTensor([(1 / video_fps) * i for i in range(nframes)])

    indices = torch.linspace(0, nframes - 1, nframes).round().long()
    frames = vr.get_batch(indices.tolist()).asnumpy()
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # T, C, H, W
    timestamps = timestamps[indices]
    return frames, timestamps

def save_keyframes(
    video_path,
    frame_sim_thereshold=MIN_FRAME_SIMILARITY,
    frame_sim_func=get_frame_sim_clip,
    memory_cmp: Optional[bool] = None,
    memory_paths: Optional[List[str]] = None,
    keyframe_profile: Optional[str] = None,
    keyframe_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    settings = load_keyframe_settings(
        profile=keyframe_profile,
        config_path=keyframe_config_path,
    )
    st = time.time()
    frames, timestamps = read_video(video_path)
    quality_model = _get_quality_model(device="cuda")
    threshold = float(frame_sim_thereshold)
    if frame_sim_thereshold == MIN_FRAME_SIMILARITY:
        threshold = settings["min_frame_similarity"]
    while True:
        keyframe_indices = extract_keyframe_indices(
            frames,
            quality_model,
            threshold,
            frame_sim_func,
            image_factor=settings["image_factor"],
            video_min_pixels=settings["video_min_pixels"],
            video_max_pixels=settings["video_max_pixels"],
            max_ratio=settings["max_ratio"],
            quality_threshold=settings["hpsv3_quality_threshold"],
        )
        if len(keyframe_indices) > settings["max_keyframe_num"]:
            threshold -= settings["adaptive_alpha"]
        else:
            break
    # keyframe_indices = [1, 40, 80]
    logger.info(f"Read video: {video_path=}, {keyframe_indices=}, time={time.time() - st:.3f}s")
    keyframes = frames[keyframe_indices]
    compare_with_history = bool(settings["compare_with_history"]) if memory_cmp is None else bool(memory_cmp)
    if compare_with_history:
        if memory_paths is None:
            memory_paths = sorted(glob.glob(os.path.join(os.path.dirname(video_path), "*keyframe*.jpg")))
        memory_bank = []
        for p in memory_paths or []:
            memory_frame = cv2.imread(p)
            memory_frame = cv2.cvtColor(memory_frame, cv2.COLOR_BGR2RGB)
            memory_frame = torch.tensor(memory_frame).permute(2, 0, 1)
            memory_bank.append(memory_frame)
    saved_keyframes = []
    for i, keyframe in enumerate(keyframes):
        if compare_with_history:
            pass_flag = False
            for memory in memory_bank:
                sim = frame_sim_func(keyframe, memory)
                if sim > settings["min_frame_similarity"]:
                    pass_flag = True
                    break
            if pass_flag:
                continue
        keyframe = keyframe.permute(1, 2, 0).numpy()
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2BGR)
        keyframe_path = video_path.replace(".mp4", f"_keyframe{i}.jpg")
        cv2.imwrite(keyframe_path, keyframe)
        saved_keyframes.append(keyframe_path)
    last_frame = frames[-1]
    last_frame = last_frame.permute(1, 2, 0).numpy()
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_RGB2BGR)
    last_frame_path = os.path.join(os.path.dirname(video_path), "last_frame.jpg")
    cv2.imwrite(last_frame_path, last_frame)

    last_frames = frames[-5:]
    motion_frames_path = os.path.join(os.path.dirname(video_path), "motion_frames.mp4")
    _, c, h, w = last_frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 5
    writer = cv2.VideoWriter(motion_frames_path, fourcc, fps, (w, h))
    for frame in last_frames:
        frame = frame.permute(1, 2, 0).numpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return {
        "keyframe_indices": [int(index) for index in keyframe_indices],
        "keyframe_paths": saved_keyframes,
        "last_frame_path": last_frame_path,
        "motion_frames_path": motion_frames_path,
    }
