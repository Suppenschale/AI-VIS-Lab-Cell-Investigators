from __future__ import annotations
 
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
 
import cv2
import numpy as np

@dataclass(frozen=True)
class Sample:
    name: str
    image_path: Path
    mask_path: Path
 
 
def _natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]
 
 
def find_samples(
    folder: Union[str, Path],
    img_suffix: str = "_img",
    mask_suffix: str = "_masks",
    extensions: Sequence[str] = (".png", ".tif", ".tiff"),
    recursive: bool = False,
    verbose: bool = True):
    
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"No folder: {folder}")
    exts = {e.lower() for e in extensions}
    files = folder.rglob("*") if recursive else folder.iterdir()
 
    imgs: dict[tuple[Path, str], Path] = {}
    masks: dict[tuple[Path, str], Path] = {}
    for f in files:
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        stem = f.stem
        if stem.endswith(mask_suffix):
            target, key = masks, (f.parent, stem[: -len(mask_suffix)])
        elif stem.endswith(img_suffix):
            target, key = imgs, (f.parent, stem[: -len(img_suffix)])
        else:
            continue
        if key in target:
            raise ValueError(f"Unclear: {target[key]} und {f}")
        target[key] = f
 
    samples = [
        Sample(name=key[1], image_path=p, mask_path=masks[key])
        for key, p in imgs.items()
        if key in masks
    ]
    samples.sort(key=lambda s: (str(s.image_path.parent), _natural_key(s.name)))
 
    if verbose:
        no_mask = sorted((k[1] for k in imgs if k not in masks), key=_natural_key)
        no_img = sorted((k[1] for k in masks if k not in imgs), key=_natural_key)
        print(f"{len(samples)} pairs found {folder}")
        if no_mask:
            print(f"  Img without Mask ({len(no_mask)}): {no_mask[:10]}{' ...' if len(no_mask) > 10 else ''}")
        if no_img:
            print(f"  Mask without Bild ({len(no_img)}): {no_img[:10]}{' ...' if len(no_img) > 10 else ''}")
    return samples
 
 
def _read(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"Error: {path}")
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
    return arr
 
 
def load_sample(sample: Sample):
    img = _read(sample.image_path).astype(np.float32)
    mask = _read(sample.mask_path)
    if mask.ndim != 2:
        raise ValueError(
            f"Mask {sample.mask_path} has shape {mask.shape}. Expected H x W with label-IDs."
            "Check if it is label mask."
        )
    if img.shape[:2] != mask.shape:
        raise ValueError(f"Size mismatch: Img: {img.shape} \t mask: {mask.shape} ({sample.name})")
    return img, mask.astype(np.int32)