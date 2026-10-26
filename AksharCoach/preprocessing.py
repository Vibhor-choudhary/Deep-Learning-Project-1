"""Shared, deterministic preprocessing for training, testing, and uploads."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_SIZE = 32
MEAN, STD = 0.5, 0.5

# Folder identifiers are the primary labels. Native glyphs are a verified display aid.
CHARACTER_GLYPHS = ["क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श", "ष", "स", "ह", "क्ष", "त्र", "ज्ञ"]


def base_transform():
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


def train_transform():
    return transforms.Compose([transforms.Grayscale(), transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomAffine(8, translate=(0.06, 0.06)), transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])


def upload_to_tensor(image: Image.Image) -> torch.Tensor:
    """Turn a user photo or scan into the white-stroke-on-black DHCD convention."""
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    arr = np.asarray(image)
    # Photographs are generally dark ink on light paper; invert when background is light.
    if arr.mean() > 127:
        image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
    return base_transform()(image).unsqueeze(0)


def glyph_for_folder(folder: str) -> str:
    if folder.startswith("digit_"):
        return "०१२३४५६७८९"[int(folder.split("_")[1])]
    if folder.startswith("character_"):
        return CHARACTER_GLYPHS[int(folder.split("_")[1]) - 1]
    return "?"


def label_records(classes):
    return [{"index": i, "folder": name, "glyph": glyph_for_folder(name)} for i, name in enumerate(classes)]


def save_label_mapping(classes, path: Path):
    path.write_text(json.dumps(label_records(classes), ensure_ascii=False, indent=2), encoding="utf-8")
