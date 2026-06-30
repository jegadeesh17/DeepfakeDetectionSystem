import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from dotenv import load_dotenv

from src.model_builder import ARCHITECTURES, get_device, get_model

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_PATHS = [
    MODEL_DIR / "final_deepfake_detector.pth",
    MODEL_DIR / "best_model.pth",
]
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def _load_state_dict(checkpoint_path: Path) -> Tuple[dict, Optional[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint.get("architecture")
    return checkpoint, None


def _count_key_mismatches(model: torch.nn.Module, state_dict: dict) -> int:
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    return len(model_keys ^ ckpt_keys)


def _infer_architecture(state_dict: dict) -> str:
    env_arch = os.getenv("MODEL_ARCH")
    if env_arch in ARCHITECTURES:
        return env_arch

    best_arch = None
    best_mismatches = float("inf")
    for arch in ARCHITECTURES:
        model = get_model(arch)
        mismatches = _count_key_mismatches(model, state_dict)
        if mismatches < best_mismatches:
            best_mismatches = mismatches
            best_arch = arch
        if mismatches == 0:
            return arch

    if best_arch is not None and best_mismatches == 0:
        return best_arch

    raise ValueError(
        "Could not infer model architecture from checkpoint. "
        f"Set MODEL_ARCH to one of {ARCHITECTURES} in your .env file."
    )


def _resolve_architecture(state_dict: dict, checkpoint_arch: Optional[str]) -> str:
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)
        arch = metadata.get("architecture")
        if arch in ARCHITECTURES:
            return arch

    if checkpoint_arch in ARCHITECTURES:
        return checkpoint_arch

    return _infer_architecture(state_dict)


def find_checkpoint_path() -> Optional[Path]:
    for path in CHECKPOINT_PATHS:
        if path.exists():
            return path
    return None


def load_model(device: Optional[torch.device] = None) -> Tuple[torch.nn.Module, str, torch.device]:
    checkpoint_path = find_checkpoint_path()
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No model checkpoint found. Place a .pth file in {MODEL_DIR} "
            "(final_deepfake_detector.pth or best_model.pth)."
        )

    state_dict, checkpoint_arch = _load_state_dict(checkpoint_path)
    architecture = _resolve_architecture(state_dict, checkpoint_arch)

    if device is None:
        device = get_device()

    model = get_model(architecture)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, architecture, device


def save_checkpoint(model: torch.nn.Module, architecture: str, path: Optional[Path] = None) -> Path:
    path = path or (MODEL_DIR / "final_deepfake_detector.pth")
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, path)

    metadata = {"architecture": architecture, "checkpoint": path.name}
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return path
