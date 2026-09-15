"""
Configuration settings for Deepfake Detection System.
Uses pydantic-settings for robust environment configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_ROOT: Path = _BASE_DIR
    MODEL_DIR: Path = _BASE_DIR / "models"
    MODEL_ARCH: str = Field(default="EfficientNet", description="Backbone architecture: EfficientNet or ResNet")
    CHECKPOINT_PATH: Path = _BASE_DIR / "models" / "final_deepfake_detector.pth"
    PORT: int = Field(default=8004, description="FastAPI service port")
    HOST: str = Field(default="0.0.0.0", description="FastAPI service host")
    DEVICE: str = Field(default="cpu", description="Inference device: cpu or cuda")
    THRESHOLD: float = Field(default=0.5, description="Classification decision threshold for FAKE label")


_settings_instance = None


def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


settings = get_settings()
