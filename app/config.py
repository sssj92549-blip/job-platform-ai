"""配置加载：default.yml提供默认值，local.yml仅覆盖本机私有配置。"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr

ROOT = Path(__file__).resolve().parent.parent


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Server(Section):
    host: str = "127.0.0.1"
    port: int = Field(default=7999, ge=1, le=65535)
    internal_token: SecretStr = SecretStr("")


class Storage(Section):
    uploads_root: Path
    chroma_path: Path
    model_cache: Path


class DeepSeek(Section):
    base_url: str = "https://api.deepseek.com"
    model: str
    api_key: SecretStr = SecretStr("")
    timeout_seconds: float = Field(default=90, gt=0, le=300)
    max_tokens: int = Field(default=8192, ge=512, le=32768)


class Embedding(Section):
    model: str
    collection: str
    chunk_chars: int = Field(default=200, ge=32, le=200)


class Ocr(Section):
    language: str = "ch"
    detection_model: str
    recognition_model: str
    dpi: int = Field(default=150, ge=72, le=200)
    text_threshold: int = Field(default=20, ge=1, le=100)


class Settings(Section):
    server: Server
    storage: Storage
    deepseek: DeepSeek
    embedding: Embedding
    ocr: Ocr


def merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_settings(root: Path = ROOT) -> Settings:
    """不打印配置，避免将密钥暴露在启动日志中。相对路径以项目目录解析。"""
    data: dict[str, Any] = yaml.safe_load((root / "default.yml").read_text("utf-8"))
    if (root / "local.yml").is_file():
        data = merge(data, yaml.safe_load((root / "local.yml").read_text("utf-8")) or {})
    settings = Settings.model_validate(data)
    for key in ("uploads_root", "chroma_path", "model_cache"):
        path = getattr(settings.storage, key)
        setattr(
            settings.storage,
            key,
            (root / path).resolve() if not path.is_absolute() else path.resolve(),
        )
    return settings
