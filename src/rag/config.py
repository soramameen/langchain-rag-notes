import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "rag"
CONFIG_FILE = CONFIG_DIR / "config.json"


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_notes_dirs(override_dirs: list[str] | None = None) -> list[str]:
    if override_dirs:
        return override_dirs
    config = load_config()
    dirs = config.get("notes_dirs")
    if not dirs:
        return []
    return dirs


def get_agent_notes_dirs(override_dirs: list[str] | None = None) -> list[str]:
    if override_dirs:
        return override_dirs
    config = load_config()
    dirs = config.get("agent_notes_dirs")
    if not dirs:
        return []
    return dirs


def get_config_value(key: str, default=None):
    config = load_config()
    return config.get(key, default)


def get_default_skills_dir() -> Path | None:
    """Detect the pi-agent skills directory."""
    # Explicit env var takes precedence
    if env_dir := os.getenv("PI_SKILLS_DIR"):
        path = Path(env_dir).expanduser()
        if path.exists():
            return path

    # Common pi paths
    candidates = [
        Path.home() / ".pi" / "agent" / "skills",
        Path.home() / ".agents" / "skills",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
