"""Prompt versions under test, stored as ``prompts/<name>.md`` at the repository root."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
PROMPT_NAMES = ("v1", "v2")


def load_prompt(name: str) -> str:
    if name not in PROMPT_NAMES:
        raise ValueError(f"unknown prompt {name!r}, expected one of {PROMPT_NAMES}")
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
