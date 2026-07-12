"""Shared helpers used by stage implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ConfigurationError, parse_csv


VALID_PLACEMENTS = {"start", "end", "center", "random"}


def require(section: dict[str, str], name: str, section_name: str) -> str:
    value = section.get(name)
    if value is None or not value.strip():
        raise ConfigurationError(f"Missing required option [{section_name}] {name}")
    return value


def integer(section: dict[str, str], name: str, section_name: str, default: int | None = None) -> int:
    value = section.get(name)
    if value is None:
        if default is None:
            raise ConfigurationError(f"Missing required option [{section_name}] {name}")
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"[{section_name}] {name} must be an integer, got {value!r}") from exc


def optional_integer(section: dict[str, str], name: str, section_name: str) -> int | None:
    if name not in section:
        return None
    return integer(section, name, section_name)


def number(section: dict[str, str], name: str, section_name: str, default: float | None = None) -> float:
    value = section.get(name)
    if value is None:
        if default is None:
            raise ConfigurationError(f"Missing required option [{section_name}] {name}")
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"[{section_name}] {name} must be a number, got {value!r}") from exc


def optional_number(section: dict[str, str], name: str, section_name: str) -> float | None:
    if name not in section:
        return None
    return number(section, name, section_name)


def boolean(section: dict[str, str], name: str, section_name: str, default: bool | None = None) -> bool:
    value = section.get(name)
    if value is None:
        if default is None:
            raise ConfigurationError(f"Missing required option [{section_name}] {name}")
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "yes", "true", "on"}:
        return True
    if normalized in {"0", "no", "false", "off"}:
        return False
    raise ConfigurationError(f"[{section_name}] {name} must be a boolean, got {value!r}")


def placement(section: dict[str, str], section_name: str, default: str = "random") -> str:
    value = section.get("placement", default).strip().lower()
    if value not in VALID_PLACEMENTS:
        raise ConfigurationError(f"[{section_name}] placement must be one of {', '.join(sorted(VALID_PLACEMENTS))}")
    return value


def csv_option(section: dict[str, str], name: str, section_name: str, *, required: bool = True) -> list[str]:
    value = section.get(name)
    if value is None:
        if required:
            raise ConfigurationError(f"Missing required option [{section_name}] {name}")
        return []
    values = parse_csv(value)
    if required and not values:
        raise ConfigurationError(f"[{section_name}] {name} cannot be empty")
    return values


def stage_work_path(ctx: Any, filename: str) -> Path:
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    return ctx.work_dir / filename
