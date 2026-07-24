"""Configuration loading and typed access for the INI pipeline."""

from __future__ import annotations

import configparser
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ConfigurationError(ValueError):
    """Raised when an INI file cannot describe a valid pipeline."""


def parse_json(value: str, field: str, expected_type: type | tuple[type, ...] | None = None) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{field} must be valid JSON: {exc.msg}") from exc
    if expected_type is not None and not isinstance(parsed, expected_type):
        expected_name = (
            "/".join(item.__name__ for item in expected_type)
            if isinstance(expected_type, tuple)
            else expected_type.__name__
        )
        raise ConfigurationError(f"{field} must decode to {expected_name}, got {type(parsed).__name__}")
    return parsed


def parse_csv(value: str) -> list[str]:
    """Parse comma/newline-separated identifiers while accepting empty lines."""
    return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]


def parse_step_groups(value: str) -> list[list[str]]:
    """Parse sequential pipeline steps with optional bracketed parallel groups.

    ``a, b, c`` produces three sequential groups. ``[a, b], c`` runs ``a``
    and ``b`` together, waits for both, and then runs ``c``.
    """

    groups: list[list[str]] = []
    token: list[str] = []
    parallel: list[str] | None = None

    def finish_token(*, required: bool = False) -> None:
        nonlocal token
        name = "".join(token).strip()
        token = []
        if not name:
            if required:
                raise ConfigurationError("[steps] parallel groups cannot contain empty stage names")
            return
        if parallel is None:
            groups.append([name])
        else:
            parallel.append(name)

    for character in value:
        if character == "[":
            if parallel is not None:
                raise ConfigurationError("[steps] nested parallel groups are not supported")
            if "".join(token).strip():
                raise ConfigurationError("[steps] '[' must begin a new parallel group")
            token = []
            parallel = []
        elif character == "]":
            if parallel is None:
                raise ConfigurationError("[steps] has an unmatched ']'")
            finish_token(required=True)
            if not parallel:
                raise ConfigurationError("[steps] parallel groups cannot be empty")
            groups.append(parallel)
            parallel = None
        elif character == ",":
            finish_token(required=parallel is not None)
        elif character == "\n" and parallel is None:
            finish_token()
        else:
            token.append(character)

    if parallel is not None:
        raise ConfigurationError("[steps] has an unclosed '[' parallel group")
    finish_token()
    if not groups:
        raise ConfigurationError("[steps] steps cannot be empty")
    return groups


@dataclass(frozen=True)
class IniConfig:
    """A loaded config and the directory against which relative paths resolve."""

    path: Path
    parser: configparser.ConfigParser
    base_dir: Path | None = None

    @property
    def root(self) -> Path:
        """Directory used to resolve relative paths.

        Slurm workers read an immutable copy of the submitted configuration
        from the experiment work directory.  Keeping the original directory
        separately lets that copy retain the same relative-path semantics as
        the user's source INI file.
        """

        return self.base_dir or self.path.parent

    def has_section(self, name: str) -> bool:
        return self.parser.has_section(name)

    def require_section(self, name: str) -> None:
        if not self.has_section(name):
            raise ConfigurationError(f"Missing required section [{name}]")

    def section(self, name: str) -> dict[str, str]:
        self.require_section(name)
        try:
            return dict(self.parser.items(name, raw=False))
        except configparser.InterpolationError as exc:
            raise ConfigurationError(f"Invalid interpolation in [{name}]: {exc}") from exc

    def get(self, section: str, option: str, *, required: bool = True, fallback: str | None = None) -> str | None:
        self.require_section(section)
        if not self.parser.has_option(section, option):
            if required:
                raise ConfigurationError(f"Missing required option [{section}] {option}")
            return fallback
        try:
            return self.parser.get(section, option)
        except configparser.InterpolationError as exc:
            raise ConfigurationError(f"Invalid interpolation in [{section}] {option}: {exc}") from exc

    def getint(self, section: str, option: str, *, required: bool = True, fallback: int | None = None) -> int | None:
        value = self.get(section, option, required=required)
        if value is None:
            return fallback
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigurationError(f"[{section}] {option} must be an integer, got {value!r}") from exc

    def getfloat(self, section: str, option: str, *, required: bool = True, fallback: float | None = None) -> float | None:
        value = self.get(section, option, required=required)
        if value is None:
            return fallback
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigurationError(f"[{section}] {option} must be a number, got {value!r}") from exc

    def getboolean(self, section: str, option: str, *, required: bool = True, fallback: bool | None = None) -> bool | None:
        self.require_section(section)
        if not self.parser.has_option(section, option):
            if required:
                raise ConfigurationError(f"Missing required option [{section}] {option}")
            return fallback
        try:
            return self.parser.getboolean(section, option)
        except ValueError as exc:
            value = self.parser.get(section, option, raw=False)
            raise ConfigurationError(f"[{section}] {option} must be a boolean, got {value!r}") from exc

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def resolved_sections(self) -> dict[str, dict[str, str]]:
        return {name: self.section(name) for name in self.parser.sections()}

    def write_resolved(self, path: Path) -> None:
        """Write an INI snapshot with all interpolation values already resolved."""
        output = configparser.ConfigParser(interpolation=None)
        for name in self.parser.sections():
            output.add_section(name)
            for key, value in self.section(name).items():
                output.set(name, key, value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            output.write(handle)


def load_ini_config(path: str | Path, *, base_dir: str | Path | None = None) -> IniConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    parser = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation(),
        strict=True,
        empty_lines_in_values=True,
    )
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            parser.read_file(handle)
    except (OSError, configparser.Error) as exc:
        raise ConfigurationError(f"Could not read INI configuration {config_path}: {exc}") from exc
    resolved_base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    return IniConfig(path=config_path, parser=parser, base_dir=resolved_base)


def ensure_known_options(section: str, values: dict[str, str], allowed: Iterable[str], *, prefixes: Iterable[str] = ()) -> None:
    allowed_set = set(allowed)
    prefixes_tuple = tuple(prefixes)
    unknown = sorted(
        key for key in values
        if key not in allowed_set and not key.startswith(prefixes_tuple)
    )
    if unknown:
        raise ConfigurationError(f"Unknown option(s) in [{section}]: {', '.join(unknown)}")
