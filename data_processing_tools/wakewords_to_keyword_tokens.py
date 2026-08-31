#!/usr/bin/env python3
"""Convert wake-word text and phoneme mappings into CTC keyword-token JSON."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def normalize_phrase(value: str) -> str:
    """Normalize Unicode and whitespace without changing displayed words."""

    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def phrase_key(value: str, *, case_sensitive: bool) -> str:
    normalized = normalize_phrase(value)
    return normalized if case_sensitive else normalized.casefold()


def make_keyword_id(display_text: str) -> str:
    """Create the underscore-separated identifier used by keyword JSON."""

    identifier = re.sub(r"[^\w]+", "_", display_text.casefold(), flags=re.UNICODE).strip("_")
    if not identifier:
        raise ValueError(f"Cannot generate a keyword id from {display_text!r}")
    return identifier


def read_wakewords(path: Path, *, case_sensitive: bool = False) -> list[str]:
    wakewords: list[str] = []
    seen: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            display_text = normalize_phrase(line)
            if not display_text:
                continue
            key = phrase_key(display_text, case_sensitive=case_sensitive)
            if key in seen:
                raise ValueError(
                    f"Duplicate wake word {display_text!r} on line {line_number} of {path}; "
                    f"first seen on line {seen[key]}"
                )
            seen[key] = line_number
            wakewords.append(display_text)
    if not wakewords:
        raise ValueError(f"Wake-word file contains no non-empty phrases: {path}")
    return wakewords


def read_phoneme_dictionary(
    path: Path, *, case_sensitive: bool = False
) -> dict[str, tuple[str, ...]]:
    pronunciations: dict[str, tuple[str, ...]] = {}
    source_lines: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if "\t" not in stripped:
                raise ValueError(
                    f"Invalid pronunciation line {line_number} of {path}: expected "
                    "<wake word><TAB><space-separated phonemes>"
                )
            phrase_text, phoneme_text = stripped.split("\t", 1)
            display_text = normalize_phrase(phrase_text)
            phonemes = tuple(phoneme_text.split())
            if not display_text or not phonemes:
                raise ValueError(f"Empty wake word or phoneme sequence on line {line_number} of {path}")
            key = phrase_key(display_text, case_sensitive=case_sensitive)
            if key in pronunciations:
                raise ValueError(
                    f"Duplicate pronunciation for {display_text!r} on line {line_number} of {path}; "
                    f"first seen on line {source_lines[key]}"
                )
            pronunciations[key] = phonemes
            source_lines[key] = line_number
    if not pronunciations:
        raise ValueError(f"Pronunciation dictionary contains no entries: {path}")
    return pronunciations


def read_token_table(path: Path) -> dict[str, int]:
    token_ids: dict[str, int] = {}
    id_tokens: dict[int, str] = {}
    source_lines: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            fields = stripped.split()
            if len(fields) != 2:
                raise ValueError(
                    f"Invalid token line {line_number} of {path}: expected <token><SPACE_OR_TAB><integer id>"
                )
            token, raw_id = fields
            try:
                token_id = int(raw_id)
            except ValueError as exc:
                raise ValueError(f"Invalid token id {raw_id!r} on line {line_number} of {path}") from exc
            if token_id < 0:
                raise ValueError(f"Negative token id {token_id} on line {line_number} of {path}")
            if token in token_ids:
                raise ValueError(
                    f"Duplicate token {token!r} on line {line_number} of {path}; "
                    f"first seen on line {source_lines[token]}"
                )
            if token_id in id_tokens:
                raise ValueError(
                    f"Token id {token_id} is assigned to both {id_tokens[token_id]!r} and {token!r} in {path}"
                )
            token_ids[token] = token_id
            id_tokens[token_id] = token
            source_lines[token] = line_number
    if not token_ids:
        raise ValueError(f"Token table contains no entries: {path}")
    return token_ids


def build_keyword_config(
    wakewords: list[str],
    pronunciations: dict[str, tuple[str, ...]],
    token_ids: dict[str, int],
    *,
    case_sensitive: bool = False,
) -> dict[str, list[dict[str, object]]]:
    keywords: list[dict[str, object]] = []
    generated_ids: dict[str, str] = {}
    missing_pronunciations: list[str] = []
    unknown_phonemes: dict[str, list[str]] = {}

    for display_text in wakewords:
        key = phrase_key(display_text, case_sensitive=case_sensitive)
        phonemes = pronunciations.get(key)
        if phonemes is None:
            missing_pronunciations.append(display_text)
            continue
        unknown = sorted({phoneme for phoneme in phonemes if phoneme not in token_ids})
        if unknown:
            unknown_phonemes[display_text] = unknown
            continue
        keyword_id = make_keyword_id(display_text)
        if keyword_id in generated_ids:
            raise ValueError(
                f"Wake words {generated_ids[keyword_id]!r} and {display_text!r} both generate id {keyword_id!r}"
            )
        generated_ids[keyword_id] = display_text
        keywords.append(
            {
                "id": keyword_id,
                "display_text": display_text,
                "token_ids": [token_ids[phoneme] for phoneme in phonemes],
            }
        )

    problems: list[str] = []
    if missing_pronunciations:
        problems.append("missing pronunciation: " + ", ".join(repr(item) for item in missing_pronunciations))
    for display_text, phonemes in unknown_phonemes.items():
        problems.append(
            f"unknown phoneme(s) for {display_text!r}: " + ", ".join(repr(item) for item in phonemes)
        )
    if problems:
        raise ValueError("Cannot build keyword config; " + "; ".join(problems))
    return {"keywords": keywords}


def read_keyword_variants(paths: list[Path]) -> list[tuple[str, str, tuple[str, ...]]]:
    """Read explicit ``id<TAB>display_text<TAB>phonemes`` keyword variants."""

    variants: list[tuple[str, str, tuple[str, ...]]] = []
    seen_ids: dict[str, tuple[Path, int]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split("\t")
                if len(fields) != 3:
                    raise ValueError(
                        f"Invalid keyword variant line {line_number} of {path}: expected "
                        "<id><TAB><display text><TAB><space-separated phonemes>"
                    )
                keyword_id = fields[0].strip()
                display_text = normalize_phrase(fields[1])
                phonemes = tuple(fields[2].split())
                if not keyword_id or not display_text or not phonemes:
                    raise ValueError(f"Empty field on keyword variant line {line_number} of {path}")
                if keyword_id in seen_ids:
                    first_path, first_line = seen_ids[keyword_id]
                    raise ValueError(
                        f"Duplicate keyword variant id {keyword_id!r} on line {line_number} of {path}; "
                        f"first seen on line {first_line} of {first_path}"
                    )
                seen_ids[keyword_id] = (path, line_number)
                variants.append((keyword_id, display_text, phonemes))
    if not variants:
        raise ValueError("Keyword variant files contain no entries")
    return variants


def build_keyword_variant_config(
    variants: list[tuple[str, str, tuple[str, ...]]],
    token_ids: dict[str, int],
) -> dict[str, list[dict[str, object]]]:
    """Build keyword JSON while allowing several IDs to share display text."""

    unknown_phonemes: dict[str, list[str]] = {}
    keywords: list[dict[str, object]] = []
    for keyword_id, display_text, phonemes in variants:
        unknown = sorted({phoneme for phoneme in phonemes if phoneme not in token_ids})
        if unknown:
            unknown_phonemes[keyword_id] = unknown
            continue
        keywords.append(
            {
                "id": keyword_id,
                "display_text": display_text,
                "token_ids": [token_ids[phoneme] for phoneme in phonemes],
            }
        )
    if unknown_phonemes:
        details = "; ".join(
            f"unknown phoneme(s) for {keyword_id!r}: "
            + ", ".join(repr(item) for item in phonemes)
            for keyword_id, phonemes in unknown_phonemes.items()
        )
        raise ValueError(f"Cannot build keyword config; {details}")
    return {"keywords": keywords}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map wake words through a phoneme dictionary and WeNet token table into keyword-token JSON."
    )
    parser.add_argument("--wakewords", help="Text file with one wake-word phrase per line")
    parser.add_argument(
        "--phoneme-dict",
        help="UTF-8 dictionary with <wake word><TAB><space-separated phonemes>",
    )
    parser.add_argument(
        "--keyword-variants",
        action="append",
        default=[],
        help=(
            "Explicit UTF-8 variant file with "
            "<id><TAB><display text><TAB><space-separated phonemes>. "
            "May be repeated; cannot be combined with --wakewords/--phoneme-dict."
        ),
    )
    parser.add_argument(
        "--tokens",
        required=True,
        help="WeNet token table with one <token><SPACE_OR_TAB><integer id> entry per line",
    )
    parser.add_argument("--output-json", required=True, help="Destination keyword-token JSON file")
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Require wake-word spelling case to match the pronunciation dictionary exactly",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens_path = Path(args.tokens).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()

    variant_mode = bool(args.keyword_variants)
    if variant_mode and (args.wakewords or args.phoneme_dict):
        raise ValueError("--keyword-variants cannot be combined with --wakewords or --phoneme-dict")
    if not variant_mode and not (args.wakewords and args.phoneme_dict):
        raise ValueError("provide --keyword-variants, or provide both --wakewords and --phoneme-dict")

    required_paths: list[tuple[str, Path]] = [("token table", tokens_path)]
    variant_paths = [Path(value).expanduser().resolve() for value in args.keyword_variants]
    if variant_mode:
        required_paths.extend(("keyword variant file", path) for path in variant_paths)
    else:
        wakewords_path = Path(args.wakewords).expanduser().resolve()
        phoneme_dict_path = Path(args.phoneme_dict).expanduser().resolve()
        required_paths.extend(
            (("wake-word file", wakewords_path), ("phoneme dictionary", phoneme_dict_path))
        )
    for name, path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    token_ids = read_token_table(tokens_path)
    if variant_mode:
        result = build_keyword_variant_config(read_keyword_variants(variant_paths), token_ids)
    else:
        wakewords = read_wakewords(wakewords_path, case_sensitive=args.case_sensitive)
        pronunciations = read_phoneme_dictionary(phoneme_dict_path, case_sensitive=args.case_sensitive)
        result = build_keyword_config(
            wakewords,
            pronunciations,
            token_ids,
            case_sensitive=args.case_sensitive,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['keywords'])} keyword(s) to {output_path}")


if __name__ == "__main__":
    main()
