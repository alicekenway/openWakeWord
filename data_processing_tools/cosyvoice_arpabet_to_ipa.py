#!/usr/bin/env python3
"""Convert bracketed CosyVoice/CMU ARPAbet variants to WeNet IPA tokens.

The input JSON contains a ``variants`` list.  Every item needs ``id``,
``display_text``, and a fully bracketed ``arpabet`` expression.  The output is
the three-column variant TSV accepted by ``wakewords_to_keyword_tokens.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ARPABET_TOKEN = re.compile(r"\[([A-Za-z]+)([012]?)\]")

PHONE_MAP: dict[str, tuple[str, ...]] = {
    "AA": ("ɑː",), "AE": ("æ",), "AO": ("ɔː",), "AW": ("aʊ",),
    "AY": ("aɪ",), "B": ("b",), "CH": ("tʃ",), "D": ("d",),
    "DH": ("ð",), "EH": ("ɛ",), "EY": ("eɪ",), "F": ("f",),
    "G": ("ɡ",), "HH": ("h",), "IH": ("ɪ",), "JH": ("dʒ",),
    "K": ("k",), "L": ("l",), "M": ("m",), "N": ("n",),
    "NG": ("ŋ",), "OW": ("oʊ",), "OY": ("ɔɪ",), "P": ("p",),
    "R": ("ɹ",), "S": ("s",), "SH": ("ʃ",), "T": ("t",),
    "TH": ("θ",), "UH": ("ʊ",), "UW": ("uː",), "V": ("v",),
    "W": ("w",), "Y": ("j",), "Z": ("z",), "ZH": ("ʒ",),
}


def arpabet_phone_to_ipa(phone: str, stress: str) -> tuple[str, ...]:
    if phone == "AH":
        return ("ə",) if stress == "0" else ("ʌ",)
    if phone == "ER":
        return ("ɚ",) if stress == "0" else ("ɜː", "ɹ")
    if phone == "IY":
        return ("i",) if stress == "0" else ("iː",)
    try:
        return PHONE_MAP[phone]
    except KeyError as exc:
        raise ValueError(f"Unsupported ARPAbet phone {phone!r}") from exc


def convert_arpabet(expression: str) -> tuple[str, ...]:
    matches = list(ARPABET_TOKEN.finditer(expression))
    if not matches:
        raise ValueError(f"No bracketed ARPAbet phones found in {expression!r}")
    remainder = ARPABET_TOKEN.sub("", expression)
    if remainder.strip():
        raise ValueError(f"Unexpected text outside ARPAbet brackets: {remainder!r}")
    ipa: list[str] = []
    for match in matches:
        ipa.extend(arpabet_phone_to_ipa(match.group(1).upper(), match.group(2)))
    return tuple(ipa)


def read_token_symbols(path: Path) -> set[str]:
    symbols: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Invalid token table line {line_number} of {path}")
            symbols.add(fields[0])
    if not symbols:
        raise ValueError(f"Token table contains no entries: {path}")
    return symbols


def convert_variants(raw: Any, token_symbols: set[str]) -> list[tuple[str, str, tuple[str, ...]]]:
    values = raw.get("variants") if isinstance(raw, dict) else None
    if not isinstance(values, list) or not values:
        raise ValueError("Variant JSON must contain a non-empty 'variants' list")
    result: list[tuple[str, str, tuple[str, ...]]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"variants[{index}] must be an object")
        keyword_id = str(value.get("id", "")).strip()
        display_text = " ".join(str(value.get("display_text", "")).split())
        arpabet = str(value.get("arpabet", "")).strip()
        if not keyword_id or not display_text or not arpabet:
            raise ValueError(f"variants[{index}] needs id, display_text, and arpabet")
        if keyword_id in seen_ids:
            raise ValueError(f"Duplicate variant id {keyword_id!r}")
        seen_ids.add(keyword_id)
        ipa = convert_arpabet(arpabet)
        unknown = sorted(set(ipa) - token_symbols)
        if unknown:
            raise ValueError(f"Variant {keyword_id!r} produces unknown IPA token(s): {unknown}")
        result.append((keyword_id, display_text, ipa))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output-tsv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = json.loads(args.input_json.read_text(encoding="utf-8"))
    variants = convert_variants(raw, read_token_symbols(args.tokens))
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{keyword_id}\t{display_text}\t{' '.join(ipa)}" for keyword_id, display_text, ipa in variants]
    args.output_tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(variants)} variant(s) to {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
