from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing_tools.wakewords_to_keyword_tokens import (  # noqa: E402
    build_keyword_config,
    build_keyword_variant_config,
    read_keyword_variants,
    read_phoneme_dictionary,
    read_token_table,
    read_wakewords,
)


def write_text(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    return path


def test_builds_keyword_token_json_from_phonemes(tmp_path: Path) -> None:
    wakewords = read_wakewords(write_text(tmp_path / "wakewords.txt", "Hey Frank\nhello   lynn\n"))
    pronunciations = read_phoneme_dictionary(
        write_text(
            tmp_path / "pronunciations.txt",
            "hey frank\tHH EY F R AE NG K\nHello Lynn\tHH AH L OW L IH N\n",
        )
    )
    token_ids = read_token_table(
        write_text(
            tmp_path / "tokens.txt",
            "<blank> 0\n<unk> 1\nHH 10\nEY 11\nF 12\nR 13\nAE 14\nNG 15\nK 16\n"
            "AH 17\nL 18\nOW 19\nIH 20\nN 21\n",
        )
    )

    assert build_keyword_config(wakewords, pronunciations, token_ids) == {
        "keywords": [
            {
                "id": "hey_frank",
                "display_text": "Hey Frank",
                "token_ids": [10, 11, 12, 13, 14, 15, 16],
            },
            {
                "id": "hello_lynn",
                "display_text": "hello lynn",
                "token_ids": [10, 17, 18, 19, 18, 20, 21],
            },
        ]
    }


def test_reports_all_missing_pronunciations_and_unknown_phonemes(tmp_path: Path) -> None:
    wakewords = ["known phrase", "missing phrase"]
    pronunciations = read_phoneme_dictionary(
        write_text(tmp_path / "pronunciations.txt", "known phrase\tHH NOT_A_TOKEN\n")
    )

    with pytest.raises(ValueError) as error:
        build_keyword_config(wakewords, pronunciations, {"HH": 10})

    message = str(error.value)
    assert "missing phrase" in message
    assert "NOT_A_TOKEN" in message


def test_rejects_duplicate_normalized_pronunciations(tmp_path: Path) -> None:
    path = write_text(
        tmp_path / "pronunciations.txt",
        "Hey Frank\tHH EY\nhey   frank\tHH EY\n",
    )

    with pytest.raises(ValueError, match="Duplicate pronunciation"):
        read_phoneme_dictionary(path)


def test_builds_multiple_variants_with_shared_display_text(tmp_path: Path) -> None:
    first = write_text(
        tmp_path / "first.tsv",
        "hello_loncin_aa_n\tHello Loncin\th ə l oʊ l ɑː n s ɪ n\n",
    )
    second = write_text(
        tmp_path / "second.tsv",
        "hello_loncin_ao_ng\tHello Loncin\th ə l oʊ l ɔː ŋ s ɪ n\n",
    )
    variants = read_keyword_variants([first, second])
    symbols = "h ə l oʊ ɑː ɔː n ŋ s ɪ".split()
    result = build_keyword_variant_config(variants, {symbol: index for index, symbol in enumerate(symbols)})

    assert [item["id"] for item in result["keywords"]] == [
        "hello_loncin_aa_n",
        "hello_loncin_ao_ng",
    ]
    assert {item["display_text"] for item in result["keywords"]} == {"Hello Loncin"}


def test_variant_files_reject_duplicate_ids(tmp_path: Path) -> None:
    variants = write_text(
        tmp_path / "variants.tsv",
        "same\tHello One\th ə\nsame\tHello Two\th ə\n",
    )
    with pytest.raises(ValueError, match="Duplicate keyword variant id"):
        read_keyword_variants([variants])
