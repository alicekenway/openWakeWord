from __future__ import annotations

import pytest

from data_processing_tools.cosyvoice_arpabet_to_ipa import convert_arpabet, convert_variants


def test_converts_all_loncin_variants() -> None:
    assert convert_arpabet("[HH][AH0][L][OW1][L][AA1][N][S][IH0][N]") == (
        "h", "ə", "l", "oʊ", "l", "ɑː", "n", "s", "ɪ", "n"
    )
    assert convert_arpabet("[HH][AH0][L][OW1][L][AO1][N][S][IH0][N]") == (
        "h", "ə", "l", "oʊ", "l", "ɔː", "n", "s", "ɪ", "n"
    )
    assert convert_arpabet("[HH][AH0][L][OW1][L][AO1][NG][S][IH0][N]") == (
        "h", "ə", "l", "oʊ", "l", "ɔː", "ŋ", "s", "ɪ", "n"
    )


def test_rejects_unbracketed_or_unknown_phones() -> None:
    with pytest.raises(ValueError, match="outside ARPAbet"):
        convert_arpabet("Hello [L][AA1][N]")
    with pytest.raises(ValueError, match="Unsupported"):
        convert_arpabet("[NOTAPHONE]")


def test_validates_ipa_against_token_inventory() -> None:
    raw = {
        "variants": [
            {
                "id": "hello_loncin",
                "display_text": "Hello Loncin",
                "arpabet": "[HH][AH0][L][OW1]",
            }
        ]
    }
    with pytest.raises(ValueError, match="unknown IPA"):
        convert_variants(raw, {"h", "ə", "l"})
