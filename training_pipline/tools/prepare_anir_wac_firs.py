#!/usr/bin/env python3
"""Prepare ANIR V-Class speech paths as 16 kHz WAC-compatible FIR files."""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


FIR_NAME = re.compile(r"(?:^|/)lsp_(\d{2})_mic_(\d{2})_enh\.fir$")
SOURCE_URL = "https://dss-kiel.de/index.php/media-center/data-bases/anir-corpus/download"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-rate", type=int, default=48_000)
    parser.add_argument("--target-rate", type=int, default=16_000)
    return parser.parse_args()


def read_anir_fir(payload: bytes, name: str) -> np.ndarray:
    if len(payload) < 8:
        raise ValueError(f"ANIR FIR is too short: {name}")
    (tap_count,) = struct.unpack_from("<i", payload, 0)
    expected_bytes = 4 + tap_count * 4
    if tap_count < 1 or len(payload) != expected_bytes:
        raise ValueError(
            f"ANIR FIR payload mismatch for {name}: "
            f"taps={tap_count}, expected_bytes={expected_bytes}, bytes={len(payload)}"
        )
    coefficients = np.frombuffer(payload, dtype="<f4", count=tap_count, offset=4).copy()
    if not np.isfinite(coefficients).all() or not np.any(coefficients):
        raise ValueError(f"ANIR FIR has invalid coefficients: {name}")
    return coefficients


def write_wac_fir(path: Path, coefficients: np.ndarray, sample_rate: int) -> None:
    values = np.asarray(coefficients, dtype="<f4")
    path.write_bytes(
        struct.pack("<i", values.size)
        + values.tobytes()
        + struct.pack("<i", sample_rate)
    )


def main() -> None:
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.source_rate < 1 or args.target_rate < 1:
        raise ValueError("Sample rates must be positive")
    if args.source_rate % args.target_rate:
        raise ValueError("This converter requires source_rate to be divisible by target_rate")
    if not archive.is_file():
        raise FileNotFoundError(archive)

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.fir"))
    if existing:
        raise FileExistsError(
            f"Output directory already contains {len(existing)} FIR files: {output_dir}"
        )

    selected: list[tuple[str, int, int]] = []
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            match = FIR_NAME.search(name)
            if not match:
                continue
            speaker, microphone = (int(value) for value in match.groups())
            if 12 <= speaker <= 17 and 0 <= microphone <= 18:
                selected.append((name, speaker, microphone))
        selected.sort(key=lambda item: (item[1], item[2], item[0]))
        if len(selected) != 114:
            raise ValueError(f"Expected 114 mouth-to-standard-microphone FIRs, found {len(selected)}")

        output_paths: list[Path] = []
        downsample = args.source_rate // args.target_rate
        source_taps = 0
        target_taps = 0
        for name, _speaker, _microphone in selected:
            source = read_anir_fir(handle.read(name), name)
            converted = resample_poly(source, up=1, down=downsample).astype(np.float32)
            source_taps = int(source.size)
            target_taps = int(converted.size)
            output_path = output_dir / Path(name).name
            write_wac_fir(output_path, converted, args.target_rate)
            output_paths.append(output_path)

    list_path = output_dir / "fir.list"
    list_path.write_text(
        "".join(f"{path.resolve()}\n" for path in output_paths),
        encoding="utf-8",
    )
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    (output_dir / "SOURCE.md").write_text(
        "\n".join(
            [
                "# ANIR V-Class FIR subset",
                "",
                f"- Source: {SOURCE_URL}",
                f"- Source archive: `{archive.name}`",
                f"- Source archive SHA-256: `{archive_sha256}`",
                f"- Source sample rate: {args.source_rate} Hz",
                f"- Prepared sample rate: {args.target_rate} Hz",
                "- Selection: HATS mouth loudspeakers 12-17 to standard car microphones 0-18.",
                "- Files: 114 WAC-compatible FIRs with an embedded 16000 Hz trailer.",
                "",
                "The ANIR authors request notification and citation when the database is used",
                "in published work; see the source page above.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"Prepared {len(output_paths)} FIRs in {output_dir}; "
        f"taps={source_taps}->{target_taps}; "
        f"list={list_path}"
    )


if __name__ == "__main__":
    main()
