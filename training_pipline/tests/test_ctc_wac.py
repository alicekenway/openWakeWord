"""Focused tests for the opt-in frozen-CTC + WAC pipeline."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wuw_training.artifacts import write_json, write_jsonl  # noqa: E402
from wuw_training.ctc_wac import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    CtcWacFeatureBlock,
    Stage1Contract,
    StreamingCtcStage1,
    ctc_keyword_score_trace,
    feature_bundle_paths,
    keyword_token_fingerprint,
    load_keywords,
    rank_keyword_scores,
)
from wuw_training.runner import PipelineRunner  # noqa: E402
from wuw_training.config import load_ini_config  # noqa: E402
from wuw_training.stages.train import FeatureBlock  # noqa: E402
from wuw_training.stages.testing import _ctc_wac_record  # noqa: E402


def _keyword_file(root: Path) -> Path:
    path = root / "keywords.json"
    write_json(
        path,
        {
            "keywords": [
                {"id": "wake_a", "display_text": "wake a", "token_ids": [1, 2], "threshold": -2.0},
                {"id": "wake_b", "display_text": "wake b", "token_ids": [2, 1], "threshold": -2.0},
            ]
        },
    )
    return path


def _write_bundle(path: Path, *, label: int, keywords_path: Path, seed: int) -> None:
    keywords = load_keywords(keywords_path)
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((4, 6, 5), dtype=np.float32)
    # Three rows pass stage 1. The final row must remain in the artifact but be
    # dropped only by the CTC-WAC train loader.
    scores = np.asarray(
        [[-1.0, -4.0], [-4.0, -1.1], [-1.2, -3.0], [-4.0, -3.0]], dtype=np.float32
    )
    top, margin, winner = rank_keyword_scores(scores)
    onehot = np.zeros_like(scores)
    onehot[np.arange(scores.shape[0]), winner] = 1.0
    paths = feature_bundle_paths(path)
    np.save(paths.features, features)
    np.save(paths.all_scores, scores)
    np.save(paths.top_score, top.reshape(-1, 1))
    np.save(paths.margin, margin.reshape(-1, 1))
    np.save(paths.winner_onehot, onehot)
    write_jsonl(paths.rows, [{"row": index, "label": label} for index in range(features.shape[0])])
    write_json(
        paths.summary,
        {
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "feature_count": int(features.shape[0]),
            "feature_shape": [6, 5],
            "keyword_count": len(keywords),
            "keyword_token_fingerprint": keyword_token_fingerprint(keywords),
            "error_count": 0,
        },
    )


def test_ctc_viterbi_trace_prefers_the_right_token_order() -> None:
    probabilities = np.asarray(
        [
            [0.05, 0.90, 0.05],  # token 1
            [0.75, 0.10, 0.15],  # blank separates the labels
            [0.05, 0.05, 0.90],  # token 2
        ],
        dtype=np.float32,
    )
    log_probs = np.log(probabilities)
    right = ctc_keyword_score_trace(log_probs, [1, 2], blank_id=0)
    wrong = ctc_keyword_score_trace(log_probs, [2, 1], blank_id=0)
    repeated = ctc_keyword_score_trace(log_probs[[0, 1, 0]], [1, 1], blank_id=0)
    assert np.isfinite(right).all()
    assert right[-1] > wrong[-1]
    assert np.isfinite(repeated[-1])


def test_generated_contract_fields_and_first_chunk_attention_mask(tmp_path: Path) -> None:
    contract_path = tmp_path / "stage1.contract.json"
    write_json(
        contract_path,
        {
            "sample_rate": 16000,
            "fbank": {"num_mel_bins": 80, "dither": 0.0},
            "chunk_frames": 67,
            "chunk_stride_frames": 64,
            "minimum_input_frames": 7,
            "pad_final_chunk": False,
            "initial_offset": 64,
            "inputs": {"features": "chunk", "offset": "offset"},
            "outputs": {"encoder": "encoder_out", "ctc_log_probs": "ctc_log_probs"},
            "ctc_output_is_log_probs": True,
            "attention_mask": {"input": "att_mask", "cache_frames": 64, "chunk_frames": 16},
        },
    )
    contract = Stage1Contract.from_json(contract_path)
    assert contract.chunk_frames == 67
    assert contract.chunk_stride_frames == 64
    assert contract.minimum_input_frames == 7
    assert contract.pad_final_chunk is False

    # This small unit test does not need ONNX Runtime.  It checks the mask
    # policy used by WeNet's own fixed-cache ONNX simulation.
    runner = StreamingCtcStage1.__new__(StreamingCtcStage1)
    runner.contract = contract
    runner._offset = contract.initial_offset
    runner._chunks_run = 0
    first = runner._streaming_attention_mask()
    assert not first[..., :64].any()
    assert first[..., 64:].all()
    runner._chunks_run = 1
    assert runner._streaming_attention_mask().all()


def test_bundle_keeps_all_rows_but_loader_applies_stage1_threshold(tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    bundle = tmp_path / "features.npy"
    _write_bundle(bundle, label=1, keywords_path=keywords_path, seed=7)
    block = FeatureBlock("feature.positive", bundle, 1, "train", (6, 5), 4)
    loaded = CtcWacFeatureBlock.from_feature_block(block, load_keywords(keywords_path))
    assert loaded.input_count == 4
    assert loaded.retained_count == 3
    assert loaded.filtering_summary()["dropped_rows"] == 1
    # Thresholds may change without invalidating frozen CTC encoder features:
    # only id/token mapping must stay the same.
    changed = tmp_path / "keywords_changed_threshold.json"
    write_json(
        changed,
        {
            "keywords": [
                {"id": "wake_a", "display_text": "wake a", "token_ids": [1, 2], "threshold": -0.9},
                {"id": "wake_b", "display_text": "wake b", "token_ids": [2, 1], "threshold": -0.9},
            ]
        },
    )
    changed_loaded = CtcWacFeatureBlock.from_feature_block(block, load_keywords(changed))
    assert changed_loaded.retained_count == 0


def test_cascade_record_feeds_all_four_wac_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    keywords_path = _keyword_file(tmp_path)
    keywords = load_keywords(keywords_path)

    class FakeStage1:
        contract = SimpleNamespace(sample_rate=16000, blank_id=0)

        def infer_fbank(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            encoder = np.arange(20, dtype=np.float32).reshape(4, 5)
            probabilities = np.asarray(
                [[0.05, 0.90, 0.05], [0.80, 0.10, 0.10], [0.05, 0.05, 0.90], [0.8, 0.1, 0.1]],
                dtype=np.float32,
            )
            return encoder, np.log(probabilities)

    class FakeStage2:
        def __init__(self) -> None:
            self.feed: dict[str, np.ndarray] | None = None

        def run(self, _outputs: object, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            self.feed = feed
            return [np.asarray([[0.75]], dtype=np.float32)]

    monkeypatch.setattr("wuw_training.stages.testing.load_audio", lambda path, sample_rate: np.zeros(32000, dtype=np.float32))
    monkeypatch.setattr(
        "wuw_training.stages.testing.audio_to_fbank",
        lambda audio, contract: np.zeros((4, 80), dtype=np.float32),
    )
    stage2 = FakeStage2()
    detail = _ctc_wac_record(
        record={"id": "one", "path": str(tmp_path / "audio.wav"), "text": "wake a"},
        stage1=FakeStage1(),  # type: ignore[arg-type]
        keywords=keywords,
        stage2=stage2,
        time_steps=6,
        feature_dim=5,
        expected_label=1,
    )
    assert detail["stage1_candidate_count"] >= 1
    assert stage2.feed is not None
    assert set(stage2.feed) == {"encoder_features", "top_score", "margin", "winner_onehot"}
    assert stage2.feed["encoder_features"].shape == (1, 6, 5)
    assert stage2.feed["winner_onehot"].shape == (1, 2)


def test_ctc_wac_train_and_export_four_input_onnx(tmp_path: Path) -> None:
    has_onnx = importlib.util.find_spec("onnx") is not None
    has_ort = importlib.util.find_spec("onnxruntime") is not None
    if has_ort:
        import onnxruntime as ort
    else:
        ort = None
    keywords_path = _keyword_file(tmp_path)
    names = {
        "positive_train": (1, "train", 11),
        "negative_train": (0, "train", 12),
        "positive_dev": (1, "dev", 13),
        "negative_dev": (0, "dev", 14),
    }
    paths: dict[str, Path] = {}
    for name, (label, _split, seed) in names.items():
        path = tmp_path / f"{name}.npy"
        _write_bundle(path, label=label, keywords_path=keywords_path, seed=seed)
        paths[name] = path
    config = tmp_path / "pipeline.ini"
    config.write_text(
        f"""[main]
experiment_dir = {tmp_path / 'experiment'}
model_name = test_ctc_wac
sample_rate = 16000
seed = 7

[steps]
steps = {"train, export" if has_onnx else "train"}

[feature.positive_train]
output_file = {paths['positive_train']}
label = 1
split = train

[feature.negative_train]
output_file = {paths['negative_train']}
label = 0
split = train

[feature.positive_dev]
output_file = {paths['positive_dev']}
label = 1
split = dev

[feature.negative_dev]
output_file = {paths['negative_dev']}
label = 0
split = dev

[train]
structure = ctc_wac
keywords = {keywords_path}
train = feature.positive_train, feature.negative_train
dev = feature.positive_dev, feature.negative_dev
batch.feature.positive_train = 2
batch.feature.negative_train = 2
steps = 1
phase_step_ratios = [1.0]
phase_learning_rates = [0.001]
wac.frame_hidden = 8
wac.frame_layers = 1
wac.head_hidden = 8
wac.dropout = 0.0
max_negative_weight = 1
checkpoint_interval_steps = 1
keep_checkpoints = 1
log_interval_steps = 1
resume = yes
model_checkpoint_dir = ${{main:experiment_dir}}/checkpoints
output_model = ${{main:experiment_dir}}/model.pt
output_summary = ${{main:experiment_dir}}/training.json

[export]
input_model = ${{train:output_model}}
output_model = ${{main:experiment_dir}}/model.onnx
verify = yes
opset_version = 13
""",
        encoding="utf-8",
    )
    result = PipelineRunner(load_ini_config(config)).run()
    assert result["steps"]["train"]["result"]["structure"] == "ctc_wac"
    if not has_onnx or ort is None:
        return
    session = ort.InferenceSession(str(tmp_path / "experiment" / "model.onnx"), providers=["CPUExecutionProvider"])
    assert {item.name for item in session.get_inputs()} == {
        "encoder_features",
        "top_score",
        "margin",
        "winner_onehot",
    }
    output = session.run(
        None,
        {
            "encoder_features": np.zeros((1, 6, 5), dtype=np.float32),
            "top_score": np.zeros((1, 1), dtype=np.float32),
            "margin": np.zeros((1, 1), dtype=np.float32),
            "winner_onehot": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
    )[0]
    assert output.shape == (1, 1)
    assert 0.0 <= float(output[0, 0]) <= 1.0
