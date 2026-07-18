"""Export a trained PyTorch wake-word checkpoint to a verified ONNX model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..artifacts import read_json, write_json
from ..config import ConfigurationError
from ..ctc_wac import make_ctc_wac_model
from .common import boolean, integer, require


def _input_model(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "input_model", ctx.step))


def _output_model(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_model", ctx.step))


def _output_summary(ctx: Any) -> Path:
    value = ctx.section.get("output_summary")
    return ctx.config.resolve_path(value) if value else _output_model(ctx).with_suffix(".export.json")


def validate(ctx: Any) -> None:
    if integer(ctx.section, "opset_version", ctx.step, 13) < 11:
        raise ConfigurationError(f"[{ctx.step}] opset_version must be >= 11")
    _input_model(ctx)
    _output_model(ctx)


def input_paths(ctx: Any) -> list[Path]:
    return [_input_model(ctx)]


def output_paths(ctx: Any) -> list[Path]:
    return [_output_model(ctx), _output_summary(ctx)]


def validate_outputs(ctx: Any) -> bool:
    model, summary = output_paths(ctx)
    if not model.is_file() or not summary.is_file() or model.stat().st_size == 0:
        return False
    try:
        value = read_json(summary)
        return value.get("output_model") == str(model)
    except Exception:
        return False


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or "model_state_dict" not in value:
        raise ConfigurationError(f"Training model is not a pipeline checkpoint with model_state_dict: {path}")
    return value


def _verify(model: torch.nn.Module, output: Path, input_shape: tuple[int, ...]) -> float:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is required to verify exported models") from exc
    rng = np.random.default_rng(1337)
    example = rng.standard_normal((1, *input_shape), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(example)).detach().cpu().numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: example})[0]
    difference = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"ONNX verification failed; maximum absolute difference is {difference}")
    return difference


def _verify_ctc_wac(
    model: torch.nn.Module,
    output: Path,
    *,
    time_steps: int,
    feature_dim: int,
    keyword_count: int,
) -> float:
    """Verify the four-input WAC ONNX interface against PyTorch."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("ONNX Runtime is required to verify exported models") from exc
    rng = np.random.default_rng(1337)
    features = rng.standard_normal((2, time_steps, feature_dim), dtype=np.float32)
    top_score = rng.standard_normal((2, 1), dtype=np.float32)
    margin = rng.standard_normal((2, 1), dtype=np.float32)
    winner = np.zeros((2, keyword_count), dtype=np.float32)
    winner[np.arange(2), np.arange(2) % keyword_count] = 1.0
    with torch.inference_mode():
        expected = model(
            torch.from_numpy(features),
            torch.from_numpy(top_score),
            torch.from_numpy(margin),
            torch.from_numpy(winner),
        ).detach().cpu().numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {
            "encoder_features": features,
            "top_score": top_score,
            "margin": margin,
            "winner_onehot": winner,
        },
    )[0]
    difference = float(np.max(np.abs(expected - actual)))
    if not np.allclose(expected, actual, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f"CTC-WAC ONNX verification failed; maximum absolute difference is {difference}")
    return difference


def _run_ctc_wac_export(ctx: Any, checkpoint: dict[str, Any]) -> dict[str, Any]:
    try:
        time_steps = int(checkpoint["time_steps"])
        feature_dim = int(checkpoint["feature_dim"])
        keyword_count = int(checkpoint["keyword_count"])
        model_config = dict(checkpoint["model_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"CTC-WAC checkpoint has incomplete architecture metadata: {_input_model(ctx)}"
        ) from exc
    normalization = dict(checkpoint.get("scalar_normalization", {}))
    network = make_ctc_wac_model(
        feature_dim=feature_dim,
        keyword_count=keyword_count,
        model_config=model_config,
        score_mean=float(normalization.get("score_mean", 0.0)),
        score_std=float(normalization.get("score_std", 1.0)),
        margin_mean=float(normalization.get("margin_mean", 0.0)),
        margin_std=float(normalization.get("margin_std", 1.0)),
    ).cpu().eval()
    network.load_state_dict(checkpoint["model_state_dict"])
    output = _output_model(ctx)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.onnx")
    examples = (
        torch.rand((1, time_steps, feature_dim), dtype=torch.float32),
        torch.rand((1, 1), dtype=torch.float32),
        torch.rand((1, 1), dtype=torch.float32),
        torch.nn.functional.one_hot(torch.zeros(1, dtype=torch.long), num_classes=keyword_count).to(torch.float32),
    )
    opset = integer(ctx.section, "opset_version", ctx.step, 13)
    torch.onnx.export(
        network,
        examples,
        temporary,
        input_names=["encoder_features", "top_score", "margin", "winner_onehot"],
        output_names=["wake_probability"],
        dynamic_axes={
            "encoder_features": {0: "batch"},
            "top_score": {0: "batch"},
            "margin": {0: "batch"},
            "winner_onehot": {0: "batch"},
            "wake_probability": {0: "batch"},
        },
        opset_version=opset,
    )
    os.replace(temporary, output)
    maximum_difference = None
    if boolean(ctx.section, "verify", ctx.step, True):
        maximum_difference = _verify_ctc_wac(
            network,
            output,
            time_steps=time_steps,
            feature_dim=feature_dim,
            keyword_count=keyword_count,
        )
    payload = {
        "structure": "ctc_wac",
        "input_model": str(_input_model(ctx)),
        "output_model": str(output),
        "input_shape": [time_steps, feature_dim],
        "keyword_count": keyword_count,
        "keyword_ids": list(checkpoint.get("keyword_ids", [])),
        "keywords_fingerprint": checkpoint.get("keywords_fingerprint"),
        "model_config": model_config,
        "opset_version": opset,
        "verified": boolean(ctx.section, "verify", ctx.step, True),
        "maximum_absolute_difference": maximum_difference,
        "inputs": ["encoder_features", "top_score", "margin", "winner_onehot"],
        "output": "wake_probability",
    }
    write_json(_output_summary(ctx), payload)
    return payload


def run(ctx: Any) -> dict[str, Any]:
    checkpoint = _load_checkpoint(_input_model(ctx))
    if checkpoint.get("structure") == "ctc_wac":
        return _run_ctc_wac_export(ctx, checkpoint)
    from openwakeword.train import Model as TrainModel

    try:
        input_shape = tuple(int(value) for value in checkpoint["input_shape"])
        model_type = str(checkpoint["model_type"])
        layer_size = int(checkpoint["layer_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Training checkpoint has incomplete architecture metadata: {_input_model(ctx)}") from exc

    trainer = TrainModel(
        n_classes=int(checkpoint.get("n_classes", 1)),
        input_shape=input_shape,
        model_type=model_type,
        layer_dim=layer_size,
        seconds_per_example=float(checkpoint.get("seconds_per_example", 0.0)) or None,
        model_kwargs=dict(checkpoint.get("model_config", {})),
    )
    network = trainer.model.cpu().eval()
    network.load_state_dict(checkpoint["model_state_dict"])
    output = _output_model(ctx)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.onnx")
    example = torch.rand((1, *input_shape), dtype=torch.float32)
    requested_opset = integer(ctx.section, "opset_version", ctx.step, 13)
    # PyTorch exports MultiheadAttention through scaled_dot_product_attention,
    # which was added to ONNX in opset 14.
    effective_opset = max(requested_opset, 14) if model_type == "attention" else requested_opset
    torch.onnx.export(
        network,
        (example,),
        temporary,
        opset_version=effective_opset,
    )
    os.replace(temporary, output)

    maximum_difference = None
    if boolean(ctx.section, "verify", ctx.step, True):
        maximum_difference = _verify(network, output, input_shape)
    payload = {
        "input_model": str(_input_model(ctx)),
        "output_model": str(output),
        "input_shape": list(input_shape),
        "model_type": model_type,
        "layer_size": layer_size,
        "requested_opset_version": requested_opset,
        "opset_version": effective_opset,
        "verified": boolean(ctx.section, "verify", ctx.step, True),
        "maximum_absolute_difference": maximum_difference,
    }
    write_json(_output_summary(ctx), payload)
    return payload
