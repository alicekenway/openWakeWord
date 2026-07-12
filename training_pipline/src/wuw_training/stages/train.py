"""Resumable three-phase training from feature-stage NPY outputs."""

from __future__ import annotations

import copy
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as functional

from ..artifacts import file_signature, hash_payload, read_json, write_json
from ..config import ConfigurationError, parse_json
from .common import boolean, csv_option, integer, number, optional_integer, optional_number, require


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    path: Path
    label: int
    split: str
    shape: tuple[int, ...]
    rows: int


def _feature_block(ctx: Any, name: str) -> FeatureBlock:
    if not name.startswith("feature."):
        raise ConfigurationError(f"[{ctx.step}] feature reference must name a feature.* block, got {name!r}")
    values = ctx.config.section(name)
    output = values.get("output_file")
    if not output:
        raise ConfigurationError(f"[{name}] is missing output_file")
    try:
        label = int(values.get("label", ""))
    except ValueError as exc:
        raise ConfigurationError(f"[{name}] label must be 0 or 1") from exc
    split = values.get("split", "").lower()
    if label not in {0, 1} or split not in {"train", "dev", "test", "false_positive"}:
        raise ConfigurationError(f"[{name}] must define label = 0|1 and a valid split")
    path = ctx.config.resolve_path(output)
    if path.is_file():
        try:
            array = np.load(path, mmap_mode="r")
            if array.ndim < 2 or array.shape[0] < 1:
                raise ValueError("expected a non-empty feature array")
            return FeatureBlock(name, path, label, split, tuple(int(v) for v in array.shape[1:]), int(array.shape[0]))
        except Exception as exc:
            raise ConfigurationError(f"[{name}] output_file is not a valid non-empty NPY array: {path}: {exc}") from exc
    # A preceding feature stage may create this later in the declared pipeline.
    return FeatureBlock(name, path, label, split, (), 0)


def _references(ctx: Any, option: str) -> list[str]:
    return csv_option(ctx.section, option, ctx.step)


def _blocks(ctx: Any, option: str) -> list[FeatureBlock]:
    return [_feature_block(ctx, name) for name in _references(ctx, option)]


def _phase_plan(ctx: Any) -> list[dict[str, float | int]]:
    base_steps = integer(ctx.section, "steps", ctx.step)
    if base_steps < 1:
        raise ConfigurationError(f"[{ctx.step}] steps must be >= 1")
    ratios = parse_json(ctx.section.get("phase_step_ratios", "[1.0, 0.1, 0.1]"), f"[{ctx.step}] phase_step_ratios", list)
    rates = parse_json(ctx.section.get("phase_learning_rates", "[0.0001, 0.00001, 0.000001]"), f"[{ctx.step}] phase_learning_rates", list)
    if not ratios or len(ratios) != len(rates):
        raise ConfigurationError(f"[{ctx.step}] phase_step_ratios and phase_learning_rates must be non-empty lists of equal length")
    plan: list[dict[str, float | int]] = []
    for index, (ratio, rate) in enumerate(zip(ratios, rates)):
        try:
            ratio_value = float(ratio)
            rate_value = float(rate)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"[{ctx.step}] invalid phase {index} ratio or learning rate") from exc
        if ratio_value <= 0 or rate_value <= 0:
            raise ConfigurationError(f"[{ctx.step}] phase ratios and learning rates must be > 0")
        plan.append({"steps": max(1, int(round(base_steps * ratio_value))), "learning_rate": rate_value})
    return plan


def _json_int_list(ctx: Any, key: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = ctx.section.get(key)
    if raw is None:
        return default
    value = parse_json(raw, f"[{ctx.step}] {key}", list)
    if not value:
        raise ConfigurationError(f"[{ctx.step}] {key} cannot be empty")
    try:
        values = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"[{ctx.step}] {key} must be a JSON integer list") from exc
    return values


def _json_bool_list(ctx: Any, key: str, default: tuple[bool, ...]) -> tuple[bool, ...]:
    raw = ctx.section.get(key)
    if raw is None:
        return default
    value = parse_json(raw, f"[{ctx.step}] {key}", list)
    if not value or not all(isinstance(item, bool) for item in value):
        raise ConfigurationError(f"[{ctx.step}] {key} must be a non-empty JSON boolean list")
    return tuple(value)


def _model_config(ctx: Any, input_shape: tuple[int, ...] | tuple[()]) -> dict[str, Any]:
    """Translate documented INI model.* options into a classifier-head config."""
    model_type = ctx.section.get("model_type", "dnn")
    if model_type in {"dnn", "rnn"}:
        return {}

    channels = integer(ctx.section, "model.channels", ctx.step, 128)
    expansion = integer(ctx.section, "model.expansion", ctx.step, 1)
    dropout = number(ctx.section, "model.dropout", ctx.step, 0.05)
    classifier_hidden = integer(ctx.section, "model.classifier_hidden", ctx.step, 64)
    if channels < 1 or expansion < 1 or classifier_hidden < 1:
        raise ConfigurationError(f"[{ctx.step}] model.channels, model.expansion, and model.classifier_hidden must be >= 1")
    if not 0.0 <= dropout < 1.0:
        raise ConfigurationError(f"[{ctx.step}] model.dropout must be in [0, 1)")
    input_dim = int(input_shape[-1]) if input_shape else None

    if model_type == "cnn":
        kernels = _json_int_list(ctx, "model.cnn_kernels", (3, 5, 3, 3))
        dilations = _json_int_list(ctx, "model.cnn_dilations", (1, 1, 2, 4))
        use_se = _json_bool_list(ctx, "model.cnn_use_se", (False, False, True, True))
        if len(kernels) != len(dilations) or len(kernels) != len(use_se):
            raise ConfigurationError(f"[{ctx.step}] CNN kernel, dilation, and SE lists must have equal lengths")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ConfigurationError(f"[{ctx.step}] model.cnn_kernels must contain positive odd values")
        if any(dilation < 1 for dilation in dilations):
            raise ConfigurationError(f"[{ctx.step}] model.cnn_dilations must contain values >= 1")
        config: dict[str, Any] = {
            "channels": channels,
            "expansion": expansion,
            "dropout": dropout,
            "kernels": kernels,
            "dilations": dilations,
            "use_se": use_se,
            "classifier_hidden": classifier_hidden,
        }
        if input_dim is not None:
            config["input_dim"] = input_dim
        return config

    if model_type == "attention":
        kernels = _json_int_list(ctx, "model.attention_local_kernels", (3, 3))
        dilations = _json_int_list(ctx, "model.attention_local_dilations", (1, 2))
        use_se = _json_bool_list(ctx, "model.attention_local_use_se", (False, False))
        heads = integer(ctx.section, "model.attention_num_heads", ctx.step, 4)
        ff_multiplier = integer(ctx.section, "model.attention_ff_multiplier", ctx.step, 2)
        configured_steps = optional_integer(ctx.section, "model.attention_time_steps", ctx.step)
        if heads < 1:
            raise ConfigurationError(f"[{ctx.step}] model.attention_num_heads must be >= 1")
        if channels % heads != 0:
            raise ConfigurationError(f"[{ctx.step}] model.channels must be divisible by model.attention_num_heads")
        if ff_multiplier < 1:
            raise ConfigurationError(f"[{ctx.step}] model.attention_ff_multiplier must be >= 1")
        if configured_steps is not None and configured_steps < 1:
            raise ConfigurationError(f"[{ctx.step}] model.attention_time_steps must be >= 1")
        if len(kernels) != len(dilations) or len(kernels) != len(use_se):
            raise ConfigurationError(f"[{ctx.step}] attention local kernel, dilation, and SE lists must have equal lengths")
        if any(kernel < 1 or kernel % 2 == 0 for kernel in kernels):
            raise ConfigurationError(f"[{ctx.step}] model.attention_local_kernels must contain positive odd values")
        if any(dilation < 1 for dilation in dilations):
            raise ConfigurationError(f"[{ctx.step}] model.attention_local_dilations must contain values >= 1")
        if input_shape and configured_steps is not None and configured_steps != input_shape[0]:
            raise ConfigurationError(
                f"[{ctx.step}] model.attention_time_steps={configured_steps} does not match feature time dimension {input_shape[0]}"
            )
        config = {
            "channels": channels,
            "expansion": expansion,
            "dropout": dropout,
            "num_heads": heads,
            "ff_multiplier": ff_multiplier,
            "local_kernels": kernels,
            "local_dilations": dilations,
            "local_use_se": use_se,
            "classifier_hidden": classifier_hidden,
        }
        if input_dim is not None:
            config["input_dim"] = input_dim
        if input_shape:
            config["time_steps"] = int(input_shape[0])
        elif configured_steps is not None:
            config["time_steps"] = configured_steps
        return config
    raise ConfigurationError(f"[{ctx.step}] model_type must be dnn, rnn, cnn, or attention")


def _batch_count(ctx: Any, block_name: str) -> int:
    key = f"batch.{block_name}"
    if key not in ctx.section:
        raise ConfigurationError(f"[{ctx.step}] is missing {key} for training block {block_name}")
    value = integer(ctx.section, key, ctx.step)
    if value < 1:
        raise ConfigurationError(f"[{ctx.step}] {key} must be >= 1")
    return value


def _checkpoint_dir(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "model_checkpoint_dir", ctx.step))


def _output_model(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_model", ctx.step))


def _output_summary(ctx: Any) -> Path:
    return ctx.config.resolve_path(require(ctx.section, "output_summary", ctx.step))


def _training_log(ctx: Any) -> Path:
    configured = ctx.section.get("training_log_file")
    return ctx.config.resolve_path(configured) if configured else _output_summary(ctx).with_suffix(".jsonl")


def _normalise_config_for_fingerprint(
    ctx: Any,
    train: list[FeatureBlock],
    dev: list[FeatureBlock],
    false_positive: list[FeatureBlock],
) -> str:
    return hash_payload(
        {
            "section": ctx.section,
            "train": [block.name for block in train],
            "dev": [block.name for block in dev],
            "false_positive": [block.name for block in false_positive],
            "pipeline_training_schema": 1,
        }
    )


def _feature_signature(blocks: Iterable[FeatureBlock]) -> str:
    return hash_payload(
        [
            {
                "name": block.name,
                "label": block.label,
                "split": block.split,
                "file": file_signature(block.path),
            }
            for block in blocks
        ]
    )


def validate(ctx: Any) -> None:
    train = _blocks(ctx, "train")
    dev = _blocks(ctx, "dev")
    false_positive = _blocks(ctx, "false_positive")
    if not any(block.label == 1 for block in train) or not any(block.label == 0 for block in train):
        raise ConfigurationError(f"[{ctx.step}] train must contain at least one positive and one negative feature block")
    if not any(block.label == 1 for block in dev) or not any(block.label == 0 for block in dev):
        raise ConfigurationError(f"[{ctx.step}] dev must contain at least one positive and one negative feature block")
    if any(block.label != 0 for block in false_positive):
        raise ConfigurationError(f"[{ctx.step}] false_positive may contain only label = 0 feature blocks")
    for block in train:
        if block.split != "train":
            raise ConfigurationError(f"[{ctx.step}] train block {block.name} has split={block.split!r}, expected train")
        _batch_count(ctx, block.name)
    for block in dev:
        if block.split != "dev":
            raise ConfigurationError(f"[{ctx.step}] dev block {block.name} has split={block.split!r}, expected dev")
    for block in false_positive:
        if block.split not in {"dev", "false_positive"}:
            raise ConfigurationError(f"[{ctx.step}] false_positive block {block.name} must use split=dev or false_positive")
    train_names = {block.name for block in train}
    extra_batch_options = [
        key
        for key in ctx.section
        if key.startswith("batch.") and key[len("batch."):] not in train_names
    ]
    if extra_batch_options:
        raise ConfigurationError(f"[{ctx.step}] batch option(s) do not name a listed train block: {', '.join(extra_batch_options)}")
    shapes = {block.shape for block in [*train, *dev, *false_positive] if block.shape}
    if len(shapes) > 1:
        raise ConfigurationError(f"[{ctx.step}] all feature arrays must have the same shape, found {sorted(shapes)}")
    if ctx.section.get("model_type", "dnn") not in {"dnn", "rnn", "cnn", "attention"}:
        raise ConfigurationError(f"[{ctx.step}] model_type must be dnn, rnn, cnn, or attention")
    known_shape = next((block.shape for block in [*train, *dev, *false_positive] if block.shape), ())
    _model_config(ctx, known_shape)
    _phase_plan(ctx)
    if number(ctx.section, "max_negative_weight", ctx.step, 100.0) < 1:
        raise ConfigurationError(f"[{ctx.step}] max_negative_weight must be >= 1")
    if integer(ctx.section, "checkpoint_interval_steps", ctx.step, 500) < 1:
        raise ConfigurationError(f"[{ctx.step}] checkpoint_interval_steps must be >= 1")
    if integer(ctx.section, "keep_checkpoints", ctx.step, 3) < 1:
        raise ConfigurationError(f"[{ctx.step}] keep_checkpoints must be >= 1")
    if integer(ctx.section, "log_interval_steps", ctx.step, 100) < 1:
        raise ConfigurationError(f"[{ctx.step}] log_interval_steps must be >= 1")
    validation_interval = optional_integer(ctx.section, "validation_interval_steps", ctx.step)
    if validation_interval is not None and validation_interval < 1:
        raise ConfigurationError(f"[{ctx.step}] validation_interval_steps must be >= 1")
    if boolean(ctx.section, "require_cuda", ctx.step, False) and not torch.cuda.is_available():
        raise ConfigurationError(f"[{ctx.step}] require_cuda=yes but CUDA is unavailable")
    _checkpoint_dir(ctx)
    _output_model(ctx)
    _output_summary(ctx)


def input_paths(ctx: Any) -> list[Path]:
    return [block.path for block in [*_blocks(ctx, "train"), *_blocks(ctx, "dev"), *_blocks(ctx, "false_positive")]]


def output_paths(ctx: Any) -> list[Path]:
    return [_output_model(ctx), _output_summary(ctx)]


def validate_outputs(ctx: Any) -> bool:
    model_path, summary_path = output_paths(ctx)
    if not model_path.is_file() or not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
        return summary.get("output_model") == str(model_path) and int(summary.get("completed_phases", 0)) >= 1
    except Exception:
        return False


class NpyBatchSampler:
    def __init__(self, groups: list[tuple[FeatureBlock, int]], seed: int):
        self.groups = []
        for block, batch_count in groups:
            values = np.load(block.path, mmap_mode="r")
            self.groups.append((block, int(batch_count), values))
        self.rng = np.random.default_rng(seed)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        arrays: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for block, batch_count, values in self.groups:
            indices = self.rng.integers(0, int(values.shape[0]), size=batch_count)
            arrays.append(np.asarray(values[indices], dtype=np.float32))
            labels.append(np.full(batch_count, block.label, dtype=np.float32))
        features = np.concatenate(arrays, axis=0)
        target = np.concatenate(labels, axis=0)
        order = self.rng.permutation(target.shape[0])
        return features[order], target[order]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.rng.bit_generator.state))

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state


def _cpu_state_dict(network: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in network.state_dict().items()}


def _weighted_probability_bce(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    negative_weight: float,
) -> torch.Tensor:
    """Compute probability-space BCE safely outside mixed-precision autocast.

    All openWakeWord classifier wrappers include their sigmoid because exported
    models and runtime callers consume probabilities. PyTorch deliberately
    rejects ``binary_cross_entropy`` under autocast, so keep the model forward
    mixed precision but perform this loss calculation in float32.
    """
    with torch.autocast(device_type=predictions.device.type, enabled=False):
        probabilities = predictions.float()
        targets = labels.float()
        weights = torch.where(
            targets == 0,
            torch.full_like(targets, float(negative_weight)),
            torch.ones_like(targets),
        )
        return functional.binary_cross_entropy(probabilities, targets, weight=weights)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ResumableAutoTrainer:
    """The legacy auto-training policy with explicit, durable state transitions."""

    def __init__(
        self,
        ctx: Any,
        train_blocks: list[FeatureBlock],
        dev_blocks: list[FeatureBlock],
        false_positive_blocks: list[FeatureBlock],
    ):
        from openwakeword.train import Model as TrainModel

        self.ctx = ctx
        self.train_blocks = train_blocks
        self.dev_blocks = dev_blocks
        self.false_positive_blocks = false_positive_blocks
        self.feature_shape = train_blocks[0].shape
        self.model_config = _model_config(ctx, self.feature_shape)
        self.plan = _phase_plan(ctx)
        self.config_fingerprint = _normalise_config_for_fingerprint(ctx, train_blocks, dev_blocks, false_positive_blocks)
        self.inputs_fingerprint = _feature_signature([*train_blocks, *dev_blocks, *false_positive_blocks])
        self.seed = integer(ctx.config.section("main"), "seed", "main", 1337)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_amp = boolean(ctx.section, "mixed_precision", ctx.step, False) and self.device.type == "cuda"
        self.model_wrapper = TrainModel(
            n_classes=1,
            input_shape=self.feature_shape,
            model_type=ctx.section.get("model_type", "dnn"),
            layer_dim=integer(ctx.section, "layer_size", ctx.step, 64),
            seconds_per_example=1280 * self.feature_shape[0] / integer(ctx.config.section("main"), "sample_rate", "main", 16000),
            model_kwargs=self.model_config,
        )
        self.network = self.model_wrapper.model.to(self.device)
        self.optimizer = self.model_wrapper.optimizer
        self.optimizer.param_groups[0]["lr"] = float(self.plan[0]["learning_rate"])
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except (AttributeError, TypeError):  # PyTorch compatibility path
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.sampler = NpyBatchSampler([(block, _batch_count(ctx, block.name)) for block in train_blocks], self.seed)
        self.phase_index = 0
        self.phase_step = 0
        self.global_step = 0
        self.current_max_negative_weight = number(ctx.section, "max_negative_weight", ctx.step, 100.0)
        self.history: dict[str, list[float]] = defaultdict(list)
        self.best_model_states: list[dict[str, torch.Tensor]] = []
        self.best_model_scores: list[dict[str, float]] = []
        self.started_at = time.time()
        self.log_path = _training_log(ctx)
        self.log_session = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"

    @property
    def checkpoint_dir(self) -> Path:
        return _checkpoint_dir(self.ctx)

    def _log_event(self, event: str, message: str, **values: Any) -> None:
        record = {
            "event": event,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session": self.log_session,
            "elapsed_seconds": round(time.time() - self.started_at, 6),
            "global_step": self.global_step,
            "phase_index": self.phase_index,
            "phase_step": self.phase_step,
            **values,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
        print(message, flush=True)

    def _checkpoint_payload(self) -> dict[str, Any]:
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return {
            "schema_version": 1,
            "config_fingerprint": self.config_fingerprint,
            "inputs_fingerprint": self.inputs_fingerprint,
            "model_state_dict": _cpu_state_dict(self.network),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "phase_index": self.phase_index,
            "phase_step": self.phase_step,
            "global_step": self.global_step,
            "current_max_negative_weight": self.current_max_negative_weight,
            "history": dict(self.history),
            "best_model_states": self.best_model_states,
            "best_model_scores": self.best_model_scores,
            "sampler_state": self.sampler.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng,
        }

    def _save_checkpoint(self) -> None:
        payload = self._checkpoint_payload()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        numbered = self.checkpoint_dir / f"checkpoint_{self.global_step:09d}.pt"
        _atomic_torch_save(numbered, payload)
        _atomic_torch_save(self.checkpoint_dir / "latest.pt", payload)
        keep = integer(self.ctx.section, "keep_checkpoints", self.ctx.step, 3)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        for stale in checkpoints[:-keep]:
            stale.unlink()

    def _restore_optimizer_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _try_resume(self) -> bool:
        candidates = [self.checkpoint_dir / "latest.pt"] + sorted(self.checkpoint_dir.glob("checkpoint_*.pt"), reverse=True)
        visited: set[Path] = set()
        for candidate in candidates:
            if candidate in visited or not candidate.is_file():
                continue
            visited.add(candidate)
            try:
                state = _torch_load(candidate)
                if not isinstance(state, dict):
                    continue
                if state.get("config_fingerprint") != self.config_fingerprint or state.get("inputs_fingerprint") != self.inputs_fingerprint:
                    continue
                self.network.load_state_dict(state["model_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                self._restore_optimizer_device()
                if self.use_amp and state.get("scaler_state_dict"):
                    self.scaler.load_state_dict(state["scaler_state_dict"])
                self.phase_index = int(state["phase_index"])
                self.phase_step = int(state["phase_step"])
                self.global_step = int(state["global_step"])
                self.current_max_negative_weight = float(state["current_max_negative_weight"])
                self.history = defaultdict(list, {key: list(value) for key, value in state.get("history", {}).items()})
                self.best_model_states = list(state.get("best_model_states", []))
                self.best_model_scores = list(state.get("best_model_scores", []))
                self.sampler.load_state_dict(state["sampler_state"])
                random.setstate(state["python_random_state"])
                np.random.set_state(state["numpy_random_state"])
                torch.set_rng_state(state["torch_rng_state"])
                if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
                return True
            except Exception:
                continue
        return False

    def _seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _learning_rate(self, phase_step: int, phase_steps: int, target: float) -> float:
        warmup = int(phase_steps * number(self.ctx.section, "warmup_ratio", self.ctx.step, 0.2))
        hold = int(phase_steps * number(self.ctx.section, "hold_ratio", self.ctx.step, 1.0 / 3.0))
        if warmup > 0 and phase_step < warmup:
            return target * (phase_step / warmup)
        decay_steps = max(1, phase_steps - warmup - hold)
        if phase_step <= warmup + hold:
            return target
        progress = min(1.0, (phase_step - warmup - hold) / decay_steps)
        return 0.5 * target * (1.0 + float(np.cos(np.pi * progress)))

    def _validation_schedule(self, phase_index: int, phase_steps: int) -> set[int]:
        interval = optional_integer(self.ctx.section, "validation_interval_steps", self.ctx.step)
        if interval is not None:
            return set(range(interval - 1, phase_steps, interval)) | {phase_steps - 1}
        points = max(1, integer(self.ctx.section, "validation_points", self.ctx.step, 20))
        start = int(phase_steps * 0.75) if phase_index == 0 else 0
        return set(int(value) for value in np.linspace(start, phase_steps - 1, points, dtype=np.int64)) | {phase_steps - 1}

    def _scores(self, blocks: list[FeatureBlock]) -> tuple[int, int, int, int, int, int]:
        total = 0
        correct = 0
        positives = 0
        positive_detected = 0
        negatives = 0
        false_positives = 0
        self.network.eval()
        with torch.inference_mode():
            for block in blocks:
                values = np.load(block.path, mmap_mode="r")
                for start in range(0, int(values.shape[0]), 8192):
                    batch = np.array(values[start:start + 8192], dtype=np.float32, copy=True)
                    x = torch.from_numpy(batch).to(self.device)
                    predictions = self.network(x).reshape(-1) >= 0.5
                    count = int(predictions.numel())
                    total += count
                    if block.label == 1:
                        positives += count
                        positive_detected += int(predictions.sum().item())
                        correct += int(predictions.sum().item())
                    else:
                        negatives += count
                        fp = int(predictions.sum().item())
                        false_positives += fp
                        correct += count - fp
        return total, correct, positive_detected, positives, false_positives, negatives

    def _validate(self) -> dict[str, float]:
        total, correct, detected, positives, val_fp, _ = self._scores(self.dev_blocks)
        _, _, _, _, fp_count, fp_rows = self._scores(self.false_positive_blocks)
        sample_rate = integer(self.ctx.config.section("main"), "sample_rate", "main", 16000)
        seconds_per_example = 1280 * self.feature_shape[0] / sample_rate
        configured_hours = optional_number(self.ctx.section, "false_positive_hours", self.ctx.step)
        validation_hours = configured_hours if configured_hours is not None else (fp_rows * seconds_per_example / 3600.0)
        score = {
            "global_step": float(self.global_step),
            "val_accuracy": (correct / total) if total else 0.0,
            "val_recall": (detected / positives) if positives else 0.0,
            "val_n_fp": float(val_fp),
            "val_fp_per_hr": (fp_count / validation_hours) if validation_hours else 0.0,
        }
        for key, value in score.items():
            self.history[key].append(float(value))
        recalls = np.asarray(self.history["val_recall"], dtype=float)
        n_fps = np.asarray(self.history["val_n_fp"], dtype=float)
        eligible = (
            score["val_n_fp"] <= float(np.percentile(n_fps, 50))
            and score["val_recall"] >= float(np.percentile(recalls, 5))
        )
        if eligible:
            self.best_model_states.append(_cpu_state_dict(self.network))
            self.best_model_scores.append(dict(score))
            limit = integer(self.ctx.section, "max_best_models", self.ctx.step, 24)
            if len(self.best_model_states) > limit:
                self.best_model_states.pop(0)
                self.best_model_scores.pop(0)
        self.network.train()
        return score

    def _train_one_step(self, phase_steps: int, learning_rate: float) -> dict[str, float | int]:
        features, labels = self.sampler.next_batch()
        x = torch.from_numpy(features).to(self.device)
        y = torch.from_numpy(labels).to(self.device)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == "cuda":
            autocast = torch.autocast(device_type="cuda", enabled=self.use_amp)
        else:
            autocast = torch.autocast(device_type="cpu", enabled=False)
        with autocast:
            predictions = self.network(x).reshape(-1)
        probabilities = predictions.float()
        keep = ((y == 0) & (probabilities >= 0.001)) | ((y == 1) & (probabilities < 0.999))
        phase_progress = self.phase_step / max(1, phase_steps - 1)
        negative_weight = 1.0 + (self.current_max_negative_weight - 1.0) * phase_progress
        if not bool(keep.any()):
            self.history["loss"].append(0.0)
            return {
                "loss": 0.0,
                "kept_examples": 0,
                "batch_examples": int(y.numel()),
                "negative_weight": negative_weight,
            }
        loss = _weighted_probability_bce(predictions[keep], y[keep], negative_weight)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        loss_value = float(loss.detach().cpu().item())
        self.history["loss"].append(loss_value)
        return {
            "loss": loss_value,
            "kept_examples": int(keep.sum().item()),
            "batch_examples": int(y.numel()),
            "negative_weight": negative_weight,
        }

    def _advance_phase(self, latest_score: dict[str, float] | None) -> None:
        target = number(self.ctx.section, "target_false_positives_per_hour", self.ctx.step, 0.5)
        multiplier = number(self.ctx.section, "negative_weight_multiplier", self.ctx.step, 2.0)
        if latest_score is not None and latest_score["val_fp_per_hr"] > target:
            self.current_max_negative_weight *= multiplier
        self.phase_index += 1
        self.phase_step = 0

    def _final_state(self) -> dict[str, torch.Tensor]:
        if not self.best_model_states:
            return _cpu_state_dict(self.network)
        scores = self.best_model_scores
        accuracy_cutoff = float(np.percentile([score["val_accuracy"] for score in scores], 90))
        recall_cutoff = float(np.percentile([score["val_recall"] for score in scores], 90))
        fp_cutoff = float(np.percentile([score["val_fp_per_hr"] for score in scores], 10))
        candidates = [
            state
            for state, score in zip(self.best_model_states, scores)
            if score["val_accuracy"] >= accuracy_cutoff
            and score["val_recall"] >= recall_cutoff
            and score["val_fp_per_hr"] <= fp_cutoff
        ]
        if not candidates:
            candidates = [self.best_model_states[-1]]
        averaged: dict[str, torch.Tensor] = {}
        for key in candidates[0]:
            values = [state[key] for state in candidates]
            if torch.is_floating_point(values[0]):
                averaged[key] = torch.stack(values).mean(dim=0)
            else:
                averaged[key] = values[0].clone()
        return averaged

    def run(self, *, resume: bool) -> dict[str, Any]:
        resumed = self._try_resume() if resume else False
        if not resumed:
            self._seed_everything()
        latest_score: dict[str, float] | None = None
        checkpoint_interval = integer(self.ctx.section, "checkpoint_interval_steps", self.ctx.step, 500)
        log_interval = integer(self.ctx.section, "log_interval_steps", self.ctx.step, 100)
        recent_losses: list[float] = []
        self._log_event(
            "run_start",
            (
                f"[train] start device={self.device} amp={self.use_amp} resumed={resumed} "
                f"global_step={self.global_step} log={self.log_path}"
            ),
            resumed=resumed,
            device=str(self.device),
            mixed_precision=self.use_amp,
            log_file=str(self.log_path),
        )
        self.network.train()
        while self.phase_index < len(self.plan):
            phase = self.plan[self.phase_index]
            phase_steps = int(phase["steps"])
            validation_steps = self._validation_schedule(self.phase_index, phase_steps)
            self._log_event(
                "phase_start",
                (
                    f"[train] phase={self.phase_index + 1}/{len(self.plan)} "
                    f"phase_step={self.phase_step}/{phase_steps} target_lr={float(phase['learning_rate']):.8g}"
                ),
                phase_number=self.phase_index + 1,
                phase_count=len(self.plan),
                phase_steps=phase_steps,
                target_learning_rate=float(phase["learning_rate"]),
            )
            while self.phase_step < phase_steps:
                learning_rate = self._learning_rate(self.phase_step, phase_steps, float(phase["learning_rate"]))
                completed_phase_index = self.phase_step
                step_result = self._train_one_step(phase_steps, learning_rate)
                recent_losses.append(float(step_result["loss"]))
                self.phase_step += 1
                self.global_step += 1
                should_log = (
                    self.global_step == 1
                    or self.global_step % log_interval == 0
                    or self.phase_step == phase_steps
                )
                if should_log:
                    mean_loss = float(np.mean(recent_losses)) if recent_losses else float(step_result["loss"])
                    self._log_event(
                        "train_step",
                        (
                            f"[train] phase={self.phase_index + 1}/{len(self.plan)} "
                            f"step={self.phase_step}/{phase_steps} global_step={self.global_step} "
                            f"loss={float(step_result['loss']):.6f} mean_loss={mean_loss:.6f} "
                            f"lr={learning_rate:.8g} kept={int(step_result['kept_examples'])}/"
                            f"{int(step_result['batch_examples'])}"
                        ),
                        phase_number=self.phase_index + 1,
                        phase_count=len(self.plan),
                        phase_steps=phase_steps,
                        loss=float(step_result["loss"]),
                        mean_loss=mean_loss,
                        learning_rate=learning_rate,
                        negative_weight=float(step_result["negative_weight"]),
                        kept_examples=int(step_result["kept_examples"]),
                        batch_examples=int(step_result["batch_examples"]),
                    )
                    recent_losses.clear()
                if completed_phase_index in validation_steps:
                    latest_score = self._validate()
                    self._log_event(
                        "validation",
                        (
                            f"[validation] phase={self.phase_index + 1}/{len(self.plan)} "
                            f"global_step={self.global_step} accuracy={latest_score['val_accuracy']:.6f} "
                            f"recall={latest_score['val_recall']:.6f} val_fp={latest_score['val_n_fp']:.0f} "
                            f"fp_per_hr={latest_score['val_fp_per_hr']:.6f}"
                        ),
                        phase_number=self.phase_index + 1,
                        phase_count=len(self.plan),
                        **latest_score,
                    )
                if self.global_step % checkpoint_interval == 0:
                    self._save_checkpoint()
                    self._log_event(
                        "checkpoint",
                        f"[train] checkpoint global_step={self.global_step} dir={self.checkpoint_dir}",
                        checkpoint_dir=str(self.checkpoint_dir),
                    )
            if latest_score is None:
                latest_score = self._validate()
                self._log_event(
                    "validation",
                    (
                        f"[validation] phase={self.phase_index + 1}/{len(self.plan)} "
                        f"global_step={self.global_step} accuracy={latest_score['val_accuracy']:.6f} "
                        f"recall={latest_score['val_recall']:.6f} val_fp={latest_score['val_n_fp']:.0f} "
                        f"fp_per_hr={latest_score['val_fp_per_hr']:.6f}"
                    ),
                    phase_number=self.phase_index + 1,
                    phase_count=len(self.plan),
                    **latest_score,
                )
            self._advance_phase(latest_score)
            self._save_checkpoint()
            self._log_event(
                "checkpoint",
                f"[train] phase checkpoint global_step={self.global_step} dir={self.checkpoint_dir}",
                checkpoint_dir=str(self.checkpoint_dir),
            )

        final_state = self._final_state()
        self.network.load_state_dict(final_state)
        model_path = _output_model(self.ctx)
        payload = {
            "schema_version": 1,
            "model_state_dict": _cpu_state_dict(self.network),
            "input_shape": list(self.feature_shape),
            "model_type": self.ctx.section.get("model_type", "dnn"),
            "layer_size": integer(self.ctx.section, "layer_size", self.ctx.step, 64),
            "model_config": self.model_config,
            "n_classes": 1,
            "seconds_per_example": 1280 * self.feature_shape[0] / integer(self.ctx.config.section("main"), "sample_rate", "main", 16000),
            "model_name": self.ctx.config.section("main").get("model_name", "wakeword_model"),
            "config_fingerprint": self.config_fingerprint,
            "inputs_fingerprint": self.inputs_fingerprint,
        }
        _atomic_torch_save(model_path, payload)
        summary = {
            "output_model": str(model_path),
            "model_checkpoint_dir": str(self.checkpoint_dir),
            "resumed": resumed,
            "completed_phases": len(self.plan),
            "global_steps": self.global_step,
            "phase_plan": self.plan,
            "input_shape": list(self.feature_shape),
            "model_type": payload["model_type"],
            "layer_size": payload["layer_size"],
            "model_config": self.model_config,
            "train_blocks": [block.name for block in self.train_blocks],
            "dev_blocks": [block.name for block in self.dev_blocks],
            "false_positive_blocks": [block.name for block in self.false_positive_blocks],
            "effective_batch_counts": {block.name: _batch_count(self.ctx, block.name) for block in self.train_blocks},
            "history": dict(self.history),
            "best_model_scores": self.best_model_scores,
            "training_log_file": str(self.log_path),
            "elapsed_seconds": time.time() - self.started_at,
            "device": str(self.device),
            "mixed_precision": self.use_amp,
            "config_fingerprint": self.config_fingerprint,
            "inputs_fingerprint": self.inputs_fingerprint,
        }
        write_json(_output_summary(self.ctx), summary)
        self._log_event(
            "run_complete",
            f"[train] complete global_step={self.global_step} model={model_path}",
            output_model=str(model_path),
            output_summary=str(_output_summary(self.ctx)),
        )
        return summary


def run(ctx: Any) -> dict[str, Any]:
    train = _blocks(ctx, "train")
    dev = _blocks(ctx, "dev")
    false_positive = _blocks(ctx, "false_positive")
    # The runner makes sure producer stages have completed before this point.
    unresolved = [block.path for block in [*train, *dev, *false_positive] if not block.path.is_file()]
    if unresolved:
        raise FileNotFoundError(f"Training feature file(s) are missing: {', '.join(str(path) for path in unresolved)}")
    # Re-read blocks now that preceding feature steps exist and include their shapes.
    train = _blocks(ctx, "train")
    dev = _blocks(ctx, "dev")
    false_positive = _blocks(ctx, "false_positive")
    trainer = ResumableAutoTrainer(ctx, train, dev, false_positive)
    result = trainer.run(resume=boolean(ctx.section, "resume", ctx.step, True) and not ctx.force)
    if not validate_outputs(ctx):
        raise RuntimeError("Training output validation failed")
    return result
