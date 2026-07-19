#!/usr/bin/env python3
"""Export one WeNet streaming CTC graph for the CTC + WAC wake-word path.

The generated ONNX graph exposes both values needed by the pipeline:

* ``encoder_out`` is the frozen acoustic feature consumed by WAC stage 2.
* ``ctc_log_probs`` is used by the stage-1 CTC keyword scorer.

The command also writes the matching stage-1 contract JSON.  Users therefore
do not need to copy ONNX tensor names or cache shapes by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic

from wenet.utils.init_model import init_model


def add_meta_data(filename: Path, meta_data: Dict[str, Any]) -> None:
    """Replace the custom metadata in an ONNX model."""

    model = onnx.load(str(filename))
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = str(key)
        meta.value = str(value)
    onnx.checker.check_model(model)
    onnx.save(model, str(filename))


class WuwStage1Onnx(torch.nn.Module):
    """A single streaming graph that returns encoder features and CTC scores."""

    def __init__(self, encoder: torch.nn.Module, ctc: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.ctc = ctc

    def forward(
        self,
        chunk: torch.Tensor,
        offset: torch.Tensor,
        required_cache_size: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_mask: torch.Tensor,
    ):
        encoder_out, next_att_cache, next_cnn_cache = self.encoder.forward_chunk(
            xs=chunk,
            offset=offset,
            required_cache_size=required_cache_size,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            att_mask=att_mask,
        )
        ctc_log_probs = self.ctc.log_softmax(encoder_out)
        return encoder_out, ctc_log_probs, next_att_cache, next_cnn_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a streaming WeNet CTC model for WUW stage 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="./final.pt", help="WeNet checkpoint")
    parser.add_argument("--config", default="./train.yaml", help="WeNet training YAML")
    parser.add_argument("--output-dir", default=".", help="Directory for ONNX and contract files")
    parser.add_argument("--output-prefix", default="stage1-wuw", help="Output filename prefix")
    parser.add_argument("--chunk-size", type=int, default=16, help="Encoder output frames per chunk")
    parser.add_argument("--left-chunks", type=int, default=4, help="Number of cached encoder chunks")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--blank-id", type=int, default=0)
    parser.add_argument(
        "--token-file",
        default=None,
        help="Optional token.txt used to fingerprint the stage-1 CTC vocabulary in the contract",
    )
    parser.add_argument("--opset-version", type=int, default=13)
    parser.add_argument("--num-mel-bins", type=int, default=None, help="Override YAML fbank dimension")
    parser.add_argument(
        "--dither",
        type=float,
        default=0.0,
        help="Inference fbank dither; keep zero for deterministic stage-1 features",
    )
    parser.add_argument("--skip-quantize", action="store_true", help="Do not create the int8 model")
    parser.add_argument("--skip-verify", action="store_true", help="Do not run ONNX Runtime smoke checks")
    parser.add_argument(
        "--strict-verify",
        action="store_true",
        help="Require close PyTorch/FP32 ONNX numerical parity, not only valid outputs",
    )
    return parser


def _fbank_config(
    configs: dict[str, Any], override_bins: int | None, inference_dither: float
) -> dict[str, float | int]:
    dataset = configs.get("dataset_conf", {})
    fbank = dataset.get("fbank_conf", {}) if isinstance(dataset, dict) else {}
    if not isinstance(fbank, dict):
        fbank = {}
    return {
        "num_mel_bins": int(override_bins or fbank.get("num_mel_bins", 80)),
        "frame_length_ms": float(fbank.get("frame_length", 25.0)),
        "frame_shift_ms": float(fbank.get("frame_shift", 10.0)),
        # Training YAML often uses dither=1.0 as augmentation.  Wake-word
        # feature generation and evaluation must be deterministic, so the
        # exporter uses the explicit inference value instead.
        "dither": float(inference_dither),
    }


def _complete_model_dimensions(
    configs: dict[str, Any], *, config_file: Path, checkpoint: Path
) -> None:
    """Fill dimensions that WeNet adds to the YAML during training.

    A recipe YAML normally only describes the feature extractor and tokenizer.
    WeNet's training setup turns those settings into ``input_dim`` and
    ``output_dim`` before it saves the runtime YAML beside each checkpoint.
    ``init_model`` requires the completed values, so an exporter that also
    accepts the original recipe needs to perform the same completion.
    """

    sources: dict[str, str] = {}
    sidecar_path = checkpoint.with_suffix(".yaml")
    sidecar: dict[str, Any] = {}
    if sidecar_path.is_file():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.load(handle, Loader=yaml.FullLoader)
        if isinstance(loaded, dict):
            sidecar = loaded

    for name in ("input_dim", "output_dim", "vocab_size"):
        if name not in configs and name in sidecar:
            configs[name] = sidecar[name]
            sources[name] = str(sidecar_path)

    if "input_dim" not in configs:
        dataset = configs.get("dataset_conf")
        if isinstance(dataset, dict):
            for feature_name in ("fbank_conf", "log_mel_spectrogram_conf", "mfcc_conf"):
                feature = dataset.get(feature_name)
                if isinstance(feature, dict) and "num_mel_bins" in feature:
                    configs["input_dim"] = feature["num_mel_bins"]
                    sources["input_dim"] = f"dataset_conf.{feature_name}.num_mel_bins"
                    break

    if "output_dim" not in configs and "vocab_size" in configs:
        configs["output_dim"] = configs["vocab_size"]
        sources["output_dim"] = "vocab_size"

    # Old checkpoints may not have a YAML sidecar.  For character/BPE recipes,
    # the symbol table is the same vocabulary used to construct the CTC head.
    if "output_dim" not in configs:
        tokenizer = configs.get("tokenizer_conf")
        symbol_table_value = tokenizer.get("symbol_table_path") if isinstance(tokenizer, dict) else None
        if symbol_table_value:
            symbol_table = Path(symbol_table_value).expanduser()
            if not symbol_table.is_absolute():
                symbol_table = config_file.parent / symbol_table
            if symbol_table.is_file():
                with symbol_table.open("r", encoding="utf-8") as handle:
                    vocab_size = sum(1 for line in handle if line.strip())
                configs["output_dim"] = vocab_size
                sources["output_dim"] = str(symbol_table)

    missing = [name for name in ("input_dim", "output_dim") if name not in configs]
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"WeNet config is missing {names}. Use the completed train/epoch YAML "
            f"saved beside the checkpoint, or provide feature and tokenizer settings "
            f"that allow the exporter to infer them."
        )

    for name in ("input_dim", "output_dim"):
        try:
            value = int(configs[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"WeNet config {name} must be a positive integer") from error
        if value < 1:
            raise ValueError(f"WeNet config {name} must be a positive integer")
        configs[name] = value
        if name in sources:
            print(f"Inferred {name}={value} from {sources[name]}")

    if "vocab_size" in configs and int(configs["vocab_size"]) != configs["output_dim"]:
        raise ValueError(
            "WeNet config output_dim and vocab_size disagree: "
            f"{configs['output_dim']} != {configs['vocab_size']}"
        )
    configs.setdefault("vocab_size", configs["output_dim"])


def _onnx_names(path: Path) -> tuple[set[str], set[str]]:
    model = onnx.load(str(path), load_external_data=False)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = {item.name for item in model.graph.input if item.name not in initializer_names}
    outputs = {item.name for item in model.graph.output}
    return inputs, outputs


def _write_contract(
    path: Path,
    *,
    model_path: Path,
    sample_rate: int,
    fbank: dict[str, float | int],
    decoding_window: int,
    input_stride: int,
    minimum_input_frames: int,
    encoder_chunk_size: int,
    required_cache_size: int,
    initial_offset: int,
    blank_id: int,
    att_cache_shape: tuple[int, ...],
    cnn_cache_shape: tuple[int, ...],
    encoder_output_size: int,
    vocab_size: int,
    subsampling_factor: int,
    token_table_fingerprint: str | None,
) -> dict[str, Any]:
    graph_inputs, graph_outputs = _onnx_names(model_path)
    required_outputs = {"encoder_out", "ctc_log_probs", "next_att_cache", "next_cnn_cache"}
    missing_outputs = sorted(required_outputs - graph_outputs)
    if missing_outputs:
        raise RuntimeError(f"Exported ONNX is missing WUW output(s): {', '.join(missing_outputs)}")
    inputs: dict[str, str] = {"features": "chunk"}
    if "offset" in graph_inputs:
        inputs["offset"] = "offset"
    constants: dict[str, Any] = {}
    if "required_cache_size" in graph_inputs:
        constants["required_cache_size"] = required_cache_size
    attention_mask = None
    if "att_mask" in graph_inputs:
        attention_mask = {
            "input": "att_mask",
            "cache_frames": required_cache_size,
            "chunk_frames": encoder_chunk_size,
        }
    cache_inputs = []
    if "att_cache" in graph_inputs:
        cache_inputs.append(
            {
                "input": "att_cache",
                "output": "next_att_cache",
                "shape": list(att_cache_shape),
                "dtype": "float32",
            }
        )
    if "cnn_cache" in graph_inputs:
        cache_inputs.append(
            {
                "input": "cnn_cache",
                "output": "next_cnn_cache",
                "shape": list(cnn_cache_shape),
                "dtype": "float32",
            }
        )
    contract: dict[str, Any] = {
        "schema_version": 2,
        "sample_rate": sample_rate,
        "fbank": fbank,
        # The ONNX input window overlaps.  These are fbank-frame counts,
        # whereas encoder_chunk_size is in subsampled encoder frames.
        "chunk_frames": decoding_window,
        "chunk_stride_frames": input_stride,
        "minimum_input_frames": minimum_input_frames,
        "pad_final_chunk": False,
        "initial_offset": initial_offset,
        "blank_id": blank_id,
        "input_layout": "BTF",
        "inputs": inputs,
        "outputs": {
            "encoder": "encoder_out",
            "ctc_log_probs": "ctc_log_probs",
        },
        "ctc_output_is_log_probs": True,
        "encoder_frame_shift_ms": float(fbank["frame_shift_ms"]) * int(subsampling_factor),
        "encoder_output_size": int(encoder_output_size),
        "vocab_size": int(vocab_size),
        "subsampling_factor": int(subsampling_factor),
        "encoder_chunk_frames": int(encoder_chunk_size),
        "constant_inputs": constants,
        "cache_inputs": cache_inputs,
    }
    if attention_mask is not None:
        contract["attention_mask"] = attention_mask
    if token_table_fingerprint is not None:
        contract["token_table_fingerprint"] = token_table_fingerprint
    path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return contract


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _runtime_feed(
    session: ort.InferenceSession,
    *,
    chunk: torch.Tensor,
    offset: torch.Tensor,
    required_cache_size: torch.Tensor,
    att_cache: torch.Tensor,
    cnn_cache: torch.Tensor,
    att_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    available = {item.name for item in session.get_inputs()}
    values = {
        "chunk": _numpy(chunk),
        "offset": _numpy(offset),
        "required_cache_size": _numpy(required_cache_size),
        "att_cache": _numpy(att_cache),
        "cnn_cache": _numpy(cnn_cache),
        "att_mask": _numpy(att_mask),
    }
    return {name: value for name, value in values.items() if name in available}


def _verify_model(
    path: Path,
    *,
    chunk: torch.Tensor,
    offset: torch.Tensor,
    required_cache_size: torch.Tensor,
    att_cache: torch.Tensor,
    cnn_cache: torch.Tensor,
    att_mask: torch.Tensor,
    expected: tuple[torch.Tensor, ...] | None,
    strict: bool = False,
) -> None:
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = _runtime_feed(
        session,
        chunk=chunk,
        offset=offset,
        required_cache_size=required_cache_size,
        att_cache=att_cache,
        cnn_cache=cnn_cache,
        att_mask=att_mask,
    )
    output_names = [item.name for item in session.get_outputs()]
    required = ["encoder_out", "ctc_log_probs", "next_att_cache", "next_cnn_cache"]
    if output_names != required:
        raise RuntimeError(f"Unexpected ONNX outputs for {path}: {output_names}")
    outputs = session.run(None, feed)
    if outputs[0].ndim != 3 or outputs[1].ndim != 3 or outputs[0].shape[:2] != outputs[1].shape[:2]:
        raise RuntimeError(f"Encoder and CTC outputs have incompatible shapes in {path}")
    if not all(np.isfinite(value).all() for value in outputs):
        raise RuntimeError(f"ONNX Runtime produced NaN or infinity for {path}")
    if expected is not None:
        for index, reference in enumerate(expected):
            difference = np.abs(_numpy(reference) - outputs[index])
            if difference.size:
                print(
                    f"  parity {required[index]}: max_abs={float(difference.max()):.6g}, "
                    f"mean_abs={float(difference.mean()):.6g}"
                )
            else:
                print(f"  parity {required[index]}: both outputs are empty")
            if strict:
                np.testing.assert_allclose(
                    _numpy(reference), outputs[index], rtol=1.0e-3, atol=1.0e-5
                )
    print(f"Verified {path.name}: encoder={outputs[0].shape}, ctc={outputs[1].shape}")


@torch.no_grad()
def main() -> None:
    args = _parser().parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")
    if args.left_chunks < 1:
        raise ValueError("--left-chunks must be >= 1 for the fixed-cache WUW exporter")
    if args.sample_rate < 1 or args.opset_version < 11:
        raise ValueError("Invalid sample rate or ONNX opset")

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    config_file = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"WeNet checkpoint does not exist: {checkpoint}")
    if not config_file.is_file():
        raise FileNotFoundError(f"WeNet config does not exist: {config_file}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with config_file.open("r", encoding="utf-8") as handle:
        configs = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(configs, dict):
        raise ValueError(f"WeNet config must contain a YAML mapping: {config_file}")
    _complete_model_dimensions(configs, config_file=config_file, checkpoint=checkpoint)
    torch_model, configs = init_model(args, configs)
    torch_model = torch_model.cpu().eval()

    # Efficient Conformer needs its global streaming chunk size fixed before
    # tracing.  The normal WeNet forward_chunk_by_chunk(use_onnx=True) path
    # performs the same setup.
    set_global_chunk_size = getattr(torch_model.encoder, "set_global_chunk_size", None)
    efficient_onnx_mode = callable(set_global_chunk_size)
    if efficient_onnx_mode:
        set_global_chunk_size(chunk_size=args.chunk_size)

    encoder_conf = configs["encoder_conf"]
    head = int(encoder_conf["attention_heads"])
    num_blocks = int(encoder_conf["num_blocks"])
    output_size = int(encoder_conf["output_size"])
    cnn_module_kernel = int(encoder_conf.get("cnn_module_kernel", 1))
    right_context = int(torch_model.right_context())
    subsampling_factor = int(torch_model.encoder.embed.subsampling_rate)
    context = right_context + 1
    decoding_window = (args.chunk_size - 1) * subsampling_factor + context
    input_stride = args.chunk_size * subsampling_factor
    required_cache_size_value = args.chunk_size * args.left_chunks
    # Efficient Conformer follows its built-in ONNX simulation and starts at
    # zero.  The normal Conformer/Transformer graph needs the fixed dummy
    # cache placed before the first real position.
    initial_offset = 0 if efficient_onnx_mode else required_cache_size_value
    fbank = _fbank_config(configs, args.num_mel_bins, args.dither)

    att_cache = torch.zeros(
        num_blocks,
        head,
        required_cache_size_value,
        output_size // head * 2,
        dtype=torch.float32,
    )
    cnn_cache = torch.zeros(
        num_blocks,
        1,
        output_size,
        cnn_module_kernel - 1,
        dtype=torch.float32,
    )
    att_mask = torch.ones(
        1,
        1,
        required_cache_size_value + args.chunk_size,
        dtype=torch.bool,
    )
    att_mask[:, :, :required_cache_size_value] = False
    chunk = torch.rand(
        1,
        decoding_window,
        int(fbank["num_mel_bins"]),
        dtype=torch.float32,
    )
    # Scalars make the generated contract/runtime interface simpler than the
    # previous one-element vectors.
    offset = torch.tensor(initial_offset, dtype=torch.int64)
    required_cache_size = torch.tensor(required_cache_size_value, dtype=torch.int64)

    model = WuwStage1Onnx(torch_model.encoder, torch_model.ctc).eval()
    output_fp32 = output_dir / f"{args.output_prefix}.onnx"
    output_int8 = output_dir / f"{args.output_prefix}.int8.onnx"
    output_contract = output_dir / f"{args.output_prefix}.contract.json"

    print(f"Exporting stage-1 WUW model to {output_fp32}")
    torch.onnx.export(
        model,
        (chunk, offset, required_cache_size, att_cache, cnn_cache, att_mask),
        str(output_fp32),
        opset_version=args.opset_version,
        export_params=True,
        do_constant_folding=True,
        input_names=[
            "chunk",
            "offset",
            "required_cache_size",
            "att_cache",
            "cnn_cache",
            "att_mask",
        ],
        output_names=[
            "encoder_out",
            "ctc_log_probs",
            "next_att_cache",
            "next_cnn_cache",
        ],
        dynamic_axes={
            "chunk": {1: "T_INPUT"},
            "att_cache": {2: "T_CACHE"},
            "att_mask": {2: "T_MASK"},
            "encoder_out": {0: "N", 1: "T_ENCODER"},
            "ctc_log_probs": {0: "N", 1: "T_ENCODER"},
            "next_att_cache": {2: "T_CACHE_OUT"},
        },
    )

    meta_data = {
        "model_type": "wenet_ctc_wuw_stage1",
        "version": "2",
        "model_author": "wenet",
        "comment": "streaming encoder feature plus CTC log probabilities",
        "url": os.environ.get("WENET_URL", ""),
        "encoder_chunk_size": args.chunk_size,
        "input_chunk_frames": decoding_window,
        "input_stride_frames": input_stride,
        "left_chunks": args.left_chunks,
        "head": head,
        "num_blocks": num_blocks,
        "output_size": output_size,
        "cnn_module_kernel": cnn_module_kernel,
        "right_context": right_context,
        "subsampling_factor": subsampling_factor,
        "vocab_size": int(torch_model.ctc.ctc_lo.weight.shape[0]),
        "blank_id": args.blank_id,
    }
    add_meta_data(output_fp32, meta_data)
    token_table_fingerprint = None
    if args.token_file:
        token_file = Path(args.token_file).expanduser().resolve()
        if not token_file.is_file():
            raise FileNotFoundError(f"--token-file does not exist: {token_file}")
        token_table_fingerprint = hashlib.sha256(token_file.read_bytes()).hexdigest()
    contract = _write_contract(
        output_contract,
        model_path=output_fp32,
        sample_rate=args.sample_rate,
        fbank=fbank,
        decoding_window=decoding_window,
        input_stride=input_stride,
        minimum_input_frames=context,
        encoder_chunk_size=args.chunk_size,
        required_cache_size=required_cache_size_value,
        initial_offset=initial_offset,
        blank_id=args.blank_id,
        att_cache_shape=tuple(att_cache.shape),
        cnn_cache_shape=tuple(cnn_cache.shape),
        encoder_output_size=output_size,
        vocab_size=int(torch_model.ctc.ctc_lo.weight.shape[0]),
        subsampling_factor=subsampling_factor,
        token_table_fingerprint=token_table_fingerprint,
    )

    expected = model(chunk, offset, required_cache_size, att_cache, cnn_cache, att_mask)
    if not args.skip_verify:
        _verify_model(
            output_fp32,
            chunk=chunk,
            offset=offset,
            required_cache_size=required_cache_size,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            att_mask=att_mask,
            expected=expected,
            strict=args.strict_verify,
        )

    if not args.skip_quantize:
        print(f"Quantizing stage-1 WUW model to {output_int8}")
        quantize_dynamic(
            model_input=str(output_fp32),
            model_output=str(output_int8),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        add_meta_data(output_int8, {**meta_data, "quantization": "dynamic_qint8_matmul"})
        fp32_interface = _onnx_names(output_fp32)
        int8_interface = _onnx_names(output_int8)
        if fp32_interface != int8_interface:
            raise RuntimeError("Quantization changed the stage-1 ONNX input/output interface")
        if not args.skip_verify:
            _verify_model(
                output_int8,
                chunk=chunk,
                offset=offset,
                required_cache_size=required_cache_size,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=att_mask,
                expected=None,
            )

    print("\nStage-1 WUW export complete")
    print(f"  FP32 model: {output_fp32}")
    if not args.skip_quantize:
        print(f"  INT8 model: {output_int8}")
    print(f"  Contract:   {output_contract}")
    print(f"  Encoder output: [1, T, {output_size}]")
    print(f"  CTC output:     [1, T, {meta_data['vocab_size']}]")
    print(f"  Contract inputs: {contract['inputs']}")


if __name__ == "__main__":
    main()
