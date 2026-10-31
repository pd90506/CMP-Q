"""
GPTQ / GPTAQ baselines: layer-by-layer quantization.

Usage:
    python -m baselines.run --method gptq  [--config config/config.yaml] [--set section.key=value ...]
    python -m baselines.run --method gptaq [--config config/config.yaml] [--set section.key=value ...]

Reads model, calibration data and quantizer settings from the shared config file
(quantization.bits / groupsize / act_order / percdamp; GPTAQ also quantization.mse_clip).
Each linear layer's inputs are collected with its predecessors already quantized. GPTAQ additionally needs the same inputs from
the full-precision model, so it also runs the FP16 chain.
"""
import argparse
import gc
import glob
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from dotenv import load_dotenv

# transformers imports TransformGetItemToIndex from a path that newer torch reorganized (same shim as main.py).
import torch._dynamo._trace_wrapped_higher_order_op as _thop
if not hasattr(_thop, "TransformGetItemToIndex"):
    from torch._higher_order_ops.flex_attention import TransformGetItemToIndex as _T
    _thop.TransformGetItemToIndex = _T

from baselines.gptaq import GPTAQ
from baselines.gptq import GPTQ
from cmpq.core.artifact_manager import ArtifactManager
from cmpq.core.config import CompressorConfig
from cmpq.core.propagation_map import WeightNode, get_weight_queue
from cmpq.utils.device import empty_cache, get_device
from cmpq.utils.efficiency import EfficiencyTracker
from cmpq.utils.model_loader import capture_layer0_inputs, load_calibration_samples, load_model

logger = logging.getLogger("Baseline")

GPTAQ_ALPHA = 0.25  # cross-term scale, reference (ResComp) default


def _capture_fp_inputs(
    layer: torch.nn.Module,
    layer_idx: int,
    layer_nodes: List[WeightNode],
    layer_input: torch.Tensor,
    outs: torch.Tensor,
    layer_kwargs: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """FP16-chain forward of one layer; returns each node's input checkpoint on CPU."""
    ckpts: Dict[str, torch.Tensor] = {f"L{layer_idx}.input": layer_input.clone().cpu()}
    needed = {n.input_checkpoint for n in layer_nodes if n.takes_hidden_input} - {f"L{layer_idx}.input"}
    captured: Dict[str, List[torch.Tensor]] = {k: [] for k in needed}

    def make_hook(key: str):
        def hook(_, inp, out):
            captured[key].append(inp[0].data.clone().cpu())
        return hook

    handles, seen = [], set()
    for node in layer_nodes:
        if node.input_checkpoint in needed and node.input_checkpoint not in seen:
            seen.add(node.input_checkpoint)
            handles.append(node.module.register_forward_hook(make_hook(node.input_checkpoint)))
    with torch.no_grad():
        for j in range(layer_input.shape[0]):
            outs[j] = layer(layer_input[j:j+1], **layer_kwargs)[0]
    for h in handles:
        h.remove()

    for key, tensors in captured.items():
        if tensors:
            ckpts[key] = torch.cat(tensors, dim=0).to(dtype=layer_input.dtype)
    return ckpts


def _quantize_node_gptq(cfg: CompressorConfig, node: WeightNode, layer, inps, outs, layer_kwargs) -> None:
    solver = GPTQ(node.module, name=node.node_id, wbits=cfg.quantization.bits)
    handle = node.module.register_forward_hook(lambda _, inp, out: solver.add_batch(inp[0].data))
    with torch.no_grad():
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j:j+1], **layer_kwargs)[0]
    handle.remove()
    logger.info(f"  Quantizing {node.node_id} (gptq)")
    solver.fasterquant(
        percdamp=cfg.quantization.percdamp,
        groupsize=cfg.quantization.groupsize,
        actorder=cfg.quantization.act_order,
    )
    del solver
    empty_cache()


def _quantize_node_gptaq(cfg: CompressorConfig, node: WeightNode, layer, inps, outs, layer_kwargs,
                         fp_ckpts: Dict[str, torch.Tensor]) -> None:
    """Pair each node's quantized-chain input X_q with its FP16-chain input X_fp."""
    solver = GPTAQ(node.module, name=node.node_id, wbits=cfg.quantization.bits)
    q_inputs: List[torch.Tensor] = []
    handle = node.module.register_forward_hook(lambda _, inp, out: q_inputs.append(inp[0].data.clone().cpu()))
    with torch.no_grad():
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j:j+1], **layer_kwargs)[0]
    handle.remove()

    quant_inp = torch.cat(q_inputs, dim=0)
    del q_inputs
    fp_inp = fp_ckpts.get(node.input_checkpoint) if node.takes_hidden_input else None
    if fp_inp is not None and fp_inp.shape != quant_inp.shape:
        logger.warning(f"  {node.node_id}: shape mismatch q={quant_inp.shape} vs fp={fp_inp.shape}, using X_fp = X_q")
        fp_inp = None
    for k in range(quant_inp.shape[0]):
        qk = quant_inp[k:k+1].to(solver.dev)
        solver.add_batch(qk, fp_inp[k:k+1].to(solver.dev) if fp_inp is not None else qk)
    del quant_inp

    logger.info(f"  Quantizing {node.node_id} (gptaq, α={GPTAQ_ALPHA})")
    solver.fasterquant(
        percdamp=cfg.quantization.percdamp,
        groupsize=cfg.quantization.groupsize,
        actorder=cfg.quantization.act_order,
        mse_clip=cfg.quantization.mse_clip,
        gptaq_alpha=GPTAQ_ALPHA,
    )
    del solver
    empty_cache()


def quantize(cfg: CompressorConfig, artifacts: ArtifactManager) -> None:
    method = cfg.experiment.method
    assert method in ("gptq", "gptaq"), f"baselines/run.py runs gptq or gptaq, got {method!r}"
    use_fp_chain = method == "gptaq"

    eff = EfficiencyTracker(cfg, method=method)
    eff.start("model_load_s")
    model, tokenizer = load_model(cfg)
    eff.stop("model_load_s")
    eff.start("calib_capture_s")
    samples = load_calibration_samples(cfg, tokenizer)
    dev = get_device()
    layers = model.model.layers
    weight_queue = get_weight_queue(model)

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)
    inps, layer_kwargs = capture_layer0_inputs(model, samples, dev, cfg.model.seq_len)
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    empty_cache()

    inps_fp = inps.clone() if use_fp_chain else None
    outs = torch.zeros_like(inps)
    eff.stop("calib_capture_s")

    eff.start("layer_loop_s")
    loop_t0 = time.perf_counter()
    for layer_idx in range(len(layers)):
        logger.info(f"=== Layer {layer_idx}/{len(layers) - 1} ===")
        layer = layers[layer_idx].to(dev)
        layer_nodes = [n for n in weight_queue if n.layer_idx == layer_idx]

        fp_ckpts: Dict[str, torch.Tensor] = {}
        if use_fp_chain:
            eff.start("ref_forward_s")
            fp_ckpts = _capture_fp_inputs(layer, layer_idx, layer_nodes, inps_fp, outs, layer_kwargs)
            eff.stop("ref_forward_s")
            inps_fp, outs = outs, inps_fp

        for node in layer_nodes:
            if use_fp_chain:
                _quantize_node_gptaq(cfg, node, layer, inps, outs, layer_kwargs, fp_ckpts)
            else:
                _quantize_node_gptq(cfg, node, layer, inps, outs, layer_kwargs)

        with torch.no_grad():
            for j in range(inps.shape[0]):
                outs[j] = layer(inps[j:j+1], **layer_kwargs)[0]
        inps, outs = outs, inps

        layers[layer_idx] = layer.cpu()
        del layer, fp_ckpts
        gc.collect()
        empty_cache()
        done, elapsed = layer_idx + 1, time.perf_counter() - loop_t0
        logger.info(f"  Progress: {done}/{len(layers)} layers, elapsed {elapsed:.0f}s")
    eff.stop("layer_loop_s")
    eff.finish()

    save_path = artifacts.compressed_model_path
    if os.path.isdir(save_path):  # stale shards would shadow the new weights at load time
        for pattern in ("model.safetensors", "model-*-of-*.safetensors", "model.safetensors.index.json"):
            for stale in glob.glob(os.path.join(save_path, pattern)):
                os.remove(stale)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved {method} model to {save_path}")
    eff.write()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="GPTQ / GPTAQ baselines")
    parser.add_argument("--method", choices=["gptq", "gptaq"], required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--set", metavar="section.key=value", nargs="*", dest="overrides", default=[],
                        help="Dot-notation overrides, e.g. --set quantization.bits=3 model.name=Qwen/Qwen3-8B-Base")
    args = parser.parse_args()

    cfg = CompressorConfig.load_from_yaml(
        args.config, overrides=args.overrides + [f"experiment.method={args.method}"])
    if cfg.experiment.name in ("auto", ""):
        cfg.experiment.name = cfg.auto_name()
    artifacts = ArtifactManager(cfg.experiment.output_dir, cfg.experiment.name)

    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(artifacts.get_log_file("Baseline"))]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger.info(f"Experiment: {cfg.experiment.name} ({args.method})")

    quantize(cfg, artifacts)


if __name__ == "__main__":
    main()
