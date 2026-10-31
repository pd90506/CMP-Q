import gc
import zlib
import json
import torch
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any
from .core.config import CompressorConfig
from .core.artifact_manager import ArtifactManager
from .core.propagation_map import get_weight_queue, WeightNode
from .utils.model_loader import load_model, load_calibration_samples, capture_layer0_inputs
from .utils.gptq_rev import GPTQREV
from .utils.quip_sharp import E8PCodebook, quip_quantize, random_hadamard
from .utils.device import empty_cache, get_device
from .utils.efficiency import EfficiencyTracker


class CMPQPipeline:
    """
    Single-pass continuous quantization pipeline using Compensatory Quantization (CMP-Q).

    Follows the GPTQ CPU-offloading pattern: one layer on GPU at a time.
    GPU memory budget: ~3 activation buffers + 1 layer weights + Hessian.
    """
    def __init__(self, config: CompressorConfig, artifact_manager: ArtifactManager):
        self.config = config
        self.artifacts = artifact_manager
        self.logger = self._setup_logger()

    def _setup_logger(self):
        stage_name = self.__class__.__name__
        logger = logging.getLogger(stage_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        for h in logger.handlers[:]:
            logger.removeHandler(h)

        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(self.artifacts.get_log_file(stage_name))

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    def run(self):
        self.logger.info("Starting CMP-Q Pipeline (GPTQ-style CPU offloading)")
        assert self.config.experiment.method in ("cmpq", "quip"), (
            f"main.py runs method=cmpq or quip; run GPTQ/GPTAQ with baselines/run.py "
            f"(got {self.config.experiment.method!r})"
        )

        eff = EfficiencyTracker(self.config, method=self.config.experiment.method)
        eff.start("model_load_s")
        model, tokenizer = load_model(self.config)
        eff.stop("model_load_s")
        eff.start("calib_capture_s")
        samples = self._prepare_samples(tokenizer)
        dev = get_device()
        layers = model.model.layers
        n_layers = len(layers)

        # Build weight queue
        weight_queue = get_weight_queue(model)
        self.logger.info(f"Built weight queue: {len(weight_queue)} nodes across {n_layers} layers")

        # --- Capture layer-0 inputs (embed + layer0 on GPU temporarily) ---
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        layers[0] = layers[0].to(dev)
        inps, layer_kwargs = capture_layer0_inputs(model, samples, dev, self.config.model.seq_len)
        layers[0] = layers[0].cpu()
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()

        # Three buffers on GPU: inps_ref, inps_quant, outs (~2.1 GB each for 8B model)
        inps_ref = inps.clone()
        inps_quant = inps
        outs = torch.zeros_like(inps_ref)
        n_samples = inps_ref.shape[0]
        del inps
        eff.stop("calib_capture_s")

        # --- Layer-by-layer quantization ---
        eff.start("layer_loop_s")
        loop_t0 = time.perf_counter()
        for layer_idx in range(n_layers):
            self.logger.info(f"=== Layer {layer_idx}/{n_layers-1} ===")
            layer = layers[layer_idx].to(dev)
            layer_nodes = [n for n in weight_queue if n.layer_idx == layer_idx]

            # Step 1: Reference forward (FP16 weights) → capture ref_ckpts on CPU.
            # cumulative ΔX: FP16 forward fed by inps_ref (FP16-chain accumulated).
            eff.start("ref_forward_s")
            ref_ckpts = self._capture_ref_checkpoints(
                layer, layer_idx, layer_nodes, inps_ref, outs, layer_kwargs
            )
            eff.stop("ref_forward_s")
            inps_ref, outs = outs, inps_ref  # Advance ref path
            self.logger.info(f"  Ref forward done, {len(ref_ckpts)} checkpoints captured")

            # Step 2: Quantize all weights (Hessian + V + GPTQ per node)
            self._quantize_layer_weights(
                layer, layer_idx, layer_nodes, inps_quant, outs,
                layer_kwargs, ref_ckpts, n_samples
            )

            # Step 3: Final forward with quantized weights → advance quant path
            with torch.no_grad():
                for j in range(n_samples):
                    outs[j] = layer(inps_quant[j:j+1], **layer_kwargs)[0]
            inps_quant, outs = outs, inps_quant

            # Step 4: CPU offload
            layers[layer_idx] = layer.cpu()
            del layer, ref_ckpts
            gc.collect()
            torch.cuda.empty_cache()
            done, elapsed = layer_idx + 1, time.perf_counter() - loop_t0
            self.logger.info(
                f"  Progress: {done}/{n_layers} layers, elapsed {elapsed:.0f}s, "
                f"{elapsed / done:.1f}s/layer, ETA {elapsed / done * (n_layers - done):.0f}s"
            )
        eff.stop("layer_loop_s")
        eff.finish()

        # --- Save compressed model ---
        save_path = self.artifacts.compressed_model_path
        # Prevent shadow-model bug: if an old safetensors file (single or
        # shards) exists in save_path, delete them BEFORE save_pretrained so
        # they can't shadow the new weights at load time. HF prioritizes a
        # single model.safetensors over shards, so leaving both causes
        # from_pretrained() to silently load stale weights.
        import glob
        import os as _os
        if _os.path.isdir(save_path):
            for pattern in ("model.safetensors", "model-*-of-*.safetensors",
                             "model.safetensors.index.json"):
                for stale in glob.glob(_os.path.join(save_path, pattern)):
                    _os.remove(stale)
                    self.logger.info(f"Removed stale weight file: {stale}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        self.logger.info(f"Saved CMP-Q compressed model to {save_path}")
        eff.write()
        return True

    # ------------------------------------------------------------------
    # Per-layer helpers
    # ------------------------------------------------------------------

    def _capture_ref_checkpoints(
        self,
        layer: torch.nn.Module,
        layer_idx: int,
        layer_nodes: List[WeightNode],
        layer_input_ref: torch.Tensor,
        outs: torch.Tensor,
        layer_kwargs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Run FP16 forward, capture ref activations at each checkpoint."""
        ref_ckpts: Dict[str, torch.Tensor] = {}

        # Store layer input on CPU
        ref_ckpts[f"L{layer_idx}.input"] = layer_input_ref.clone().cpu()

        # Only capture checkpoints that differ from L{i}.input AND are actually
        # used by a V-bearing node.
        needed = {
            n.input_checkpoint for n in layer_nodes if n.takes_hidden_input
        } - {f"L{layer_idx}.input"}
        captured: Dict[str, List[torch.Tensor]] = {k: [] for k in needed}
        seen_ckpts: set = set()  # deduplicate: only capture from first weight per checkpoint

        def make_hook(ckpt_key: str):
            def hook(_, inp, out):
                captured[ckpt_key].append(inp[0].data.clone().cpu())
            return hook

        handles = []
        for node in layer_nodes:
            ckpt = node.input_checkpoint
            if ckpt in needed and ckpt not in seen_ckpts:
                seen_ckpts.add(ckpt)
                handles.append(node.module.register_forward_hook(make_hook(ckpt)))

        with torch.no_grad():
            for j in range(layer_input_ref.shape[0]):
                outs[j] = layer(layer_input_ref[j:j+1], **layer_kwargs)[0]

        for h in handles:
            h.remove()

        dtype = layer_input_ref.dtype
        for ckpt, tensors in captured.items():
            if tensors:
                ref_ckpts[ckpt] = torch.cat(tensors, dim=0).to(dtype=dtype)

        return ref_ckpts

    def _quantize_layer_weights(
        self,
        layer: torch.nn.Module,
        layer_idx: int,
        layer_nodes: List[WeightNode],
        inps_quant: torch.Tensor,
        outs: torch.Tensor,
        layer_kwargs: Dict[str, Any],
        ref_ckpts: Dict[str, torch.Tensor],
        n_samples: int,
    ) -> None:
        """Quantize each weight node: collect Hessian via forward → accumulate V from ΔX → quantize."""
        for node in layer_nodes:
            # Collect Hessian via forward with hook
            gptq = GPTQREV(node.module, name=node.node_id, wbits=self.config.quantization.bits)

            quant_ckpt_tensors: List[torch.Tensor] = []
            # ΔX (and V) is only needed when the damping is on or ρ statistics are logged;
            # with λ=0 (GPTQ, plain QuIP#) skip it entirely.
            need_v = self.config.cmpq.lambda_val != 0.0 or self.config.cmpq.delta_stats
            capture_quant = node.takes_hidden_input and need_v

            def hessian_hook(_, inp, out):
                gptq.add_batch(inp[0].data)
                if capture_quant:
                    quant_ckpt_tensors.append(inp[0].data.clone().cpu())

            handle = node.module.register_forward_hook(hessian_hook)
            with torch.no_grad():
                for j in range(n_samples):
                    outs[j] = layer(inps_quant[j:j+1], **layer_kwargs)[0]
            handle.remove()

            # Accumulate V from ΔX (streaming: one sample's ΔX at a time, V on device)
            V_matrix = None
            if capture_quant and quant_ckpt_tensors:
                ref_inp = ref_ckpts.get(node.input_checkpoint)
                if ref_inp is not None:
                    for k, qt in enumerate(quant_ckpt_tensors):
                        ref_slice = ref_inp[k:k+1]
                        if qt.shape != ref_slice.shape:
                            break
                        delta_k = (qt.float() - ref_slice.float()).to(gptq.dev)
                        gptq.add_batch_delta(delta_k)
                        del delta_k
                    V_matrix = gptq.V  # reference; freed in gptq.free()
                del quant_ckpt_tensors

            if self.config.cmpq.delta_stats and V_matrix is not None:
                gptq.v_spectrum = self._v_spectrum_stats(gptq.H, V_matrix)

            if self.config.experiment.method == "quip":
                self._quip_quantize_node(gptq, node, V_matrix)
                if self.config.cmpq.delta_stats:
                    self._log_delta_stats(gptq)
                gptq.free()
                del V_matrix
                gc.collect()
                empty_cache()
                continue

            # Quantize
            self.logger.info(
                f"  Quantizing {node.node_id} (λ={self.config.cmpq.lambda_val}, "
                f"V={'yes' if V_matrix is not None else 'no'})"
            )
            # RHT: run CMPQ on QuIP#-style incoherence-processed weights.
            # W_r = S_out W S_inᵀ, H_r = S_in H S_inᵀ, V_r = S_in V S_inᵀ; undo after.
            rht = self.config.quantization.rht
            if rht:
                gen = torch.Generator().manual_seed(
                    self.config.experiment.seed * 1_000_003 + zlib.crc32(node.node_id.encode()))
                W0 = node.module.weight.data
                S_in = random_hadamard(W0.shape[1], gen, gptq.dev)
                S_out = random_hadamard(W0.shape[0], gen, gptq.dev)
                node.module.weight.data = S_out @ W0.float() @ S_in.T  # fp32 until un-rotated
                gptq.H = S_in @ gptq.H @ S_in.T
                if V_matrix is not None:
                    V_matrix = S_in @ V_matrix @ S_in.T
            gptq.fasterquant(
                cmpq_lambda=self.config.cmpq.lambda_val,
                percdamp=self.config.quantization.percdamp,
                groupsize=self.config.quantization.groupsize,
                V_matrix=V_matrix,
                actorder=self.config.quantization.act_order,
                mse_clip=self.config.quantization.mse_clip,
            )
            if rht:
                Wq = node.module.weight.data
                node.module.weight.data = (S_out.T @ Wq.float() @ S_in).to(W0.dtype)
                del S_in, S_out, W0
            if self.config.cmpq.delta_stats:
                self._log_delta_stats(gptq)
            gptq.free()
            del V_matrix
            gc.collect()
            empty_cache()

    def _log_delta_stats(self, gptq: GPTQREV) -> None:
        """Append this node's ΔX zero-mean statistics to diagnostics/<experiment>_deltastats.jsonl.

        Read by scripts/rho_report.py. Failures here must never abort a quantization run.
        """
        try:
            stats = gptq.delta_stats()
            if stats is None:
                return
            stats["experiment"] = self.config.experiment.name
            stats["model"] = self.config.model.name
            stats["bits"] = self.config.quantization.bits
            stats["lambda"] = self.config.cmpq.lambda_val
            stats.update(getattr(gptq, "v_spectrum", None) or {})
            out_dir = Path(__file__).resolve().parent.parent / "diagnostics"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{self.config.experiment.name}_deltastats.jsonl"
            with open(path, "a") as f:
                f.write(json.dumps(stats) + "\n")
        except Exception as e:  # diagnostics must not break the run
            self.logger.warning(f"  delta_stats failed for {gptq.name}: {e}")

    @staticmethod
    def _v_spectrum_stats(H: torch.Tensor, V: torch.Tensor) -> Dict[str, float]:
        """Rotation-invariant shape of V relative to H (same in any RHT basis).

        tr_ratio = tr V / tr H; pr_norm = (tr V)² / (‖V‖_F² · d) ∈ [1/d, 1]
        (1 = isotropic, 1/d = rank one); align = ⟨H, V⟩_F / (‖H‖_F ‖V‖_F).
        """
        with torch.no_grad():
            Hf, Vf = H.float(), V.float()
            trH, trV = torch.diag(Hf).sum(), torch.diag(Vf).sum()
            v_fro2 = Vf.pow(2).sum().clamp_min(1e-30)
            return {
                "tr_ratio": float(trV / trH.clamp_min(1e-30)),
                "pr_norm": float(trV.pow(2) / (v_fro2 * Vf.shape[0])),
                "align": float((Hf * Vf).sum() / (Hf.norm() * v_fro2.sqrt()).clamp_min(1e-30)),
            }

    def _quip_quantize_node(self, gptq: GPTQREV, node: WeightNode, V_matrix: torch.Tensor | None) -> None:
        """QuIP# (no FT) on one node, with the CMPQ primary Hessian when λ>0:
        Ĥ = H + λ·(tr H / tr V)·V. λ=0 or V=None → plain QuIP#."""
        cfg = self.config
        H = gptq.H
        lam = float(cfg.cmpq.lambda_val)
        if V_matrix is not None and lam != 0.0:
            r = (torch.diag(H).sum() / torch.diag(V_matrix).sum().clamp_min(1e-12)).item()
            H = H + (lam * r) * V_matrix
        if getattr(self, "_quip_cb", None) is None:
            self._quip_cb = E8PCodebook(cfg.quantization.bits, gptq.dev)
        # Same random Hadamard signs per node in every arm (QuIP# vs QuIP#+CMPQ).
        gen = torch.Generator().manual_seed(cfg.experiment.seed * 1_000_003 + zlib.crc32(node.node_id.encode()))
        W = node.module.weight.data
        hatW, proxy = quip_quantize(W.to(gptq.dev), H, self._quip_cb, gen)
        node.module.weight.data = hatW.to(W.dtype).reshape(W.shape)
        del gptq.H
        self.logger.info(
            f"  [QuIP#] {node.node_id}: E8P{'+RVQ' if cfg.quantization.bits == 3 else ''} "
            f"{cfg.quantization.bits}b, λ={lam:g}, V={'yes' if V_matrix is not None else 'no'}, "
            f"proxy={proxy:.4e}"
        )

    def _prepare_samples(self, tokenizer):
        return load_calibration_samples(self.config, tokenizer)
