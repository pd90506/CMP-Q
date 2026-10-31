"""
Propagation map for CMP-Q: defines topological order of weights for quantization.
"""
from dataclasses import dataclass
from typing import List, Protocol

import torch.nn as nn


@dataclass
class WeightNode:
    """A single weight (linear layer) in the computational graph."""
    layer_idx: int
    node_id: str           # e.g. "L0.attn.q_proj", "L0.mlp.gate_proj"
    module: nn.Module      # the nn.Linear
    takes_hidden_input: bool  # True => accumulate V (cross-layer noise signal)
    input_checkpoint: str  # cache key for quantized-input lookup


class PropagationMap(Protocol):
    """Protocol for propagation map implementations."""

    def get_weight_queue(self, model: nn.Module) -> List[WeightNode]:
        """Return weights in topological (propagation) order."""
        ...


def get_weight_queue(model: nn.Module) -> List[WeightNode]:
    """
    Dispatch to the correct propagation map based on model architecture.
    Raises ValueError for unsupported model types.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    if model_type in ("llama", "qwen2", "qwen3", "mistral", "gemma", "gemma2"):
        return LlamaPropagationMap().get_weight_queue(model)
    else:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            "Add a PropagationMap subclass and register it in get_weight_queue()."
        )


class LlamaPropagationMap:
    """
    Propagation map for LLaMA-style decoder layers (LLaMA, Qwen2, Qwen3,
    Mistral, Gemma). These all share the same dense-MLP structure:
      self_attn:  q_proj, k_proj, v_proj, o_proj
      mlp:        gate_proj, up_proj, down_proj  (SwiGLU)

    V assignment:
      - q/k/v_proj:   input is the post-input_layernorm hidden state.
      - gate/up_proj: input is post-post_attention_layernorm hidden state.
      - o_proj:       input is the attention head output
                      (FP16 chain vs quant chain still disagree because upstream
                      noise propagates through this layer's attention before
                      hitting o_proj). Worth a V term.
      - down_proj:    input is SiLU(gate)*up intermediate; same reasoning.

    All 7 projections get takes_hidden_input=True — the ΔX at o_proj and
    down_proj is the upstream drift propagated through this layer's attention
    or MLP up to the o/down projection input.
    """

    def get_weight_queue(self, model: nn.Module) -> List[WeightNode]:
        queue: List[WeightNode] = []
        layers = model.model.layers

        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn

            # q, k, v projections — take the normed hidden state
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is not None and isinstance(proj, nn.Linear):
                    queue.append(WeightNode(
                        layer_idx=layer_idx,
                        node_id=f"L{layer_idx}.attn.{proj_name}",
                        module=proj,
                        takes_hidden_input=True,
                        input_checkpoint=f"L{layer_idx}.after_input_norm",
                    ))

            # o_proj — takes concatenated attention head outputs (within-layer).
            # ΔX here reflects upstream drift propagated through q/k/v + attention.
            o_proj = getattr(attn, "o_proj", None)
            if o_proj is not None and isinstance(o_proj, nn.Linear):
                queue.append(WeightNode(
                    layer_idx=layer_idx,
                    node_id=f"L{layer_idx}.attn.o_proj",
                    module=o_proj,
                    takes_hidden_input=True,
                    input_checkpoint=f"L{layer_idx}.attn.after_heads",
                ))

            # MLP: gate_proj, up_proj — take the post-attention-normed hidden state
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            for proj_name in ("gate_proj", "up_proj"):
                proj = getattr(mlp, proj_name, None)
                if proj is not None and isinstance(proj, nn.Linear):
                    queue.append(WeightNode(
                        layer_idx=layer_idx,
                        node_id=f"L{layer_idx}.mlp.{proj_name}",
                        module=proj,
                        takes_hidden_input=True,
                        input_checkpoint=f"L{layer_idx}.after_post_attn_norm",
                    ))

            # down_proj — takes SiLU(gate)*up activation (within-layer).
            # ΔX here reflects upstream drift propagated through gate_proj + up_proj.
            down_proj = getattr(mlp, "down_proj", None)
            if down_proj is not None and isinstance(down_proj, nn.Linear):
                queue.append(WeightNode(
                    layer_idx=layer_idx,
                    node_id=f"L{layer_idx}.mlp.down_proj",
                    module=down_proj,
                    takes_hidden_input=True,
                    input_checkpoint=f"L{layer_idx}.mlp.after_gate_up",
                ))

        return queue
