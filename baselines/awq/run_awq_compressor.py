"""AWQ baseline runner using vLLM's `llm-compressor` (current community-recommended AWQ impl).

This replaces the patched `mit-han-lab/llm-awq` runner for Qwen3, which proved to
be catastrophically broken due to scale-search / clip-search interaction. `llm-compressor` has first-class Qwen3 support and an
AWQModifier with `duo_scaling` option default.

Calibration: C4 raw text (matches our GPTQ / CMP-Q baselines), NOT the UltraChat
chat-template preprocessing from llm-compressor's default examples — those are
for instruction-tuned models; Qwen3-8B-Base is a base model.

Output: standard HF bf16 safetensors (save_compressed=False) so our existing
evaluation.evaluate pipeline loads it directly without the compressed-tensors
unpacking path.
"""
import argparse
import gc
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationScheme, QuantizationStrategy, QuantizationType,
)


def _build_c4_calib(tokenizer, n_samples: int, seqlen: int, seed: int = 42):
    """Raw C4 text concatenated + chunked to `n_samples` blocks of `seqlen` tokens.
    Matches the convention of our GPTQ/CMP-Q calibration (model_loader._get_c4):
    most C4 docs are shorter than 2048 tokens, so we concat-then-split rather
    than filter for long-enough individual docs.
    """
    ds = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    ds = ds.shuffle(seed=seed)

    needed = n_samples * seqlen
    buf: list[int] = []
    for ex in ds:
        ids = tokenizer(ex["text"], return_tensors=None, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        if len(buf) >= needed:
            break
    assert len(buf) >= needed, f"C4 subset yielded only {len(buf)} tokens; need {needed}"

    rows = []
    for i in range(n_samples):
        chunk = buf[i * seqlen : (i + 1) * seqlen]
        rows.append({"input_ids": chunk, "attention_mask": [1] * seqlen})

    from datasets import Dataset
    return Dataset.from_list(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B-Base")
    p.add_argument("--bits", type=int, required=True, help="only 4 is currently supported by W4A16 scheme")
    p.add_argument("--groupsize", type=int, default=128)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--duo-scaling", default="both", choices=["both", "true", "false"],
                   help="llm-compressor AWQModifier option. 'both' = try both and pick best (2x cost).")
    p.add_argument("--scheme", default="W4A16_ASYM",
                   help="W4A16 / W4A16_ASYM / W8A16 ... ; asym matches our GPTQ/CMP-Q zero_point=True")
    args = p.parse_args()

    # For W4 we use the named preset string (e.g. "W4A16_ASYM"); for W3/W2 the
    # compressed-tensors preset table has no entry, so we pass a custom
    # QuantizationScheme via `config_groups` — the AWQModifier / QuantizationMixin
    # accepts either `scheme` (preset name or dict of {preset: targets}) or
    # `config_groups` ({group_name: QuantizationScheme}), but not both.
    scheme_arg: object = None
    config_groups_arg: object = None
    if args.bits == 4:
        scheme_arg = args.scheme
        scheme_label = args.scheme
    elif args.bits in (2, 3):
        config_groups_arg = {
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=QuantizationArgs(
                    num_bits=args.bits,
                    type=QuantizationType.INT,
                    strategy=QuantizationStrategy.GROUP,
                    group_size=args.groupsize,
                    symmetric=False,
                    dynamic=False,
                ),
            )
        }
        scheme_label = f"W{args.bits}A16_ASYM(custom)"
    else:
        raise NotImplementedError(f"bits={args.bits} not supported")

    print(f"=== llm-compressor AWQ ({scheme_label}, gs={args.groupsize}, duo_scaling={args.duo_scaling}) ===")
    print(f"Model: {args.model}")
    print(f"Calib: {args.n_samples} C4 samples × {args.seqlen} tokens")
    print(f"Output: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", trust_remote_code=True)

    print("=== Building C4 calibration set ===")
    ds = _build_c4_calib(tokenizer, n_samples=args.n_samples, seqlen=args.seqlen)

    duo_map = {"both": "both", "true": True, "false": False}
    # For MoE models, the router (mlp.gate — a Linear to num_experts) must be
    # kept FP16 by convention (AutoGPTQ / llm-compressor / AWQ). Matching the
    # CMP-Q propagation map (Qwen3MoePropagationMap), which also skips the
    # router. `re:.*\.mlp\.gate$` matches Qwen3MoE / Mixtral router naming
    # and is a no-op on dense models (no linear is named `.mlp.gate`).
    awq_kwargs = dict(
        ignore=["lm_head", r"re:.*\.mlp\.gate$"],
        targets=["Linear"],
        duo_scaling=duo_map[args.duo_scaling],
    )
    if scheme_arg is not None:
        awq_kwargs["scheme"] = scheme_arg
    else:
        awq_kwargs["config_groups"] = config_groups_arg
    recipe = [AWQModifier(**awq_kwargs)]

    print("=== oneshot calibration ===")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.seqlen,
        num_calibration_samples=args.n_samples,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Remove any stale weight files so a newer sharded save can't be shadowed
    # by a lingering single-file model.safetensors (HF load priority issue).
    import glob
    for pattern in ("model.safetensors", "model-*-of-*.safetensors",
                     "model.safetensors.index.json"):
        for stale in glob.glob(str(Path(args.output_dir) / pattern)):
            os.remove(stale)
            print(f"Removed stale weight file: {stale}")
    # save_compressed=False → plain bf16 safetensors, our evaluate.py loads it as-is.
    model.save_pretrained(args.output_dir, save_compressed=False)
    tokenizer.save_pretrained(args.output_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== Done. Saved to {args.output_dir} ===")


if __name__ == "__main__":
    main()
