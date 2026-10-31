"""
Zero-shot evaluation using lm-evaluation-harness.

Supports: PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, BoolQ, MMLU

Usage:
    python evaluation/zero_shot.py /path/to/model --tasks piqa arc_easy arc_challenge hellaswag winogrande boolq
"""
import argparse
import sys
from pathlib import Path

try:
    from lm_eval import evaluator
except ImportError:
    print("Error: lm-evaluation-harness not installed.")
    print("Install with: pip install lm-eval")
    sys.exit(1)


def evaluate_zero_shot(
    model_path: str,
    tasks: list[str],
    num_fewshot: int = 0,
    device_map: str = "auto",
    dtype: str = "bfloat16",
    batch_size: int | str = "auto",
    add_bos_token: bool = False,
) -> dict:
    """
    Evaluate model on zero-shot benchmarks via lm-evaluation-harness.

    Args:
        model_path: Path to model or HuggingFace model name
        tasks: List of task names (e.g., ['arc_easy', 'arc_challenge', 'piqa'])
        num_fewshot: Number of few-shot examples (0 for zero-shot)
        device_map: Device map strategy for model loading
        dtype: Model dtype string (e.g. 'bfloat16', 'float16')
        batch_size: Batch size for evaluation ('auto' or int)

    Returns:
        Dictionary of results from lm-evaluation-harness
    """
    if isinstance(tasks, str):
        tasks = [tasks]

    model_args = f"pretrained={model_path},trust_remote_code=True,dtype={dtype},device_map={device_map}"
    if add_bos_token:
        # LLaMA/Mistral-style models rely on the first token as an attention sink.
        # lm-eval omits BOS by default, which quantized checkpoints tolerate far
        # worse than their BF16 counterparts.
        model_args += ",add_bos_token=True"

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )

    return results


def print_results(results: dict):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print("Zero-Shot Evaluation Results")
    print(f"{'='*60}")

    if "results" in results:
        for task, metrics in results["results"].items():
            print(f"\n{task}:")
            for key, val in sorted(metrics.items()):
                if key.endswith(",none"):
                    short_key = key.replace(",none", "")
                    if isinstance(val, float):
                        print(f"  {short_key}: {val:.4f}")
                    else:
                        print(f"  {short_key}: {val}")

    print(f"\n{'='*60}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate models on zero-shot benchmarks"
    )
    parser.add_argument(
        "model_path",
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"],
        help="Tasks to evaluate (default: piqa arc_easy arc_challenge hellaswag winogrande boolq)",
    )
    parser.add_argument(
        "--num-fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples (default: 0 for zero-shot)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map strategy (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--add-bos-token",
        action="store_true",
        help="Prepend BOS to each request (LLaMA/Mistral attention-sink behaviour)",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size — 'auto' or integer (default: auto)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save results to JSON file (explicit path)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory for auto-named results (default: eval_results/)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        help="Prefix for output filename (e.g. 'qwen3-8b-gptq-w3g128' -> qwen3-8b-gptq-w3g128_zero_shot.json)",
    )
    args = parser.parse_args()

    batch_size = args.batch_size if args.batch_size == "auto" else int(args.batch_size)

    results = evaluate_zero_shot(
        model_path=args.model_path,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        device_map=args.device_map,
        dtype=args.dtype,
        batch_size=batch_size,
        add_bos_token=args.add_bos_token,
    )

    print_results(results)

    out_path = args.output
    if out_path is None and args.output_prefix:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = args.output_dir / f"{args.output_prefix}_zero_shot.json"

    if out_path:
        import json
        serializable_results = {
            "model_path": args.model_path,
            "experiment_name": args.output_prefix or args.model_path,
            "benchmark": "zero_shot",
            "tasks": args.tasks,
            "num_fewshot": args.num_fewshot,
            "results": results.get("results", {}),
            "versions": results.get("versions", {}),
            "config": results.get("config", {}),
        }
        with open(out_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
