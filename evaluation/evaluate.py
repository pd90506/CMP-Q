"""
Main evaluation script for CMP-Q quantized models.

Supports:
- Perplexity: WikiText-2, C4
- Zero-shot: ARC-E, ARC-C, PIQA, WinoGrande, HellaSwag, MMLU (via lm-evaluation-harness)
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation.perplexity import evaluate_wikitext2, evaluate_c4, load_model_and_tokenizer


def evaluate_perplexity_benchmarks(
    model_path: str,
    benchmarks: list[str],
    seq_len: int = 2048,
    device_map: str = "auto",
):
    """
    Evaluate perplexity on multiple benchmarks.

    Args:
        model_path: Path to model
        benchmarks: List of benchmark names ('wikitext2', 'c4')
        seq_len: Sequence length
        device_map: Device map
    """
    # Load model once for all benchmarks
    print(f"Loading model: {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path, device_map=device_map)

    results = {}

    for bench in benchmarks:
        if bench.lower() in ("wikitext2", "wikitext"):
            print(f"\n{'='*60}")
            print(f"Evaluating on WikiText-2")
            print(f"{'='*60}")
            ppl, tokens = evaluate_wikitext2(model, tokenizer, seq_len=seq_len)
            results["wikitext2"] = {"ppl": ppl, "tokens": tokens}

        elif bench.lower() in ("c4", "c4-val", "allenai/c4"):
            print(f"\n{'='*60}")
            print(f"Evaluating on C4 (validation, 256 samples)")
            print(f"{'='*60}")
            ppl, tokens = evaluate_c4(model, tokenizer, seq_len=seq_len)
            results["c4"] = {"ppl": ppl, "tokens": tokens}

    return results


def print_results(model_path: str, results: dict):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_path}")
    print(f"{'='*60}")
    for bench, metrics in results.items():
        print(f"\n{bench.upper()}:")
        print(f"  Perplexity: {metrics['ppl']:.4f}")
        print(f"  Tokens: {metrics['tokens']:,}")
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CMP-Q quantized models on benchmarks"
    )
    parser.add_argument(
        "model_path",
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["wikitext2"],
        choices=["wikitext2", "c4"],
        help="Benchmarks to evaluate (default: wikitext2)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map strategy (default: auto)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to experiment config YAML (used to derive experiment name for output filename)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Explicit output JSON path (overrides auto-naming; only for single benchmark)",
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
        help="Prefix for output filenames (e.g. 'qwen3-8b-base_fp16' → qwen3-8b-base_fp16_wikitext2.json)",
    )
    args = parser.parse_args()

    results = evaluate_perplexity_benchmarks(
        model_path=args.model_path,
        benchmarks=args.benchmarks,
        seq_len=args.seq_len,
        device_map=args.device_map,
    )

    print_results(args.model_path, results)

    # Determine experiment name for auto-naming
    experiment_name = args.output_prefix
    if experiment_name is None and args.config:
        from cmpq.core.config import CompressorConfig
        cfg = CompressorConfig.load_from_yaml(str(args.config))
        experiment_name = cfg.experiment.name

    import json

    for bench in args.benchmarks:
        # Determine output path for this benchmark
        if args.output and len(args.benchmarks) == 1:
            out_path = args.output
        elif experiment_name:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / f"{experiment_name}_{bench}.json"
        else:
            continue  # no config, no --output: skip saving

        output_data = {
            "model_path": args.model_path,
            "experiment_name": experiment_name,
            "benchmark": bench,
            "results": results.get(bench, {}),
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
