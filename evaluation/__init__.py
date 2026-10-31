"""
CMP-Q Evaluation Package

Provides evaluation utilities for quantized models:
- Perplexity on WikiText-2, C4
- Zero-shot benchmarks via lm-evaluation-harness

Usage:
    # Perplexity evaluation
    python -m evaluation.evaluate /path/to/model --benchmarks wikitext2 c4

    # Direct perplexity script
    python evaluation/perplexity.py /path/to/model --dataset wikitext --subset wikitext-2-v1
"""
