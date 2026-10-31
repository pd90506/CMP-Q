"""Table of QuIP# vs QuIP# + CMPQ runs (eval_results/<model>_quip_w<b>_lambda<λ>[suffix]_*.json).

ZS = camera-ready mix average: acc_norm for ARC-c/ARC-e/HellaSwag/PIQA, acc for BoolQ/WinoGrande.
Usage: python scripts/quip_report.py [eval_dir] [suffix]
"""
import json
import sys
from pathlib import Path

EVAL = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_results")
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
TASKS = [("arc_challenge", "acc_norm"), ("arc_easy", "acc_norm"), ("boolq", "acc"),
         ("hellaswag", "acc_norm"), ("piqa", "acc_norm"), ("winogrande", "acc")]


def load(prefix: str, kind: str) -> dict | None:
    p = EVAL / f"{prefix}_{kind}.json"
    return json.loads(p.read_text()) if p.exists() else None


print("| model | bits | λ | WT2 | C4 | ARC-c | ARC-e | BoolQ | HS | PIQA | WG | ZS |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for model in ("qwen3-8b-base", "qwen3-14b-base"):
    for bits in (2, 3):
        for lam in ("0", "0.01", "0.1", "0.5", "1"):
            pre = f"{model}_quip_w{bits}_lambda{lam}{SUFFIX}"
            wt, c4, zs = load(pre, "wikitext2"), load(pre, "c4"), load(pre, "zero_shot")
            ppl = [f"{d['results']['ppl']:.3f}" if d else "—" for d in (wt, c4)]
            if zs:
                accs = [100 * zs["results"][t][f"{m},none"] for t, m in TASKS]
                cells = [f"{a:.2f}" for a in accs] + [f"**{sum(accs) / len(accs):.2f}**"]
            else:
                cells = ["—"] * 7
            name = "QuIP#" if lam == "0" else f"+CMPQ {lam}"
            print(f"| {model} | W{bits} | {name} | " + " | ".join(ppl + cells) + " |")
