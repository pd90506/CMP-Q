#!/usr/bin/env python3
"""Summarize the ΔX zero-mean diagnostic (rho) across models.

rho = T||mu||^2 / sum_t ||dX_t||^2  is the fraction of tr(V) explained by a
nonzero mean of the upstream noise. rho ~ 0 means the zero-mean assumption
behind Eq. (8) is empirically harmless; rho -> 1 means V is dominated by a
systematic shift and misestimates the covariance.

Reads every diagnostics/*_deltastats.jsonl written by cmpq.pipeline when
`cmpq.delta_stats=true`, and prints per-model / per-node-type summary tables.

Usage:  python scripts/rho_report.py [glob]
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

NODE_TYPES = [
    "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m = re.match(r"L(\d+)\.(.+)", r["name"])
        if not m:
            continue
        r["layer"], r["node"] = int(m.group(1)), m.group(2)
        rows.append(r)
    return rows


def short(model: str) -> str:
    return model.split("/")[-1]


def main(pattern: str) -> None:
    data: dict[str, list[dict]] = {}
    for f in sorted(glob.glob(pattern)):
        rows = load(f)
        if rows:
            data[short(rows[0]["model"])] = rows
    if not data:
        print(f"no diagnostics matched {pattern!r}")
        return

    print("=" * 78)
    print("rho = fraction of tr(V) explained by a NONZERO MEAN   (0 = assumption exact)")
    print("=" * 78)
    print(f"{'model':<24}{'bits':>5}{'nodes':>7}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}")
    for m, rows in data.items():
        rho = sorted(r["rho"] for r in rows)
        p90 = rho[min(int(0.9 * len(rho)), len(rho) - 1)]
        print(f"{m:<24}{rows[0]['bits']:>5}{len(rows):>7}"
              f"{st.median(rho):>9.4f}{st.mean(rho):>9.4f}{p90:>9.4f}{max(rho):>9.4f}")

    print("\n" + "=" * 78)
    print("median rho by NODE TYPE")
    print("=" * 78)
    print(f"{'model':<24}" + "".join(f"{t.split('.')[1].replace('_proj',''):>9}" for t in NODE_TYPES))
    for m, rows in data.items():
        cells = []
        for t in NODE_TYPES:
            sel = [r["rho"] for r in rows if r["node"] == t]
            cells.append(f"{st.median(sel):>9.4f}" if sel else f"{'-':>9}")
        print(f"{m:<24}" + "".join(cells))

    print("\n" + "=" * 78)
    print("median rho by DEPTH quartile")
    print("=" * 78)
    print(f"{'model':<24}{'Q1(early)':>11}{'Q2':>11}{'Q3':>11}{'Q4(late)':>11}")
    for m, rows in data.items():
        n_layers = max(r["layer"] for r in rows) + 1
        cells = []
        for q in range(4):
            sel = [r["rho"] for r in rows
                   if q * n_layers / 4 <= r["layer"] < (q + 1) * n_layers / 4]
            cells.append(f"{st.median(sel):>11.4f}" if sel else f"{'-':>11}")
        print(f"{m:<24}" + "".join(cells))

    incomplete = [m for m, rows in data.items()
                  if len(rows) < 7 * (max(r["layer"] for r in rows) + 1) - 7]
    if incomplete:
        print("\nNOTE: still accumulating (job may be running): " + ", ".join(incomplete))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    main(sys.argv[1] if len(sys.argv) > 1 else str(root / "diagnostics" / "*_deltastats.jsonl"))
