#!/bin/bash
# reproduce.sh — the experiment sets reported in the paper, run sequentially on one GPU.
#
# Usage:
#   bash scripts/reproduce.sh <set> [--dry-run]
#
# Sets:
#   main        Qwen3-8B/14B-Base, W2/W3 G128: GPTQ, GPTAQ, CMP-Q λ ∈ {0.01, 0.1, 0.5, 1}
#   families    Other model families at W2/W3 G128: GPTQ, GPTAQ, CMP-Q λ=0.1
#   seeds       Seed study, W2G128: CMP-Q λ=0.1 and GPTAQ × seeds {42, 0, 1, 7}
#   efficiency  Wall-clock / peak memory, W3G128: GPTQ, GPTAQ, CMP-Q
#               (results: eval_results/<name>_efficiency.json)
#   rho         ΔX zero-mean statistics (rho) and V spectrum, W2G128 λ=0.1
#               (results: diagnostics/*_deltastats.jsonl; summarize with scripts/rho_report.py)
#   quip        QuIP# (no fine-tuning) vs QuIP# + CMP-Q, Qwen3-8B/14B-Base, 2/3 bits
#               (summarize with scripts/quip_report.py)
#
# Each run is quantize + evaluate (scripts/run.sh). Models are downloaded from the
# HuggingFace Hub; set HF_TOKEN for gated models (Llama, Gemma, Mistral).
set -euo pipefail
cd "$(dirname "$0")/.."

SET="${1:?usage: bash scripts/reproduce.sh <main|families|seeds|efficiency|rho|quip> [--dry-run]}"
DRY_RUN=0
[ "${2:-}" = "--dry-run" ] && DRY_RUN=1

CMPQ=(experiment.method=cmpq quantization.mse_clip=true)
GPTQ=(experiment.method=gptq)                                # baselines/run.py
GPTAQ=(experiment.method=gptaq quantization.mse_clip=true)   # baselines/run.py

run() {  # run [SUFFIX=...] override...
    local suffix=""
    if [[ "${1:-}" == SUFFIX=* ]]; then suffix="${1#SUFFIX=}"; shift; fi
    echo "+ SUFFIX='${suffix}' bash scripts/run.sh $*"
    if [ "${DRY_RUN}" -eq 0 ]; then
        SUFFIX="${suffix}" bash scripts/run.sh "$@"
    fi
}

case "${SET}" in
main)
    for model in Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base; do
        for bits in 3 2; do
            q=(model.name=${model} quantization.bits=${bits})
            run "${q[@]}" "${GPTQ[@]}"
            run "${q[@]}" "${GPTAQ[@]}"
            for lam in 0.01 0.1 0.5 1; do
                run "${q[@]}" "${CMPQ[@]}" cmpq.lambda_val=${lam}
            done
        done
    done ;;
families)
    for model in Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407 google/gemma-2-9b mistralai/Mistral-7B-v0.3; do
        for bits in 2 3; do
            q=(model.name=${model} quantization.bits=${bits})
            run "${q[@]}" "${GPTQ[@]}"
            run "${q[@]}" "${GPTAQ[@]}"
            run "${q[@]}" "${CMPQ[@]}" cmpq.lambda_val=0.1
        done
    done ;;
seeds)
    # Mistral-Nemo prepends BOS; add calibration.add_bos=true to use BOS-first calibration windows.
    for model in Qwen/Qwen2.5-7B mistralai/Mistral-Nemo-Base-2407; do
        for s in 42 0 1 7; do
            q=(model.name=${model} quantization.bits=2 experiment.seed=${s})
            run SUFFIX=_seed${s} "${q[@]}" "${CMPQ[@]}" cmpq.lambda_val=0.1
            run SUFFIX=_seed${s} "${q[@]}" "${GPTAQ[@]}"
        done
    done ;;
efficiency)
    # Measure with nothing else running on the GPU.
    for row in "Qwen/Qwen3-8B-Base 1" "Qwen/Qwen3-14B-Base 0.5"; do
        read -r model lam <<< "${row}"
        q=(model.name=${model} quantization.bits=3)
        run SUFFIX=_eff "${q[@]}" "${GPTQ[@]}"
        run SUFFIX=_eff "${q[@]}" "${GPTAQ[@]}"
        run SUFFIX=_eff "${q[@]}" "${CMPQ[@]}" cmpq.lambda_val=${lam}
    done ;;
rho)
    for model in Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base meta-llama/Llama-3.1-8B mistralai/Mistral-7B-v0.3; do
        SKIP_ZS=1 run model.name=${model} quantization.bits=2 "${CMPQ[@]}" cmpq.lambda_val=0.1 cmpq.delta_stats=true
    done
    [ "${DRY_RUN}" -eq 1 ] || python scripts/rho_report.py ;;
quip)
    for model in Qwen/Qwen3-8B-Base Qwen/Qwen3-14B-Base; do
        for bits in 2 3; do
            for lam in 0 0.01 0.1 0.5 1; do
                # λ=0 is plain QuIP#; λ>0 feeds H + λ·(tr H / tr V)·V into QuIP#'s BlockLDLQ.
                run model.name=${model} experiment.method=quip quantization.bits=${bits} cmpq.lambda_val=${lam}
            done
        done
    done
    [ "${DRY_RUN}" -eq 1 ] || python scripts/quip_report.py eval_results ;;
*)
    echo "unknown set: ${SET}" >&2; exit 2 ;;
esac
