#!/bin/bash
# run.sh — quantize one model, then evaluate it (WikiText-2 / C4 perplexity + 6-task zero-shot).
# experiment.method=cmpq|quip runs main.py; experiment.method=gptq|gptaq runs baselines/run.py.
#
# Usage:
#   bash scripts/run.sh [section.key=value ...]
#
# Every argument is a --set override for config/config.yaml (primary CMP-Q recipe:
# W3G128, full V, trace normalization, λ=0.1, MSE clipping). Examples:
#   bash scripts/run.sh model.name=Qwen/Qwen3-8B-Base quantization.bits=2 cmpq.lambda_val=0.1
#   bash scripts/run.sh experiment.method=gptq
#   bash scripts/run.sh experiment.method=gptaq quantization.bits=2
#
# Environment:
#   CONFIG    config file (default: config/config.yaml)
#   SUFFIX    appended to the auto-generated experiment name (e.g. _seed0)
#   EXP_DIR   where quantized models are written (default: ./experiments)
#   EVAL_DIR  where result JSONs are written (default: ./eval_results)
#   SKIP_ZS=1 skip the zero-shot evaluation
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-config/config.yaml}"
EXP_DIR="${EXP_DIR:-./experiments}"
EVAL_DIR="${EVAL_DIR:-./eval_results}"
PYTHON="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

OVERRIDES=("experiment.output_dir=${EXP_DIR}" "$@")

# Resolve the method and the experiment name exactly as the entry point would, then append SUFFIX.
read -r METHOD NAME < <("${PYTHON}" - "${CONFIG}" "${OVERRIDES[@]}" <<'EOF'
import sys
from cmpq.core.config import CompressorConfig
cfg = CompressorConfig.load_from_yaml(sys.argv[1], overrides=sys.argv[2:])
print(cfg.experiment.method, cfg.experiment.name)
EOF
)
[ -n "${NAME:-}" ] || { echo "could not resolve the config" >&2; exit 1; }
NAME="${NAME}${SUFFIX:-}"
MODEL_PATH="${EXP_DIR}/${NAME}/models/compressed_model"

echo "=== Quantize: ${NAME} ($(date)) ==="
case "${METHOD}" in
    gptq|gptaq) "${PYTHON}" -m baselines.run --method "${METHOD}" --config "${CONFIG}" \
                    --set "${OVERRIDES[@]}" "experiment.name=${NAME}" ;;
    *)          "${PYTHON}" main.py --config "${CONFIG}" --set "${OVERRIDES[@]}" "experiment.name=${NAME}" ;;
esac

echo "=== Perplexity: ${NAME} ($(date)) ==="
"${PYTHON}" -m evaluation.evaluate "${MODEL_PATH}" \
    --benchmarks wikitext2 c4 --output-dir "${EVAL_DIR}" --output-prefix "${NAME}"

if [ "${SKIP_ZS:-0}" != 1 ]; then
    echo "=== Zero-shot: ${NAME} ($(date)) ==="
    "${PYTHON}" evaluation/zero_shot.py "${MODEL_PATH}" \
        --output-dir "${EVAL_DIR}" --output-prefix "${NAME}"
fi
echo "=== Done: ${NAME} ($(date)) ==="
