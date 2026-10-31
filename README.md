# CMP-Q: Compensatory Quantization via Input-Aware Hessian Damping

Code for the NeurIPS 2026 paper *CMPQ: Compensatory Quantization via Input-Aware Hessian Damping*
by Deng Pan, Grigorii Khvatskii, Xiaobao Huang, Peiyu Li, Shangqian Gao, Ting Hua and Nitesh V. Chawla
(University of Notre Dame, Florida State University).

CMP-Q is a post-training, weight-only quantization method for LLMs. It keeps GPTQ's
column-wise solver unchanged and modifies only the Hessian it inverts. When a layer is
quantized, its input already carries the error of every layer quantized before it.
CMP-Q measures that upstream noise and adds it to the Hessian as a damping term, so
error compensation is steered away from input directions dominated by noise.

## Method

For a linear layer with calibration inputs `X` (from the already-quantized model)
and the same inputs `X_ref` from the full-precision model:

```
ΔX = X − X_ref                           upstream quantization noise (cumulative)
H  = (2/N) · X  Xᵀ                        GPTQ Hessian
V  = (2/N) · ΔX ΔXᵀ                       noise covariance
Ĥ  = H + λ · (tr H / tr V) · V             CMP-Q Hessian (full V, trace normalization)
```

`Ĥ` replaces `H` before the Cholesky factorization; the rest of GPTQ (column order,
groups, error propagation) is untouched. With trace normalization, `λ` is the damping
energy as a fraction of `tr H`, so it transfers across layers, models and bit-widths.
**With λ = 0, CMP-Q is exactly GPTQ** (`tests/test_full_v.py` checks this bit for bit).

CMP-Q results use per-group MSE clipping in the quantizer (`quantization.mse_clip=true`),
and so does the GPTAQ baseline.

## Installation

```bash
git clone <this repo> && cd cmpq
conda create -n cmpq python=3.11 -y && conda activate cmpq
pip install -r requirements.txt
```

Python ≥ 3.10 and PyTorch ≥ 2.1. A CUDA GPU is needed for 7B+ models (one 48 GB GPU
fits up to ~14B); the code also runs on Apple MPS or CPU for small models. Set
`HF_TOKEN` (in the environment or a `.env` file) for gated models.

## Quick start

Smoke test on Qwen3-0.6B (a few minutes on a laptop):

```bash
python main.py --config config/debug_config.yaml
```

Quantize and evaluate one model (WikiText-2 / C4 perplexity and six zero-shot tasks:
ARC-c, ARC-e, BoolQ, HellaSwag, PIQA, WinoGrande):

```bash
bash scripts/run.sh model.name=Qwen/Qwen3-8B-Base quantization.bits=2 cmpq.lambda_val=0.1
```

The quantized model (fake-quantized, saved in the original dtype) goes to
`experiments/<name>/models/compressed_model`, and result JSONs to `eval_results/`.
`<name>` is derived from the config, e.g. `qwen3-8b-base_cmpq_w2g128_lambda0.1_trace_full_mseclip`.

## Configuration

`config/config.yaml` holds the CMP-Q recipe; any field can be overridden with
`--set section.key=value` (in `main.py` and `baselines/run.py`) or as plain arguments
(in `scripts/run.sh`).

| key | values | meaning |
|---|---|---|
| `experiment.method` | `cmpq` · `quip` · `gptq` · `gptaq` | `cmpq`/`quip`: `main.py`; `gptq`/`gptaq`: `baselines/run.py` |
| `quantization.bits`, `.groupsize` | e.g. `2`/`3`/`4`, `128` | weight bit-width and group size |
| `quantization.percdamp` | default `0.01` | GPTQ's standard diagonal damping |
| `quantization.mse_clip` | `true` · `false` | per-group MSE clip search (CMP-Q, GPTAQ) |
| `quantization.rht` | `true` · `false` | QuIP#-style random Hadamard rotation of W, H and V (CMP-Q) |
| `cmpq.lambda_val` | e.g. `0.01`–`1` | damping strength λ (fraction of tr H) |
| `calibration.add_bos` | `true` · `false` | start each calibration window with BOS |
| `cmpq.delta_stats` | `true` · `false` | log ΔX mean statistics (ρ) to `diagnostics/` |
| `experiment.seed` | int | calibration sampling and rotation seed |

Calibration defaults to 128 random 2048-token windows from C4.

## Reproducing the paper

`scripts/reproduce.sh` runs each experiment set sequentially (add `--dry-run` to list
the runs first):

| set | what it runs |
|---|---|
| `main` | Qwen3-8B/14B-Base, W2/W3: GPTQ, GPTAQ, CMP-Q λ ∈ {0.01, 0.1, 0.5, 1} |
| `families` | Qwen2.5-7B, Mistral-Nemo-12B, Gemma-2-9B, Mistral-7B-v0.3 at W2/W3 |
| `seeds` | W2 seed study (seeds 42, 0, 1, 7) for CMP-Q and GPTAQ |
| `efficiency` | wall-clock and peak GPU memory of GPTQ, GPTAQ and CMP-Q |
| `rho` | how much of tr(V) comes from a nonzero noise mean (ρ) |
| `quip` | QuIP# (no fine-tuning) with and without the CMP-Q Hessian |

```bash
bash scripts/reproduce.sh main --dry-run
bash scripts/reproduce.sh main
```

## Baselines

GPTQ and GPTAQ live in `baselines/` with their own solvers and layer-wise runner, and use
the same calibration data as CMP-Q:

```bash
python -m baselines.run --method gptq  --set model.name=Qwen/Qwen3-8B-Base quantization.bits=3
python -m baselines.run --method gptaq --set model.name=Qwen/Qwen3-8B-Base quantization.bits=2
```

QuIP# (E8P codebook, BlockLDLQ, no fine-tuning) is `main.py` with `experiment.method=quip`
and `cmpq.lambda_val=0`; with λ > 0 the same run uses the CMP-Q Hessian. AWQ uses vLLM's
llm-compressor through `baselines/awq/run_awq_compressor.py` (`pip install llmcompressor`).

## Repository layout

```
main.py                       CMP-Q (and QuIP#) entry point
config/                       default and debug configs
cmpq/
  pipeline.py                 layer-by-layer CMP-Q pipeline: FP16 and quantized chains, ΔX
  core/config.py              config dataclasses and experiment naming
  core/propagation_map.py     order of linear layers inside a decoder block
  utils/gptq_rev.py           CMP-Q solver (H, V, Ĥ, GPTQ column loop) and ρ statistics
  utils/quip_sharp.py         QuIP# (E8P, BlockLDLQ, random Hadamard) solver
  utils/quantization.py       uniform quantizer with MSE clip search
  utils/efficiency.py         timing and peak-memory tracking
  utils/model_loader.py       model and calibration-data loading
evaluation/                   perplexity (WikiText-2, C4) and lm-eval zero-shot
baselines/
  run.py                      GPTQ / GPTAQ entry point (layer-wise runner)
  gptq.py, gptaq.py           GPTQ and GPTAQ solvers
  awq/                        AWQ runner (llm-compressor)
scripts/                      run.sh, reproduce.sh, rho_report.py, quip_report.py
tests/                        unit tests (pytest)
```

Each layer's inputs are collected with its predecessors in the same block already
quantized, so the Hessian matches what the deployed model sees.

## Tests

```bash
pytest tests
```

## License

[MIT](LICENSE)

## Citation

```bibtex
@inproceedings{pan2026cmpq,
  title     = {CMPQ: Compensatory Quantization via Input-Aware Hessian Damping},
  author    = {Pan, Deng and Khvatskii, Grigorii and Huang, Xiaobao and Li, Peiyu and
               Gao, Shangqian and Hua, Ting and Chawla, Nitesh V.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
