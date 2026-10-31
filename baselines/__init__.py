"""
Baselines.

gptq.py / gptaq.py : GPTQ and GPTAQ solvers;   run.py : their layer-wise runner
    python -m baselines.run --method gptq|gptaq [--set section.key=value ...]
awq/               : AWQ via vLLM's llm-compressor

QuIP# (no fine-tuning) runs in the CMP-Q pipeline (main.py, experiment.method=quip,
cmpq.lambda_val=0) because QuIP# + CMP-Q shares its code path.
"""
