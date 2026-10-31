from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os

@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    seed: int = 42
    method: str = "cmpq"

@dataclass
class ModelConfig:
    name: str
    hf_token: Optional[str] = None
    device_map: str = "auto"
    seq_len: int = 512

@dataclass
class DataConfig:
    dataset_name: str
    subset: str = "en"
    n_samples: int = 1536
    batch_size: int = 1
    num_workers: int = 4

@dataclass
class CalibrationConfig:
    # Replace the first token of every calibration window with BOS, for tokenizers
    # that prepend BOS by default (Llama-2, Mistral, ...). Off by default so old
    # results reproduce. No-op for tokenizers that do not add BOS (Qwen).
    add_bos: bool = False

@dataclass
class QuantizationConfig:
    bits: int = 4
    groupsize: int = 128
    sym: bool = True
    act_order: bool = True
    rht: bool = False   # QuIP#-style random Hadamard incoherence on W/H/V before the CMP-Q solver
    percdamp: float = 0.01   # GPTQ's standard diagonal damping
    mse_clip: bool = False   # per-group MSE clip search in Quantizer.find_params

@dataclass
class CMPQConfig:
    lambda_val: float = 1.0
    delta_stats: bool = False     # ΔX zero-mean (rho) and V-spectrum statistics, written to diagnostics/

@dataclass
class CompressorConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    quantization: QuantizationConfig
    cmpq: CMPQConfig
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    @classmethod
    def load_from_yaml(cls, path: str, overrides: list[str] | None = None) -> 'CompressorConfig':
        """
        Load config from YAML, then apply optional dot-notation overrides.

        Each override is a string of the form ``section.key=value``, e.g.:
            ``quantization.bits=3``
            ``experiment.name=qwen3_8b_w3g64``

        Values are parsed as YAML scalars (so ``true``/``false`` become bools,
        integers stay integers, etc.).
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Resolve ${VAR} tokens in string values using environment variables.
        import re
        def _resolve(value):
            if isinstance(value, str):
                value = re.sub(
                    r'\$\{(\w+)\}',
                    lambda m: os.environ.get(m.group(1), m.group(0)),
                    value,
                )
                if re.fullmatch(r'\$\{\w+\}', value):
                    return None  # unset variable (e.g. no HF token for a public model)
            return value

        model_cfg = raw_config.get('model', {})
        model_cfg = {k: _resolve(v) for k, v in model_cfg.items()}
        raw_config['model'] = model_cfg

        # Apply dot-notation overrides (e.g. "quantization.bits=3")
        if overrides:
            for override in overrides:
                if '=' not in override:
                    raise ValueError(f"Invalid --set override (expected section.key=value): {override!r}")
                dotted_key, _, raw_value = override.partition('=')
                parts = dotted_key.strip().split('.')
                if len(parts) != 2:
                    raise ValueError(f"Override key must be section.key, got: {dotted_key!r}")
                section, key = parts
                value = yaml.safe_load(raw_value)  # parse as YAML scalar
                if section not in raw_config:
                    raw_config[section] = {}
                raw_config[section][key] = value

        cfg = cls(
            experiment=ExperimentConfig(**raw_config.get('experiment', {})),
            model=ModelConfig(**raw_config.get('model', {})),
            data=DataConfig(**raw_config.get('data', {})),
            quantization=QuantizationConfig(**raw_config.get('quantization', {})),
            cmpq=CMPQConfig(**raw_config.get('cmpq', {})),
            calibration=CalibrationConfig(**(raw_config.get('calibration') or {})),
        )
        # Auto-generate experiment name if set to "auto"
        if cfg.experiment.name == "auto":
            cfg.experiment.name = cfg.auto_name()
        return cfg

    def auto_name(self) -> str:
        """Generate experiment name from config: {model}_{method}_w{bits}g{groupsize}[_lambda{val}]"""
        # Short model name: "Qwen/Qwen3-8B" -> "qwen3-8b"
        model_short = self.model.name.split("/")[-1].lower()
        method = self.experiment.method.lower()
        bits = self.quantization.bits
        gs = self.quantization.groupsize
        name = f"{model_short}_{method}_w{bits}g{gs}"
        if method == "quip":
            # QuIP# E8P has no groups. λ=0 is plain QuIP#; λ>0 feeds the CMP-Q
            # Hessian H + λ·(trH/trV)·V into BlockLDLQ.
            name = f"{model_short}_quip_w{bits}_lambda{self.cmpq.lambda_val:g}"
        if method == "gptaq":
            if self.quantization.mse_clip:
                name += "_mseclip"
        if method == "cmpq":
            lv = self.cmpq.lambda_val
            # Format lambda: 1.0 -> "1.0", 0.5 -> "0.5", 0 -> "0"
            name += f"_lambda{lv:g}"
            name += "_trace_full"  # full V, trace-normalized λ
            if self.quantization.mse_clip:
                name += "_mseclip"
        if self.quantization.rht:
            name += "_rht"
        # Calibration dataset suffix (default c4 → no suffix)
        if "hellaswag" in self.data.dataset_name.lower():
            name += "_hswgcal"
        # BOS-first calibration windows: always the last auto_name token, so job
        # suffixes (e.g. _seed42) come after it.
        if self.calibration.add_bos:
            name += "_bos"
        return name
