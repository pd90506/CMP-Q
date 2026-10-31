"""Wall-time / peak-GPU-memory accounting for quantization runs.

Reads clocks and CUDA memory counters. torch.cuda.synchronize() is called before
every timer read so that queued kernels are charged to the phase that issued them.
"""
import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _cuda() -> bool:
    return torch.cuda.is_available()


def _sync() -> None:
    if _cuda():
        torch.cuda.synchronize()


class EfficiencyTracker:
    """Accumulates named wall-time phases and peak GPU memory for one run.

    Phases (seconds):
      model_load_s     load_model() (not part of quantization)
      calib_capture_s  calibration sampling + layer-0 input capture
      ref_forward_s    FP reference forward passes (accumulated over layers;
                       runs INSIDE the layer loop, so also counted in layer_loop_s)
      layer_loop_s     the per-layer quantization loop
    Derived:
      calibration_incl_ref_s = calib_capture_s + ref_forward_s
      total_quant_s          = calib_capture_s + layer_loop_s  (excludes load/save/eval)
    """

    def __init__(self, config: Any, method: str) -> None:
        self.meta: dict[str, Any] = {
            "experiment_name": config.experiment.name,
            "method": method,
            "model": config.model.name,
            "bits": config.quantization.bits,
            "groupsize": config.quantization.groupsize,
            "lambda": config.cmpq.lambda_val if method not in ("gptq", "gptaq") else None,
            "mse_clip": bool(config.quantization.mse_clip) and method != "gptq",
            "seed": config.experiment.seed,
            "n_samples": config.data.n_samples,
            "seq_len": config.model.seq_len,
            "hostname": socket.gethostname(),
            "job_id": os.environ.get("JOB_ID"),
            "torch_version": torch.__version__,
        }
        if _cuda():
            props = torch.cuda.get_device_properties(0)
            self.meta["gpu_name"] = props.name
            self.meta["gpu_mem_total_gb"] = round(props.total_memory / 1024**3, 2)
            torch.cuda.reset_peak_memory_stats()
        else:
            self.meta["gpu_name"] = None
            self.meta["gpu_mem_total_gb"] = None
        self.times: dict[str, float] = {}
        self._open: dict[str, float] = {}

    def start(self, key: str) -> None:
        _sync()
        self._open[key] = time.perf_counter()

    def stop(self, key: str) -> None:
        _sync()
        self.times[key] = self.times.get(key, 0.0) + time.perf_counter() - self._open.pop(key)

    def finish(self) -> dict[str, Any]:
        """Freeze timers and read peak memory. Call before saving the model."""
        _sync()
        t = self.times
        rec = dict(self.meta)
        for k in ("model_load_s", "calib_capture_s", "ref_forward_s", "layer_loop_s"):
            rec[k] = round(t[k], 3) if k in t else None
        rec["calibration_incl_ref_s"] = round(t.get("calib_capture_s", 0.0) + t.get("ref_forward_s", 0.0), 3)
        rec["total_quant_s"] = round(t.get("calib_capture_s", 0.0) + t.get("layer_loop_s", 0.0), 3)
        if _cuda():
            rec["max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            rec["max_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        else:
            rec["max_memory_allocated_gb"] = None
            rec["max_memory_reserved_gb"] = None
        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.record = rec
        print(
            f"EFFICIENCY | exp={rec['experiment_name']} method={rec['method']} "
            f"gpu={rec['gpu_name']} calib={rec['calib_capture_s']}s "
            f"ref_fwd={rec['ref_forward_s']}s loop={rec['layer_loop_s']}s "
            f"total_quant={rec['total_quant_s']}s "
            f"peak_alloc={rec['max_memory_allocated_gb']}GB "
            f"peak_reserved={rec['max_memory_reserved_gb']}GB",
            flush=True,
        )
        return rec

    def write(self, out_dir: Optional[Path] = None) -> None:
        """Write eval_results/<experiment_name>_efficiency.json. Never raises."""
        try:
            out_dir = Path(out_dir) if out_dir else _REPO_ROOT / "eval_results"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{self.record['experiment_name']}_efficiency.json"
            with open(path, "w") as f:
                json.dump(self.record, f, indent=2)
            logger.info(f"Wrote efficiency record to {path}")
        except Exception as e:  # accounting must never fail a quantization run
            logger.warning(f"Failed to write efficiency record: {e}")
