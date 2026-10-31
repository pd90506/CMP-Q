"""
CMP-Q revised GPTQ solver.

Extends the standard GPTQ Hessian with an input-aware damping term derived
from upstream quantization noise. When the damping is disabled (V None or
cmpq_lambda=0), this reduces to standard GPTQ exactly.

    Ĥ = H + λ·(tr(H)/tr(V))·V,   H = (2/N)·X·Xᵀ,   V = (2/N)·ΔX·ΔXᵀ
Convention: λ = tr(damping)/tr(H) exactly.

GPTQ column-wise loop is unchanged; only the pre-Cholesky H_hat differs.
    δ_F = -(w_q - quant(w_q)) / [Ĥ⁻¹]_{qq} · (Ĥ⁻¹)_{:,q}
"""
import math
import torch
import torch.nn as nn
import logging
from .quantization import Quantizer
from .device import synchronize, empty_cache

logger = logging.getLogger(__name__)


class GPTQREV:
    def __init__(self, layer, name: str, wbits: int):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.V: torch.Tensor | None = None        # full-mode only, lazily allocated
        self.delta_nsamples = 0
        # Zero-mean statistics of ΔX (for rho).
        self.delta_sum: torch.Tensor | None = None   # Σ_t δ_t   ∈ R^{d_in}
        self.delta_sq_per_ch: torch.Tensor | None = None  # Σ_t δ_t²  (per channel)
        self.delta_sq_sum = 0.0                      # Σ_t ‖δ_t‖²
        self.delta_tokens = 0                        # T
        self.quantizer = Quantizer()
        self.name = name
        self.wbits = wbits

    def add_batch(self, inp: torch.Tensor) -> None:
        """Accumulate Hessian H = 2·X·Xᵀ from a batch of inputs."""
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))

        inp = inp.t()  # [d_in, n_tokens]

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def add_batch_delta(self, delta: torch.Tensor) -> None:
        """Accumulate full V = 2·ΔX·ΔXᵀ streamingly (full mode only).

        Mirrors `add_batch` exactly so the 2/N normalization matches H.
        """
        if delta.dim() == 2:
            delta = delta.unsqueeze(0)
        tmp = delta.shape[0]
        if delta.dim() == 3:
            delta = delta.reshape((-1, delta.shape[-1]))

        delta = delta.t()  # [d_in, n_tokens]

        # --- zero-mean statistics of ΔX (for rho) ---
        d32 = delta.float()
        acc = torch.float32 if self.dev.type == "mps" else torch.float64  # MPS has no float64
        if self.delta_sum is None:
            self.delta_sum = torch.zeros(self.columns, device=self.dev, dtype=acc)
        if self.delta_sq_per_ch is None:
            self.delta_sq_per_ch = torch.zeros(self.columns, device=self.dev, dtype=acc)
        self.delta_sum += d32.sum(dim=1).to(acc)
        self.delta_sq_per_ch += d32.pow(2).sum(dim=1).to(acc)
        self.delta_sq_sum += float(d32.pow(2).sum())
        self.delta_tokens += d32.shape[1]
        del d32
        # -------------------------------------------------------------------------

        if self.V is None:
            self.V = torch.zeros((self.columns, self.columns), device=self.dev)

        self.V *= self.delta_nsamples / (self.delta_nsamples + tmp)
        self.delta_nsamples += tmp

        delta = math.sqrt(2 / self.delta_nsamples) * delta.float()
        self.V += delta.matmul(delta.t())

    def delta_stats(self) -> dict | None:
        """Zero-mean diagnostics for the upstream noise ΔX (Eq. 8).

        The EIV model assumes ξ is zero-mean, which makes V = ΔX·ΔXᵀ the plug-in
        estimator of its covariance. If the mean is not zero, V conflates a
        systematic shift with genuine covariance. Since

            tr(ΔX·ΔXᵀ) = T‖μ‖² + tr(centered second moment),   μ = (1/T)Σ_t δ_t,

        the quantity `rho = T‖μ‖² / Σ_t‖δ_t‖²` is exactly the fraction of the
        damping term's trace that is explained by a nonzero mean. rho ≈ 0 means
        the zero-mean assumption is empirically harmless; rho → 1 means V is
        dominated by a systematic shift and misestimates the covariance.

        Returns None when no ΔX was observed (e.g. the first block, where the
        quantized and FP16 chains still coincide).
        """
        if self.delta_sum is None or self.delta_tokens == 0 or self.delta_sq_sum == 0.0:
            return None
        T = self.delta_tokens
        mu = self.delta_sum / T
        rho = float(self.delta_sum.pow(2).sum()) / (T * self.delta_sq_sum)
        # Per-channel version: fraction of each channel's V-diagonal due to the mean.
        per_ch = (T * mu.pow(2)) / self.delta_sq_per_ch.clamp_min(1e-30)
        q = torch.tensor([0.5, 0.95], device=per_ch.device, dtype=per_ch.dtype)
        med, p95 = torch.quantile(per_ch, q).tolist()
        return {
            "name": self.name,
            "d_in": int(self.columns),
            "tokens": int(T),
            "rho": rho,                                   # global bias fraction of tr(V)
            "rho_ch_median": med,                         # per-channel bias fraction
            "rho_ch_p95": p95,
            "delta_rms": (self.delta_sq_sum / (T * self.columns)) ** 0.5,
            "mu_rms": float(mu.pow(2).mean()) ** 0.5,
        }

    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        V_matrix: torch.Tensor | None = None,
        cmpq_lambda: float = 1.0,
        mse_clip: bool = False,
    ):
        """
        Quantize layer weights using CMP-Q modified GPTQ.

        Args:
            V_matrix: Noise covariance V ∈ R^{d_in × d_in}, typically gptq.V after
                      streaming accumulation via add_batch_delta. If None or
                      cmpq_lambda=0, behaves identically to standard GPTQ.
            cmpq_lambda: Damping strength λ (fraction of tr(H)).
            mse_clip: Per-group MSE clip search in Quantizer.find_params.
        """
        self.layer.to(self.dev)
        W = self.layer.weight.data.clone().float()

        self.quantizer.configure(self.wbits, mse_clip=mse_clip)
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = getattr(self, "H", None)
        if H is None:
            logger.warning(f"{self.name}: Hessian not available (already freed?), skipping")
            return None, None, None, None
        if hasattr(self, "H"):
            del self.H

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            if V_matrix is not None:
                V_matrix = V_matrix[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        diag = torch.arange(self.columns, device=self.dev)

        # Trace ratio r = tr(H)/tr(V), computed BEFORE percdamp (from raw Hessian)
        trace_ratio = 1.0
        if V_matrix is not None and cmpq_lambda != 0.0:
            assert V_matrix.shape == (self.columns, self.columns), (
                f"V_matrix shape mismatch: got {tuple(V_matrix.shape)}, "
                f"expect ({self.columns}, {self.columns})"
            )
            V_matrix = V_matrix.to(self.dev)
            tr_V = torch.diag(V_matrix).sum().clamp_min(1e-12)
            trace_ratio = (torch.diag(H).sum() / tr_V).item()

        # Standard GPTQ percdamp
        damp = percdamp * torch.mean(torch.diag(H))
        H[diag, diag] += damp

        # CMP-Q: add λ·(tr(H)/tr(V))·V before Cholesky
        # (clean convention: λ = damping_trace / H_trace exactly)
        if V_matrix is not None and cmpq_lambda != 0.0:
            H = H + (cmpq_lambda * trace_ratio) * V_matrix

        # Cholesky — MPS can fail on near-rank-deficient matrices; retry on CPU.
        def _cholesky_inv(mat: torch.Tensor) -> torch.Tensor:
            L = torch.linalg.cholesky(mat)
            return torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)

        try:
            Hinv = _cholesky_inv(H)
        except RuntimeError:
            if H.device.type == "mps":
                logger.warning(f"Cholesky failed on MPS for {self.name}, retrying on CPU")
                try:
                    Hinv = _cholesky_inv(H.cpu()).to(H.device)
                except RuntimeError:
                    logger.warning(f"Hessian not positive definite for {self.name}, using RTN fallback")
                    return None, None, None, None
            else:
                logger.warning(f"Hessian not positive definite for {self.name}, using RTN fallback")
                return None, None, None, None

        scale = []
        zero = []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1

                q_r = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q_r

                Losses1[:, i] = (w - q_r) ** 2 / d ** 2
                err1 = (w - q_r) / d

                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = torch.tensor(
            [i // groupsize for i in range(self.columns)],
            dtype=torch.int32,
            device=Q.device,
        )

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        logger.info(f"[CMP-Q] Quantized {self.name} to {self.wbits} bits. Error: {error:.4f}")

        if not scale:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        return torch.cat(scale, dim=1), torch.cat(zero, dim=1), g_idx, error

    def free(self) -> None:
        # Idempotent: safe to call after fasterquant() (which already del's self.H)
        # and safe to call before fasterquant() (H still exists).
        if hasattr(self, "H") and self.H is not None:
            del self.H
        if getattr(self, "V", None) is not None:
            del self.V
            self.V = None
        # ΔX diagnostics: drop the d_in-sized accumulators (stats already extracted).
        self.delta_sum = None
        self.delta_sq_per_ch = None
        empty_cache()
